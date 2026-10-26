"""One optimization round on the dev machine: judge, commit, submit, gate, log, accept or revert."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import Attempt, ExperimentLog
from kernel_agent_loop.core.gate import AcceptanceGate, Decision
from kernel_agent_loop.core.static_judge import StaticJudge
from kernel_agent_loop.core.types import EvalResult, Verdict
from kernel_agent_loop.evald.bundle import make_bundle
from kernel_agent_loop.evald.diagnose import RULES
from kernel_agent_loop.evald.result_codec import per_shape_to_dict


class AlreadyTried(Exception):
    pass


@dataclass(frozen=True)
class RoundOutcome:
    commit: str
    decision: Decision
    result: EvalResult | None


def run_round(
    campaign: Campaign,
    *,
    client,
    log: ExperimentLog,
    gate: AcceptanceGate,
    lever: str,
    hypothesis: str,
    target: str = "",
) -> RoundOutcome:
    """Evaluate the working-tree impl against campaign.best and apply the verdict.

    The incumbent shipped in the bundle is read from git at campaign.best, never from the agent's files.
    INVALID leaves the working tree as is so the same candidate can be resubmitted.
    `target` names the symptom code (evald.diagnose.RULES) the lever is aimed at, for tuning statistics.
    """
    if target and target not in RULES:
        raise ValueError(f"unknown symptom code {target!r}; known: {', '.join(RULES)}")
    if log.tried(campaign.campaign_id, lever=lever, parent=campaign.best):
        raise AlreadyTried(f"lever {lever!r} already tried on {campaign.best[:12]}")
    parent = campaign.best
    judged = StaticJudge(campaign.op, campaign.backend).check(_changed_files(campaign))
    sha = campaign.commit_candidate(lever=lever, hypothesis=hypothesis)
    if not judged.ok:
        decision = Decision(Verdict.REJECT, judged.reasons)
        _record(log, campaign, sha, parent, lever, hypothesis, target, decision, judged.reasons, None)
        campaign.revert()
        return RoundOutcome(sha, decision, None)

    meta = {
        "campaign": campaign.campaign_id,
        "commit": sha,
        "parent": parent,
        "op": campaign.op,
        "backend": campaign.backend,
        "lever": lever,
        "hypothesis": hypothesis,
    }
    bundle = make_bundle(
        campaign.repo, campaign.op, campaign.backend, meta, incumbent=campaign.impl_files(parent)
    )
    result = client.submit_sync(bundle)
    decision = gate.decide(result)
    _record(log, campaign, sha, parent, lever, hypothesis, target, decision, result.symptoms, result)
    if decision.verdict is Verdict.ACCEPT:
        campaign.accept(sha)
    elif decision.verdict is not Verdict.INVALID:
        campaign.revert()
    return RoundOutcome(sha, decision, result)


def _changed_files(campaign: Campaign) -> dict[str, str]:
    impl = Path(campaign.repo) / "ops" / campaign.op / "impls" / campaign.backend
    return {
        path.relative_to(campaign.repo).as_posix(): path.read_text(errors="replace")
        for path in sorted(impl.rglob("*"))
        if path.is_file()
    }


def _record(log, campaign, sha, parent, lever, hypothesis, target, decision, symptoms, result) -> None:
    per_shape = per_shape_to_dict(result.per_shape) if result is not None else {}
    log.record(
        Attempt(
            campaign=campaign.campaign_id,
            commit=sha,
            parent=parent,
            lever=lever,
            hypothesis=hypothesis,
            verdict=decision.verdict.value,
            symptoms=tuple(symptoms) + tuple(r for r in decision.reasons if r not in symptoms),
            per_shape=per_shape,
            target=target,
            profiled=bool(result is not None and result.profile),
        )
    )
