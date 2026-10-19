"""Example: submit a candidate bundle to thor-003 via ssh nvidia@tz -p 2240."""

from pathlib import Path

from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import Attempt, ExperimentLog
from kernel_agent_loop.core.gate import AcceptanceGate
from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.static_judge import StaticJudge
from kernel_agent_loop.core.types import Verdict
from kernel_agent_loop.evald.bundle import make_bundle
from kernel_agent_loop.evald.client import EvalClient
from kernel_agent_loop.evald.env_guard import EnvGuard

REPO = Path(__file__).resolve().parents[2]
REMOTE_CONFIG = "/home/nvidia/kal/evald.config.json"
EVALD_CMD = "/home/nvidia/kal/venv/bin/python -m kernel_agent_loop.evald.serve"


def main() -> None:
    campaign = Campaign.load(REPO, "rms-thor")  # or Campaign.create(...)
    lever, hypothesis = "naive", "thor smoke"
    sha = campaign.commit_candidate(lever=lever, hypothesis=hypothesis)
    bundle = make_bundle(
        REPO,
        campaign.op,
        campaign.backend,
        {
            "campaign": campaign.campaign_id,
            "commit": sha,
            "parent": campaign.best,
            "op": campaign.op,
            "backend": campaign.backend,
            "lever": lever,
            "hypothesis": hypothesis,
        },
    )
    client = EvalClient(
        mode="ssh",
        host="tz",
        user="nvidia",
        port=2240,
        remote_config=REMOTE_CONFIG,
        evald_cmd=EVALD_CMD,
        remote_workdir="/home/nvidia/kal/inbox",
    )
    result = client.submit_sync(bundle)
    decision = AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0).decide(result)
    print(decision)
    if decision.verdict is Verdict.ACCEPT:
        campaign.accept(sha)
    elif decision.verdict is not Verdict.INVALID:
        campaign.revert()


if __name__ == "__main__":
    main()
