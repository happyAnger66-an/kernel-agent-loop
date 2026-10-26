"""Rank levers per symptom from the experiment log: which fixes actually worked, seeded by the rule table.

Each attempt that named a `target` symptom counts toward that (symptom, lever) pair. Ranking uses the
Laplace-smoothed accept rate (accepts + 1) / (tries + 2), so an untried seed lever scores 0.5: below
levers that have worked, above ones that have failed. `resolved` counts profiled, correct attempts whose
own symptoms no longer carry the target code; a speedup without that means the gain came from elsewhere.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from kernel_agent_loop.core.experiment_log import Attempt, ExperimentLog
from kernel_agent_loop.evald.diagnose import RULES, symptom_codes

_JUDGED = ("accept", "revert", "reject")
_CORRECT = ("accept", "revert")


@dataclass(frozen=True)
class LeverStats:
    lever: str
    description: str
    tries: int = 0
    accepts: int = 0
    profiled: int = 0
    resolved: int = 0
    median_gain: float | None = None

    @property
    def score(self) -> float:
        return (self.accepts + 1) / (self.tries + 2)


def rank_levers(attempts: list[Attempt], code: str) -> list[LeverStats]:
    """Seed levers of `code` plus any lever tried against it, best first."""
    seeds = dict(RULES[code].levers) if code in RULES else {}
    tried: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        if attempt.target == code and attempt.verdict in _JUDGED:
            tried.setdefault(attempt.lever, []).append(attempt)
    stats = []
    for lever in list(seeds) + [name for name in tried if name not in seeds]:
        runs = tried.get(lever, [])
        correct = [a for a in runs if a.verdict in _CORRECT]
        profiled = [a for a in correct if a.profiled]
        gains = [g for g in (_geomean_speedup(a.per_shape) for a in correct) if g is not None]
        stats.append(LeverStats(
            lever=lever,
            description=seeds.get(lever, ""),
            tries=len(runs),
            accepts=sum(a.verdict == "accept" for a in runs),
            profiled=len(profiled),
            resolved=sum(code not in symptom_codes(a.symptoms) for a in profiled),
            median_gain=statistics.median(gains) if gains else None,
        ))
    return sorted(stats, key=lambda s: (-s.score, -(s.median_gain or 0.0)))


def current_symptoms(attempts: list[Attempt], campaign: str) -> tuple[str, ...]:
    """Symptoms of the campaign's latest accepted attempt, i.e. of its current best."""
    for attempt in reversed(attempts):
        if attempt.campaign == campaign and attempt.verdict == "accept":
            return attempt.symptoms
    return ()


def suggest(attempts: list[Attempt], campaign: str) -> dict[str, list[LeverStats]]:
    """{symptom code of the current best: ranked levers}, in rule-table order."""
    present = symptom_codes(current_symptoms(attempts, campaign))
    return {code: rank_levers(attempts, code) for code in RULES if code in present}


def format_ranking(code: str, ranked: list[LeverStats]) -> str:
    lines = [f"[{code}]"]
    for s in ranked:
        gain = "-" if s.median_gain is None else f"{s.median_gain:.3f}x"
        resolved = f"{s.resolved}/{s.profiled}" if s.profiled else "-"
        text = f"  {s.score:.2f}  {s.lever:<22} tries={s.tries} accepts={s.accepts} resolved={resolved} gain={gain}"
        lines.append(text + (f"  {s.description}" if s.description and not s.tries else ""))
    return "\n".join(lines)


def _geomean_speedup(per_shape: dict) -> float | None:
    ratios = [
        perf["incumbent_us"] / perf["candidate_us"]
        for perf in per_shape.values()
        if perf.get("incumbent_us") and perf.get("candidate_us")
    ]
    if not ratios:
        return None
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank levers for a campaign's current symptoms")
    parser.add_argument("--log", required=True, type=Path, help="ExperimentLog sqlite path")
    parser.add_argument("--campaign", help="rank for this campaign's current best symptoms")
    parser.add_argument("--symptom", action="append", default=[], help="rank for this symptom code")
    args = parser.parse_args(argv)

    attempts = ExperimentLog(args.log).attempts()
    rankings = {code: rank_levers(attempts, code) for code in args.symptom}
    if args.campaign:
        current = current_symptoms(attempts, args.campaign)
        print(f"current best symptoms of {args.campaign}:")
        for line in current:
            print(f"  {line[:160]}")
        rankings.update(suggest(attempts, args.campaign))
    for code, ranked in rankings.items():
        print(format_ranking(code, ranked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
