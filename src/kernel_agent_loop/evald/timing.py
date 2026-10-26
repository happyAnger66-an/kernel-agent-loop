import numpy as np

from kernel_agent_loop.core.types import ShapePerf


def interleaved_bench(candidate, incumbent, *, rounds: int, reps: int) -> ShapePerf:
    candidate_rounds = []
    incumbent_rounds = []
    for _ in range(rounds):
        candidate_rounds.append(float(np.median([candidate() for _ in range(reps)])))
        if incumbent is not None:
            incumbent_rounds.append(float(np.median([incumbent() for _ in range(reps)])))
    candidate_us = float(np.median(candidate_rounds))
    if incumbent is None:
        return ShapePerf(candidate_us, None, _noise_pct(candidate_rounds, candidate_us))
    incumbent_us = float(np.median(incumbent_rounds))
    noise = max(_noise_pct(incumbent_rounds, incumbent_us), _noise_pct(candidate_rounds, candidate_us))
    return ShapePerf(candidate_us, incumbent_us, noise)


def _noise_pct(round_medians: list[float], median: float) -> float:
    if median == 0:
        return 0.0
    return (max(round_medians) - min(round_medians)) / median * 100
