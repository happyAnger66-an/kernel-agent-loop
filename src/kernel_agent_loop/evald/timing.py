import numpy as np

from kernel_agent_loop.core.types import ShapePerf

_CONFIDENCE = 0.95
_RESAMPLES = 2000


def interleaved_bench(candidate, incumbent, *, rounds: int, reps: int) -> ShapePerf:
    """Alternate candidate and incumbent each round.

    Each candidate sample is paired with the incumbent sample at the same position in its round.
    The speedup interval is a bootstrap confidence interval of the median of those paired ratios.
    """
    candidate_rounds = []
    incumbent_rounds = []
    ratios = []
    for _ in range(rounds):
        cand = [candidate() for _ in range(reps)]
        candidate_rounds.append(float(np.median(cand)))
        if incumbent is not None:
            inc = [incumbent() for _ in range(reps)]
            incumbent_rounds.append(float(np.median(inc)))
            ratios.extend(i / c for i, c in zip(inc, cand))
    candidate_us = float(np.median(candidate_rounds))
    if incumbent is None:
        return ShapePerf(candidate_us, None, _spread_pct(candidate_rounds, candidate_us))
    return _paired_perf(candidate_rounds, incumbent_rounds, ratios)


def paired_bench(pair, *, rounds: int, reps: int) -> ShapePerf:
    """Like interleaved_bench, but each sample `pair()` returns (candidate_us, incumbent_us) measured together."""
    candidate_rounds = []
    incumbent_rounds = []
    ratios = []
    for _ in range(rounds):
        samples = [pair() for _ in range(reps)]
        candidate_rounds.append(float(np.median([c for c, _ in samples])))
        incumbent_rounds.append(float(np.median([i for _, i in samples])))
        ratios.extend(i / c for c, i in samples)
    return _paired_perf(candidate_rounds, incumbent_rounds, ratios)


def _paired_perf(candidate_rounds: list[float], incumbent_rounds: list[float], ratios: list[float]) -> ShapePerf:
    candidate_us = float(np.median(candidate_rounds))
    incumbent_us = float(np.median(incumbent_rounds))
    lo, hi = _median_ci(ratios)
    center = float(np.median(ratios))
    noise = (hi - lo) / 2 / center * 100 if center else 0.0
    return ShapePerf(candidate_us, incumbent_us, noise, speedup_lo=lo, speedup_hi=hi)


def _median_ci(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    medians = np.median(rng.choice(data, size=(_RESAMPLES, data.size), replace=True), axis=1)
    tail = (1 - _CONFIDENCE) / 2 * 100
    return float(np.percentile(medians, tail)), float(np.percentile(medians, 100 - tail))


def _spread_pct(round_medians: list[float], median: float) -> float:
    if median == 0:
        return 0.0
    return (max(round_medians) - min(round_medians)) / median * 100
