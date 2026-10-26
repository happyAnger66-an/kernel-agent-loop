import math
from dataclasses import dataclass

from kernel_agent_loop.core.types import EvalResult, ShapePerf, Verdict
from kernel_agent_loop.evald.env_guard import EnvGuard


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reasons: tuple[str, ...] = ()


class AcceptanceGate:
    def __init__(self, env_guard: EnvGuard, min_gain_pct: float) -> None:
        self.env_guard = env_guard
        self.min_gain_pct = min_gain_pct

    def decide(self, result: EvalResult) -> Decision:
        if not result.compile_ok:
            return Decision(Verdict.REJECT, ("compile failed",))
        if not result.correct:
            return Decision(Verdict.REJECT, ("incorrect output",))
        env_reasons = self.env_guard.check(result.env_before, result.env_after)
        if env_reasons:
            return Decision(Verdict.INVALID, tuple(env_reasons))
        if all(perf.incumbent_us is None for perf in result.per_shape.values()):
            return Decision(Verdict.ACCEPT)
        return self._compare(result.per_shape)

    def _compare(self, per_shape: dict[str, ShapePerf]) -> Decision:
        """Each shape is judged against its own interval, so one noisy shape cannot block the rest.

        Accept only if no shape is significantly slower, at least one is significantly faster by
        more than min_gain_pct, and the geometric-mean speedup exceeds 1 + min_gain_pct.
        """
        reasons = []
        speedups = []
        min_gain = 1 + self.min_gain_pct / 100
        significant_gain = False
        for label, perf in per_shape.items():
            assert perf.incumbent_us is not None
            speedups.append(perf.incumbent_us / perf.candidate_us)
            lo, hi = perf.speedup_interval()
            if hi < 1:
                reasons.append(f"{label} slower than incumbent beyond noise (speedup interval {lo:.3f}-{hi:.3f})")
            if lo > min_gain:
                significant_gain = True
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        if not significant_gain:
            reasons.append(f"no shape faster than {min_gain:.3f} beyond noise")
        if geomean <= min_gain:
            reasons.append(f"geometric-mean speedup {geomean:.4f} within {min_gain:.4f}")
        if reasons:
            return Decision(Verdict.REVERT, tuple(reasons))
        return Decision(Verdict.ACCEPT)
