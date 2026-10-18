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
        reasons = []
        speedups = []
        noises = [self.min_gain_pct]
        for label, perf in per_shape.items():
            assert perf.incumbent_us is not None
            speedups.append(perf.incumbent_us / perf.candidate_us)
            noises.append(perf.noise_pct)
            limit = perf.incumbent_us * (1 + perf.noise_pct / 100)
            if perf.candidate_us > limit:
                reasons.append(f"{label} slower than incumbent beyond noise")
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        threshold = 1 + max(noises) / 100
        if geomean <= threshold:
            reasons.append(f"geometric-mean speedup {geomean:.4f} within {threshold:.4f}")
        if reasons:
            return Decision(Verdict.REVERT, tuple(reasons))
        return Decision(Verdict.ACCEPT)
