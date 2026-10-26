from dataclasses import dataclass, field
from enum import Enum


class Verdict(Enum):
    REJECT = "reject"
    INVALID = "invalid"
    ACCEPT = "accept"
    REVERT = "revert"


@dataclass(frozen=True)
class EnvSnapshot:
    power_mode: str
    gpu_clock_mhz: int
    temp_c: float
    throttled: bool


@dataclass(frozen=True)
class ShapePerf:
    candidate_us: float
    incumbent_us: float | None
    noise_pct: float
    speedup_lo: float | None = None
    speedup_hi: float | None = None
    floor_us: float | None = None

    def headroom(self) -> float | None:
        """candidate_us / floor_us: how far the candidate sits above the measured touch-kernel floor."""
        if self.floor_us is None or self.floor_us <= 0:
            return None
        return self.candidate_us / self.floor_us

    def speedup_interval(self) -> tuple[float, float]:
        """Confidence interval of incumbent/candidate; derived from noise_pct when not measured."""
        if self.speedup_lo is not None and self.speedup_hi is not None:
            return self.speedup_lo, self.speedup_hi
        assert self.incumbent_us is not None
        speedup = self.incumbent_us / self.candidate_us
        widen = 1 + self.noise_pct / 100
        return speedup / widen, speedup * widen


@dataclass(frozen=True)
class EvalResult:
    commit: str
    compile_ok: bool
    correct: bool
    env_before: EnvSnapshot
    env_after: EnvSnapshot
    per_shape: dict[str, ShapePerf] = field(default_factory=dict)
    symptoms: tuple[str, ...] = ()
    profile: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Shape:
    label: str
    dims: dict
