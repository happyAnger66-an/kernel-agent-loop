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


@dataclass(frozen=True)
class EvalResult:
    commit: str
    compile_ok: bool
    correct: bool
    env_before: EnvSnapshot
    env_after: EnvSnapshot
    per_shape: dict[str, ShapePerf] = field(default_factory=dict)


@dataclass(frozen=True)
class Shape:
    label: str
    dims: dict
