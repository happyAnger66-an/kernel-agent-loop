"""Serialize EvalResult for SSH / file writeback."""

from __future__ import annotations

from kernel_agent_loop.core.types import EnvSnapshot, EvalResult, ShapePerf


def eval_result_to_dict(result: EvalResult) -> dict:
    return {
        "commit": result.commit,
        "compile_ok": result.compile_ok,
        "correct": result.correct,
        "env_before": _env_to_dict(result.env_before),
        "env_after": _env_to_dict(result.env_after),
        "per_shape": per_shape_to_dict(result.per_shape),
        "symptoms": list(result.symptoms),
    }


def per_shape_to_dict(per_shape: dict[str, ShapePerf]) -> dict:
    return {
        label: {
            "candidate_us": perf.candidate_us,
            "incumbent_us": perf.incumbent_us,
            "noise_pct": perf.noise_pct,
            "speedup_lo": perf.speedup_lo,
            "speedup_hi": perf.speedup_hi,
        }
        for label, perf in per_shape.items()
    }


def eval_result_from_dict(data: dict) -> EvalResult:
    return EvalResult(
        commit=data["commit"],
        compile_ok=bool(data["compile_ok"]),
        correct=bool(data["correct"]),
        env_before=_env_from_dict(data["env_before"]),
        env_after=_env_from_dict(data["env_after"]),
        per_shape={
            label: ShapePerf(
                candidate_us=float(perf["candidate_us"]),
                incumbent_us=_optional_float(perf["incumbent_us"]),
                noise_pct=float(perf["noise_pct"]),
                speedup_lo=_optional_float(perf.get("speedup_lo")),
                speedup_hi=_optional_float(perf.get("speedup_hi")),
            )
            for label, perf in data.get("per_shape", {}).items()
        },
        symptoms=tuple(str(s) for s in data.get("symptoms", ())),
    )


def _env_to_dict(env: EnvSnapshot) -> dict:
    return {
        "power_mode": env.power_mode,
        "gpu_clock_mhz": env.gpu_clock_mhz,
        "temp_c": env.temp_c,
        "throttled": env.throttled,
    }


def _env_from_dict(data: dict) -> EnvSnapshot:
    return EnvSnapshot(
        power_mode=data["power_mode"],
        gpu_clock_mhz=int(data["gpu_clock_mhz"]),
        temp_c=float(data["temp_c"]),
        throttled=bool(data["throttled"]),
    )


def _optional_float(value) -> float | None:
    return None if value is None else float(value)
