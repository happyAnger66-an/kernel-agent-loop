"""模块: kernel_agent_loop.core.gate
功能: 接受门禁按编译、正确性、环境、加速和单 shape 退化的顺序判定。
"""

from dataclasses import replace

import pytest

from kernel_agent_loop.core.gate import AcceptanceGate, Verdict
from kernel_agent_loop.core.types import EnvSnapshot, EvalResult, ShapePerf
from kernel_agent_loop.evald.env_guard import EnvGuard

STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)


def perf(speedup: float, noise_pct: float = 2.0) -> ShapePerf:
    return ShapePerf(candidate_us=100.0 / speedup, incumbent_us=100.0, noise_pct=noise_pct)


def result(per_shape=None, *, compile_ok=True, correct=True, env_after=STEADY) -> EvalResult:
    return EvalResult(
        commit="c1",
        compile_ok=compile_ok,
        correct=correct,
        env_before=STEADY,
        env_after=env_after,
        per_shape=per_shape if per_shape is not None else {"s": perf(1.2)},
    )


@pytest.fixture
def gate() -> AcceptanceGate:
    return AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0)


# 用例: 编译失败必须拒绝，不看性能
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_compile_failure_is_rejected(gate):
    assert gate.decide(result(compile_ok=False)).verdict is Verdict.REJECT


# 用例: 结果错误必须拒绝，即使加速达到 3 倍
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_incorrect_output_is_rejected_even_when_fast(gate):
    d = gate.decide(result({"s": perf(3.0)}, correct=False))
    assert d.verdict is Verdict.REJECT


# 用例: 发生降频时判作废，不能接受
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_throttled_run_is_invalid_not_accepted(gate):
    d = gate.decide(result({"s": perf(1.5)}, env_after=replace(STEADY, throttled=True)))
    assert d.verdict is Verdict.INVALID
    assert d.reasons


# 用例: 评测前后温漂超过 5°C 时判作废
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_temperature_drift_makes_run_invalid(gate):
    d = gate.decide(result({"s": perf(1.5)}, env_after=replace(STEADY, temp_c=STEADY.temp_c + 10)))
    assert d.verdict is Verdict.INVALID


# 用例: 没有 incumbent 的第一个正确候选直接接受
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_first_candidate_without_incumbent_is_accepted(gate):
    first = ShapePerf(candidate_us=100.0, incumbent_us=None, noise_pct=2.0)
    assert gate.decide(result({"s": first})).verdict is Verdict.ACCEPT


# 用例: 每个 shape 的加速都超过噪声时接受
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_gain_above_noise_on_every_shape_is_accepted(gate):
    d = gate.decide(result({"a": perf(1.10), "b": perf(1.08)}))
    assert d.verdict is Verdict.ACCEPT


# 用例: 加速落在噪声以内时回退
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_gain_within_noise_is_reverted(gate):
    d = gate.decide(result({"a": perf(1.015), "b": perf(1.015)}))
    assert d.verdict is Verdict.REVERT


# 用例: 噪声很小时仍要求至少 1% 收益，否则回退
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_min_gain_applies_when_noise_is_tiny(gate):
    d = gate.decide(result({"a": perf(1.005, noise_pct=0.1)}))
    assert d.verdict is Verdict.REVERT


# 用例: 单个 shape 明显变慢就回退，即使平均加速很大
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_regression_on_one_shape_is_reverted_despite_big_average_gain(gate):
    d = gate.decide(result({"a": perf(1.30), "b": perf(0.92)}))
    assert d.verdict is Verdict.REVERT
    assert any("b" in r for r in d.reasons)


# 用例: 单个 shape 变慢但没超过自身噪声时，不阻止接受
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_slowdown_within_noise_does_not_block_acceptance(gate):
    d = gate.decide(result({"a": perf(1.12), "b": perf(1.10), "c": perf(0.99)}))
    assert d.verdict is Verdict.ACCEPT


def measured(speedup: float, lo: float, hi: float) -> ShapePerf:
    return ShapePerf(candidate_us=100.0 / speedup, incumbent_us=100.0, noise_pct=0.0, speedup_lo=lo, speedup_hi=hi)


# 用例: 只有一个 shape 明确变快、其余持平且不退化时接受，其他 shape 的大噪声不拖累它
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_targeted_gain_is_accepted_when_other_shapes_hold(gate):
    d = gate.decide(result({
        "noisy": measured(1.00, 0.80, 1.25),
        "target": measured(1.14, 1.10, 1.18),
        "flat": measured(1.00, 0.995, 1.005),
    }))
    assert d.verdict is Verdict.ACCEPT


# 用例: 所有 shape 的区间都跨过 1+最小收益时，没有显著变快，回退
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_no_significant_gain_is_reverted(gate):
    d = gate.decide(result({"a": measured(1.15, 0.95, 1.35), "b": measured(1.00, 0.99, 1.01)}))
    assert d.verdict is Verdict.REVERT
    assert any("no shape faster" in r for r in d.reasons)


# 用例: 单个 shape 显著变快但几何平均不到 1+最小收益时回退
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_tiny_geomean_is_reverted_even_with_one_significant_gain(gate):
    flat = {f"s{i}": measured(1.0, 0.999, 1.001) for i in range(9)}
    d = gate.decide(result({**flat, "target": measured(1.05, 1.04, 1.06)}))
    assert d.verdict is Verdict.REVERT
    assert any("geometric-mean" in r for r in d.reasons)


# 用例: 某个 shape 比实测硬件下限的 0.8 倍还快，判拒绝待人工复核，即使没有 incumbent
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%、下限比 0.8
def test_candidate_far_below_hardware_floor_is_rejected(gate):
    cheat = ShapePerf(candidate_us=5.0, incumbent_us=None, noise_pct=1.0, floor_us=10.0)
    d = gate.decide(result({"s": cheat}))
    assert d.verdict is Verdict.REJECT
    assert any("hardware floor" in r for r in d.reasons)


# 用例: 略低于下限（0.9 倍）在容差内，不触发拒绝，正常比较
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%、下限比 0.8
def test_candidate_slightly_below_floor_is_judged_normally(gate):
    near = replace(measured(1.2, 1.15, 1.25), floor_us=100.0 / 1.2 / 0.9)
    assert gate.decide(result({"s": near})).verdict is Verdict.ACCEPT


# 用例: 区间整体低于 1 的 shape 判为退化，即使另一个 shape 大幅变快
# 参数: gate 是 AcceptanceGate，温漂上限 5°C、最小收益 1%
def test_interval_below_one_is_a_regression(gate):
    d = gate.decide(result({"a": measured(1.5, 1.4, 1.6), "b": measured(0.97, 0.96, 0.98)}))
    assert d.verdict is Verdict.REVERT
    assert any(r.startswith("b ") for r in d.reasons)
