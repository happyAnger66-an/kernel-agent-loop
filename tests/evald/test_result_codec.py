"""模块: kernel_agent_loop.evald.result_codec
功能: EvalResult 与 JSON 字典互转，供 SSH 回写。
"""

from kernel_agent_loop.core.types import EnvSnapshot, EvalResult, ShapePerf
from kernel_agent_loop.evald.result_codec import eval_result_from_dict, eval_result_to_dict


# 用例: 往返序列化保持字段
# 参数: 无
def test_eval_result_roundtrip():
    env = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=48.0, throttled=False)
    original = EvalResult(
        commit="abc",
        compile_ok=True,
        correct=True,
        env_before=env,
        env_after=env,
        per_shape={"r1": ShapePerf(candidate_us=10.0, incumbent_us=12.0, noise_pct=1.5)},
    )
    restored = eval_result_from_dict(eval_result_to_dict(original))
    assert restored == original
