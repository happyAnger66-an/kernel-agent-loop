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
        per_shape={"r1": ShapePerf(candidate_us=10.0, incumbent_us=12.0, noise_pct=1.5, speedup_lo=1.1, speedup_hi=1.3)},
    )
    restored = eval_result_from_dict(eval_result_to_dict(original))
    assert restored == original


# 用例: 正确性失败的症状经序列化后保留，旧格式缺字段时为空
# 参数: 无
def test_eval_result_symptoms_roundtrip():
    env = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=48.0, throttled=False)
    original = EvalResult(
        commit="abc",
        compile_ok=True,
        correct=False,
        env_before=env,
        env_after=env,
        symptoms=("mismatch r1-h4096: max_abs=1.0",),
    )
    data = eval_result_to_dict(original)
    assert eval_result_from_dict(data) == original
    del data["symptoms"]
    assert eval_result_from_dict(data).symptoms == ()
