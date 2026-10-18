"""模块: kernel_agent_loop.evald.env_guard
功能: 降频、功耗模式变化、频率变化和温漂超过阈值时环境检查要报出来。
"""

from dataclasses import replace

from kernel_agent_loop.core.types import EnvSnapshot
from kernel_agent_loop.evald.env_guard import EnvGuard, parse_gpu_env_line

STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)
guard = EnvGuard(max_temp_drift_c=5.0)


# 用例: 评测前后环境快照完全一致时检查通过
# 参数: 无
def test_identical_snapshots_pass():
    assert guard.check(STEADY, STEADY) == []


# 用例: 评测前或评测后发生降频都要报出来
# 参数: 无
def test_throttling_before_or_after_is_reported():
    assert guard.check(replace(STEADY, throttled=True), STEADY)
    assert guard.check(STEADY, replace(STEADY, throttled=True))


# 用例: 功耗模式在评测期间变化要报出来
# 参数: 无
def test_power_mode_change_is_reported():
    assert guard.check(STEADY, replace(STEADY, power_mode="15W"))


# 用例: GPU 频率在评测期间变化要报出来
# 参数: 无
def test_clock_change_is_reported():
    assert guard.check(STEADY, replace(STEADY, gpu_clock_mhz=1200))


# 用例: 温漂超过 5°C 上限要报出来
# 参数: 无
def test_temperature_drift_beyond_limit_is_reported():
    assert guard.check(STEADY, replace(STEADY, temp_c=56.0))


# 用例: 温漂在 5°C 以内时检查通过
# 参数: 无
def test_temperature_drift_within_limit_passes():
    assert guard.check(STEADY, replace(STEADY, temp_c=54.0)) == []


# 用例: nvidia-smi 的一行能解析成功耗模式、频率、温度，以及未降频
# 参数: 无
def test_parse_gpu_env_line_reads_a_steady_sample():
    parsed = parse_gpu_env_line("P0, 1575, 50, Not Active")
    assert parsed == EnvSnapshot("P0", 1575, 50.0, False)


# 用例: clocks_event_reasons.active 不是未激活时记为降频
# 参数: 无
def test_parse_gpu_env_line_marks_active_throttle():
    parsed = parse_gpu_env_line("P2, 1200, 71, 0x0000000000000004")
    assert parsed.throttled
