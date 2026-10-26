"""模块: kernel_agent_loop.evald.timing
功能: 交替计时每轮两边都跑，噪声取逐轮中位数极差，线性漂移下加速比仍准。
"""

import pytest

from kernel_agent_loop.evald.timing import interleaved_bench


def constant(us, log=None, tag=None):
    def run():
        if log is not None:
            log.append(tag)
        return us
    return run


def from_sequence(values):
    it = iter(values)
    return lambda: next(it)


# 用例: 每一轮里候选和 incumbent 都各跑 reps 次
# 参数: 无
def test_every_round_runs_both_sides():
    log = []
    interleaved_bench(constant(80, log, "c"), constant(100, log, "i"), rounds=3, reps=2)
    rounds = [log[i:i + 4] for i in range(0, len(log), 4)]
    assert len(rounds) == 3
    assert all(sorted(r) == ["c", "c", "i", "i"] for r in rounds)


# 用例: 耗时恒定时中位数就是该值，噪声为 0
# 参数: 无
def test_steady_timings_give_exact_medians_and_zero_noise():
    perf = interleaved_bench(constant(80), constant(100), rounds=5, reps=3)
    assert perf.candidate_us == 80
    assert perf.incumbent_us == 100
    assert perf.noise_pct == 0


# 用例: 配对比值有抖动时，加速比区间包住真实比值且有宽度，noise_pct 是区间半宽
# 参数: 无
def test_jittery_pairs_give_an_interval_around_the_true_ratio():
    perf = interleaved_bench(
        constant(80), from_sequence([100, 108, 92, 100, 104, 96, 100, 100, 112, 88]), rounds=10, reps=1
    )
    assert perf.speedup_lo < 1.25 < perf.speedup_hi
    assert perf.noise_pct == pytest.approx((perf.speedup_hi - perf.speedup_lo) / 2 / 1.25 * 100)


# 用例: 耗时恒定时区间退化成一个点
# 参数: 无
def test_steady_pairs_give_a_point_interval():
    perf = interleaved_bench(constant(80), constant(100), rounds=5, reps=3)
    assert perf.speedup_lo == perf.speedup_hi == pytest.approx(1.25)


# 用例: 两边耗时同步线性变慢时，交替计时得出的加速比误差在 1% 以内
# 参数: 无
def test_interleaving_cancels_slow_drift():
    calls = [0]

    def drifting(base):
        def run():
            calls[0] += 1
            return base * (1 + 0.01 * calls[0])
        return run

    perf = interleaved_bench(drifting(90), drifting(100), rounds=20, reps=1)
    assert perf.incumbent_us / perf.candidate_us == pytest.approx(100 / 90, rel=0.01)


# 用例: 没有 incumbent 时噪声改由候选自己的各轮中位数计算
# 参数: 无
def test_without_incumbent_noise_comes_from_candidate():
    perf = interleaved_bench(from_sequence([100, 104, 96]), None, rounds=3, reps=1)
    assert perf.incumbent_us is None
    assert perf.candidate_us == 100
    assert perf.noise_pct == pytest.approx(8.0)


# 用例: 配对采样时两侧共享同一份布局抖动，比值抵消它，区间收窄到真实比值
# 参数: 无
def test_paired_samples_cancel_shared_jitter():
    from kernel_agent_loop.evald.timing import paired_bench

    factors = iter([1.0, 1.3, 0.8, 1.2, 1.0, 0.9, 1.25, 1.1, 0.85, 1.0] * 3)

    def pair():
        f = next(factors)
        return 80 * f, 100 * f

    perf = paired_bench(pair, rounds=10, reps=3)
    assert perf.speedup_lo == pytest.approx(1.25)
    assert perf.speedup_hi == pytest.approx(1.25)
