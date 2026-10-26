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


# 用例: 噪声是 incumbent 各轮中位数的极差除以中位数
# 参数: 无
def test_noise_is_the_spread_of_incumbent_round_medians():
    perf = interleaved_bench(constant(80), from_sequence([100, 102, 98, 100, 100]), rounds=5, reps=1)
    assert perf.incumbent_us == 100
    assert perf.noise_pct == pytest.approx(4.0)


# 用例: 候选一侧更抖时，噪声取两侧中较大的那个
# 参数: 无
def test_noise_takes_the_noisier_side():
    perf = interleaved_bench(from_sequence([80, 92, 80, 80, 80]), constant(100), rounds=5, reps=1)
    assert perf.noise_pct == pytest.approx(15.0)


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
