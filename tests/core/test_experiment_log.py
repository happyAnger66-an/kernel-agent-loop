"""模块: kernel_agent_loop.core.experiment_log
功能: 尝试记录可回读、重新打开后仍在、按 campaign 隔离，并按父版本去重手段。
"""

from kernel_agent_loop.core.experiment_log import Attempt, ExperimentLog


def attempt(commit, parent, lever, verdict="revert", **kw):
    return Attempt(
        campaign=kw.get("campaign", "rms-1"),
        commit=commit,
        parent=parent,
        lever=lever,
        hypothesis=f"try {lever}",
        verdict=verdict,
        symptoms=("long_scoreboard",),
        per_shape={"r1-h4096": {"candidate_us": 9.5, "incumbent_us": 10.0, "noise_pct": 1.2}},
    )


# 用例: 同一 campaign 的尝试按写入顺序读回
# 参数: tmp_path 是存放 sqlite 日志的临时目录
def test_attempts_are_returned_in_insertion_order(tmp_path):
    log = ExperimentLog(tmp_path / "log.sqlite")
    log.record(attempt("a1", "a0", "vectorize-loads", "accept"))
    log.record(attempt("a2", "a1", "unroll-reduction", "reject"))
    assert [a.commit for a in log.attempts("rms-1")] == ["a1", "a2"]


# 用例: 写入的尝试读回来字段完全一致
# 参数: tmp_path 是存放 sqlite 日志的临时目录
def test_recorded_attempt_round_trips(tmp_path):
    log = ExperimentLog(tmp_path / "log.sqlite")
    original = attempt("a1", "a0", "vectorize-loads")
    log.record(original)
    assert log.attempts("rms-1") == [original]


# 用例: 重新打开日志文件后，已写入的尝试还在
# 参数: tmp_path 是存放 sqlite 日志的临时目录
def test_log_survives_reopen(tmp_path):
    ExperimentLog(tmp_path / "log.sqlite").record(attempt("a1", "a0", "vectorize-loads"))
    assert len(ExperimentLog(tmp_path / "log.sqlite").attempts("rms-1")) == 1


# 用例: 查询一个 campaign 不会返回另一个 campaign 的尝试
# 参数: tmp_path 是存放 sqlite 日志的临时目录
def test_campaigns_are_isolated(tmp_path):
    log = ExperimentLog(tmp_path / "log.sqlite")
    log.record(attempt("a1", "a0", "vectorize-loads", campaign="rms-1"))
    log.record(attempt("b1", "b0", "vectorize-loads", campaign="rms-2"))
    assert [a.commit for a in log.attempts("rms-2")] == ["b1"]


# 用例: 同一个父版本上试过的手段能查到，换父版本或换手段则查不到
# 参数: tmp_path 是存放 sqlite 日志的临时目录
def test_tried_dedupes_a_lever_on_the_same_parent_only(tmp_path):
    log = ExperimentLog(tmp_path / "log.sqlite")
    log.record(attempt("a1", "a0", "vectorize-loads", "revert"))
    assert log.tried("rms-1", lever="vectorize-loads", parent="a0")
    assert not log.tried("rms-1", lever="vectorize-loads", parent="a5")
    assert not log.tried("rms-1", lever="unroll-reduction", parent="a0")
