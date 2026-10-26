"""模块: kernel_agent_loop.tuning
功能: 按症状从实验日志统计各手段的接受率、症状消除和收益，并以规则表为种子排序。
"""

import sqlite3

from kernel_agent_loop.core.experiment_log import Attempt, ExperimentLog
from kernel_agent_loop.tuning import current_symptoms, main, rank_levers, suggest

UNDERFILLED = "[underfilled-gpu] r1-h4096: 1 blocks for 20 SMs, headroom 1.44x. Try: ..."
NARROW = "[narrow-global-access] r1-h4096: 2.0 sectors per global load request. Try: ..."


def attempt(lever, verdict, *, target="underfilled-gpu", symptoms=(), speedup=1.0, profiled=True,
            campaign="c1", commit="x"):
    per_shape = {"r1": {"candidate_us": 10.0 / speedup, "incumbent_us": 10.0, "noise_pct": 1.0}}
    return Attempt(campaign=campaign, commit=commit, parent="p", lever=lever, hypothesis="h", verdict=verdict,
                   symptoms=tuple(symptoms), per_shape=per_shape, target=target, profiled=profiled)


def by_lever(ranked):
    return {s.lever: s for s in ranked}


# 用例: 试过且被接受的手段排最前，没试过的种子手段 0.5 居中，试过失败的排后
# 参数: 无
def test_rank_orders_worked_untried_failed():
    attempts = [
        attempt("warp-per-row", "accept", speedup=1.3),
        attempt("split-row-blocks", "revert", speedup=0.95, symptoms=(UNDERFILLED,)),
    ]
    ranked = rank_levers(attempts, "underfilled-gpu")
    assert [s.lever for s in ranked] == ["warp-per-row", "split-row-blocks"]
    ranked = rank_levers(attempts[1:], "underfilled-gpu")
    assert [s.lever for s in ranked] == ["warp-per-row", "split-row-blocks"]
    assert by_lever(ranked)["warp-per-row"].score == 0.5 and by_lever(ranked)["warp-per-row"].tries == 0


# 用例: 症状消除只统计有 profile 的正确尝试；Agent 自创手段也进排名
# 参数: 无
def test_resolved_counts_profiled_correct_attempts_and_custom_levers_appear():
    attempts = [
        attempt("warp-per-row", "accept", speedup=1.2),                          # gone -> resolved
        attempt("warp-per-row", "revert", symptoms=(UNDERFILLED,)),               # still there
        attempt("warp-per-row", "accept", profiled=False),                        # unknown
        attempt("warp-per-row", "reject", symptoms=("mismatch r1",)),             # incorrect
        attempt("cluster-2", "accept", speedup=1.1),
        attempt("vector-16b", "accept", target="narrow-global-access"),           # other symptom
        attempt("warp-per-row", "invalid"),                                       # not judged
    ]
    stats = by_lever(rank_levers(attempts, "underfilled-gpu"))
    s = stats["warp-per-row"]
    assert (s.tries, s.accepts, s.profiled, s.resolved) == (4, 2, 2, 1)
    assert stats["cluster-2"].description == "" and stats["cluster-2"].accepts == 1
    assert "vector-16b" not in stats


# 用例: 中位收益取正确尝试的几何平均加速
# 参数: 无
def test_median_gain_over_correct_attempts():
    attempts = [attempt("warp-per-row", "accept", speedup=1.3), attempt("warp-per-row", "revert", speedup=0.9),
                attempt("warp-per-row", "accept", speedup=1.1)]
    assert abs(by_lever(rank_levers(attempts, "underfilled-gpu"))["warp-per-row"].median_gain - 1.1) < 1e-9


# 用例: 当前 best 的症状取该 campaign 最近一次接受；suggest 只给当前存在的症状排序
# 参数: 无
def test_suggest_uses_latest_accepted_symptoms():
    attempts = [
        attempt("naive", "accept", target="", symptoms=(NARROW, UNDERFILLED)),
        attempt("vector-16b", "accept", target="narrow-global-access", symptoms=(UNDERFILLED,)),
        attempt("bad", "revert", target="", symptoms=(NARROW,)),
        attempt("other", "accept", target="", symptoms=(NARROW,), campaign="c2"),
    ]
    assert current_symptoms(attempts, "c1") == (UNDERFILLED,)
    assert list(suggest(attempts, "c1")) == ["underfilled-gpu"]


# 用例: 旧库（没有 target/profiled 列）打开时自动补列，旧记录取默认值
# 参数: tmp_path 临时目录
def test_old_log_schema_is_migrated(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, campaign TEXT NOT NULL, "
               "commit_sha TEXT NOT NULL, parent TEXT NOT NULL, lever TEXT NOT NULL, hypothesis TEXT NOT NULL, "
               "verdict TEXT NOT NULL, symptoms TEXT NOT NULL, per_shape TEXT NOT NULL)")
    db.execute("INSERT INTO attempts (campaign, commit_sha, parent, lever, hypothesis, verdict, symptoms, per_shape)"
               " VALUES ('c1', 'a', 'p', 'naive', 'h', 'accept', '[]', '{}')")
    db.commit()
    db.close()
    log = ExperimentLog(path)
    log.record(attempt("warp-per-row", "accept"))
    old, new = log.attempts("c1")
    assert (old.target, old.profiled) == ("", False)
    assert (new.target, new.profiled) == ("underfilled-gpu", True)


# 用例: CLI 打印当前症状和排序
# 参数: tmp_path 临时目录；capsys 捕获输出
def test_cli_prints_ranking(tmp_path, capsys):
    log = ExperimentLog(tmp_path / "log.sqlite")
    log.record(attempt("naive", "accept", target="", symptoms=(UNDERFILLED,)))
    log.record(attempt("warp-per-row", "accept", speedup=1.3, symptoms=(UNDERFILLED,)))
    assert main(["--log", str(tmp_path / "log.sqlite"), "--campaign", "c1"]) == 0
    out = capsys.readouterr().out
    assert "[underfilled-gpu]" in out and "warp-per-row" in out and "split-row-blocks" in out
