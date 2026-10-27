"""模块: kernel_agent_loop.knowledge.cell_result
功能: 用工单和尝试归纳格子结果。四种结论对不上契约时拒绝，并且不从 per_shape 推端到端毫秒。
"""

import json

import pytest

from kernel_agent_loop.core.experiment_log import Attempt
from kernel_agent_loop.knowledge.cell_result import AttemptNote, CellResultError, induce_cell_result


def _attempt(verdict="accept", commit="4567def", parent="0123abc"):
    return Attempt(
        campaign="cell-1",
        commit=commit,
        parent=parent,
        lever="fuse-rmsnorm",
        hypothesis="少一次激活往返",
        verdict=verdict,
        symptoms=("long_scoreboard",),
        per_shape={"r1": {"candidate_us": 9.5, "incumbent_us": 10.0, "noise_pct": 1.2}},
    )


def _note(**overrides):
    base = dict(kind="draft", expected_gain_ms=0.6, stage_reached="e2e_bench", e2e_ms=80.0)
    base.update(overrides)
    return AttemptNote(**base)


def _start():
    return {
        "commit": "0123abc",
        "e2e_ms": 90.585,
        "segments_ms": {"dit.denoise": 45.044},
        "noise_pct": 1.5,
    }


def _final():
    point = _start()
    point["commit"] = "4567def"
    point["e2e_ms"] = 80.0
    return point


def _induce(monkeypatch, **overrides):
    monkeypatch.delenv("AIINFRAKB_ROOT", raising=False)
    base = dict(
        ticket={"ticket": "gr00t/forge/schedule/dit.denoise"},
        attempts=[_attempt()],
        notes=[_note()],
        outcome="apply",
        stop_reason="stalled",
        start=_start(),
        final=_final(),
        env={"hardware": "thor"},
        finished_at="2026-10-26T16:00:00+08:00",
    )
    base.update(overrides)
    return induce_cell_result(**base)


# 用例: apply 带上日志里的提交、父提交、手段、假设和判定，调用方补的字段也在
# 参数: monkeypatch 清掉 AIINFRAKB_ROOT；一次 accept，日志里的症状和 per_shape 不进结果
def test_apply_copies_log_fields_and_ignores_per_shape(monkeypatch):
    result = _induce(monkeypatch)
    item = result["attempts"][0]
    assert item["commit"] == "4567def"
    assert item["parent"] == "0123abc"
    assert item["lever"] == "fuse-rmsnorm"
    assert item["hypothesis"] == "少一次激活往返"
    assert item["verdict"] == "accept"
    assert item["kind"] == "draft"
    assert item["e2e_ms"] == 80.0
    blob = json.dumps(result)
    assert "long_scoreboard" not in blob
    assert "9.5" not in blob
    assert result["ticket"] == "gr00t/forge/schedule/dit.denoise"
    assert result["outcome"] == "apply"


# 用例: 调用方没给 e2e_ms 时，结果里不出现该字段
# 参数: note.e2e_ms 为 None，per_shape 里仍有 9.5 微秒
def test_missing_e2e_ms_is_not_invented_from_per_shape(monkeypatch):
    result = _induce(monkeypatch, notes=[_note(e2e_ms=None)])
    assert "e2e_ms" not in result["attempts"][0]


# 用例: attempts 与 notes 数量不一致时拒绝
# 参数: 一条尝试，零条 note
def test_notes_must_align_with_attempts(monkeypatch):
    with pytest.raises(CellResultError, match="数量必须相同"):
        _induce(monkeypatch, notes=[])


# 用例: 四种结论缺了契约要求的字段时拒绝
# 参数: 见各 kwargs，monkeypatch 清掉 AIINFRAKB_ROOT
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"final": None}, "final"),
        ({"attempts": [_attempt(verdict="revert")], "notes": [_note()]}, "accept"),
        (
            {
                "outcome": "partial",
                "final": None,
                "remaining": [{"lever": "x", "hypothesis": "再试一次", "expected_gain_ms": 0.2}],
            },
            "final",
        ),
        ({"outcome": "partial"}, "remaining"),
        (
            {
                "outcome": "reject",
                "final": None,
                "attempts": [_attempt(verdict="reject")],
                "notes": [_note()],
            },
            "software",
        ),
        (
            {
                "outcome": "reject",
                "final": None,
                "conditions": {"software": {"mlc": "905482a73"}},
            },
            "accept",
        ),
        ({"outcome": "measure", "stop_reason": "measure_only", "final": None}, "attempts"),
        (
            {
                "outcome": "measure",
                "stop_reason": "exhausted",
                "attempts": [],
                "notes": [],
                "final": None,
            },
            "measure_only",
        ),
    ],
)
def test_outcome_rules_reject_incomplete_results(monkeypatch, kwargs, message):
    with pytest.raises(CellResultError, match=message):
        _induce(monkeypatch, **kwargs)


# 用例: partial、reject、measure 在字段齐的时候能通过契约
# 参数: monkeypatch 清掉 AIINFRAKB_ROOT
def test_partial_reject_and_measure_validate(monkeypatch):
    remaining = [{"lever": "widen-load", "hypothesis": "加宽访存", "expected_gain_ms": 0.2}]
    partial = _induce(monkeypatch, outcome="partial", remaining=remaining)
    assert partial["remaining"] == remaining
    rejected = _induce(
        monkeypatch,
        outcome="reject",
        final=None,
        attempts=[_attempt(verdict="reject", commit="89ab012")],
        notes=[_note()],
        conditions={"software": {"cuda": "12.8"}},
    )
    assert rejected["attempts"][0]["verdict"] == "reject"
    measured = _induce(
        monkeypatch,
        outcome="measure",
        stop_reason="measure_only",
        attempts=[],
        notes=[],
        final=None,
    )
    assert measured["attempts"] == []
    assert measured["stop_reason"] == "measure_only"
