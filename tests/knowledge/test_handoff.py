"""模块: kernel_agent_loop.knowledge.handoff
功能: 格子结果经 kb propose 写成 lesson 草稿，钉在同一格，留在 inbox，不自动接受。
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kernel_agent_loop.knowledge.cell_result import CellResultError, induce_cell_result
from kernel_agent_loop.knowledge.handoff import propose_argv, propose_cell_result


def _kb_cmd(src: Path):
    py = shutil.which("python3.12")
    kb = src / "tools" / "kb.py"
    if py is None or not kb.is_file():
        pytest.skip("需要 python3.12 和 AiInfraKB 的 tools/kb.py")
    return [py, str(kb)]


def _measure(monkeypatch):
    monkeypatch.delenv("AIINFRAKB_ROOT", raising=False)
    return induce_cell_result(
        {"ticket": "gr00t/forge/profile/e2e"},
        [],
        [],
        outcome="measure",
        stop_reason="measure_only",
        start={
            "commit": "0123abc",
            "e2e_ms": 90.585,
            "segments_ms": {"dit.denoise": 45.044},
            "noise_pct": 1.0,
        },
        env={"hardware": "thor"},
        finished_at="2026-10-26T16:00:00+08:00",
    )


def _mini_kb(tmp_path, src: Path) -> Path:
    root = tmp_path / "kb"
    root.mkdir()
    shutil.copy(src / "kb.toml", root / "kb.toml")
    shutil.copytree(src / "schema", root / "schema")
    shutil.copytree(src / "vocab", root / "vocab")
    shutil.copytree(src / "map", root / "map")
    for sub in ("facts", "lessons", "techniques", "benchmarks", "decisions"):
        (root / "cards" / sub).mkdir(parents=True)
    (root / "archive").mkdir()
    (root / "evidence").mkdir()
    return root


# 用例: propose 参数是 lesson，地图钉在工单那一格，claim 带端到端毫秒
# 参数: 无
def test_propose_argv_pins_the_ticket_cell():
    result = {
        "ticket": "gr00t/forge/profile/e2e",
        "outcome": "measure",
        "start": {"e2e_ms": 90.585},
        "env": {"hardware": "thor"},
    }
    argv = propose_argv(result, author="agent:kal/test", evidence_text="只记账\n")
    assert argv[:3] == ["propose", "--type", "lesson"]
    assert "--map-node" in argv and argv[argv.index("--map-node") + 1] == "profile"
    assert argv[argv.index("--map-module") + 1] == "e2e"
    assert argv[argv.index("--map-outcome") + 1] == "measure"
    claim = argv[argv.index("--claim") + 1]
    assert "90.585" in claim
    assert "accept" not in argv


# 用例: 草稿进 inbox，kb next 把这一格标成 pending，cards/ 里没有新卡
# 参数: tmp_path 是临时知识库；kb_src 是 AiInfraKB 正本；monkeypatch 清掉 AIINFRAKB_ROOT
def test_propose_cell_result_lands_in_inbox_and_blocks_next(tmp_path, kb_src, monkeypatch):
    root = _mini_kb(tmp_path, kb_src)
    cmd = _kb_cmd(kb_src)
    result = _measure(monkeypatch)
    rel = propose_cell_result(
        result,
        kb_root=root,
        author="agent:kal/test",
        evidence_text="只记账，没有改代码\n",
        kb_cmd=cmd,
    )
    assert rel == "inbox/kb-0001.md"
    assert (root / rel).is_file()
    assert not any((root / "cards").rglob("*.md"))
    proc = subprocess.run(
        [*cmd, "--root", str(root), "next", "--hardware", "thor", "--model", "gr00t", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    chosen = json.loads(proc.stdout)
    assert {"node": "profile", "module": "e2e", "card": "kb-0001"} in chosen["pending"]
    assert (chosen["next"]["node"], chosen["next"]["module"]) != ("profile", "e2e")


# 用例: 对不上契约的结果在调用 kb 之前拒绝
# 参数: outcome 非法；kb_src 是 AiInfraKB 正本；kb_cmd 指向不存在的程序，被调用就会失败
def test_propose_cell_result_rejects_invalid_result_before_kb(tmp_path, kb_src):
    root = _mini_kb(tmp_path, kb_src)
    result = {"ticket": "gr00t/forge/profile/e2e", "outcome": "nope"}
    with pytest.raises(CellResultError, match="契约"):
        propose_cell_result(
            result,
            kb_root=root,
            author="agent:kal/test",
            evidence_text="只记账\n",
            kb_cmd=["/nonexistent/kb"],
        )
