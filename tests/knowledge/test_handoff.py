"""模块: kernel_agent_loop.knowledge.handoff
功能: 格子结果经 kb return 交回 AiInfraKB：成卡由知识库负责，草稿钉在同一格、留在 inbox，不自动接受；
成功后释放这一格的占位。
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kernel_agent_loop.knowledge.cell_result import CellResultError, induce_cell_result
from kernel_agent_loop.knowledge.handoff import return_argv, return_cell_result

AUTHOR = "agent:kal/test"


def _kb_cmd(src: Path):
    py = shutil.which("python3.12")
    kb = src / "tools" / "kb.py"
    if py is None or not kb.is_file():
        pytest.skip("需要 python3.12 和 AiInfraKB 的 tools/kb.py")
    if not (src / "tools" / "kb_return.py").is_file():
        pytest.skip(f"{src} 没有 kb return")
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


# 用例: kb return 从标准输入读结果，带 author、note，并要 JSON 回执
# 参数: note 为空时不传 --note
def test_return_argv_reads_stdin():
    assert return_argv(author=AUTHOR, note="计时表") == [
        "return", "--file", "-", "--author", AUTHOR, "--note", "计时表", "--json",
    ]
    assert "--note" not in return_argv(author=AUTHOR, note=None)


# 用例: 草稿进 inbox，kb next 把这一格标成 pending，cards/ 里没有新卡；证据里有原始 JSON；同一 agent 的占位被释放
# 参数: tmp_path 是临时知识库；kb_src 是 AiInfraKB 正本；monkeypatch 清掉 AIINFRAKB_ROOT
def test_return_cell_result_lands_in_inbox_and_releases_lease(tmp_path, kb_src, monkeypatch):
    root = _mini_kb(tmp_path, kb_src)
    lease = root / "leases" / "gr00t__forge__profile__e2e.yaml"
    lease.parent.mkdir()
    lease.write_text(f"holder: {AUTHOR}\nexpires: '2099-01-01T00:00:00+08:00'\n", encoding="utf-8")
    cmd = _kb_cmd(kb_src)
    result = _measure(monkeypatch)
    receipt = return_cell_result(result, kb_root=root, author=AUTHOR, note="只记账，没有改代码", kb_cmd=cmd)
    assert receipt["path"] == "inbox/kb-0001.md"
    assert receipt["lease"] == "released"
    assert (root / receipt["path"]).is_file()
    assert not lease.exists()
    assert not any((root / "cards").rglob("*.md"))
    summary = (root / "evidence" / "kb-0001" / "summary.md").read_text(encoding="utf-8")
    assert "只记账，没有改代码" in summary
    assert json.loads(summary.split("```json\n", 1)[1].split("\n```", 1)[0]) == result
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
def test_return_cell_result_rejects_invalid_result_before_kb(tmp_path, kb_src):
    root = _mini_kb(tmp_path, kb_src)
    result = {"ticket": "gr00t/forge/profile/e2e", "outcome": "nope"}
    with pytest.raises(CellResultError, match="契约"):
        return_cell_result(result, kb_root=root, author=AUTHOR, kb_cmd=["/nonexistent/kb"])
