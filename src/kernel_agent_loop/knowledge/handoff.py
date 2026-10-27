"""把格子结果交回 AiInfraKB。成卡由知识库的 kb return 负责；不接受，不移出 inbox/。"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from kernel_agent_loop.knowledge.cell_result import validate_cell_result
from kernel_agent_loop.knowledge.contracts import check_contracts


class HandoffError(Exception):
    """kb return 没有收下这份结果。"""


def return_cell_result(
    result: dict,
    *,
    kb_root: str | Path,
    author: str,
    note: str | None = None,
    kb_cmd: list[str] | None = None,
) -> dict:
    """调用 kb return，返回回执 {"id", "path", "lease"}。author 要与领工单的 agent 相同，占位才会释放。"""
    check_contracts(kb_root)
    validate_cell_result(result)
    cmd = list(kb_cmd) if kb_cmd is not None else default_kb_cmd()
    argv = [*cmd, "--root", str(kb_root), *return_argv(author=author, note=note)]
    proc = subprocess.run(argv, input=json.dumps(result, ensure_ascii=False), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HandoffError((proc.stderr or proc.stdout).strip())
    receipt = json.loads(proc.stdout)
    if not str(receipt.get("path", "")).startswith("inbox/"):
        raise HandoffError(f"kb return 没有写入 inbox/: {receipt}")
    return receipt


def return_argv(*, author: str, note: str | None) -> list[str]:
    argv = ["return", "--file", "-", "--author", author]
    if note:
        argv += ["--note", note]
    return [*argv, "--json"]


def default_kb_cmd() -> list[str]:
    sibling = Path(__file__).resolve().parents[4] / "AiInfraKB" / "tools" / "kb.py"
    py = shutil.which("python3.12") or sys.executable
    if sibling.is_file():
        return [py, str(sibling)]
    return ["kb"]
