"""把格子结果写成 AiInfraKB 的 lesson 草稿。不接受，不移出 inbox/。"""

import shutil
import subprocess
import sys
from pathlib import Path

from kernel_agent_loop.knowledge.cell_result import CellResultError, validate_cell_result
from kernel_agent_loop.knowledge.contracts import check_contracts


class HandoffError(Exception):
    """kb propose 没有写成草稿。"""


def propose_cell_result(
    result: dict,
    *,
    kb_root: str | Path,
    author: str,
    evidence_text: str,
    kb_cmd: list[str] | None = None,
) -> str:
    """调用 kb propose --type lesson。返回 inbox 里的相对路径。"""
    check_contracts(kb_root)
    validate_cell_result(result)
    if not evidence_text.strip():
        raise HandoffError("evidence_text 不能为空")
    cmd = list(kb_cmd) if kb_cmd is not None else default_kb_cmd()
    argv = [*cmd, "--root", str(kb_root), *propose_argv(result, author=author, evidence_text=evidence_text)]
    proc = subprocess.run(argv, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HandoffError((proc.stderr or proc.stdout).strip())
    rel = proc.stdout.strip()
    if not rel.startswith("inbox/"):
        raise HandoffError(f"kb propose 没有写入 inbox/: {rel}")
    return rel


def propose_argv(result: dict, *, author: str, evidence_text: str) -> list[str]:
    model, _track, node, module = _split_ticket(result.get("ticket"))
    point = result.get("final") or result.get("start") or {}
    e2e = point.get("e2e_ms")
    if not isinstance(e2e, (int, float)) or isinstance(e2e, bool):
        raise CellResultError("格子结果缺少端到端毫秒，claim 需要数字")
    return [
        "propose",
        "--type",
        "lesson",
        "--title",
        f"{model} {node}×{module} {result.get('outcome')}",
        "--claim",
        f"端到端 {e2e} ms，结论 {result.get('outcome')}",
        "--hardware",
        str((result.get("env") or {}).get("hardware")),
        "--model",
        model,
        "--map-node",
        node,
        "--map-module",
        module,
        "--map-outcome",
        str(result.get("outcome")),
        "--evidence-text",
        evidence_text,
        "--author",
        author,
    ]


def default_kb_cmd() -> list[str]:
    sibling = Path(__file__).resolve().parents[4] / "AiInfraKB" / "tools" / "kb.py"
    py = shutil.which("python3.12") or sys.executable
    if sibling.is_file():
        return [py, str(sibling)]
    return ["kb"]


def _split_ticket(ticket) -> tuple[str, str, str, str]:
    parts = str(ticket or "").split("/")
    if len(parts) != 4 or any(not part for part in parts):
        raise CellResultError(f"ticket 必须是模型/场景/节点/模块: {ticket}")
    return parts[0], parts[1], parts[2], parts[3]
