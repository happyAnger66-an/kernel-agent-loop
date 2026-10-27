"""本仓 contracts/ 是 AiInfraKB schema 的副本。正本在知识库。"""

import os
from pathlib import Path

CONTRACT_NAMES = ("ticket.schema.json", "cell-result.schema.json")


class ContractMismatch(Exception):
    """副本和 AiInfraKB 正本不一致，或正本缺文件。"""


def contracts_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
    path = root / "contracts"
    if not path.is_dir():
        raise ContractMismatch("找不到 contracts/，请在仓库里使用这份代码")
    return path


def check_contracts(kb_root: str | Path | None = None) -> None:
    """比对副本和 AiInfraKB 的 schema/。

    传入 kb_root 时总是比对。不传时只在设置了 AIINFRAKB_ROOT 时比对；没设置就跳过。
    """
    if kb_root is None:
        env = os.environ.get("AIINFRAKB_ROOT")
        if not env:
            return
        kb_root = env
    root = Path(kb_root)
    local = contracts_dir()
    for name in CONTRACT_NAMES:
        theirs = root / "schema" / name
        if not theirs.is_file():
            raise ContractMismatch(f"AiInfraKB 缺少 schema/{name}")
        if (local / name).read_bytes() != theirs.read_bytes():
            raise ContractMismatch(f"contracts/{name} 与 AiInfraKB schema/{name} 不一致")
