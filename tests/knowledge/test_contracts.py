"""模块: kernel_agent_loop.knowledge.contracts
功能: 设置了知识库根目录时，contracts/ 必须与 AiInfraKB schema 字节一致。
"""

import shutil

import pytest

from kernel_agent_loop.knowledge.contracts import ContractMismatch, check_contracts, contracts_dir


# 用例: 仓库里的副本和 AiInfraKB 正本一致
# 参数: kb_src 是 AiInfraKB 正本目录（AIINFRAKB_ROOT 或同级目录）
def test_local_contracts_match_aiinfrakb(kb_src):
    check_contracts(kb_src)


# 用例: 正本被改过一个字节时拒绝
# 参数: tmp_path 里放一份 schema，cell-result 末尾多一个换行
def test_check_contracts_rejects_a_different_byte(tmp_path):
    schema = tmp_path / "schema"
    schema.mkdir()
    for name in ("ticket.schema.json", "cell-result.schema.json"):
        shutil.copy(contracts_dir() / name, schema / name)
    changed = schema / "cell-result.schema.json"
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ContractMismatch, match="cell-result"):
        check_contracts(tmp_path)


# 用例: 没设置 AIINFRAKB_ROOT 时不比对
# 参数: monkeypatch 删掉该环境变量
def test_check_contracts_skips_when_root_is_unset(monkeypatch):
    monkeypatch.delenv("AIINFRAKB_ROOT", raising=False)
    check_contracts()
