"""模块: kernel_agent_loop.evald.diagnose 与 AiInfraKB vocab/kernel_symptoms.yaml
功能: RULES 的症状码是 kernel 卡 kernel.symptom 的正本；知识库词表必须与它的键完全一致。
"""

import os
import re
from pathlib import Path

import pytest

from kernel_agent_loop.evald.diagnose import RULES

VOCAB = Path("vocab") / "kernel_symptoms.yaml"


def _kb_root() -> Path:
    env = os.environ.get("AIINFRAKB_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2].parent / "AiInfraKB"


def vocab_keys(path: Path) -> set[str]:
    return set(re.findall(r"(?m)^([a-z0-9][a-z0-9-]*):", path.read_text(encoding="utf-8")))


# 用例: RULES 的键与知识库症状词表一致
# 参数: AIINFRAKB_ROOT，未设置时用同级目录的 AiInfraKB；正本还没有这份词表时跳过
def test_rules_match_kb_symptom_vocab():
    path = _kb_root() / VOCAB
    if not path.is_file():
        pytest.skip(f"找不到 {path}")
    keys = vocab_keys(path)
    assert keys == set(RULES), f"只在 RULES: {sorted(set(RULES) - keys)}；只在词表: {sorted(keys - set(RULES))}"


# 用例: 词表多一个键或少一个键都能被比出来；注释和缩进行不算键
# 参数: tmp_path 里写一份词表
def test_vocab_keys_detects_drift(tmp_path):
    path = tmp_path / "kernel_symptoms.yaml"
    lines = ["# 注释: 不是键", *(f"{code}: {{signal: ncu}}" for code in RULES), "  nested: {}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert vocab_keys(path) == set(RULES)
    path.write_text("\n".join(lines[:-2]) + "\nnew-symptom: {signal: ncu}\n", encoding="utf-8")
    assert vocab_keys(path) != set(RULES)
