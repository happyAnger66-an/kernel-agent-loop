"""从开发机经 SSH 向 Thor 提交正确/错误 RMSNorm 候选，核对 Gate 判定。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from kernel_agent_loop.core.gate import AcceptanceGate
from kernel_agent_loop.core.types import Verdict
from kernel_agent_loop.evald.bundle import make_bundle
from kernel_agent_loop.evald.client import EvalClient
from kernel_agent_loop.evald.env_guard import EnvGuard

REPO = Path(__file__).resolve().parents[2]
CLIENT = EvalClient(
    mode="ssh",
    host="tz",
    user="nvidia",
    port=2240,
    remote_config="/home/nvidia/kal/evald.config.json",
    evald_cmd="/home/nvidia/kal/venv/bin/python -m kernel_agent_loop.evald.serve",
    remote_workdir="/home/nvidia/kal/inbox",
)


def _bundle(impl_dir: Path, *, commit: str) -> bytes:
    """impl_dir 含 kernel.cu 与 launch.py；打成只含实现目录的候选包。"""
    tmp = Path(tempfile.mkdtemp(prefix="kal-smoke-"))
    try:
        dest = tmp / "ops/rmsnorm/impls/cuda"
        dest.mkdir(parents=True)
        shutil.copy2(impl_dir / "kernel.cu", dest / "kernel.cu")
        shutil.copy2(impl_dir / "launch.py", dest / "launch.py")
        return make_bundle(
            tmp,
            "rmsnorm",
            "cuda",
            {
                "campaign": "smoke",
                "commit": commit,
                "parent": commit,
                "op": "rmsnorm",
                "backend": "cuda",
                "lever": "smoke",
                "hypothesis": "phase-2.5",
            },
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    gate = AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0)
    src = REPO / "ops/rmsnorm/impls/cuda"
    correct = (src / "kernel.cu").read_text()
    wrong_text = correct.replace(
        "    const float value = __half2float(row_x[i]) * inv * __half2float(weight[i]);\n"
        "    row_y[i] = __float2half(value);",
        "    row_y[i] = row_x[i];",
    )
    wrong_dir = Path(tempfile.mkdtemp(prefix="kal-wrong-"))
    try:
        (wrong_dir / "kernel.cu").write_text(wrong_text)
        shutil.copy2(src / "launch.py", wrong_dir / "launch.py")

        print("== wrong candidate ==")
        wrong_result = CLIENT.submit_sync(_bundle(wrong_dir, commit="wrong1"))
        wrong_decision = gate.decide(wrong_result)
        print(wrong_result.compile_ok, wrong_result.correct, wrong_decision.verdict)
        assert wrong_result.compile_ok and not wrong_result.correct
        assert wrong_decision.verdict is Verdict.REJECT

        print("== correct candidate ==")
        ok_result = CLIENT.submit_sync(_bundle(src, commit="ok1"))
        ok_decision = gate.decide(ok_result)
        print(ok_result.compile_ok, ok_result.correct, sorted(ok_result.per_shape), ok_decision.verdict)
        assert ok_result.compile_ok and ok_result.correct
        assert ok_decision.verdict is Verdict.ACCEPT
        assert "hidden-r16-h2048" in ok_result.per_shape

        print("smoke_ssh: OK")
    finally:
        shutil.rmtree(wrong_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
