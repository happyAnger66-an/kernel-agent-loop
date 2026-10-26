"""模块: kernel_agent_loop.evald.client
功能: 本地模式 EvalClient 同步提交候选包并返回 EvalResult。
"""

import shutil
from pathlib import Path

from kernel_agent_loop.backends.cuda import Artifact, BuildError
from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import EnvSnapshot
from kernel_agent_loop.evald.bundle import make_bundle
from kernel_agent_loop.evald.client import EvalClient
from kernel_agent_loop.evald.config import EvaldConfig
from conftest import small_ops_root


STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)
IMPL = Path("ops/rmsnorm/impls/cuda")


class FakeBackend:
    def __init__(self, spec) -> None:
        self.spec = spec

    def build(self, source_dir: Path) -> Artifact:
        text = (Path(source_dir) / "kernel.cu").read_text()
        if "CANDIDATE" not in text:
            raise BuildError("error: not a candidate")
        return Artifact(cubin_path=Path(source_dir) / "kernel.cu", source_dir=Path(source_dir))

    def load(self, artifact: Artifact):
        spec = self.spec

        def run(inputs, outputs, shape) -> None:
            for name, value in spec.reference(inputs).items():
                outputs[name][...] = value

        return run


# 用例: local 模式 submit_sync 返回正确结果
# 参数: op_repo 临时仓库；repo_root 提供 OpSpec；tmp_path 工作区
def test_local_client_submit_sync(op_repo, repo_root, tmp_path):
    trusted = tmp_path / "trusted"
    shutil.copytree(op_repo, trusted, ignore=shutil.ignore_patterns(".git"))
    shutil.copy2(repo_root / "ops/rmsnorm/spec.py", trusted / "ops/rmsnorm/spec.py")
    (op_repo / IMPL / "kernel.cu").write_text("CANDIDATE_OK\n")
    spec = load_opspec(small_ops_root(repo_root), "rmsnorm")
    config = EvaldConfig(
        trusted=trusted,
        work_root=tmp_path / "jobs",
        lock_path=tmp_path / "gpu.lock",
        ops_root=small_ops_root(repo_root),
        isolate=False,
        rounds=2,
        reps=1,
    )
    client = EvalClient(
        mode="local",
        config=config,
        backend=FakeBackend(spec),
        env_probe=lambda: STEADY,
    )
    bundle = make_bundle(
        op_repo,
        "rmsnorm",
        "cuda",
        {
            "campaign": "rms-1",
            "commit": "c1",
            "parent": "c1",
            "op": "rmsnorm",
            "backend": "cuda",
            "lever": "naive",
            "hypothesis": "unit",
        },
    )
    result = client.submit_sync(bundle)
    assert result.compile_ok and result.correct
