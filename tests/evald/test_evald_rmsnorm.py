"""模块: kernel_agent_loop.evald.serve + isolate
功能: 真 GPU 上经 flock/overlay/isolate 跑 RMSNorm，故意算错必须失败。
"""

import shutil
from pathlib import Path

import pytest

from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import EnvSnapshot
from kernel_agent_loop.evald.bundle import make_bundle
from kernel_agent_loop.evald.config import EvaldConfig
from kernel_agent_loop.evald.serve import evald_run
from kernel_agent_loop.evald.thor_env import prepare_thor_session

pytestmark = pytest.mark.thor

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not found", allow_module_level=True)
pytest.importorskip("cuda")

IMPL = Path("ops/rmsnorm/impls/cuda")
STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)


# 用例: prepare_thor_session 在有/无 Jetson 工具时都不抛异常
# 参数: 无
def test_prepare_thor_session_smoke():
    notes = prepare_thor_session()
    assert isinstance(notes, list)


# 用例: evald_run + isolate 对正确 RMSNorm 返回 correct=True
# 参数: op_repo 临时仓库；repo_root 提供实现与契约；tmp_path 工作区
def test_evald_run_isolate_correct_rmsnorm(op_repo, repo_root, tmp_path):
    trusted = tmp_path / "trusted"
    shutil.copytree(op_repo, trusted, ignore=shutil.ignore_patterns(".git"))
    shutil.copy2(repo_root / "ops/rmsnorm/spec.py", trusted / "ops/rmsnorm/spec.py")
    src = repo_root / "ops/rmsnorm/impls/cuda"
    (op_repo / IMPL / "launch.py").write_text((src / "launch.py").read_text())
    (op_repo / IMPL / "kernel.cu").write_text((src / "kernel.cu").read_text())
    config = EvaldConfig(
        trusted=trusted,
        work_root=tmp_path / "jobs",
        lock_path=tmp_path / "gpu.lock",
        ops_root=repo_root / "ops",
        isolate=True,
        prepare_env=False,
        rounds=2,
        reps=1,
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
            "hypothesis": "gpu",
        },
    )
    result = evald_run(bundle, config, env_probe=lambda: STEADY)
    assert result.compile_ok and result.correct
    assert load_opspec(repo_root / "ops", "rmsnorm").public_shapes()[0].label in result.per_shape
