"""模块: kernel_agent_loop.evald.serve
功能: evald_run 在 flock 下展开候选包并测出 EvalResult。
"""

import shutil
import time
from pathlib import Path

from kernel_agent_loop.backends.cuda import Artifact, BuildError
from kernel_agent_loop.core.types import EnvSnapshot, Verdict
from kernel_agent_loop.core.gate import AcceptanceGate
from kernel_agent_loop.evald.bundle import make_bundle
from kernel_agent_loop.evald.config import EvaldConfig
from kernel_agent_loop.evald.env_guard import EnvGuard
from kernel_agent_loop.evald.serve import evald_run


STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)
IMPL = Path("ops/rmsnorm/impls/cuda")


class FakeBackend:
    def __init__(self, spec) -> None:
        self.spec = spec

    def build(self, source_dir: Path) -> Artifact:
        text = (Path(source_dir) / "kernel.cu").read_text()
        if "COMPILE_FAIL" in text or "CANDIDATE" not in text:
            raise BuildError("error: not a candidate")
        return Artifact(cubin_path=Path(source_dir) / "kernel.cu", source_dir=Path(source_dir))

    def load(self, artifact: Artifact):
        text = artifact.cubin_path.read_text()
        spec = self.spec

        def run(inputs, outputs, shape) -> None:
            if "SLOW" in text:
                time.sleep(0.02)
            if "WRONG" in text:
                for array in outputs.values():
                    array.fill(0)
                return
            for name, value in spec.reference(inputs).items():
                outputs[name][...] = value

        return run


def _trusted(op_repo: Path, repo_root: Path, tmp_path: Path) -> Path:
    trusted = tmp_path / "trusted"
    shutil.copytree(op_repo, trusted, ignore=shutil.ignore_patterns(".git"))
    shutil.copy2(repo_root / "ops/rmsnorm/spec.py", trusted / "ops/rmsnorm/spec.py")
    return trusted


def _config(trusted: Path, tmp_path: Path, repo_root: Path, **kwargs) -> EvaldConfig:
    defaults = dict(
        trusted=trusted,
        work_root=tmp_path / "jobs",
        lock_path=tmp_path / "gpu.lock",
        ops_root=repo_root / "ops",
        isolate=False,
        prepare_env=False,
        rounds=3,
        reps=1,
        hidden_shapes=(),
    )
    defaults.update(kwargs)
    return EvaldConfig(**defaults)


def _bundle(dev: Path, *, commit: str = "c1", parent: str = "c0") -> bytes:
    meta = {
        "campaign": "rms-1",
        "commit": commit,
        "parent": parent,
        "op": "rmsnorm",
        "backend": "cuda",
        "lever": "naive",
        "hypothesis": "unit",
    }
    return make_bundle(dev, "rmsnorm", "cuda", meta)


# 用例: 正确候选经 evald_run 得到 compile_ok 且 correct
# 参数: op_repo 临时仓库；repo_root 提供真实 OpSpec；tmp_path 工作区
def test_evald_run_accepts_correct_candidate(op_repo, repo_root, tmp_path):
    from kernel_agent_loop.core.opspec import load_opspec

    trusted = _trusted(op_repo, repo_root, tmp_path)
    (op_repo / IMPL / "kernel.cu").write_text("CANDIDATE_OK\n")
    spec = load_opspec(repo_root / "ops", "rmsnorm")
    result = evald_run(
        _bundle(op_repo, commit="c1", parent="c1"),
        _config(trusted, tmp_path, repo_root),
        env_probe=lambda: STEADY,
        backend=FakeBackend(spec),
    )
    assert result.compile_ok and result.correct
    assert result.commit == "c1"
    gate = AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0)
    assert gate.decide(result).verdict is Verdict.ACCEPT


# 用例: 算错候选 correct=False
# 参数: op_repo 临时仓库；repo_root 提供真实 OpSpec；tmp_path 工作区
def test_evald_run_rejects_wrong_candidate(op_repo, repo_root, tmp_path):
    from kernel_agent_loop.core.opspec import load_opspec

    trusted = _trusted(op_repo, repo_root, tmp_path)
    (op_repo / IMPL / "kernel.cu").write_text("CANDIDATE_WRONG\n")
    spec = load_opspec(repo_root / "ops", "rmsnorm")
    result = evald_run(
        _bundle(op_repo),
        _config(trusted, tmp_path, repo_root),
        env_probe=lambda: STEADY,
        backend=FakeBackend(spec),
    )
    assert result.compile_ok is True
    assert result.correct is False


# 用例: 隐藏 shape 会并入评测集合
# 参数: op_repo 临时仓库；repo_root 提供真实 OpSpec；tmp_path 工作区
def test_evald_run_includes_hidden_shapes(op_repo, repo_root, tmp_path):
    from kernel_agent_loop.core.opspec import load_opspec
    from kernel_agent_loop.core.types import Shape

    trusted = _trusted(op_repo, repo_root, tmp_path)
    (op_repo / IMPL / "kernel.cu").write_text("CANDIDATE_OK\n")
    spec = load_opspec(repo_root / "ops", "rmsnorm")
    hidden = (Shape("hidden-r2-h16", {"rows": 2, "hidden": 16}),)
    result = evald_run(
        _bundle(op_repo, commit="c1", parent="c1"),
        _config(trusted, tmp_path, repo_root, hidden_shapes=hidden),
        env_probe=lambda: STEADY,
        backend=FakeBackend(spec),
    )
    assert "hidden-r2-h16" in result.per_shape
