"""模块: kernel_agent_loop.evald.evaluate isolate
功能: isolate=True 时通过 fake worker 子进程完成一轮评测。
"""

from pathlib import Path

from kernel_agent_loop.backends.cuda import Artifact, BuildError
from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import ExperimentLog
from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import EnvSnapshot, Verdict
from kernel_agent_loop.evald.evaluate import evaluate


IMPL = "ops/rmsnorm/impls/cuda"
STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)


class FakeBackend:
    def build(self, source_dir: Path) -> Artifact:
        text = (Path(source_dir) / "kernel.cu").read_text()
        if "COMPILE_FAIL" in text or "CANDIDATE" not in text:
            raise BuildError("error: not a candidate")
        return Artifact(cubin_path=Path(source_dir) / "kernel.cu", source_dir=Path(source_dir))

    def load(self, artifact: Artifact):
        raise AssertionError("isolate path must not call backend.load")


# 用例: isolate + fake worker 接受第一个正确候选
# 参数: op_repo 临时仓库；repo_root 加载契约
def test_evaluate_isolate_accepts_first_correct_candidate(op_repo, repo_root):
    spec = load_opspec(repo_root / "ops", "rmsnorm")
    campaign = Campaign.create(op_repo, campaign_id="rms-iso", op="rmsnorm", backend="cuda", hardware="thor")
    (op_repo / ".kal").mkdir(exist_ok=True)
    log = ExperimentLog(op_repo / ".kal" / "experiments.sqlite")
    (op_repo / IMPL / "kernel.cu").write_text("CANDIDATE_OK\n")
    sha = campaign.commit_candidate(lever="naive", hypothesis="unit")
    outcome = evaluate(
        campaign,
        spec=spec,
        log=log,
        lever="naive",
        hypothesis="unit",
        commit=sha,
        backend=FakeBackend(),
        env_probe=lambda: STEADY,
        shapes=spec.public_shapes()[:1],
        rounds=2,
        reps=1,
        isolate=True,
        worker_args=["--fake"],
    )
    assert outcome.decision.verdict is Verdict.ACCEPT
