"""模块: kernel_agent_loop.loop
功能: 一轮优化在开发机上完成静态检查、提交、送评、门禁、记日志和接受/回退，incumbent 取自 campaign.best。
"""

import shutil
import time
from pathlib import Path

import pytest

from kernel_agent_loop.backends.cuda import Artifact, BuildError
from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import ExperimentLog
from kernel_agent_loop.core.gate import AcceptanceGate
from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import EnvSnapshot, Verdict
from kernel_agent_loop.evald.client import EvalClient
from kernel_agent_loop.evald.config import EvaldConfig
from kernel_agent_loop.evald.env_guard import EnvGuard
from kernel_agent_loop.loop import AlreadyTried, run_round

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
        text = artifact.cubin_path.read_text()
        spec = self.spec

        def run(inputs, outputs, shape) -> None:
            if "SLOW" in text:
                time.sleep(0.01)
            for name, value in spec.reference(inputs).items():
                outputs[name][...] = value

        return run


@pytest.fixture
def setup(op_repo, repo_root, tmp_path):
    spec = load_opspec(repo_root / "ops", "rmsnorm")
    trusted = tmp_path / "trusted"
    shutil.copytree(op_repo, trusted, ignore=shutil.ignore_patterns(".git"))
    config = EvaldConfig(
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
    client = EvalClient(mode="local", config=config, backend=FakeBackend(spec), env_probe=lambda: STEADY)
    campaign = Campaign.create(op_repo, campaign_id="rms-1", op="rmsnorm", backend="cuda", hardware="thor")
    log = ExperimentLog(tmp_path / "experiments.sqlite")
    gate = AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0)

    def round_(source: str, lever: str):
        (op_repo / IMPL / "kernel.cu").write_text(source)
        return run_round(campaign, client=client, log=log, gate=gate, lever=lever, hypothesis="unit")

    return campaign, log, round_, op_repo


# 用例: 更快的候选接受并前移 best；之后更慢的候选相对新 best 回退，工作区恢复成 best
# 参数: setup 提供 campaign、日志、单轮驱动和临时仓库
def test_rounds_accept_faster_and_revert_slower(setup):
    campaign, log, round_, repo = setup
    first = round_("CANDIDATE_SLOW\n", "naive")
    assert first.decision.verdict is Verdict.ACCEPT
    second = round_("CANDIDATE_OK\n", "drop-sleep")
    assert second.decision.verdict is Verdict.ACCEPT
    assert all(perf.incumbent_us is not None for perf in second.result.per_shape.values())
    assert campaign.best == second.commit
    third = round_("CANDIDATE_SLOW\n", "add-sleep")
    assert third.decision.verdict is Verdict.REVERT
    assert campaign.best == second.commit
    assert (repo / IMPL / "kernel.cu").read_text() == "CANDIDATE_OK\n"
    assert [a.verdict for a in log.attempts("rms-1")] == ["accept", "accept", "revert"]


# 用例: 静态检查不过的候选不送评，直接拒绝并回退
# 参数: setup 提供 campaign、日志、单轮驱动和临时仓库
def test_static_judge_rejects_without_submitting(setup):
    campaign, log, round_, repo = setup
    best = campaign.best
    outcome = round_('CANDIDATE_OK\n// fopen("/etc/passwd")\n', "escape")
    assert outcome.decision.verdict is Verdict.REJECT
    assert outcome.result is None
    assert campaign.best == best
    assert any("forbidden escape" in s for s in log.attempts("rms-1")[0].symptoms)


# 用例: 同一 best 上重复同一手段名直接报错，不消耗评测
# 参数: setup 提供 campaign、日志、单轮驱动和临时仓库
def test_same_lever_on_same_parent_is_refused(setup):
    _, _, round_, _ = setup
    round_("CANDIDATE_OK\n", "naive")
    assert round_("CANDIDATE_SLOW\n", "x").decision.verdict is Verdict.REVERT
    with pytest.raises(AlreadyTried):
        round_("CANDIDATE_SLOW\n", "x")
