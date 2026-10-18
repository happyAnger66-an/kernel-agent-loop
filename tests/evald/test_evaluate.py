"""模块: kernel_agent_loop.evald.evaluate
功能: 把编译、正确性、交替计时、环境快照和门禁串成一次评测，并写入实验日志。
"""

import time
from dataclasses import replace
from pathlib import Path

from conftest import git

from kernel_agent_loop.backends.cuda import Artifact, BuildError
from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import ExperimentLog
from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import EnvSnapshot, Verdict
from kernel_agent_loop.evald.evaluate import evaluate

IMPL = "ops/rmsnorm/impls/cuda"
STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)


class FakeBackend:
    """Build 只认带 CANDIDATE 标记的源码；计时用 sleep 区分快慢，不碰 GPU。"""

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


def _campaign(op_repo, repo_root):
    spec = load_opspec(repo_root / "ops", "rmsnorm")
    campaign = Campaign.create(op_repo, campaign_id="rms-1", op="rmsnorm", backend="cuda", hardware="thor")
    (op_repo / ".kal").mkdir(exist_ok=True)
    log = ExperimentLog(op_repo / ".kal" / "experiments.sqlite")
    return spec, campaign, log


def _write(op_repo, source: str) -> None:
    (op_repo / IMPL / "kernel.cu").write_text(source)


def _run(op_repo, repo_root, source: str, *, lever: str, probe=None):
    spec, campaign, log = _campaign(op_repo, repo_root)
    _write(op_repo, source)
    sha = campaign.commit_candidate(lever=lever, hypothesis="unit")
    outcome = evaluate(
        campaign,
        spec=spec,
        log=log,
        lever=lever,
        hypothesis="unit",
        commit=sha,
        backend=FakeBackend(spec),
        env_probe=probe or (lambda: STEADY),
        shapes=spec.public_shapes()[:1],
        rounds=3,
        reps=1,
        seed=0,
    )
    return campaign, log, outcome


# 用例: 编译失败时拒绝，日志记下拒绝，最优指针不动
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约
def test_compile_failure_is_rejected_and_logged(op_repo, repo_root):
    campaign, log, outcome = _run(op_repo, repo_root, "COMPILE_FAIL\n", lever="bad-syntax")
    assert outcome.decision.verdict is Verdict.REJECT
    assert outcome.result.compile_ok is False
    assert log.attempts("rms-1")[0].verdict == "reject"
    assert campaign.best != outcome.result.commit


# 用例: 算错的 kernel 拒绝，症状里带 mismatch
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约
def test_wrong_kernel_is_rejected_with_mismatch_symptom(op_repo, repo_root):
    _, log, outcome = _run(op_repo, repo_root, "CANDIDATE_WRONG\n", lever="zero-output")
    assert outcome.decision.verdict is Verdict.REJECT
    assert any("mismatch" in symptom for symptom in log.attempts("rms-1")[0].symptoms)


# 用例: 基线编不过时，第一个正确候选没有 incumbent，直接接受
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约
def test_first_correct_kernel_without_incumbent_is_accepted(op_repo, repo_root):
    _, log, outcome = _run(op_repo, repo_root, "CANDIDATE_OK\n", lever="naive")
    assert outcome.decision.verdict is Verdict.ACCEPT
    assert all(perf.incumbent_us is None for perf in outcome.result.per_shape.values())
    assert log.attempts("rms-1")[0].verdict == "accept"


# 用例: 正确但更慢的候选回退，原因里带上变慢的 shape
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约
def test_slower_correct_kernel_is_reverted(op_repo, repo_root):
    _write(op_repo, "CANDIDATE_OK\n")
    git(op_repo, "add", "-A")
    git(op_repo, "commit", "-q", "-m", "fast")
    spec, campaign, log = _campaign(op_repo, repo_root)
    _write(op_repo, "CANDIDATE_SLOW\n")
    sha = campaign.commit_candidate(lever="extra-loop", hypothesis="unit")
    outcome = evaluate(
        campaign,
        spec=spec,
        log=log,
        lever="extra-loop",
        hypothesis="unit",
        commit=sha,
        backend=FakeBackend(spec),
        env_probe=lambda: STEADY,
        shapes=spec.public_shapes()[:1],
        rounds=3,
        reps=1,
    )
    label = spec.public_shapes()[0].label
    assert outcome.decision.verdict is Verdict.REVERT
    assert any(label in reason for reason in outcome.decision.reasons)
    assert campaign.best != sha


# 用例: 评测后降频判作废，这次不占用手段名
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约
def test_throttled_environment_is_invalid_and_lever_stays_available(op_repo, repo_root):
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        if calls["n"] >= 2:
            return replace(STEADY, throttled=True)
        return STEADY

    _, log, outcome = _run(op_repo, repo_root, "CANDIDATE_OK\n", lever="naive", probe=probe)
    assert outcome.decision.verdict is Verdict.INVALID
    assert not log.tried("rms-1", lever="naive", parent=log.attempts("rms-1")[0].parent)
