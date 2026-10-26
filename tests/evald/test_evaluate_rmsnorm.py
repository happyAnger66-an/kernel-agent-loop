"""模块: kernel_agent_loop.evald.evaluate
功能: 用真的 RMSNorm cubin 走评测入口，算错必须拒绝，正确起点必须接受，故意变慢必须回退。
"""

import shutil

import pytest

from conftest import git

from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import ExperimentLog
from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import EnvSnapshot, Verdict
from kernel_agent_loop.evald.evaluate import evaluate

pytestmark = pytest.mark.thor

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not found", allow_module_level=True)
pytest.importorskip("cuda")

IMPL = "ops/rmsnorm/impls/cuda"
STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)
SPIN = """  if (threadIdx.x == 0) {
    unsigned long long start = clock64();
    while (clock64() - start < 20000ULL) {}
  }
  __syncthreads();
"""


def _install(op_repo, repo_root, kernel_text: str) -> None:
    src = repo_root / "ops" / "rmsnorm" / "impls" / "cuda"
    dest = op_repo / IMPL
    (dest / "launch.py").write_text((src / "launch.py").read_text())
    (dest / "kernel.cu").write_text(kernel_text)


def _evaluate(op_repo, repo_root, kernel_text: str, *, lever: str, pin_current_as_best: bool = False):
    spec = load_opspec(repo_root / "ops", "rmsnorm")
    if pin_current_as_best:
        _install(op_repo, repo_root, (repo_root / "ops/rmsnorm/impls/cuda/kernel.cu").read_text())
        git(op_repo, "add", "-A")
        git(op_repo, "commit", "-q", "-m", "fast baseline")
    campaign = Campaign.create(op_repo, campaign_id="rms-1", op="rmsnorm", backend="cuda", hardware="thor")
    _install(op_repo, repo_root, kernel_text)
    sha = campaign.commit_candidate(lever=lever, hypothesis="gpu")
    (op_repo / ".kal").mkdir(exist_ok=True)
    log = ExperimentLog(op_repo / ".kal" / "experiments.sqlite")
    outcome = evaluate(
        campaign,
        spec=spec,
        log=log,
        lever=lever,
        hypothesis="gpu",
        commit=sha,
        env_probe=lambda: STEADY,
        rounds=3,
        reps=1,
        seed=0,
    )
    return spec, campaign, log, outcome


# 用例: 把输出写成输入的 RMSNorm 必须被拒绝，症状含 mismatch
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约和正确 kernel
def test_wrong_rmsnorm_is_rejected(op_repo, repo_root):
    correct = (repo_root / "ops/rmsnorm/impls/cuda/kernel.cu").read_text()
    wrong = correct.replace(
        "    const float value = __half2float(row_x[i]) * inv * __half2float(weight[i]);\n"
        "    row_y[i] = __float2half(value);",
        "    row_y[i] = row_x[i];",
    )
    _, _, log, outcome = _evaluate(op_repo, repo_root, wrong, lever="copy-input")
    assert outcome.decision.verdict is Verdict.REJECT
    assert any("mismatch" in symptom for symptom in log.attempts("rms-1")[0].symptoms)


# 用例: 仓库里的正确 kernel 在没有可运行 incumbent 时被接受
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约和正确 kernel
def test_checked_in_rmsnorm_is_accepted_as_first_candidate(op_repo, repo_root):
    correct = (repo_root / "ops/rmsnorm/impls/cuda/kernel.cu").read_text()
    _, campaign, _, outcome = _evaluate(op_repo, repo_root, correct, lever="naive")
    assert outcome.decision.verdict is Verdict.ACCEPT
    assert all(perf.incumbent_us is None for perf in outcome.result.per_shape.values())
    assert campaign.best != outcome.result.commit


# 用例: 128-bit 向量化 + warp shuffle 的 RMSNorm 相对朴素 incumbent 必须被接受
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约和正确 kernel
def test_vectorized_rmsnorm_is_accepted_over_naive(op_repo, repo_root):
    fast = (repo_root / "tests/evald/rmsnorm_vec.cu").read_text()
    _, campaign, _, outcome = _evaluate(
        op_repo, repo_root, fast, lever="vec128-warp-reduce", pin_current_as_best=True
    )
    assert outcome.decision.verdict is Verdict.ACCEPT, outcome.decision.reasons
    assert all(perf.candidate_us < perf.incumbent_us for perf in outcome.result.per_shape.values())


# 用例: 在正确 kernel 上插入约 20k 周期（十几微秒）空转后，相对该 incumbent 必须回退
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约和正确 kernel
def test_slower_rmsnorm_is_reverted(op_repo, repo_root):
    correct = (repo_root / "ops/rmsnorm/impls/cuda/kernel.cu").read_text()
    slow = correct.replace("  const float inv = rsqrtf", SPIN + "  const float inv = rsqrtf")
    spec, campaign, _, outcome = _evaluate(
        op_repo, repo_root, slow, lever="busy-wait", pin_current_as_best=True
    )
    assert outcome.decision.verdict is Verdict.REVERT
    assert any(spec.public_shapes()[0].label in reason for reason in outcome.decision.reasons)
    assert campaign.best != outcome.result.commit
