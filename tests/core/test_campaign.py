"""模块: kernel_agent_loop.core.campaign
功能: campaign 分支、候选提交只含实现改动、接受移动最优指针、回退恢复实现目录。
"""

from conftest import git

from kernel_agent_loop.core.campaign import Campaign

IMPL = "ops/rmsnorm/impls/cuda"


def new_campaign(repo):
    return Campaign.create(repo, campaign_id="rms-1", op="rmsnorm", backend="cuda", hardware="thor")


# 用例: 创建 campaign 切到 kal/<id> 分支，并把最优指针钉在基线提交
# 参数: op_repo 是带一次初始提交的临时 git 仓库，含 rmsnorm 的 CUDA 实现
def test_create_checks_out_campaign_branch_and_pins_best_to_base(op_repo):
    base = git(op_repo, "rev-parse", "HEAD")
    c = new_campaign(op_repo)
    assert git(op_repo, "branch", "--show-current") == "kal/rms-1"
    assert c.best == base


# 用例: 重新加载 campaign 后算子、后端、硬件和最优指针都不变
# 参数: op_repo 是带一次初始提交的临时 git 仓库，含 rmsnorm 的 CUDA 实现
def test_campaign_round_trips_through_load(op_repo):
    c = new_campaign(op_repo)
    loaded = Campaign.load(op_repo, "rms-1")
    assert (loaded.op, loaded.backend, loaded.hardware, loaded.best) == ("rmsnorm", "cuda", "thor", c.best)


# 用例: 候选提交只包含实现目录的改动，目录外的修改不进提交
# 参数: op_repo 是带一次初始提交的临时 git 仓库，含 rmsnorm 的 CUDA 实现
def test_candidate_commit_contains_only_impl_changes(op_repo):
    c = new_campaign(op_repo)
    (op_repo / IMPL / "kernel.cu").write_text("// v1\n")
    (op_repo / "README.md").write_text("edited outside impl\n")
    sha = c.commit_candidate(lever="vectorize-loads", hypothesis="use 128-bit loads")
    changed = git(op_repo, "show", "--name-only", "--format=", sha).splitlines()
    assert changed == [f"{IMPL}/kernel.cu"]


# 用例: 候选提交说明里记下手段名和假设
# 参数: op_repo 是带一次初始提交的临时 git 仓库，含 rmsnorm 的 CUDA 实现
def test_candidate_commit_records_lever_and_hypothesis(op_repo):
    c = new_campaign(op_repo)
    (op_repo / IMPL / "kernel.cu").write_text("// v1\n")
    sha = c.commit_candidate(lever="vectorize-loads", hypothesis="use 128-bit loads")
    message = git(op_repo, "log", "-1", "--format=%B", sha)
    assert "vectorize-loads" in message and "use 128-bit loads" in message


# 用例: 接受候选后最优指针移到该提交，重新加载仍然指向它
# 参数: op_repo 是带一次初始提交的临时 git 仓库，含 rmsnorm 的 CUDA 实现
def test_accept_moves_best(op_repo):
    c = new_campaign(op_repo)
    (op_repo / IMPL / "kernel.cu").write_text("// v1\n")
    sha = c.commit_candidate(lever="vectorize-loads", hypothesis="h")
    c.accept(sha)
    assert c.best == sha
    assert Campaign.load(op_repo, "rms-1").best == sha


# 用例: 回退把实现目录恢复成最优版本，最优指针不动
# 参数: op_repo 是带一次初始提交的临时 git 仓库，含 rmsnorm 的 CUDA 实现
def test_revert_restores_impl_from_best_and_keeps_best(op_repo):
    c = new_campaign(op_repo)
    best = c.best
    (op_repo / IMPL / "kernel.cu").write_text("// slower\n")
    c.commit_candidate(lever="bad-idea", hypothesis="h")
    c.revert()
    assert (op_repo / IMPL / "kernel.cu").read_text() == "// v0\n"
    assert c.best == best
    assert git(op_repo, "diff", best, "--", IMPL) == ""


# 用例: 在 git worktree 里（.git 是文件）也能创建并重新加载 campaign
# 参数: op_repo 是临时 git 仓库；tmp_path 放 worktree
def test_campaign_works_inside_a_git_worktree(op_repo, tmp_path):
    worktree = tmp_path / "wt"
    git(op_repo, "worktree", "add", "-q", "--detach", str(worktree), "HEAD")
    created = Campaign.create(worktree, campaign_id="wt-1", op="rmsnorm", backend="cuda", hardware="thor")
    assert Campaign.load(worktree, "wt-1").best == created.best
