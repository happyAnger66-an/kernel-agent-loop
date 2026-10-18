# kernel-agent-loop

由 Agent 主导的 GPU 算子优化闭环：Agent 改 kernel，本仓库在硬件上评测并决定接受还是回退。变快且正确的改动留下，失败的尝试记进实验日志。

本仓库不调用大模型。Cursor、Claude Code、Hermes、Grok bot 负责读结果、提假设、改代码。接受与否由 `AcceptanceGate` 根据评测结果判定，不采用 Agent 自己报的计时。

目标硬件是 Jetson Thor。当前落地的后端是 CUDA C++（`nvcc -cubin`）。Triton、CuTe DSL、TIRx 还没有后端实现。

设计见 [docs/architecture.md](docs/architecture.md)，在线阅读见 [Agent 主导的算子优化闭环：整体架构](http://www.xiaoanger.com/topics/ai-native/agent-kernel/kernel-agent-loop/)。

## 当前能用什么

已经能用的是库，不是命令行。没有 `kal` CLI，没有 MCP，也没有常驻的评测守护进程。

| 能力 | 入口 |
|---|---|
| 算子契约 | `ops/<op>/spec.py`，用 `load_opspec` 加载。现有算子是 RMSNorm |
| 候选的提交、接受、回退 | `Campaign`。每个 campaign 一条分支 `kal/<id>`，最优提交记在 `.git/kal/<id>.json` |
| 正确性 | `check_correctness`：每轮新种子、输出先填 NaN、核对输入没被改写 |
| 计时统计 | `interleaved_bench`：候选和当前最优交替跑，噪声取逐轮中位数的极差 |
| 接受门禁 | `AcceptanceGate` |
| 静态检查 | `StaticJudge`：只允许改实现目录，拦截进程、网络和文件逃逸 |
| 实验日志 | `ExperimentLog`：记录每次尝试，按 campaign 和父版本去重手段 |
| 候选包 | `make_bundle` / `overlay_bundle`：包里只有实现目录 |
| CUDA 后端 | `CudaBackend`：编译 cubin 并用 `run(inputs, outputs, shape)` 启动 |

还没有的部分：ncu/SASS 症状诊断、L2 调优库、AiInfraKB 接入、Triton 和 CuTe DSL 后端。架构文档里的这些设计仍然有效，只是本阶段没有实现。

## 安装

需要 Python 3.10 以上。在仓库根目录：

```bash
git clone git@github.com:happyAnger66-an/kernel-agent-loop.git
cd kernel-agent-loop
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

不安装也可以跑测试：`pyproject.toml` 已经把 `src` 和仓库根加进 pytest 的 `pythonpath`。

```bash
python3 -m pytest
```

默认会跳过标记为 `thor` 的测试。CUDA 后端测试需要本机有 NVIDIA GPU、`nvcc` 和 `cuda-python`：

```bash
pip install -e ".[cuda,test]"
python3 -m pytest -m thor --override-ini="addopts="
```

这组测试在 RTX 4070 上通过过。Thor 上的 `-arch` 由 `CudaBackend` 在运行时向设备查询，不写死。

## 一次优化怎么走

Agent 只改 `ops/<op>/impls/<backend>/`。RMSNorm 的 CUDA 实现约定是：

```text
ops/rmsnorm/impls/cuda/
  kernel.cu    # extern "C" __global__
  launch.py    # launch_config(shape) -> kernel 名、grid、block、动态 smem、参数名
```

`launch.py` 里的参数名从输入、输出和 `shape.dims` 里取。参考实现、公开 shape、容差都在 `ops/rmsnorm/spec.py`，Agent 不能改。

```python
from pathlib import Path

from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import Attempt, ExperimentLog
from kernel_agent_loop.core.gate import AcceptanceGate
from kernel_agent_loop.core.static_judge import StaticJudge
from kernel_agent_loop.evald.env_guard import EnvGuard

repo = Path(".").resolve()
campaign = Campaign.create(repo, campaign_id="rms-1", op="rmsnorm", backend="cuda", hardware="thor")

# 只改 ops/rmsnorm/impls/cuda/，每个候选只做一个手段
judge = StaticJudge(op="rmsnorm", backend="cuda")
# verdict = judge.check({相对路径: 源码})
# verdict.ok 为 False 时不要提交

sha = campaign.commit_candidate(lever="vectorize-loads", hypothesis="用 128-bit load 提高访存宽度")

# result 由评测填入。Gate 不接受调用方自己报的耗时。
gate = AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0)
decision = gate.decide(result)
if decision.verdict.name == "ACCEPT":
    campaign.accept(sha)
else:
    campaign.revert()

ExperimentLog(repo / ".kal" / "experiments.sqlite").record(attempt)
```

`Campaign.load(repo, "rms-1")` 可以在新会话里恢复同一个 campaign。

门禁顺序是：编译失败或不正确则拒绝；降频、功耗模式变化、频率变化或温漂超过 5°C 则这次结果作废；没有当前最优版本的第一个正确候选直接接受。之后要同时满足两条才接受，否则回退：

- 各 shape 加速比的几何平均，超过 `1 + max(各 shape 噪声, 1%)`
- 没有任何 shape 比当前最优慢超过它自己的噪声

shape 集合就是这次优化要守住的范围。只优化某一个 shape 时，集合里只放那个 shape。集合之外的 shape 变慢，不在这套门禁里。

## 知识库

实验日志是现在唯一落地的一层：每次尝试的提交、父提交、手段、假设、判定和症状都会记下，包括失败。`tried(campaign, lever=..., parent=...)` 用来避免在同一个父版本上重复同一个手段。

L2 调优库和 L3 AiInfraKB 还没接入。症状字段可以写入日志，但还没有从 ncu/SASS 自动生成症状的代码。
