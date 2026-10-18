# kernel-agent-loop（给 Agent）

改 kernel 时遵守这里。架构和取舍在 `docs/architecture.md`，给人看的说明在 `README.md`。

## 可以改什么

只改 `ops/<op>/impls/<backend>/`。当前唯一的算子是 RMSNorm，唯一的后端是 CUDA：

```text
ops/rmsnorm/impls/cuda/kernel.cu
ops/rmsnorm/impls/cuda/launch.py
```

`kernel.cu` 里用 `extern "C" __global__`。`launch.py` 提供 `launch_config(shape)`，返回 `kernel`、`grid`、`block`、`smem`、`args`。`args` 里的名字必须是输入名、输出名或 `shape.dims` 的键。

不要改这些文件：

- `ops/<op>/spec.py`：参考实现、公开 shape、容差、`eps`
- `src/kernel_agent_loop/`：评测、门禁、候选包、campaign
- `tests/`：已经 review 过的契约。修 bug 时另说，不要为了让候选通过去改断言

每个候选只做一个手段（lever），并在提交说明里写明手段名和假设。

## 提交候选

没有 CLI。用库接口，在仓库根目录执行：

```python
from pathlib import Path
from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.static_judge import StaticJudge

repo = Path(".").resolve()
campaign = Campaign.load(repo, "<campaign-id>")  # 没有就 Campaign.create(...)
verdict = StaticJudge(op=campaign.op, backend=campaign.backend).check(files)
# files: {仓库相对路径: 源码}。verdict.ok 为 False 时停止，不要提交
sha = campaign.commit_candidate(lever="<手段>", hypothesis="<假设>")
```

`commit_candidate` 只会把 `ops/<op>/impls/<backend>/` 的改动放进提交。该目录以外的修改留在工作区，不会被这次提交带走。

`StaticJudge` 会拒绝实现目录以外的路径，以及源码里的 `subprocess`、`socket`、`os.environ`、`open(`、`fopen(`、`system(`。

接受或回退不要自己判断。把评测产生的 `EvalResult` 交给 `AcceptanceGate.decide`：

- `ACCEPT`：`campaign.accept(sha)`
- `REVERT` 或 `REJECT`：`campaign.revert()`，实现目录回到 `campaign.best`
- `INVALID`：环境不合格（降频、频率或功耗模式变化、温漂超过 5°C）。不要接受，也不要把它当成性能回退记进「这个手段无效」

最优提交存在 `.git/kal/<campaign-id>.json`，不在 git 历史里。新会话用 `Campaign.load` 恢复。

## 评测契约

正确性用 `check_correctness(spec, run, shapes, seed=...)`。`run(inputs, outputs, shape)` 就地写入 `outputs`。调用前输出缓冲区是 NaN；改写 `inputs` 会判失败。每个 shape 至少两轮，种子不同。

公开 shape 来自 `ops/rmsnorm/spec.py` 的 `public_shapes()`：`r1-h4096`、`r8-h4096`、`r32-h1024`。隐藏 shape 不在仓库里，评测时由调用方追加。不要在 kernel 里写死这些 label。

性能比较对象是当前最优版本，不是 PyTorch eager。加速比用同一轮交替计时的中位数。几何平均要超过 `1 + max(噪声, 1%)`，并且没有任何一个声明范围内的 shape 慢过它自己的噪声。

只优化某一个 shape 时，让 campaign 的 shape 集合只包含它。不要为了「测得更全」加进交付范围外的 shape。

## 实验日志

每次尝试都记，包括编译失败、结果错误和回退：

```python
from kernel_agent_loop.core.experiment_log import Attempt, ExperimentLog

log = ExperimentLog(repo / ".kal" / "experiments.sqlite")
if log.tried(campaign.campaign_id, lever=lever, parent=campaign.best):
    # 这个父版本上试过该手段，换一个
    ...
log.record(Attempt(...))
```

症状字符串可以填，但仓库还不会从 ncu/SASS 生成症状。不要去实现诊断或知识库，除非任务明确要求。

## 验证

```bash
python3 -m pytest
```

默认跳过 `thor`。改了 `CudaBackend` 且本机有 GPU、`nvcc`、`cuda-python` 时再跑：

```bash
python3 -m pytest -m thor --override-ini="addopts="
```

新增测试文件要有模块文档字符串（`模块:`、`功能:`）。每个 `test_*` 在装饰器上方写 `# 用例:` 和 `# 参数:`，没有形参时参数行写「无」。

## 还没实现

不要假设下面这些已经存在：`kal` 命令、MCP、evald 守护进程、Triton / CuTe DSL / TIRx 后端、ncu 症状映射、L2 调优库、AiInfraKB。
