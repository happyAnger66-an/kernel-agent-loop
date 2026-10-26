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

提交之后用 `evaluate` 评测**当前工作区**里的实现目录，不要自己报耗时：

```python
from kernel_agent_loop.core.experiment_log import ExperimentLog
from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import Verdict
from kernel_agent_loop.evald.evaluate import evaluate

(repo / ".kal").mkdir(exist_ok=True)
log = ExperimentLog(repo / ".kal" / "experiments.sqlite")
outcome = evaluate(
    campaign,
    spec=load_opspec(repo / "ops", campaign.op),
    log=log,
    lever="<手段>",
    hypothesis="<假设>",
    commit=sha,
)
if outcome.decision.verdict is Verdict.ACCEPT:
    campaign.accept(sha)
elif outcome.decision.verdict is Verdict.INVALID:
    # 环境不合格。不要 accept，也不要 revert 成“这个手段无效”
    ...
else:
    campaign.revert()
```

`evaluate` 会编译、做正确性、和 `campaign.best` 交替计时、读环境快照、调用 `AcceptanceGate`，并把判定写入 `log`。它不修改最优指针。`campaign.best` 编不过时当作没有 incumbent，第一个正确候选会被接受。默认 shape 是 `spec.public_shapes()`；隐藏 shape 由调用方追加到 `shapes`。默认环境探针是 `nvidia-smi`，频率或温度不合格会得到 `INVALID`。

在 Thor 上评测时不要本机 `evaluate`，也不要自己拼 `make_bundle` + `EvalClient`，用 `run_round` 跑一整轮：

```python
from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import ExperimentLog
from kernel_agent_loop.core.gate import AcceptanceGate
from kernel_agent_loop.evald.client import EvalClient
from kernel_agent_loop.evald.env_guard import EnvGuard
from kernel_agent_loop.loop import run_round

client = EvalClient(
    mode="ssh", host="tz", user="nvidia", port=2240,
    remote_config="/home/nvidia/kal/evald.config.json",
    evald_cmd="/home/nvidia/kal/venv/bin/python -m kernel_agent_loop.evald.serve",
    remote_workdir="/home/nvidia/kal/inbox",
)
outcome = run_round(
    Campaign.load(repo, "rms-1"), client=client, log=ExperimentLog(path),
    gate=AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0),
    lever=lever, hypothesis=hypothesis,
)
```

`run_round` 依次做：`tried` 去重（重复抛 `AlreadyTried`）→ `StaticJudge` → `commit_candidate` → 打包（附带从 git 取出的 `campaign.best` 实现作为 incumbent）→ `submit_sync` → Gate → 写日志 → `accept` / `revert`（`INVALID` 时工作区保持不动，可原样重提）。候选包不带 incumbent 时 Thor 上没有比较对象，任何正确候选都会被接受，所以不要绕过 `run_round` 自己打包。

Thor 配置示例在 `deploy/thor/`。`evald_run` 持 `flock`、overlay 候选包、可选 `nvpmodel`/`jetson_clocks`，只回传 `EvalResult`（含 `symptoms`）。

`commit_candidate` 只会把 `ops/<op>/impls/<backend>/` 的改动放进提交。该目录以外的修改留在工作区，不会被这次提交带走。

`StaticJudge` 会拒绝实现目录以外的路径，以及源码里的 `subprocess`、`socket`、`os.environ`、`open(`、`fopen(`、`system(`。

接受或回退以 `evaluate` 返回的 `decision` 为准，不要再单独用自己测的时间叫一次门禁：

- `ACCEPT`：`campaign.accept(sha)`
- `REVERT` 或 `REJECT`：`campaign.revert()`，实现目录回到 `campaign.best`
- `INVALID`：环境不合格（降频、频率或功耗模式变化、温漂超过 5°C）。不要接受，也不要把它当成性能回退。这次会记进日志，但 `tried` 不把它算成“这个手段已经试过”

最优提交存在 `.git/kal/<campaign-id>.json`，不在 git 历史里。新会话用 `Campaign.load` 恢复。

## 评测契约

正确性用 `check_correctness(spec, run, shapes, seed=...)`。`run(inputs, outputs, shape)` 就地写入 `outputs`。调用前输出缓冲区是 NaN；改写 `inputs` 会判失败。每个 shape 至少两轮，种子不同。

公开 shape 来自 `ops/rmsnorm/spec.py` 的 `public_shapes()`：`r1-h4096`、`r8-h4096`、`r32-h1024`。隐藏 shape 不在仓库里，评测时由调用方追加。不要在 kernel 里写死这些 label。

性能比较对象是当前最优版本，不是 PyTorch eager。候选和当前最优在同一进程、同一份缓冲上配对计时，每个 shape 得到加速比的 95% 置信区间。接受要同时满足：没有 shape 显著变慢（区间上界 < 1）；至少一个 shape 显著变快（区间下界 > 1.01）；几何平均 > 1.01。只改善部分 shape、其余不退化的改动可以被接受。

只优化某一个 shape 时，让 campaign 的 shape 集合只包含它。不要为了「测得更全」加进交付范围外的 shape。

## 实验日志

`evaluate` 会把每次尝试写入日志，包括编译失败、结果错误和回退。不要再自己 `record` 一次。提交前用 `tried` 跳过已经在这个父版本上试过的手段：

```python
from kernel_agent_loop.core.experiment_log import ExperimentLog

log = ExperimentLog(repo / ".kal" / "experiments.sqlite")
if log.tried(campaign.campaign_id, lever=lever, parent=campaign.best):
    # 这个父版本上试过该手段，换一个。INVALID 不算试过
    ...
```

正确性失败的类别（`mismatch`、`input_modified`、`error`）在症状字段里。仓库还不会从 ncu/SASS 生成症状。不要去实现诊断或知识库，除非任务明确要求。

## 验证

```bash
python3 -m pytest
```

默认跳过 `thor`。改了 `CudaBackend` 或 `evaluate` 的 GPU 路径，且本机有 GPU、`nvcc`、`cuda-python` 时再跑：

```bash
python3 -m pytest -m thor --override-ini="addopts="
```

新增测试文件要有模块文档字符串（`模块:`、`功能:`）。每个 `test_*` 在装饰器上方写 `# 用例:` 和 `# 参数:`，没有形参时参数行写「无」。

## 还没实现

不要假设下面这些已经存在：`kal` 命令、MCP、常驻 evald 守护进程、Triton / CuTe DSL / TIRx 后端、ncu 症状映射、L2 调优库、按工单自动搜索的循环。按次调用的 `evald_run` / `EvalClient`（含 SSH 同步）已经可用。格子结果可以经 `induce_cell_result` 归纳，再用 `propose_cell_result` 写成 AiInfraKB 的 inbox 草稿；`contracts/` 要和知识库的 schema 保持字节一致。

<!-- agent-kit:begin -->

由 agent-kit 0.1.0 生成，不要手改。规范正文在 `.cursor/rules/agent-kit/`。

- **respond-in-chinese**（always）：与用户交流一律使用简体中文；代码、标识符、提交信息按目标仓库约定
- **conventional-commit**（always）：提交时 commit 标题必须以类型前缀说明改动分类
- **test-first**（always）：实现需求前先写测试，review 通过后再编码
- **bug-regression**（always）：修 bug 必须带回归测试，复现原失败路径
- **test-layout**（globs: tests/**,**/*_test.py,**/test_*.py）：测试代码放在 tests/，按被测对象组织，不与源码混排
- **test-file-header**（globs: tests/**,**/*_test.py,**/test_*.py）：测试文件用模块文档字符串写明被测模块和覆盖的功能
- **test-case-comment**（globs: tests/**,**/*_test.py,**/test_*.py）：每个测试用例用注释写明用例功能和参数作用
- **arch-diagrams**（on-demand）：新项目立项或分层/公开类型变动时，先补架构图和类图，再写测试
- **synapsex-mlc-vla-boundary**（always）：公司级仓职责：通用 fused 核进 synapsex，DiT/模型 glue 只留 mlc-vla；勿混进上游 GR00T 或领域知识卡
- **tvm-relax-core-boundary**（always）：tvm-relax 只管核心编译
- **perf-profiler-ab**（always）：负向或未达预期的性能改动收工前，必须用 profiler（nsys/ncu 等）做 A/B，禁止只交 wall-clock
- **perf-baseline-dual-track**（always）：抬全链路 baseline 必须交 stages+e2e 与 overall+per-dim；速度轨与精度轨分开，禁止只报 denoise
- **aiinfrakb-optimization-loop**（on-demand）：在某硬件上优化某模型的推理时，必须先 kb next 只做返回的那一格，再用 kb propose 钉回地图
- **aiinfrakb-work-env**（on-demand）：连接任何机器（Thor、PRO5000、gpu-4090…）或服务（Model Forge 测试/生产…）前，必须先用 kb env 查 AiInfraKB 的环境条目并遵守 guardrails
- **atomic-commit**（always）：每完成一个原子功能单独提交一次 commit，message 写清做了什么
- **docs-sync**（always）：改了对外行为、接口或用法时，同一改动里更新对应文档
- **release-changelog**（on-demand）：给运行时仓库发版或改 release.sh、pack.sh、CHANGELOG 时，改动说明写进 CHANGELOG 并由 release.sh 门禁

<!-- agent-kit:end -->
