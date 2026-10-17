# kernel-agent-loop

由 Agent 主导的 GPU 算子优化闭环：模型提出改动，硬件评测并给出症状反馈，变快的改动留下，可复用的经验沉淀进知识库，再进入下一轮。

这个仓库本身不调用大模型。Cursor、Claude Code、Hermes、Grok bot 等外部 Agent 框架负责读反馈、提假设、改代码。本仓库通过 skills、MCP 和 CLI 给它们提供工具，并担任裁判：评测、判定接受还是回退、管理知识。是否接受只认评测守护进程写回的结果，不信 Agent 自己报告的数字。

目标硬件是 Jetson Thor。实现语言按后端插件接入，计划支持 CUDA C++、Triton、CuTe DSL 和 TIRx。

设计细节见 [docs/architecture.md](docs/architecture.md)，在线阅读见 [Agent 主导的算子优化闭环：整体架构](http://www.xiaoanger.com/topics/ai-native/agent-kernel/kernel-agent-loop/)。

## 当前状态

仓库里目前只有架构文档，CLI、评测守护进程和后端插件都还没有实现，所以现在不能安装、也不能跑优化循环。下面的安装和使用是按架构约定写的目标形态，实现落地后才会生效。

## 安装

计划中的形态是一个 Python 包，评测在 Thor 上由独立进程 `evald` 执行。

开发机：

```bash
git clone git@github.com:happyAnger66-an/kernel-agent-loop.git
cd kernel-agent-loop
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Thor 上还需要本机的 CUDA 工具链（`nvcc`）、Nsight Compute（`ncu`）以及 `cuobjdump` / `nvdisasm`。`evald` 以独占 GPU 的方式运行，评测前固定功耗模式和频率。Jetson 上跑 `ncu` 通常需要 root 或额外的 profiling 权限，部署方式在实现 `evald` 时确定。

各后端对 Thor 架构的支持程度尚未验证。工具链验证是实施计划的第 0 步；验证不过的后端先不接入。

## 使用

一次优化是一个 campaign：选定算子、后端和 shape 集，Agent 反复提交候选，系统评测后决定保留或回退。

```bash
# 建一个 campaign：算子、后端、硬件
kal campaign new rmsnorm --backend cuda --hardware thor

# Agent 只改 ops/<op>/impls/<backend>/ 下的实现，每个候选只改一个手段
kal submit --lever fuse-residual --hypothesis "融合残差，减少一次全局写"

# 看本轮结论：接受或回退、症状标签、相对上一版的差分
kal status

# 按症状查已有改法，避免重复试过的失败手段
kal kb query --symptoms long_scoreboard --backend cuda --hardware thor
```

约定：

- Agent 只能改 `ops/<op>/impls/<backend>/`。参考实现、shape、容差、基线和计时逻辑属于算子契约，Agent 不能改。
- 每个候选只改一个手段，并在提交时写明手段名和假设。失败的尝试也会记进实验日志，用来去重。
- 加速要超过噪声阈值，且所有 shape 都不退化，才会被接受。隐藏 shape 只存在于 Thor 端，用来抓住针对公开 shape 过拟合的候选。
- 数值参数（tile、stage 数）由搜索器枚举，不交给大模型。大模型只做融合、换 layout、加流水线这类结构性改动。

不支持 MCP 的框架直接调用上面的 CLI。支持 MCP 的框架走同一套命令的 MCP 封装。skills 描述工作流，由各 Agent 框架加载。

## 知识从哪来

| 层 | 内容 | 谁来写 |
|---|---|---|
| L1 实验日志 | 每次尝试的 diff、手段、假设、结果、症状，含失败 | 系统自动写入 |
| L2 调优库 | 可复用的 kernel 改法，按症状和后端索引 | 系统草拟，人工审批后进入 `knowledge/db/` |
| L3 AiInfraKB | 跨项目的事实与基准 | 写入 AiInfraKB 的 inbox，审核后生效 |

检索顺序是先用 L1 去掉已经试过的手段，再用 L2 找改法，最后用 L3 查硬件特性。用过的条目会回写是否真的带来被接受的改动，用来发现过期知识。
