# kernel-agent-loop 架构

由 Agent 主导的 GPU 算子优化闭环：**模型提出改动 → 硬件评测与 profiling → 症状反馈 → 知识检索与沉淀**，循环迭代。

本仓库本身不调用大模型。Cursor、Claude Code、Hermes、Grok bot 等外部 Agent 框架负责“读反馈、提假设、改代码”；本仓库通过 skills、MCP、CLI 给它们提供工具，并担任**裁判**：评测、判定是否接受、管理知识。

## 1. 设计原则

1. **评测器归系统，不归 Agent。** Agent 只能修改 `ops/<op>/impls/<backend>/` 下的实现源码，不能碰算子契约、测试数据、参考实现和计时逻辑。
2. **接受与否只认评测守护进程写入的结果。** Agent 报告的数字不作为依据。
3. **每个候选只改一个手段（lever）。** 方便归因，也让知识能够沉淀。
4. **失败也要记录。** 实验日志既用于去重，也是以后 SFT/RL 的训练数据。
5. **便宜的事不交给大模型。** tile、stage 数这类数值参数用搜索器；大模型只做结构性改动。
6. **知识晋升必须经人工审批。** 共享知识库沿用 AiInfraKB 的 inbox 治理流程。

## 2. 架构图

```mermaid
flowchart TB
  subgraph Frameworks["外部 Agent 框架（可替换）"]
    Cursor["Cursor"]
    Claude["Claude Code"]
    Hermes["Hermes"]
    Grok["Grok bot"]
  end

  subgraph Surface["接入面"]
    Skills["skills/<br/>optimize-loop / diagnose / distill-knowledge"]
    MCP["MCP Server<br/>CLI 的薄封装"]
    CLI["CLI<br/>campaign / submit / status / kb"]
  end

  subgraph Core["核心库（开发机）"]
    CampaignMgr["Campaign Manager<br/>候选 = git commit"]
    StaticJudge["Static Judge<br/>文件白名单 / diff 规则"]
    Gate["Acceptance Gate<br/>只认 evald 写入的结果"]
    ExpLog["L1 实验日志<br/>SQLite"]
    FB["Feedback Builder<br/>ncu + SASS → Symptoms"]
    KAdapters["Knowledge Adapters"]
  end

  subgraph Ops["算子契约（与后端无关，Agent 不可写）"]
    OpSpec["OpSpec<br/>参考实现 / shape / 容差 / 基线"]
  end

  subgraph Backends["后端插件"]
    Triton["TritonBackend"]
    Cuda["CudaBackend"]
    Cute["CuteDslBackend"]
    Tirx["TirxBackend"]
  end

  subgraph Thor["Thor 评测守护进程（Agent 不可写）"]
    Evald["evald<br/>收候选包、排队、独占 GPU"]
    Env["Env Guard<br/>功耗模式 / 锁频 / 温度"]
    Runner["隔离进程<br/>构建 → 正确性 → 计时"]
    Prof["Profiler<br/>ncu / cuobjdump / nvdisasm"]
    Hidden["隐藏 shape<br/>只存在于 Thor 端"]
  end

  subgraph Knowledge["知识层"]
    TuneDB["L2 调优库<br/>knowledge/db/"]
    KB["L3 AiInfraKB<br/>hardware: thor"]
  end

  Human["人工审批"]

  Frameworks --> Skills
  Frameworks --> MCP
  Frameworks --> CLI
  MCP --> CLI
  CLI --> CampaignMgr
  CampaignMgr --> StaticJudge
  StaticJudge -->|候选包| Evald
  Evald --> Env --> Runner --> Prof
  Runner --> OpSpec
  Runner --> Hidden
  Runner --> Backends
  Prof -->|EvalResult| ExpLog
  ExpLog --> Gate
  Gate --> CampaignMgr
  ExpLog --> FB
  FB --> CLI
  CLI --> KAdapters
  KAdapters --> ExpLog
  KAdapters --> TuneDB
  KAdapters --> KB
  KAdapters -->|候选条目| Human
  Human -->|晋升| TuneDB
  Human -->|晋升| KB
```

### 模块职责

| 模块 | 运行位置 | 职责 |
|---|---|---|
| skills | Agent 框架内 | 用中性 Markdown 描述工作流，再由 agent-kit 同步到各框架的格式 |
| MCP / CLI | 开发机 | 唯一的对外接口；MCP 只负责转发 CLI 参数 |
| Campaign Manager | 开发机 | 管理优化任务；候选就是 campaign 分支上的 commit，回退就是切回父提交 |
| Static Judge | 开发机 | 确定性检查：只允许修改白名单文件，禁止读取测试数据、硬编码 shape |
| Acceptance Gate | 开发机 | 根据 `EvalResult` 判定接受或回退 |
| 实验日志 | 开发机 | 记录每次尝试的 diff、手段、假设、结果、症状，包括失败的尝试 |
| Feedback Builder | 开发机 | 把 ncu 指标和 SASS 统计映射成症状标签，并和父候选做差分 |
| Knowledge Adapters | 开发机 | 统一查询 L1/L2/L3，回写知识使用效果，生成待晋升的候选条目 |
| OpSpec | 两端 | 定义算子契约；隐藏 shape 只放在 Thor 端 |
| Backends | Thor | 把实现源码编译成 cubin，并负责启动 kernel |
| evald | Thor | 接收候选包，固定运行环境，在隔离进程里构建、验证、计时、profiling |

## 3. 单轮循环

```mermaid
stateDiagram-v2
  [*] --> Profile: 基线 profile
  Profile --> Diagnose: 症状标签
  Diagnose --> Retrieve: 按症状检索 L1/L2/L3
  Retrieve --> Hypothesize: 选一个手段
  Hypothesize --> Edit: 结构改动（Agent）
  Hypothesize --> Tune: 数值参数（搜索器）
  Edit --> StaticJudge
  StaticJudge --> Submit: 通过
  StaticJudge --> Log: 拒绝
  Tune --> Submit
  Submit --> Build
  Build --> Correctness
  Build --> Log: 编译失败
  Correctness --> Bench
  Correctness --> Log: 结果错误
  Bench --> EnvCheck
  EnvCheck --> Bench: 降频或温漂，作废重测
  EnvCheck --> Gate
  Gate --> Accept: 超过噪声阈值，且所有 shape 不退化
  Gate --> Revert: 否则
  Accept --> Log
  Revert --> Log
  Log --> Profile: 预算未耗尽
  Log --> Distill: 收敛或到达里程碑
  Distill --> [*]
```

图中 Diagnose、Retrieve、Hypothesize、Edit 由 Agent 框架执行，其余步骤由本仓库执行。

## 4. 类图

```mermaid
classDiagram
  class OpSpec {
    <<interface>>
    +name
    +public_shapes()
    +hidden_shapes()
    +make_inputs(shape, seed)
    +reference(inputs)
    +tolerance(dtype)
    +baselines()
  }
  class KernelBackend {
    <<interface>>
    +name
    +build(source_dir, target) Artifact
    +launch(artifact, inputs, outputs)
  }
  class Artifact {
    +cubin_paths
    +launch_meta
  }
  class TritonBackend
  class CudaBackend
  class CuteDslBackend
  class TirxBackend

  class Campaign {
    +id
    +op
    +backend
    +hardware
    +best_commit
    +budget
  }
  class Candidate {
    +commit
    +parent_commit
    +lever
    +hypothesis
  }
  class StaticJudge {
    +check(candidate, campaign) Verdict
  }
  class EvalClient {
    +submit(candidate) job_id
    +result(job_id) EvalResult
  }
  class EvalResult {
    +env
    +compile_ok
    +correctness
    +perf
    +profile
  }
  class EnvSnapshot {
    +power_mode
    +clocks
    +temp_before
    +temp_after
    +throttled
  }
  class PerfStats {
    +median_us
    +noise_pct
    +ratio_vs_baseline
    +per_shape
  }
  class ProfileReport {
    +ncu_metrics
    +sass_stats
    +regs_smem
  }
  class AcceptanceGate {
    +decide(result, best) Decision
  }
  class FeedbackBuilder {
    +build(result, parent) Feedback
  }
  class Feedback {
    +symptoms
    +delta_summary
  }
  class KnowledgeSource {
    <<interface>>
    +query(symptoms, backend, hardware) list
    +feedback(entry_id, outcome)
  }
  class ExperimentLog
  class TuningDB
  class AiInfraKB

  KernelBackend <|.. TritonBackend
  KernelBackend <|.. CudaBackend
  KernelBackend <|.. CuteDslBackend
  KernelBackend <|.. TirxBackend
  KernelBackend --> Artifact
  Campaign --> OpSpec
  Campaign --> "many" Candidate
  StaticJudge --> Candidate
  EvalClient --> EvalResult
  EvalResult --> EnvSnapshot
  EvalResult --> PerfStats
  EvalResult --> ProfileReport
  AcceptanceGate --> EvalResult
  FeedbackBuilder --> Feedback
  KnowledgeSource <|.. ExperimentLog
  KnowledgeSource <|.. TuningDB
  KnowledgeSource <|.. AiInfraKB
```

## 5. 关键设计

### 5.1 算子契约与后端分离

借鉴 TIRx-kernels 的 `KernelModule` 协议，但不直接依赖它。该协议把“测什么”和“怎么实现”放在同一个模块里，这里拆成两部分：

- `OpSpec` 归系统所有：输入生成、参考实现（用 fp32/fp64 计算）、按 dtype 设定的容差、公开 shape、隐藏 shape、基线实现。
- `KernelBackend` 是插件：把实现源码编译成 `Artifact`，并负责启动 kernel。

所有后端最终都产出 cubin，因此 SASS 和 ncu 是跨后端的统一反馈来源，症状映射只写一份。

### 5.2 防作弊

| 风险 | 对策 |
|---|---|
| 复用上一次的输出 | 每次运行前用 NaN 填满输出缓冲区 |
| 针对固定输入过拟合 | 每次评测用新的随机种子；隐藏 shape 只存在 Thor 端 |
| 硬编码 shape、读取测试数据 | Static Judge 做文件白名单检查和 diff 规则扫描 |
| kernel 实际没有执行 | 用 ncu 核对 kernel launch 数量 |
| 加速比不合理 | 超过 Thor 实测 roofline 的结果直接标为可疑，转人工复核 |
| Agent 谎报结果 | Gate 只读取 evald 写入实验日志的结果 |

### 5.3 测量纪律（Thor）

- 评测前固定 `nvpmodel` 功耗模式并执行 `jetson_clocks`；每个结果都附带 `EnvSnapshot`。
- 如果评测前后温度漂移超过阈值，或者检测到降频，结果作废并重测。
- 每次计时前 flush L2，取多次运行的中位数；基线在同一会话里重新测量。
- 只有加速超过噪声阈值，并且**所有 shape 都不退化**，才接受候选。
- CPU 和 GPU 共享 LPDDR5X 带宽：评测期间保证 CPU 侧没有其他负载；roofline 用部署时自己校准的实测上限，不用规格表的数字。
- Thor 的 SM 数远少于数据中心 GPU，wave 量化的影响不同。从数据中心 kernel 借鉴代码时，不要沿用按 148 个 SM 写死的默认值。

### 5.4 反馈

Feedback Builder 只给 Agent 少量高信号信息：排在前几位的症状标签、和父候选相比的指标变化、寄存器和 shared memory 用量。症状词表沿用 `tirx-codegen-tuning` 的 snake_case 标签（如 `long_scoreboard`、`register_spill`），没有合适的词再新增。

### 5.5 知识分层

| 层 | 内容 | 写入方 | 晋升门槛 |
|---|---|---|---|
| L1 实验日志 | 每次尝试的 diff、手段、假设、结果、症状 | 系统自动写入 | 不需要 |
| L2 调优库 | 可复用的 kernel 代码改法 | Distiller 草拟，人工审批 | 正确性通过，且实测有收益 |
| L3 AiInfraKB | 跨项目事实与基准，按硬件、模型索引 | 写入 AiInfraKB 的 `inbox/` | 治理 bot 审核后人工批准 |

L2 条目沿用 `tirx-codegen-tuning` 的格式（一个条目一个文件，Symptoms 是唯一索引），另加一行 `**Backends:**`。SASS 层面的经验通常跨后端通用，标为 `any`。

Planner 检索顺序：先查 L1 去重，再查 L2 找改法，最后查 L3 了解硬件特性。检索到的条目如果最终带来被接受的改动，就回写一次成功；没起作用也要回写，用来发现过期条目。

### 5.6 适配多种 Agent 框架

- skills 只维护一份，由 agent-kit 同步到各框架。
- MCP 与 CLI 共用核心逻辑；不支持 MCP 的框架直接调用 CLI。
- 可以请另一个框架或模型对 diff 给出“第二意见”，但结果只作参考，不作为门禁。

## 6. 仓库结构

```text
kernel-agent-loop/
  docs/architecture.md
  skills/                     # optimize-loop / diagnose / distill-knowledge
  ops/<op>/spec.py            # OpSpec
  ops/<op>/impls/<backend>/   # Agent 唯一可写的区域
  knowledge/db/               # L2 调优库条目
  src/kernel_agent_loop/
    core/                     # campaign、gate、static judge、实验日志
    backends/                 # triton / cuda / cute / tirx
    evald/                    # Thor 端守护进程、环境守卫、隔离运行
    feedback/                 # ncu + SASS → symptoms
    knowledge/                # L2 与 AiInfraKB 适配
    cli.py
    mcp.py
  tests/
```

## 7. 分阶段计划

| 阶段 | 内容 | 验证 |
|---|---|---|
| 0 | 在 Thor 上验证各后端工具链，并校准带宽和算力上限 | 各后端的 vector add 都能编译运行，校准数字落盘 |
| 1 | `OpSpec`、`CudaBackend`、本地 evald、实验日志、Gate；以 RMSNorm 跑通“提交 → 评测 → 接受或回退” | 故意写错和故意变慢的候选必须被拒绝 |
| 2 | 防作弊：隐藏 shape、输出投毒、Static Judge | 一组作弊候选测试必须全部被拦下 |
| 3 | ncu + SASS 接入，Feedback Builder，L2 检索 | 已知症状的 kernel 输出预期标签 |
| 4 | skills + MCP，至少两个框架跑通完整 campaign | 同一个 campaign 分别用 Cursor 和 Claude Code 跑完 |
| 5 | Triton 后端、Distiller、AiInfraKB 回写 | 晋升条目通过 AiInfraKB lint |

## 8. 待验证风险

- Triton、CuTe DSL 对 Thor 架构的支持程度未确认，在阶段 0 验证；不支持的后端先不接入。
- Jetson 上运行 ncu 需要 root 或额外的 profiling 权限，evald 的部署方式要提前确定。
- 开发机和 Thor 之间的候选包传输方式（SSH 或 HTTP）待定。
