# vLLM 与 SGLang 并行策略异同及对显存规划器的影响

> 状态：当前设计输入，非最终方案
>
> 版本状态：已按两个目标 release 的 exact tag 重新核验
>
> 对比日期：2026-07-12；更新日期：2026-07-13
>
> 版本基线：vLLM `v0.25.0`（`702f4814fe54fabff350d43cb753ae3e47c0c276`）；SGLang `v0.5.15`（`f63458b5beaceabbd9d749b9fc956370e1b649e6`）
>
> 证据来源：[vLLM v0.25.0 并行策略源码审计](./2026-07-13-vllm-v0.25.0-parallelism-source-audit.md)；[SGLang v0.5.15 并行策略源码审计](./2026-07-12-sglang-v0.5.15-parallelism-source-audit.md)
>
> 目的：明确两个框架哪些概念可以投影到共同的显存账本，哪些语义必须由框架版本适配器独立解析。本文件不定义最终配置 schema。

不熟悉 worker、rank、process group、TP/PP/DP/EP/CP、Prefill/Decode 或 KV page 的读者，应先阅读[并行策略源码阅读基础知识](./2026-07-13-parallelism-source-reading-foundations.md)，再阅读两个框架审计，最后阅读本文。本文比较的是已经从源码解析出来的运行事实，不以参数名称猜实现。

旧的 [vLLM v0.24.0 源码审计](./2026-07-12-vllm-v0.24.0-parallelism-source-audit.md)仅保留为升级背景，不再作为当前设计证据；本文涉及 vLLM 的结论均以 `v0.25.0` exact tag 为准。

## 1. 结论先行

两个框架都使用 TP、PP、DP、EP、CP、DCP 等术语，但同名参数并不保证以下任何一项相同：

- 是否增加 GPU worker；
- 是否创建独立模型副本和 scheduler；
- 哪些模型组件被切分或复制；
- 请求及其 KV cache 归属于哪个进程；
- 通信 buffer、activation 和 CUDA Graph 的峰值如何变化；
- 参数是否真正可用于当前模型、attention backend、量化方式和部署角色。

因此，显存规划器不能把服务命令机械归一化为一组通用的 `tp/pp/dp/ep/cp` 数字，然后套用统一除法。正确的共同落点应是框架解析之后的事实，例如：

```text
物理 GPU / worker
  -> 持有哪些模型组件及 checkpoint tensor
  -> 持有哪些 cache group，以及每个 group 的 token/page 归属
  -> 属于哪些 collective group
  -> 执行哪些启动、Prefill、Decode 路径
  -> 在各路径中持有哪些 Graph、activation 和通信 workspace
```

框架适配器必须先把用户参数解析成这些事实，共享计算层才有稳定的输入。这里的“共享”是共享显存分项及归属语言，不是把两个框架压缩成同一套并行实现。

## 2. 比较口径

### 2.1 三种不同的数量不能混用

对任意部署至少要区分：

1. **物理设备数**：部署实际占用多少张 GPU。
2. **worker 数与 rank 坐标**：模型执行进程如何映射到 GPU，各 rank 参与哪些 group。
3. **服务分区数**：有多少独立 scheduler、请求队列和 KV/cache 容量域。

一个并行参数可能改变其中一个数量，而不改变另外两个。例如，DCP 通常复用已有 TP worker，不增加物理设备；普通 DP 通常增加独立服务分区；SGLang DP Attention 则在已有 TP world 内建立多个 attention 请求分区。

### 2.2 “切分”和“复制”必须按组件判断

一个物理 rank 同时承担 TP、PP、DP、EP 或 CP 坐标。不能把“TP rank 显存”“PP rank 显存”“DP rank 显存”分别计算再相加，也不能用总权重连续除以所有并行度。需要按下列组件分别确认放置规则：

- embedding、lm_head；
- attention Q/K/V/O；
- dense FFN；
- routed experts、shared experts、router；
- norm、bias、量化 scale/zero point；
- MTP 或 speculative 模块；
- KV、MLA latent、indexer、linear/recurrent state；
- Graph、activation、allocator 和 collective workspace。

### 2.3 本文的证据等级

- **源码事实**：由目标 release tag 的参数解析、process-group 构造、模型层或 cache 实现直接确认。
- **显存推论**：可以从源码事实确定分片/复制、所有权或生命周期，但具体字节仍需模型/checkpoint 数据。
- **待校准项**：实现明确受参数影响，但无法仅凭源码可靠得到峰值字节，例如 Graph、allocator 和部分通信 workspace。
- **能力状态**：某个组合能否启用，必须同时考虑模型、框架版本、设备、backend、量化和其他功能；CLI 字段存在不代表组合受支持。

## 3. 总体拓扑对比

| 主题 | vLLM `v0.25.0` | SGLang `v0.5.15` | 显存规划含义 |
|---|---|---|---|
| 基础模型执行 world | 单个 DP EngineCore 内为 `PP × PCP × TP`；整个部署还需考虑 DP | 单个 distributed world 为 `TP × PP` | 不能由同一条 `world_size = TP × PP × DP × CP` 公式覆盖 |
| 普通 DP | 每个 DP rank 是独立 EngineCore、scheduler 和 KV cache 域；MoE 又会跨 DP 建 expert group | 未启用 DP Attention 时，每个 DP rank 启动独立的 `TP × PP` 副本 | Dense 普通 DP 可视为独立容量分区，但 MoE 组件归属仍不同 |
| 组件级 DP | MoE 场景中 attention/dense 部分按 DP 复制，expert domain 跨 DP/PCP/TP | DP Attention 将 attention 的 DP/CP/TP 和 MoE 的 DP/EP/TP 分解折叠进同一个 TP world | “DP”不能直接解释为完整模型副本数 |
| PP | 增加 stage worker；各 stage 只持有部分层，首尾组件不对称 | world rank 先按 PP stage 与 TP rank组织；各 stage 持有分配层 | 两者均要按 stage 层集合求最坏卡，不能平均除 PP |
| DCP | 复用 TP ranks，切分 Decode KV，不切分权重 | 复用 TP ranks，切分 Decode KV，但 backend/cache 约束不同 | 可共享“复用 worker、改变 KV 归属”概念，不可共享支持条件和布局公式 |
| Prefill context parallel | PCP 是 worker 乘数；v0.25.0 已扩展到 executor、KV/connector/offload 等路径，但 attention backend 默认仍不支持 | Attention CP 在 TP world 内进一步分解 attention 轴 | 同名 CP 对设备数、权重和 workspace 的影响相反或不同；代码面扩大不等于可运行组合已验证 |
| EP | expert group 大小由 DP、PCP、TP 等现有维度派生；无独立用户 `ep_size` 乘数 | 用户配置 EP，但其与 MoE-DP、MoE-TP 共同分解现有 TP world | 两者 EP 都不应简单乘到 GPU 数上，但专家放置算法不同 |
| scheduler / KV 域 | 通常以 DP EngineCore 为独立域；PP workers 协同一个副本容量 | 普通 DP 为独立域；DP Attention 下每个 Attention-DP 请求/缓存域由一组 TP/CP/PP workers 协同 | 最大 token 必须按服务分区和最坏 worker分别给出 |
| PD | P/D deployment group 各自解析 topology；connector 产生额外显存项 | P/D deployment group 各自解析 topology；disaggregation mode 还改变默认 reserve/Graph 路径 | 两者都不能把 P/D 显存或 token 容量相加，但 role-specific runtime 解析不同 |

## 4. Worker、rank 与 process group

### 4.1 vLLM：DP EngineCore 外层与模型并行内层

vLLM `ParallelConfig` 的基础 worker world 为：

```text
world_size = PP × TP × PCP
```

普通内部/混合 DP 部署还会有多个 DP EngineCore，整体部署 worker 数通常为：

```text
deployment workers = DP × PP × TP × PCP
```

外部 launcher 会把 DP 纳入其 distributed world，内部 launcher 则可能把不同 DP EngineCore 作为独立进程域组织。目标版本在 external launcher 初始化时已经把 DP 乘入 `world_size`，而 `world_size_across_dp` 属性仍会再乘一次；因此该属性不能在此路径上被当作物理 worker 数。无论启动方式如何，物理 GPU 账本都必须表达 DP 坐标，并由 launcher 语义解析，不能只读取某一个派生 world-size 字段。

目标版本中模型并行初始化构造 TP、DCP、PCP、PP、DP、EP，必要时还构造 EPLB group。当一个 model-parallel world 收到非平凡 DP 轴时，rank 张量的实际 reshape 顺序是 `External-DP × DP × PP × PCP × TP`；源码紧邻注释写成 `ExternalDP × DP × PP × TP`，少列了 PCP，适配器应以 reshape 和 group 构造为准。普通非 MoE Dense DP 则是多个独立 EngineCore worlds，每个 world 内的 DP 轴为 1。其中 DCP 是对 TP ranks 的子分组，不增加 worker。

关键源码锚点：

- `vllm/config/parallel.py:791-802`：基础 world size 及 external launcher 下的 DP 处理；
- `vllm/config/parallel.py:516-520`：`world_size_across_dp` 再乘 DP 的实现；
- `vllm/distributed/parallel_state.py:1713-1954`：rank 坐标与 TP/DCP/PCP/PP/DP/EP/EPLB group 构造；
- `vllm/distributed/parallel_state.py:1889-1938`：MoE EP/EPLB group 跨 DP、PCP、TP 的形成。

### 4.2 SGLang：同一 TP world 的组件级重解释

SGLang 模型 worker 初始化的 distributed world 始终是：

```text
world_size = TP × PP
rank = TP × pp_rank + tp_rank
```

但该 `TP` 不一定表示每个组件都按同一个 TP degree 切分。启用 DP Attention 后，同一个 TP stage 内会形成两套分解：

```text
Attention: TP = Attention-DP × Attention-CP × Attention-TP
MoE:       TP = MoE-DP × EP × MoE-TP
```

也就是说，同一张 GPU 对 attention、dense FFN 和 routed experts 可能具有不同的并行坐标和参数归属。Attention-TP、Attention-CP、MoE-DP、MoE-EP、MoE-TP 是针对同一批物理 ranks 建立的不同 group，并非可以相乘成额外设备数。

关键源码锚点：

- `python/sglang/srt/model_executor/model_runner.py:1301-1323`：`TP × PP` world 及各并行参数传入；
- `python/sglang/srt/layers/dp_attention.py:245-276`：Attention-DP/CP/TP 坐标分解；
- `python/sglang/srt/distributed/parallel_state.py:2137-2293`：Attention 与 MoE 的各 process group 构造。

### 4.3 共同结论

`rank` 应理解为一个物理模型 worker 在多个 group 中的组合坐标，不是某一种并行策略独占的一份显存。规划器最终需要比较的是同一张物理 GPU 上所有并存分项的总和。

框架适配器至少要输出：

- 物理 worker 标识、node 和 local device；
- PP stage 及实际层集合；
- 每种组件对应的 shard/replica group 和 rank；
- scheduler/cache partition 标识；
- collective group 及需要计入显存的 workspace 类型；
- 各 phase 中 worker 是否参与真实 forward、占位 forward 或 connector 工作。

## 5. TP：相同名称，不同有效切分度

### 5.1 可以共享的基础事实

两个框架的常规 Transformer TP 都包含类似的列并行、行并行、vocabulary parallel 和 QKV 特殊处理。因此以下原则可以共享：

- checkpoint 总字节不等于每卡 resident 权重；
- ColumnParallelLinear 沿输出维切权重，本地输出可按需 all-gather；RowParallelLinear 沿输入维切权重，本地结果通常需要 all-reduce 或 reduce-scatter；
- Q heads 通常按有效 TP 切分；GQA/MQA 的 KV heads 在 TP 不超过 KV head 数时切分，超过后转为复制，继续增大 TP 不会继续降低本地 KV heads；
- norm、bias、量化元数据及部分共享参数可能复制；
- embedding 和 lm_head 需要考虑 vocabulary padding、tied weight 及首尾 stage；
- GQA/MQA 中 TP 大于 KV head 数时可能出现 KV head 复制饱和，不能继续按 TP 降低 KV 权重和 KV cache。

### 5.2 必须分别解析的差异

vLLM 的常规 TP group 直接决定多数 dense/attention tensor 的本地 shard；MoE expert 的有效 domain 还可能扩展到 DP、PCP 和 TP。

SGLang 启用 DP Attention 后，用户 `tp_size` 是物理 stage 宽度，但 attention 的有效 TP 为：

```text
attention_tp_size = tp_size / (dp_size × attn_cp_size)
```

MoE 组件则使用另一套：

```text
moe_tp_size = tp_size / (ep_size × moe_dp_size)
```

所以在 SGLang DP Attention 中，不能用用户 `tp_size` 直接切 attention 权重或计算每卡 KV bytes/token；也不能用 attention TP 去切 expert 权重。模型 adapter 还必须验证具体 QKV layer 构造确实使用了有效 Attention-TP，不能把该结论机械应用到所有模型实现。

### 5.3 对显存计算的影响

共享层可以接收“某 tensor 在哪些 worker 上各持有多少字节”的解析结果，但不应自行从原始 `tp_size` 猜测。模型 adapter 提供 tensor/component 结构，framework adapter 提供目标版本的 shard/replica 规则，两者共同解析 resident placement。

## 6. PP：共同原则相似，容量聚合仍需框架化

两个框架的 PP 都按层把模型分到多个 stage，但每个 stage 并非均匀同构：

- transformer layer 数可能不能整除 PP；
- embedding 通常只在首 stage；
- final norm 和 lm_head 通常只在末 stage；
- MoE/dense/MTP 等层类型可能分布不均；
- PP 中间激活和通信 buffer 只出现在特定 stage/phase；
- 各 stage 持有的 KV/cache layer 数由真实层集合决定。

因此，权重和 cache 均需先得到每个 PP stage 的实际层集合，再计算每个物理 rank。服务可启动性由显存最紧张的 worker/phase 决定，而服务的 cache blocks 或 max tokens 还可能由框架在 PP workers 间取最小值后生效。

可以共享的不是“总量除以 PP”，而是：

```text
component placement -> stage-local ledger -> per-worker peak -> replica bottleneck
```

vLLM 的 KV cache 配置会在 PP workers 间协调并受最小可用 blocks 约束；SGLang 的实际 pool 初始化、scheduler 分区及 PP 组合应由其版本适配器解析。比较文档不假设两者具有完全相同的容量协调算法。

## 7. DP、vLLM MoE DP 与 SGLang DP Attention

### 7.1 Dense 模型的普通 DP

在不涉及组件级并行时，两者的普通 DP 都可近似理解为：

- 每个 DP 副本持有完整模型逻辑，但副本内部仍可有 TP/PP；
- 每个副本有独立请求调度和 KV/cache 容量；
- 单请求只属于一个 DP 副本；
- 整个服务 aggregate capacity 是各独立副本容量之和，但单请求上限不能跨副本相加。

即便如此，命令到 worker 的展开方式仍不同：vLLM 需要区别 internal、hybrid、external load balancing/launcher；SGLang 普通 DP controller 会启动多个独立 `TP × PP` world。导入器不能只根据一个 `dp_size` 推导启动进程。

### 7.2 vLLM MoE DP

vLLM MoE 场景打破了“DP 完全互不通信”的直觉：

- attention 和非 expert 组件通常在 DP 间复制；
- expert group 可跨 `DP × PCP × TP` 形成；
- 未启用 expert parallel 时，expert tensor 仍可能在这一宽域上做 tensor sharding；
- 启用 expert parallel 后，专家按该 domain 分配；
- 为保证 MoE collective 对齐，没有真实请求的 DP rank 也可能执行 dummy forward。

因此，同一配置中的 DP 对 attention/KV 表示独立副本，对 experts 却可能是同一 collective/sharding domain。显存规划器必须按组件解释 DP，而不是给整个模型贴一个“复制”或“切分”标签。

### 7.3 SGLang DP Attention

SGLang DP Attention 不是额外启动 `DP` 份完整 `TP × PP` 模型。它把 Attention-DP 坐标折叠进已有 TP stage：

- 每个 Attention-DP 请求/缓存域接收不同请求，域内 TP/CP/PP workers 分别持有本地 KV shard；
- attention 只处理本地请求；
- attention 权重按缩小后的 Attention-TP 切分，并跨 Attention-DP 复制；
- 进入 dense FFN/MoE 前，需要 gather 或等价 collective 汇总各 DP rank 的 token；
- FFN/MoE 后再 scatter/reduce-scatter 回请求所属的 Attention-DP 域；
- FFN/MoE 的权重放置由 MoE-DP、EP、MoE-TP 和 dense TP 规则决定，而不是由 Attention-TP 决定。

源码中的 gather/scatter 路径见 `python/sglang/srt/layers/dp_attention.py:472-691`。这些操作会引入 global/local token buffer、padding 和 collective workspace，因此 DP Attention 同时改变权重、KV、activation、Graph 和 runtime reserve。

### 7.4 三者不能合并成一个 `dpMode`

| 场景 | 完整模型副本 | 独立 scheduler/KV 域 | expert 是否跨“DP”协同 | 是否改变 attention 有效 TP |
|---|---|---|---|---|
| 普通 Dense DP | 是 | 是 | 不适用 | 否 |
| vLLM MoE DP | attention 等组件是；expert 按配置重组 | 是 | 是 | 通常不因 DP 缩小常规 TP |
| SGLang DP Attention | 否，多个 attention 分区共享同一 TP stage 的 FFN/MoE 执行域 | 是 | 是 | 是 |

对 UI 和导入报告而言，需要展示用户参数经过框架解析后的实际含义，尤其是 GPU 数、attention shard 数、scheduler 数和每个 scheduler 的 KV 容量。仅显示 `DP=8` 会产生误导。

## 8. EP、MoE-DP 与 shared experts

### 8.1 vLLM

vLLM `v0.25.0` 不把用户指定的 `ep_size` 作为额外独立维度。MoE expert domain 从现有 DP、PCP、TP ranks 派生，有效宽度为 `DP × PCP × TP`。`enable_expert_parallel=false` 时，expert tensors 在这个展平 domain 上做 tensor sharding；开启后则在同一 domain 上按 expert IDs 放置，每个设备持有一组完整 expert。两种模式都不能解释成“总 MoE 权重再除一次 EP”。

EPLB、冗余专家和 elastic EP 还会改变专家副本或运行时状态；shared experts、router/gate、quantization scales 也不能自动沿用 routed expert 的分片规则。

### 8.2 SGLang

SGLang 将现有 TP stage 分解为 `MoE-DP × EP × MoE-TP`。EP、MoE-DP 和 MoE-TP 并非三个额外设备乘数。具体约束包括整除关系，以及部分组合要求 `EP × MoE-DP = TP`。

`moe_dense_tp_size` 还会改变 MoE 模型中 dense MLP 的放置；当解析为 1 时，dense MLP 在参与的 local replica 内不做 tensor sharding，具体复制域仍由 DP Attention 和模型构造决定。不同 A2A backend 也可能改变 EP 的 resolved value 和通信 workspace。目标版本还明确禁止 `MoE-DP > 1` 与 `PP > 1` 组合；未启用 DP Attention 的原生普通 DP 多节点组合也会被拒绝。

### 8.3 共同账本所需信息

对每个 MoE layer，适配器需分别输出：

- routed expert 总数、每 rank 本地 expert IDs 及副本数；
- 每个本地 expert 的 tensor shard；
- shared experts 的独立 shard/replica 规则；
- router/gate、bias、scales 和 EPLB state；
- dispatcher/A2A backend 及需要预算的 workspace；
- 空 rank/dummy forward 是否可能出现在某一 phase。

“专家参数量 / EP”只能作为非常有限的直觉，不能作为最终 resident 权重公式。

## 9. CP 与 DCP：不能只提供一个 CP 输入框

### 9.1 vLLM PCP

vLLM Prefill Context Parallel（PCP）是基础 world size 的乘数，会增加 worker，并影响 Prefill token/context 分配、KV 布局和 MoE group。相较历史版本，`v0.25.0` 已把 PCP 继续接入 Ray executor、KV cache manager/coordinator、KV connector/Mooncake 命名空间、offload 和 executor 进程命名等代码面。

但这仍不是“PCP 已普遍支持”的证据：`vllm/v1/attention/backend.py` 中 attention implementation 的 `supports_pcp` 默认值仍为 `False`，本次 exact-tag 全仓审计未找到内置实现显式声明 `supports_pcp=True`；`docs/serving/context_parallel_deployment.md` 也仍将两种 Prefill CP 路径标为 active development。hybrid、多 block-size cache，以及 sliding-window、chunked-local、Mamba 等路径也仍有限制。规划器必须以目标模型、attention backend、cache 类型及组合测试证据判定能力，不能用代码覆盖面或 CLI 参数存在代替支持结论。

MVP 若要接入 PCP，必须有 exact-version、model、attention backend 和 cache 类型的完整能力证据。否则应解析并明确标为 experimental 或 unsupported，且阻止可执行导出。

### 9.2 SGLang Attention CP

SGLang Attention CP 在已有 TP stage 内把 attention 轴分解为 DP、CP、TP，通常不增加 `TP × PP` 之外的 worker。它会改变 attention 权重/KV 的有效分片、Prefill token 分配、LSE/merge workspace，以及与 MoE-DP 的 token sharing 关系。

部分模型或 MLA/DSA 路径还可能通过参数解析自动联动 DP Attention、dense TP、EP 或 A2A backend。因此不能把用户输入的 CP 数值直接视为最终 runtime topology。

### 9.3 两个框架的 DCP

两个框架的 Decode Context Parallel 都复用 TP ranks，目标是沿 token/context 维切分 Decode KV，而不是增加 worker 或改变权重切分。这个高层概念可以共享，但以下内容必须分别实现：

- TP、KV heads 与 DCP 的整除/复制规则；
- 支持的 attention backend；
- MLA、sliding-window、hybrid/Mamba state 等 cache 类型约束；
- speculative decoding、Graph 与 PD role 兼容性；
- page/block 布局及每 rank 实际可用容量。

### 9.4 UI 与导入含义

服务命令中的 `CP` 必须解析成明确的框架字段：vLLM PCP、vLLM DCP、SGLang Attention CP 或 SGLang DCP。规划器不能提供一个通用 `CP` 数字再由计算层猜测含义。

## 10. 权重与 cache ownership 对比

### 10.1 权重

| 组件 | 可共享原则 | vLLM 特有解析 | SGLang 特有解析 |
|---|---|---|---|
| embedding / lm_head | 按 tensor、vocab shard、tie 与 PP 首尾 stage 计算 | vLLM loader 和模型 adapter 的分片/占位规则 | SGLang 模型实现、PP 及可选 DP lm_head 规则 |
| Q/K/V/O | 按实际 attention TP 与 head replication 计算 | 常规 TP、PCP/DCP 及 attention backend 布局 | Attention-DP/CP/TP 分解后的有效 TP |
| dense FFN | 按真实 row/column shard 和 replica group 计算 | 常规 TP，MoE 模型中的 dense 层仍依模型实现 | `moe_dense_tp_size` 可令 dense MLP 与用户 TP 不同 |
| routed experts | 按本地 expert IDs、expert tensor shard 计算 | domain 由 DP/PCP/TP 派生，EP 开关改变放置 | `MoE-DP × EP × MoE-TP` 分解已有 TP world |
| shared experts | 与 routed experts 分开计算 | backend/model-specific | dense TP、fusion 和模型实现可能独立 |
| norm/bias/scales | 通常小但需按 checkpoint tensor 精确计入 | loader-specific 复制与量化布局 | runner/kernel-specific 复制与量化布局 |

### 10.2 KV、latent、indexer 与 state

共同公式只能计算一个已经解析好的 cache shard：

```text
cache bytes
  = owned layers
  × owned heads/latent/state width
  × owned tokens or allocated pages
  × element bytes
  + layout/alignment metadata
```

其中每一项的“owned”都来自 topology 和 cache strategy，而不是原始并行参数的简单除法。标准 KV 的理论 payload 还必须经过 page/block 取整，不能只用请求 token 数乘每 token 字节。vLLM `v0.25.0` 中单 cache group 的 scheduler block size 会乘 `DCP × PCP`，单 rank 的最大 token ownership 则按 `ceil(max_model_len / (DCP × PCP))` 计算；多种 block size 的 hybrid cache 与非平凡 DCP/PCP 组合会被拒绝。这些是版本适配器应输出的布局事实，不是共享核心的默认规则。

主要差异包括：

- vLLM 普通 DP EngineCore 各有独立 KV；SGLang 普通 DP 也各有独立 KV，但进程组织不同；
- SGLang DP Attention 的每个 Attention-DP 请求/缓存域只拥有自己的请求 KV，域内 workers 各持本地 shard，attention 权重则按有效 Attention-TP 切分并跨域复制；
- vLLM 与 SGLang 的 DCP 都切 Decode KV，但 page/block、backend 与 cache 类型约束不同；
- PP 下每个 worker 只为本 stage 的 cache layers 分配空间，最终服务容量受最紧张 worker 或框架协调规则限制；
- MLA、indexer、linear/recurrent state 不能套用标准 K/V 两份 tensor 公式。

## 11. Scheduler、容量与结果展示

### 11.1 需要同时给出的容量

显存规划器至少应区分：

- **单 scheduler/cache partition 的 aggregate max tokens**；
- **整个 deployment group 的 aggregate max tokens**；
- **单请求 max context**；
- **给定并发和输入/输出上限时，目标服务参数是否能启动**。

### 11.2 普通 DP 的聚合

若多个 DP 副本真正独立，则整个服务 aggregate max tokens 可以对副本容量求和；但单请求 max context 仍受单个副本及其最坏 PP/TP worker 限制。负载均衡不均或 prefix cache 命中差异会影响实际利用率，但本项目不估算性能和调度效率。

### 11.3 vLLM MoE DP 的边界

即使 DP rank 各有 scheduler/KV，MoE collective 仍要求参与 ranks 协同执行。planner 可以计算静态显存容量，但不能把它推导为吞吐或无条件可线性扩展的有效服务容量。dummy forward 和 collective workspace 仍要进入相关 phase 的每卡预算。

### 11.4 SGLang DP Attention 的分区

每个 Attention-DP 请求/缓存域是独立容量分区，域内仍由 Attention-TP、Attention-CP 和 PP workers 协同；FFN/MoE 会在更大的 TP domain 汇合 token。结果页需要同时展示每个域的有效容量和整个 deployment group 的算术总容量，并明确后者不是单请求可用上下文。

SGLang 启用 DP Attention 后还会自动调整 `chunked_prefill_size` 和调度保守度。planner 应展示 resolved value，不能继续用命令中的原始 chunk size 计算 activation 和 Graph。

## 12. CUDA Graph、activation 与通信 workspace

这些显存项不能仅靠理论权重/KV公式得到准确点值，但并行策略会明确改变其适用输入和上下界。

### 12.1 共同影响因素

- 每 worker 的 Prefill token 数、Decode batch 和 chunk size；
- 本地 hidden/intermediate shape；
- PP stage 的实际层及是否为首尾 stage；
- Graph capture batch-size 集合和 padding；
- collective 算法、group size、跨节点方式；
- MoE dispatcher/A2A backend；
- speculative/MTP token 数；
- PD role 及 connector staging。

### 12.2 vLLM

vLLM adapter 需要把 DBO、PP、PCP/DCP、MoE EP/EPLB、attention backend 和 Graph capture 参数投影到版本化校准键。即使某 DP rank 没有真实请求，MoE 同步路径也可能产生 dummy activation/collective buffer。

`v0.25.0` 还必须区分两条 GPU model runner 路径：

- V1/旧 runner `vllm/v1/worker/gpu_model_runner.py` 会在初始化内存 profiling 中临时 capture 一部分 Graph，估计共享开销与每 Graph 增量；只有 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 开启时，该估计才会从可分配 KV cache 预算中扣除。
- V2 runner `vllm/v1/worker/gpu/model_runner.py` 的 `profile_cudagraph_memory()` 当前直接返回 0，因此初始化 profiling 阶段不会预扣 Graph 估值。
- 两条路径在正式 `capture_model()` 后都会以 capture 前后 free memory 差得到实际 Graph bytes；worker 会在存在预估时记录 actual/estimated 差异，并在后续建议 KV cache 大小时使用实际 Graph 占用。

因此，planner 不能给 vLLM `v0.25.0` 配置套一个不区分 runner 的 Graph 公式。V2 的选择不仅受 `VLLM_USE_V2_MODEL_RUNNER` 显式覆盖，还会由模型类型、架构和不兼容功能解析；`v0.25.0` 已扩大 dense 模型默认进入 V2 的范围。导入器应输出 resolved runner，并同时解析 Graph mode/capture sizes；缺少匹配校准时，将 Graph 作为版本化经验预算或实测覆盖项。源码锚点为 `vllm/config/vllm.py:518-574`、`vllm/v1/worker/gpu_worker.py:430-583, 723-835`、`vllm/v1/worker/gpu_model_runner.py:6498-6712` 和 `vllm/v1/worker/gpu/model_runner.py:680-727`。

### 12.3 SGLang

DP Attention 会产生 DP gather/scatter buffer并改变 Graph padding。目标版本还会在参数解析中调整 chunked prefill，并在默认静态显存 reserve 中加入与 DP、TP、PP、Decode Graph batch 和 A2A backend 相关的项。

源码锚点：

- `python/sglang/srt/server_args.py:3623-3679`：activation、Decode/Prefill Graph、DP Attention 和 A2A reserve；
- `python/sglang/srt/server_args.py:4889-4905`：DP Attention 对调度保守度和 chunked prefill 的 resolved 调整；
- `python/sglang/srt/layers/dp_attention.py:472-691`：token gather/scatter 及其 global/local buffers。

### 12.4 规划器边界

共享账本可以统一记录 `graph`、`activation`、`communication workspace`、`runtime/allocator` 等类别及上下界；具体预算模型和校准键必须包含框架版本、模型、设备、backend、topology、phase 和 resolved workload。不能用一个跨框架固定百分比代替。

## 13. PD 部署对比

### 13.1 共同原则

本项目的 PD 部署由 Prefill 和 Decode 两个 deployment group 组成，二者使用同一模型/checkpoint和同一框架版本，但可以有不同硬件、拓扑、workload 和 role-specific runtime 参数。

两组必须分别：

- 展开 worker/rank；
- 计算 resident 权重和 cache；
- 计算对应 phase 的 Graph、activation 和 runtime；
- 计算 connector 发送、接收、转换与 staging 项；
- 给出最坏卡与 max-token 容量。

P/D 显存和容量不能相加成“每卡显存”或“单请求上下文”。

### 13.2 框架差异

- vLLM 的 PD connector、KV transfer 配置和每端 worker 参与方式必须由 vLLM adapter 解析；DCP/PCP/PP 与 connector 的支持组合也要独立判断。
- SGLang 的 `disaggregation_mode=prefill|decode` 会改变默认 activation/Graph reserve 路径；P/D 的 DP Attention、CP、DCP 和 A2A 组合需要分别 resolved。
- 两者的 connector buffer 生命周期、是否与 cache pool 或 Graph 并存，不能通过统一静态常量推断，应使用已验证 connector preset、用户预算或 trace 覆盖。

### 13.3 不在本轮推导的内容

本文不把一个框架的 connector 语义类比为另一个框架，也不从拓扑推导网络带宽、传输时间或吞吐。PD 在本项目中仍只判断两端服务能否在各自显存约束下启动。

## 14. 能力状态与兼容性

### 14.1 为什么布尔白名单不够

“模型支持 DP Attention”或“框架支持 DCP”不是充分结论。实际能力至少由以下维度共同决定：

```text
framework version
× model family / architecture adapter
× checkpoint / quantization
× device platform
× attention / MoE / A2A backend
× TP / PP / DP / EP / CP / DCP combination
× speculative or MTP mode
× serving or PD role
```

SGLang DP Attention 尤其不能仅根据 CLI help 中的模型名称，或看到模型实现调用了 helper，就判定所有组合可执行。vLLM PCP 也不能根据字段和 group 已存在就判定目标 backend 可用。

### 14.2 建议保留的证据状态

这里仅确认需要多级证据表达，不在本文锁定最终字段名：

| 状态 | 含义 | 计算与导出行为 |
|---|---|---|
| 已验证支持 | 目标 tag、模型和组合有源码路径及官方测试/recipe或本项目实测证据 | 可计算；校准完整时可给出确定结论和可执行导出 |
| 实现已接入但未验证 | 必要模型结构和代码路径存在，但缺目标组合证据 | 可展示理论分项；总体通常只能判“可能”，默认阻止可执行导出 |
| 实验性 | upstream 明示开发中或需要显式实验开关 | 清楚警告；MVP 可选择统一阻止 |
| 不支持 | 必要结构缺失、显式报错或约束冲突 | 阻止计算结论或可执行导出，并给出具体原因 |

最终产品状态需要与已有的输入校验、后端兼容性、显存结论三层展示保持独立。

## 15. 哪些事实可以进入共享显存账本

共享层可以接受以下已经解析完成的事实，而不需要知道这些事实由 vLLM 还是 SGLang 的哪一个 CLI 参数产生：

1. **物理资源**：node、device、单卡物理/实际可用显存。
2. **worker 身份**：物理 GPU 映射、deployment group role、scheduler/cache partition。
3. **组件放置**：每个 worker 的 checkpoint tensor 或逻辑组件 shard、replica、PP layer 集合。
4. **cache 放置**：cache group 类型、所属层、每 token 字节、page/block 取整、容量域和 worker 间协调规则的解析结果。
5. **阶段共存关系**：初始化/Graph capture、Prefill、Decode 中哪些分项同时存在。
6. **经验预算项**：Graph、activation、runtime、allocator、communication、connector 的适用范围、上下界和证据来源。
7. **结果聚合关系**：单 worker 峰值、PP/replica bottleneck、单 scheduler 容量、deployment group aggregate capacity。
8. **能力证据**：支持状态、原因、目标版本和证据引用。

共享账本应能回答“这张 GPU 在这个 phase 为什么需要这些字节”，但不应反向解释用户 `--tp-size` 或 `--enable-dp-attention` 的框架语义。

## 16. 哪些内容必须留在 framework adapter

下列逻辑若进入共享核心，会把版本特定语义伪装成通用公式，因此必须由 vLLM/SGLang 的 exact-version adapter 负责：

- CLI/Compose/Kubernetes 参数名、默认值、alias、冲突和 resolved override；
- worker 数、launcher/controller 结构及自动 rank placement；
- process group 坐标和组件使用哪个 group；
- 用户 TP/DP/EP/CP/DCP 到有效 component parallel degree 的映射；
- model layer 的 framework-specific loader、tensor shard/replica 规则；
- scheduler 和 cache partition ownership；
- PP stage 的实际分层与 cache block 协调方式；
- Graph capture、chunked prefill、DP Attention、DBO 等参数的自动调整；
- attention/MoE/A2A backend 的 workspace 与兼容性；
- PD connector 和 disaggregation role 的运行时语义；
- 模型、量化、backend、设备和并行组合的能力判定。

模型 adapter 与 framework adapter 也不能互相替代：模型 adapter 说明模型有哪些组件、层、heads、experts 和 checkpoint tensors；framework adapter 说明这些组件在目标运行时如何放置和执行。

## 17. 对命令导入导出的影响

导入结果需要同时保留三层信息：

1. **原始字段**：包括未知参数和原始变量，保证无损往返。
2. **框架解析结果**：resolved 默认值、自动改写、冲突、能力状态和实际拓扑。
3. **显存输入事实**：worker/component/cache/phase 的解析结果。

例如，SGLang 命令中的 `tp_size=8, dp_size=8, enable_dp_attention=true` 不能在通用层解释为 64 张 GPU；其物理 worker、Attention-TP 和 scheduler 数必须由 SGLang adapter 解析。vLLM 中同样的 `tp=8, dp=8` 又可能表示 64 个模型 workers，并在 MoE experts 上形成跨 DP 的 group。

导出时应使用框架 adapter 从已验证的 resolved 配置生成命令。若 adapter 无法证明导出的参数会还原为相同的 component placement、cache partition 和显存相关 resolved values，则不能生成可执行命令。

## 18. 尚待方案设计确认的决策

以下问题已有足够源码事实作为输入，但仍属于产品/架构决策，本文不代替用户确认：

1. **vLLM PCP 的 MVP 展示边界**：源码证据尚不足以判“已验证支持”；MVP 是统一标为 `unsupported`，还是保留可解析但不可执行导出的 `experimental` 状态。
2. **DCP 的首批支持矩阵**：两个框架分别允许哪些模型、attention backend、cache 类型、TP/DCP 和 PD role 组合。
3. **SGLang DP Attention 能力分级**：首批哪些模型组合为“已验证支持”，哪些仅标记“实现已接入但未验证”。
4. **MoE 首批 topology**：vLLM EP/EPLB 与 SGLang EP/MoE-DP/MoE-TP/dense-TP 分别支持到什么范围。
5. **容量展示**：结果页如何同时呈现单 scheduler、单 deployment group、单请求 context，并避免将算术 aggregate 误读为单请求能力。
6. **经验项校准键**：Graph、activation、runtime 和 workspace 的最小维度，以及缺少校准时采用用户预算还是区间结论。
7. **能力证据治理**：由 upstream CI/recipe、源码接入、本项目实测分别授予什么状态；框架升级时如何使旧证据失效。
8. **framework adapter 输出边界**：最终是输出 component placement facts、worker ledger seeds，还是更高层的 resolved topology。应在下一轮 schema 设计中结合可测试性决定。
9. **PD connector 支持范围**：两个框架首批各支持哪个 connector preset，以及 staging 与其他 phase 项的共存规则。
10. **自动 resolved 值的 UI**：原始命令值和运行时实际值发生变化时，如何并列展示并用于报告/导出。

## 19. 设计阶段应坚持的不变量

无论上述决策如何选择，后续方案都应保持以下不变量：

- 不用总权重连续除以 TP、PP、DP、EP；
- 不把多个并行 rank 的显存相加，因为它们通常是同一物理 worker 的组合坐标；
- 不把 CLI 参数存在当成能力受支持；
- 不把普通 DP、vLLM MoE DP 和 SGLang DP Attention 当成同一语义；
- 不把 vLLM PCP、SGLang Attention CP 和两者 DCP 合并成一个 CP；
- 不把整个服务 aggregate max tokens 当成单请求 max context；
- 不把 Prefill 与 Decode deployment group 的显存或容量相加；
- 不让共享核心猜测框架默认值、自动参数改写或 process-group 布局；
- 对 Graph、activation、runtime 和通信 workspace 使用版本化证据、区间或用户预算，不伪造通用精确公式；
- 所有结论最终落到物理 GPU、scheduler/cache partition 和 phase，才能回答目标服务是否能启动。

## 20. 本文边界

本文形成下一阶段方案设计的共同语言和约束，不完成以下工作：

- 不定义最终 `Deployment Config` 或 `Resolved Deployment` 的字段结构；
- 不确定是否采用“统一 rank/phase 核心”这一具体架构；
- 不给出尚未审计模型的 DP Attention/CP/DCP 支持承诺；
- 不估算吞吐、TTFT、TPOT、网络时间或负载均衡效果；
- 不用一个框架的经验数据替代另一个框架的校准。

下一阶段应从本文件的组件归属事实出发，分别设计 vLLM 与 SGLang adapter 的解析责任，再验证两者能否稳定投影到同一种 per-device memory ledger，而不是先构造统一并行参数模型。
