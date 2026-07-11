# LLM Memory Planner 技术可行性与同类项目审计

> 状态：设计输入，非最终方案
>
> 审计日期：2026-07-11
>
> 审计范围：公开计算器、开源显存工具、vLLM/SGLang 运行时语义、浏览器端配置往返和主站集成
>
> 结论：技术可行，但不能把任一现有计算器整体作为本项目计算内核

确认记录：2026-07-11，已确认目前没有一个被审计项目能够单独满足本项目完整需求；后续方案不得以直接 fork 某一项目作为总体架构，但允许合规复用其公式、数据、测试向量和局部设计。

## 1. 执行摘要

需求在技术上可实现，但必须把不同确定性的问题拆开处理：

1. **权重和 KV/状态缓存**：在模型结构、checkpoint 元数据、dtype、后端布局、并行拓扑和取整规则完整时，可以使用可解释的确定性公式计算。
2. **激活、CUDA/XPU Graph、runtime、allocator 和通信缓冲**：不存在跨模型、后端、设备及版本可靠通用的闭式公式，需要版本化校准、用户预算或 trace 实测。
3. **部署可行性**：不能比较“总模型显存”和“所有卡总显存”。必须计算到 `pool -> node -> rank -> phase -> memory item`，并取最坏 rank、最坏阶段。
4. **命令与 manifest 往返**：可以在浏览器内安全完成，但要把静态词法解析、统一语义模型、后端参数映射和格式导出分层，绝不调用 shell。
5. **纯前端部署**：计算、解析、搜索和报告均可在浏览器完成；大规模拓扑搜索可放入 Web Worker。静态产物可以按现有主站插件约定集成。

因此推荐吸收开源项目的公式 strategy、证据模型、纯函数接口和测试向量，但独立设计核心账本与适配器边界。独立设计不是为了重新发明公式，而是因为现有项目均无法同时表达阶段共存、rank 放置、多机、PD 异构池、Graph 校准和命令无损往返。

## 2. 审计方法与证据边界

本次使用三类证据，并保持边界：

- **公开网页行为**：用于参考输入流程、结果表达和公式说明，不推断不可见实现。
- **固定 commit 开源源码**：用于判断可复用机制、理论假设和许可证。
- **固定 commit 运行时源码**：用于确认 vLLM/SGLang 的真实显存预算语义，而不是把第三方计算器当作运行时事实。

测试证据分为：

- 本次实际运行通过；
- 仅审阅仓库测试和源码；
- 运行失败或环境缺失，明确记录，不宣称通过。

## 3. 用户提供的公开工具

### 3.1 APXML VRAM Calculator

参考价值：

- 模型、量化、上下文和硬件的低门槛输入流程；
- 把权重、KV cache 和额外开销分开解释；
- 面向普通用户的 fit/no-fit 展示。

不足：

- 公开页面主要是模型级或设备级估算；
- 无法作为 rank 放置、分阶段峰值和后端版本化校准的证据；
- 不覆盖 P800、PD connector 和服务配置语义往返。

结论：参考交互层次，不复用为计算依据。

### 3.2 阿里云 PAI 显存估算

参考价值：

- 清楚说明参数量、精度、序列长度对部署和微调显存的影响；
- 适合作为公式说明和硬件选型文案参考。

不足：

- 通用产品文档不能代表具体 vLLM/SGLang 版本的 Graph、allocator 和 cache pool 行为；
- 目标包含训练/微调，与本项目只做推理显存的边界不同。

结论：参考概念说明，不采用其通用比例作为运行时峰值模型。

### 3.3 大模型 GPU 资源计算器（blog.liuts.com）

参考价值：

- 将模型参数、KV cache、并发和 GPU 数量放在同一交互中；
- 适合参考快速试算和配置对比方式。

不足：

- 总卡数或总显存级汇总不能替代最坏 rank 判断；
- 未形成可验证的后端、阶段、trace 和校准契约。

结论：参考输入密度和即时反馈，不采用其总显存判定方法。

### 3.4 KVCache.ai Calculator

公开页面背后有 MIT 开源实现，是本次审计中最有价值的 KV 公式参考。

- 仓库：[kvcache-ai/kvcache-blog](https://github.com/kvcache-ai/kvcache-blog/tree/7c6d39f2fa8678982578ca8bc694d765130337ea)
- 固定 commit：`7c6d39f2fa8678982578ca8bc694d765130337ea`
- 许可证：MIT
- 核心源码：`assets/js/kv-cache-calculator.js`
- 模型数据：`data/kv_cache_calculator/models.yaml`
- 测试：`tests/kv-cache-calculator.test.mjs`
- 本次验证：36 个 Node 测试全部通过。

可复用：

- standard MHA/GQA/MQA；
- MLA latent + decoupled RoPE；
- DSA/MLA + indexer；
- full/sliding 混合层；
- linear/recurrent state；
- MiniMax MSA 主缓存和 key-only indexer；
- 不同 cache group 的独立 dtype；
- 公式分项和来源元数据；
- golden test vectors。

不能直接复用为总显存内核：

- 只计算 cache payload，明确不覆盖 allocator/memory pool；
- 不处理权重、Graph、激活和 runtime；
- 不处理 TP/PP/DP/EP/CP、rank 放置和最坏 rank；
- 不处理阶段峰值、PD connector 和校准适用范围。

## 4. 开源同类项目源码审计

### 4.1 InferPlan

- 仓库：[rabrooks/inferplan](https://github.com/rabrooks/inferplan/tree/563a689e82ec4c625a21cefdcc6e281c4bca4765)
- 固定 commit：`563a689e82ec4c625a21cefdcc6e281c4bca4765`
- 许可证：Apache-2.0
- 主要路径：`src/engine/types.ts`、`inference.ts`、`fit.ts`、`hf.ts`
- 测试状态：仓库包含 65 个 Vitest case；本次因 npm 网络 `ECONNRESET` 未能安装依赖，未宣称实跑通过。

值得参考：

- 纯 TypeScript、无 UI 依赖的领域引擎；
- 统一使用 bytes 的纯函数接口；
- breakdown、warning 和计算结果分离；
- 最大 token、最小 GPU 等 inverse solver 模式；
- HF config 和 safetensors index fallback。

不能整体采用：

- 权重直接平均除以 `TP * PP`，无法表达复制参数和 PP 最坏 stage；
- PP 不整除只警告，结果仍使用偏低平均值；
- activation 使用固定 `6 * hidden` buffer heuristic；
- framework overhead 固定为 0.75 GiB；
- 无 Graph、EP/DP/CP、rank placement 和阶段账本；
- llm-d 模块以性能/SLO 为主，没有 connector 两端显存模型。

### 4.2 llm-cal

- 仓库：[FlyTOmeLight/llm-cal](https://github.com/FlyTOmeLight/llm-cal/tree/ca137f6b71f05d02153111dc9db18d71c0934d47)
- 固定 commit：`ca137f6b71f05d02153111dc9db18d71c0934d47`
- 许可证：Apache-2.0
- 主要路径：`architecture/profile.py`、`weight_analyzer/`、`core/evaluator.py`、`fleet/planner.py`、`command_generator/`
- 测试状态：仓库包含 46 个测试文件、约 2841 行测试；本次环境缺少 pytest，未宣称实跑通过。

值得参考：

- attention、MoE、position 等可组合 architecture traits；
- estimated parameters、observed safetensors bytes、quant fingerprint 和 reconciliation 分层；
- 每个数值携带 provenance、confidence 和解释链；
- 独立兼容性矩阵和薄命令 adapter。

不能整体采用：

- 普通 sliding window 不能表达 full/sliding 混合层；
- MLA 近似遗漏 decoupled RoPE；
- fleet 偏单机 TP，缺少 PP/EP/多机/rank；
- 使用固定 10% overhead，缺少 activation/Graph/runtime 校准；
- 命令模块只有生成，无导入和未知参数保真。

### 4.3 FitLLM Engine

- 仓库：[click6067-ship-it/fitllm-engine](https://github.com/click6067-ship-it/fitllm-engine/tree/fa4210e690480092fefd50fceade7702fd385e70)
- 固定 commit：`fa4210e690480092fefd50fceade7702fd385e70`
- 许可证：MIT
- 主要路径：`engine.js`、`vectors/fit-vectors-v1.json`、`fixtures/measured.json`
- 测试状态：14/14 conformance vectors 通过；用 `node --test test/*.test.mjs` 运行的 12 个测试通过。仓库 `npm test` 在当前 Node 22 下把目录当模块而失败。

值得参考：

- 浏览器友好的 config parsing；
- standard、MLA、sliding/full 和 hybrid linear KV；
- 权重与 KV dtype 分离；
- conformance vectors；
- resident weights 与 total-to-run 的 measurement kind 区分。

不能整体采用：

- 单一 `weights + KV + heuristic runtime + reserve`，无运行阶段；
- runtime 使用固定比例和常数；
- 多 GPU 直接汇总设备显存，无分片、放置和最坏 rank；
- measured fixtures 主要是 Apple/oMLX idle resident，不是 H20/P800 生成峰值。

### 4.4 llm-analysis

- 仓库：[cli99/llm-analysis](https://github.com/cli99/llm-analysis/tree/d841e40aec8c84b9e76ba9cfbce67418adb172bd)
- 固定 commit：`d841e40aec8c84b9e76ba9cfbce67418adb172bd`
- 许可证：Apache-2.0
- 主要路径：`llm_analysis/analysis.py`
- 测试状态：仓库包含 12 个 pytest 测试；本次环境缺少 pytest，未宣称实跑通过。

值得参考：

- 传统 Transformer 的参数、TP/PP/EP、KV 和 activation 理论；
- Prefill/Decode 分开描述的思路。

不能整体采用：

- 架构基线较旧，不覆盖 MLA、linear state、indexer、Graph 和 PD；
- PP 不整除时向下取整，可能低估最大 stage；
- 仍是 per-GPU aggregate，不是实际 rank placement；
- Python 实现不适合作为浏览器核心直接移植。

### 4.5 Hugging Face Accelerate

- 仓库：[huggingface/accelerate](https://github.com/huggingface/accelerate/tree/665444ceb62211f2b410d0d0fdb4bc013c5effdf)
- 审计 HEAD：`665444ceb62211f2b410d0d0fdb4bc013c5effdf`
- 许可证：Apache-2.0

参考范围仅限权重 tensor dtype/size、模块大小和设备映射的交叉验证。Accelerate 面向模型装载和训练/推理设备映射，不提供本项目所需的后端 Graph、Paged KV、PD 和分阶段峰值语义。

## 5. vLLM 与 SGLang 运行时事实

### 5.1 固定审计版本

- 原定适配基线 vLLM：`c227aaa3f8edd02dae4583e27246430eebabfb25`
- 原定适配基线 SGLang：`789bc3995c5aecfeab08212cd557b3d102dd4181`
- 2026-07-11 刷新的 vLLM upstream HEAD：`735def4fcf39945b6e6c24769878760e3e113b15`
- 2026-07-11 刷新的 SGLang upstream HEAD：`7b99900980df86423bf9532833968fcb0a4736b0`

两组快照的目的不同：原定基线保证早期设计可复现，刷新 HEAD 用于判断当前实现是否已经改变。正式适配器必须固定版本或版本范围；后端升级不得静默改变旧报告结果。

### 5.2 权重

权重可高确定性计算，但“参数量 × 标称位宽 / TP”不足以达到验收精度。需要考虑：

- safetensors/checkpoint 中各 tensor 的真实 dtype 和字节数；
- quant scale、zero point、codebook 和未量化 tensor；
- tied/shared tensor 是否物理共享；
- embedding、lm_head、norm、vision、MTP 等组件；
- TP/PP/EP 的分片与复制；
- PP 余数、首尾 stage 差异和实际 rank placement；
- 加载时临时峰值是否与最终 resident 权重并存。

因此权重应有两个不同的值：模型 checkpoint 总字节和特定 rank 的运行时 resident 权重。二者不能混为一个指标。

### 5.3 KV 与状态缓存

基础标准注意力公式为：

```text
bytes = tokens * sequences * layers * 2(K,V)
      * local_kv_heads * head_dim * dtype_bytes
```

但实际需要 strategy 化处理：

- GQA 中 `TP > kv_heads` 时的 KV head 复制；
- MLA latent、decoupled RoPE 和 backend-specific replication/layout；
- indexer、draft model、MTP 等额外缓存；
- full/sliding/linear/recurrent 混合 cache group；
- 每个 group 的独立 dtype；
- Paged KV 的每序列 page/block 向上取整；
- hybrid group 的对齐和 rank 间最小可用 block 数；
- PP stage 持有的实际层集合。

结论：KV payload 可以确定性计算，但必须由“模型 cache strategy + 后端 layout/sharding adapter + 拓扑 placement”共同决定，不能统一除以 TP。

### 5.4 激活、Graph 与 runtime

没有可信的跨后端通用闭式公式：

- 激活峰值受 scheduler、batched tokens、chunked prefill、attention backend 和临时 workspace 影响；
- Graph 受 capture size 集合、piecewise/full graph、模型分支、通信 capture 和后端实现影响；
- allocator reserved/allocated 差异、CUDA/XPU context、NCCL/XCCL buffer 和非框架占用随版本与环境变化。

理论公式可用作校准模型的 feature 或保守预算提示，但不能直接升级为“可运行”证据。可运行结论需要适用校准、明确用户预算或实测 trace。

### 5.5 两个后端的预算语义不能强行统一

vLLM 的 worker 会 profile 权重、torch activation peak、non-torch increase，并将 Graph 等估计纳入可用 KV 的确定过程；关键实现位于：

- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu/model_runner.py`（刷新 HEAD）
- `vllm/v1/worker/gpu_model_runner.py`（原定基线）
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/compilation/cuda_graph.py`

SGLang 的 `mem_fraction_static` 表达权重与 KV 等 static pool 预算，并保留 dynamic 内存；部分路径会在 graph capture 后重新决定 pool。关键实现位于：

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/kv_cache_builder.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/pool_configurator.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/server_args.py` 及平台分支对应模块

因此核心账本可以统一，后端预算解释和参数映射必须放在版本化 adapter 中。使用一个通用 `available KV = fraction - weights - all overhead` 公式会在至少一个后端上语义错误。

刷新 HEAD 还提供了可解析的运行时证据：vLLM 可观察 model loading、torch activation peak、non-torch increase、available KV 和 graph capture memory；SGLang 可观察 weight loading、KV pool allocation、graph capture 及 post-capture KV sizing。日志文本不是永久 API，解析器仍必须绑定后端 commit，并保留无法识别时的 unknown 状态。

## 6. PD 与异构池可行性

PD 模式可以在显存层面建模，而无需引入性能预测：

- P/D 各自拥有同构 pool、节点、rank、阶段和显存结论；
- 两池可以是 NVIDIA/P800 任意方向组合；
- connector adapter 只描述两端 resident/send/recv/staging/conversion buffer 的大小、生命周期和证据；
- trace 可以分别覆盖 P 端、D 端和 connector 项；
- 不计算带宽、传输时间和排队。

SGLang 刷新 HEAD 中已有 connector staging buffer 的实际配置和分配路径，例如 `python/sglang/srt/disaggregation/common/staging_buffer.py`。这些证据支持把 connector resident pool 建成独立显存项，但不能据此假定所有 connector 或 P800 平台分支具有相同默认值。

现有通用计算器没有完整 PD connector 显存模型。自研的优势是把 connector 当普通、可校准的 memory item，并分别挂到 P/D rank 和 phase，而不是把网络性能模型引入显存核心。

## 7. 浏览器端服务配置导入与导出

### 7.1 推荐组合

1. **YAML 解析**：采用 `yaml`（eemeli/yaml，ISC）读取 Compose/Kubernetes 多文档 YAML，保留 AST、range、comment 和 source token。
2. **CLI 词法层**：自研受限、无执行能力、lossless shell lexer，只识别空白、引号、转义、参数、赋值和操作符边界。
3. **参数语义层**：为 vLLM、SGLang 和平台分支建立版本化 argument schema，不依赖后端 Python argparse 运行时。
4. **统一中间表示**：recognized fields 写入 Deployment Config；unknown、conflict、variable 和危险结构写入带来源位置的 Extension Bag。
5. **导出层**：CLI、Docker、Compose、Kubernetes 各自从统一语义生成；兼容性阻断时只输出不可执行草稿。

### 7.2 为什么 CLI lexer 自研

不建议直接使用完整 shell parser 或执行 shell，原因是：

- 产品只需要识别部署命令，不需要实现 Bash；
- `$VAR` 必须保留待用户补值，而多数 shell parser/quote 库会进行或假定展开；
- `$(...)`、反引号、管道和重定向必须原样保留并标记，不能求值；
- round-trip 需要保留未知参数的 raw lexeme、顺序和来源；
- 浏览器端安全边界应可通过小型 lexer 的 property/fuzz test 验证。

自研 lexer 不尝试证明任意 shell 命令安全，也不将危险结构转换成可执行命令。遇到不在受限语法内的结构，整体保留为 unresolved segment，并阻止可执行导出。

可以用 `shell-quote`、`yargs-parser` 等成熟项目作为差分测试和行为参考，但不让它们决定安全边界。

### 7.3 为什么 YAML 不自研

YAML 的 anchors、aliases、多文档、block scalar、tag、comment 和位置映射复杂。成熟 `yaml` 库已提供 CST/AST 和 source token，自己实现收益低、风险高。

不建议把 `@kubernetes/client-node` 打入前端：其当前包解压体积约 48 MB，功能远超静态 manifest 解析需求。Kubernetes OpenAPI 校验可以选取必要 schema 生成轻量数据，不需要完整客户端。

### 7.4 语义往返条件

往返等价不等于字符串相同。需要定义 canonical semantic projection：

- 模型、dtype、负载和 Graph 参数；
- TP/PP/DP/EP/CP；
- pool、node、replica、device resource 和 placement；
- PD role 与 connector；
- 环境变量已解析值和原表达式；
- unknown args 的 raw token、位置、scope 和 annotation。

测试比较 canonical projection；格式、引号和参数顺序允许变化。任何未知项丢失、冲突被覆盖或危险表达式被求值都属于失败。

## 8. 纯前端与主站集成可行性

计算和解析均为本地确定性运算，纯前端技术上可行：

- TypeScript 核心可在主线程运行小规模校验；
- 拓扑搜索和大 trace 解析可放入 Web Worker；
- IndexedDB 可保存本地草稿和校准数据，不要求服务端；
- 分享 URL 只包含用户明确允许的非敏感紧凑配置；
- 导入文件使用浏览器 File API，不上传；
- 报告和 JSON 使用本地 Blob 导出。

主站已有可复用静态插件约定：

- 构建 base：`/plugins/llm-memory-planner/`；
- payload：`plugins/llm-memory-planner/`；
- wrapper：`integrations/llm-memory-planner/index.html`；
- registry：`integrations/registry/index.json`；
- 独立 `plugin.manifest.json`；
- 同步构建产物时必须避免 `rsync --delete` 删除 manifest。

本阶段只确认可行性，不修改主站。

## 9. 复用矩阵

| 能力 | 参考来源 | 决策 | 原因 |
| --- | --- | --- | --- |
| KV cache strategies | KVCache.ai | 合规移植思想、公式和测试向量 | 架构覆盖新、解释性强、MIT |
| 模型 architecture traits | llm-cal、KVCache.ai | 参考后自研 schema | 需要加入 rank、phase、backend layout 和证据范围 |
| 权重证据链 | llm-cal、Accelerate | 参考后自研 TypeScript 实现 | Python 实现和在线 API 假设不适合纯前端 |
| 纯函数与 inverse solver | InferPlan | 参考接口和测试模式 | 其平均分片和固定 overhead 不满足精度 |
| Conformance vectors | FitLLM、KVCache.ai | 纳入差分和 golden tests | 可防止公式回归，但不能证明运行时峰值 |
| 传统 TP/PP/EP 理论 | llm-analysis | 引用理论，不移植代码 | 架构较旧且存在最坏 stage 低估 |
| rank/phase memory ledger | 无完整实现 | 自研 | 是本产品正确处理多机、PD、Graph 的核心 |
| 校准适用范围与 trace override | 无完整实现 | 自研 | 验收要求 H20/P800 版本化误差边界 |
| 后端预算 adapter | vLLM、SGLang 源码 | 自研版本化 adapter | 两个后端预算语义不同 |
| YAML parser | eemeli/yaml | 直接依赖 | 完整 YAML AST 自研风险无意义 |
| CLI lossless lexer | shell parser 项目作参考 | 自研受限 lexer | 安全、不展开、unknown 保真要求特殊 |
| 主站集成 | kinchow.github.io 既有约定 | 复用现有插件结构 | 已有静态插件成功先例 |

## 10. 风险与验证策略

### 10.1 主要风险

- 新模型缓存布局变化快，`config.json` 字段并不总能唯一决定 runtime layout。
- P800 平台分支与上游 vLLM/SGLang 的参数和内存行为可能不同。
- Graph/allocator 的校准容易被驱动、runtime、backend 和 capture 配置变化污染。
- 用户命令可能包含任意 shell 结构，静态解析不能承诺完整 Bash 兼容。
- 拓扑搜索空间随并行维度和 placement 指数增长。

### 10.2 控制措施

- 能力/strategy 驱动模型，不以模型 ID 写核心分支；
- 所有 adapter、公式和 calibration 固定版本与适用范围；
- 范围外返回 unknown，不做无提示外推；
- measured trace 优先，但必须校验 measurement kind 和阶段；
- CLI 只接受受限 grammar，危险段保留并阻断执行导出；
- 搜索先做整数、整除、兼容性和容量下界剪枝，再进行 placement 评估；
- 用开源 vectors、手算 fixtures、runtime trace 和浏览器 round-trip 四层测试交叉验证。

## 11. 对方案设计的约束

后续方案至少应满足：

1. 计算核心、校准、后端、命令和 UI 模块解耦。
2. 模型优先，但用可组合 capabilities/strategies 表达模型结构。
3. rank 是显存 fit 判断的最小单元，phase 是峰值共存的最小时间单元。
4. 公式值、校准值、实测值和 unknown 使用统一但不可混淆的证据类型。
5. vLLM/SGLang/P800 分支使用版本化 adapter，不污染通用账本。
6. 导入器不执行任何输入，导出器不能绕过兼容性阻断。
7. 所有结果能追溯到输入、公式/校准版本和来源证据。

### 11.1 设备与框架边界的讨论状态

2026-07-11 方案讨论已确认：

1. P800、NVIDIA 设备属于平台/设备维度，vLLM、SGLang 属于推理框架维度；显存核心不得把四者写成互相绑定的四套计算器。
2. 模型、平台、框架、校准、命令和 UI 等模块之间需要解耦，不能为每种平台与框架组合复制一套计算公式。
3. 应提取与具体设备无关、可通过理论推导的逻辑显存负载；设备、算子实现、框架调度、Graph、allocator 和通信实现会进一步改变实际运行峰值。

尚未确认：是否采用一个统一的 `rank/phase` 显存核心作为系统中心，以及模型、后端、校准、命令和 UI 是否都应被描述为围绕该核心工作。需要继续讨论模块职责、数据流和依赖方向后再决定。

这里的“解耦”不等于忽略组合差异。某个框架在某个平台上的实际分支可能改变 cache layout、算子 workspace、参数语义或日志格式，因此允许存在显式的 integration profile：

```text
通用模型机制 + 通用理论核心
          + framework adapter
          + platform adapter
          + 必要时的 framework-platform integration profile
```

integration profile 只补充该组合的差异和兼容性证据，不复制通用权重、KV、拓扑和峰值计算逻辑。

理论核心也不应把所有 byte size 都误称为设备无关。元素数量和逻辑生命周期通常可独立计算；物理 dtype、量化 scale/zero-point、cache layout、page padding、tensor repack 和复制规则可能由 framework/platform adapter 决定。只有这些物理规则已知后，才能把逻辑负载确定地转换为实际字节。

### 11.2 Rank 与运行场景的讨论澄清

TP rank、PP rank、DP rank、EP rank 和 CP rank 不是五个需要相加的独立 rank，而是同一个逻辑执行 rank 的并行坐标。它们共同决定该 rank 持有的层、tensor shard、专家、模型副本、cache shard 和通信状态。显存项先依据完整坐标与放置规则归属到逻辑 rank，再汇总到承载该 rank 的物理设备；最终 OOM 判定单位是物理设备。

因此概念公式是：

```text
rank memory items = resolve(model, full parallel coordinates, placement,
                            framework rules, platform rules, workload scenario)

device peak = device-scoped items
            + aggregate(all rank/process items placed on this device)
```

这里不能写成 `TP rank memory + PP rank memory + DP rank memory + EP rank memory + CP rank memory`，否则同一权重或缓存会被重复计算。各并行策略对同一显存项施加的是组合后的 shard、replicate、placement 或 workspace 规则。

固定的 `initialize / graph_capture / prefill / decode / mixed` 枚举也不足以完整描述峰值。Chunked prefill、continuous batching、prefill/decode 混合调度、capture size、eager/graph 路径、batched tokens 和 KV 占用会在同一粗粒度阶段内产生不同峰值。

后续设计应区分：

- **生命周期（lifecycle）**：加载/初始化、Graph capture、稳态服务等较稳定的时间边界；
- **显存场景（memory scenario）**：带参数的具体共存条件，例如 chunked prefill 的 chunk tokens、并发序列、同时 decode tokens、KV 占用、graph/eager 模式和 capture size。

候选场景由统一工作负载约束与特定 framework/platform scheduler 规则生成，账本对每个适用场景计算设备峰值并取最大值。不能靠手写所有配置组合，也不能用一个普通 Prefill 数字覆盖 chunked prefill。trace 可直接提供实测场景并覆盖相同适用域内的估算。

### 11.3 已确认的精度与校准目标

2026-07-11 方案讨论确认，不把“经验分项误差不超过 10%、总峰值误差不超过 5%”作为首版目标。用户已有机器时，真实运行比复杂的运行时模拟更有价值；工具不应为了输出貌似精确的单点数字而枚举大量调度场景。

校准仍然保留，但目的调整为：给无法从理论确定的 activation、Graph、runtime、allocator 和通信显存建立适用域内的保守上下界，而不是追求完全准确的点估计。

判定采用区间逻辑：

```text
conservative upper bound <= safety threshold  => 可运行
conservative lower bound > physical memory    => OOM
otherwise                                     => 可能运行，需要实测
```

“临界”是“可能运行”的风险标签，不作为独立的确定结论。校准典型值可以帮助用户了解大概需要多少显存，但不得用于给出确定的“可运行”或“OOM”。

正式验收的核心目标从点估计误差改为分类可靠性：验收集合内，标记“可运行”的配置在真实机器、声明负载下不得发生显存 OOM；标记“OOM”的配置必须能够复现显存 OOM。证据不足或上下界重叠时，宁可输出“可能运行”，不追求覆盖所有配置。

这一调整同时意味着不再建立复杂的请求时间线模拟器。首版只保留影响边界的关键配置、粗粒度生命周期和必要的运行场景类别；框架/平台校准吸收难以建模的调度与算子差异，trace 或用户预算可以覆盖经验边界。

### 11.4 已确认的模型支持边界

2026-07-11 方案讨论最终确认，首版不支持用户自定义模型或自定义 checkpoint，也不提供用户侧 `config.json`、checkpoint metadata、Tensor Manifest 上传和离线 Manifest CLI。

业界工具中的“自定义”通常是以下较弱能力之一：

- APXML、liuts 等允许输入参数量、量化或自定义设备显存，使用通用公式给出粗略估算；
- FitLLM、InferPlan 等可以解析部分 Hugging Face config，但仍依赖已知架构和启发式 overhead，不能自动证明任意新模型的多机 tensor placement；
- llm-cal 对未知架构降级为权重文件大小，KV 和后端兼容性保持未知；
- KVCache.ai 使用维护者审核的模型目录和固定 formula strategy，不从任意 config 自动推导新缓存机制。

没有被审计项目能够从任意 checkpoint 自动、可靠地推导模型结构、TP/PP/EP/DP Attention 权重放置、cache layout 和后端兼容性。即使用户提供完整 tensor metadata，也只能证明有哪些 tensor，不能证明每个 tensor 在特定框架和平台上如何分片、复制或 repack。

因此首版只支持内置、固定 repository/revision/checkpoint 且已完成以下适配和验证的模型：

- 完整 checkpoint tensor metadata；
- 模型结构和 tensor placement 规则；
- KV/latent/indexer/state cache strategy；
- vLLM/SGLang 及平台版本兼容性；
- golden vectors 和真实机器验收证据。

新模型只能通过产品版本更新加入。任一模型机制或必要 adapter 未支持时明确返回 `unsupported`，不使用 tensor 名称猜测、参数量乘位宽或相近模型公式进行降级估算。服务命令中引用未支持模型时仍可静态解析并保留参数，但不能进入显存计算或可执行导出。

### 11.5 已确认的框架、并行和容量口径

2026-07-11 方案讨论确认，首版要求显式指定框架版本，并锁定当日最新稳定 release：vLLM `v0.24.0`、SGLang `v0.5.15`。不以动态“最新版”解析已有配置，也不兼容更早 release；平台分支使用独立的已适配发行版本。

“权重可放置”和“目标服务配置可启动”独立判断。服务启动必须包含目标 KV/latent/indexer/state cache pool、CUDA/XPU Graph、activation、runtime、allocator、通信和安全余量；max-token 容量在权重放置后按整个服务、每副本或调度分区、单请求三个层级展示。

负载参数按用途选填。命令中存在 max context、max running requests、max batched/prefill tokens 或 chunk size 时用于目标配置校验；未显式提供时采用固定框架/平台版本的真实默认值并标注来源，无法确定默认值时经验项保持 unknown。

Rank placement 不由用户编辑，而由版本化框架/平台 adapter 根据机器数量和 TP/PP/DP/EP/CP 自动生成。用户决定机器数量和具体部署形式，工具只校验该方案；首版取消最小机器数、最优拓扑和并行范围搜索。

SGLang DP Attention 作为首版显式能力，不按普通 DP 处理。其 Attention 与 FFN/MoE 使用不同的组件级并行规则；有效 Attention TP、KV/cache 归属、scheduler 分区、chunked prefill 默认值和 Graph/runtime 预留均由 SGLang adapter 解析。容量按 DP Attention 调度分区分别展示，再提供请求可均衡分配前提下的服务聚合容量，单请求不能跨分区使用聚合容量。

PD 部署的 Prefill、Decode 池分别展示并判定显存、缓存 tokens 和上下文容量，不生成两池相加的 max-token 总数。

## 12. 下一步

基于本审计，自顶向下讨论方案：

1. 产品架构与可信度模型；
2. 统一部署数据模型；
3. 确定性显存核心；
4. 分阶段、分 rank 拓扑规划器；
5. 校准与 trace；
6. 后端兼容性和命令往返；
7. PD 显存模型；
8. 前端信息架构；
9. 测试与校准矩阵；
10. 主站静态集成。

每一节确认后再进入下一节。全部设计确认后写入设计文档和 ADR，再制定实施计划。
