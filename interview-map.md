# 实验 → 面试题映射表（自测清单）

用法：每做完一个模块，用下面的问题自测。能 60 秒讲清 + 指出自己 repo 里的证据 = 过关。
标记 ⭐ = 高频原题。

## M0 推理数学

- ⭐ 为什么 decode 阶段是 memory-bound 而 prefill 是 compute-bound？用 arithmetic intensity 解释。
- ⭐ 一个 8B 模型、32 层、8 个 KV head、head_dim=128，context 4k、batch 32，KV cache 占多少显存？（笔算）
- ⭐ RTX 4090 上推理吞吐的理论上限怎么算？（带宽 / 每 token 必访字节数）
- ⭐ 显存怎么估算？模型权重 + KV cache + activation + 框架开销各占多少？
- FP16/BF16/FP8/INT4 模型加载后各占多少显存？

## M1 PyTorch 手写推理（scratch-decode）

- ⭐ 不用 `model.generate`，手写一遍 generate 主循环：tokenize → embed → N×(RMSNorm→QKV→RoPE→attention+KV cache→MLP) → logits → sample。
- ⭐ prefill 和 decode 的计算形态有什么区别？（GEMM vs GEMV，为什么 decode 每步只算 1 个 token）
- ⭐ KV cache 的数据结构怎么设计？past_key_values 每层怎么存、怎么 append？max length 怎么预分配？
- attention 里每一步的矩阵维度怎么变？（面试官会追着手推：B×H×S×D）
- greedy / temperature / top-k / top-p 采样各怎么做？
- 你自己测的 prefill/decode 耗时占比是多少？随 seq len 怎么变？（背下自己的数据）

## M2 Kernel 实战（Triton + CUDA）

- ⭐ 写一个 fused softmax / layernorm 的 Triton kernel。追问：BLOCK_SIZE 怎么选？num_warps/num_stages 干什么？
- ⭐ online softmax 怎么做？为什么能省一遍显存读取？（flash attention 的核心思想）
- ⭐ Triton 的 autotune 原理？tl.dot 对 block 尺寸有什么要求？
- ⭐ Triton vs CUDA C++ 的取舍？什么场景用哪个？（开发效率 vs 极致控制；serving 引擎里 attention 为什么常用 CUDA 而非 Triton）
- ⭐ GEMM 怎么优化？说出 naive → smem tiling → register tiling → tensor core 的演进，每步为什么快。
- ⭐ occupancy 是什么？register/shared memory 用量怎么影响 occupancy？
- ⭐ ncu 里你看哪些指标？（achieved occupancy、memory throughput、SM busy、warp stall reasons、L2 hit rate）
- roofline 模型是什么？给一个 kernel 的 AI 和实测 GFLOPS 判断它在 ridge 哪边。
- warp divergence、bank conflict、coalesced access 各是什么？
- ⭐ bank conflict 具体怎么发生？shared memory 32 个 bank 怎么映射地址？stride 访问什么 pattern 会 conflict？（面试官视角原题，候选人答错率高）
- ⭐ CUDA stream 的异步语义：同一 stream 里 kernel/memcpy 执行顺序？不同 stream 之间？event 怎么用来做依赖同步？
- ⭐ 寄存器溢出（register spilling）会发生什么？表现在哪里（local memory、性能骤降）？怎么从 ncu 看出来？
- kernel fusion 为什么有用？LLM 里哪些 op 通常被 fuse？（RMSNorm+quant、QKV proj、attention 全流程、silu*mul）

## M3 生产引擎（vLLM + SGLang）

- ⭐ vLLM 为什么快？讲清 PagedAttention + continuous batching + prefix caching + CUDA graph 各自的贡献。
- ⭐ PagedAttention 原理？block/page 怎么分配？解决了什么问题（内部碎片、外部碎片、prefix 共享、copy-on-write）？
- ⭐ continuous batching 原理？iteration-level scheduling 是什么？比 static batching 好在哪？
- ⭐ vLLM V1 的架构：API server → engine core → scheduler → KV cache manager → model runner → worker，各自职责？
- ⭐ scheduler 每步做什么？waiting/running 队列、token budget、抢占（preemption）策略？
- ⭐ prefix caching 怎么实现？block hash 怎么算？什么场景收益大？
- RadixAttention 和 PagedAttention 的区别？（SGLang：radix tree 存 KV，LRU 驱逐，cache-aware 路由）
- ⭐ chunked prefill 解决什么问题？max_num_batched_tokens 怎么调？
- vLLM vs SGLang vs TensorRT-LLM 定位和差异？（真题：Python 调度+Triton/CUDA kernel vs 编译固化 engine）
- KV cache 满了会发生什么？preemption 的 recompute vs swap 两种策略？
- 多 LoRA serving 原理？（LoRA 权重单独加载、按请求切换）
- ⭐ "vLLM 最大的创新点是什么"？（面试官视角原题——答案不是"快"，是 PagedAttention 把 KV 显存利用率从 ~40% 拉到 ~96%）
- ⭐ "vLLM 的调度还有什么可改进的"？（开放题，答得出 chunked prefill/priority/SLO-aware/preemption 策略任意一个方向即高分）
- ⭐ weight-only 量化为什么只量化 weight 不量化 activation？（activation 有 outlier 通道、动态范围难校准；weight 静态可离线校准——蚂蚁/阿里都问过）
- encoder-decoder 和 decoder-only 架构对推理引擎设计的影响？（KV cache 有无 cross-attention、T5 类模型的 serving 差异）

## M4 迷你 serving 引擎（⭐ 招牌项目）

- ⭐ 手写一个最小 serving engine 主循环：admit → 组 batch → forward step → sample → 流式输出 → 释放 KV。讲出你的 scheduler 逻辑。
- ⭐ 设计一个 LLM serving 系统：从请求进来到 SSE 流式返回，画出组件图。（API 层 → 请求队列 → scheduler → engine → KV pool → detokenizer → stream）
- ⭐ TTFT、TPOT、ITL、端到端延迟、吞吐分别受什么影响？怎么压测？（固定 output 长度扫并发，画曲线；为什么吞吐会饱和）
- 请求到来时机不同（Poisson 到达），continuous batching 为什么能拉高吞吐？自己 mini-serving 的数据呢？
- KV pool 满了怎么办？preempt/recompute/swap 三策略 tradeoff？
- P/D 分离（prefill/decode disaggregation）为什么存在？代价是什么？（KV 传输、两池资源配比、TPOT 稳定性）
- speculative decoding 原理？draft model 怎么选？什么 workload 收益大？
- 10 万 QPS 怎么扛？（水平扩展、路由亲和性、SLO 队列、优先级、auto-scaling——系统设计框架）
- 流式输出怎么实现？SSE vs WebSocket？detokenize 为什么要在单独线程/进程做？（增量 decode、多字节字符边界）

## M5 量化与 profiling

- ⭐ PTQ 主流量化路线对比：GPTQ / AWQ / SmoothQuant / FP8 / GGUF。各自量化对象（weight/activation）、校准需求、kernel 要求？
- ⭐ 量化为什么加速？（显存↓ → 带宽需求↓ → decode 提速；低精度 tensor core 算力更高）什么情况下量化不提速？
- ⭐ 你的量化实验数据：同模型 BF16/FP8/INT4 的显存、tok/s、精度怎么变的？（必须背下自己的数据）
- torch profiler / nsys timeline 怎么看 kernel 间隙和 launch overhead？
- ⭐ CUDA Graph 是什么？为什么对 decode 阶段收益大？（大量小 kernel 的 launch overhead）什么情况下不能用？（动态 shape）
- ncu 判断一个 kernel 是 memory-bound 还是 compute-bound 的具体步骤？

## 手写题（国内外都考的固定题型，建议白板练习）

- ⭐ 带 KV cache 的 GQA attention forward（torch 手写，不用 SDPA）
- ⭐ paged KV block allocator：alloc/free/共享前缀，处理碎片化
- ⭐ continuous batching scheduler 主循环（我们 mini-serving 就是答案）
- ⭐ top-p/top-k sampler（numpy/torch，注意数值稳定）
- rope 的手写实现 + 为什么 position id 加在 qk 上
- 计算图手推：attention 每一步的 tensor shape（B×H×S×D 全流程）

## Python/系统工程（贯穿，面试官随手考）

- ⭐ asyncio 事件循环怎么工作？serving 引擎为什么用 asyncio？哪些部分不能用 async？（GPU forward 是同步阻塞的，怎么解耦——executor/独立进程）
- GIL 是什么？对 serving 的影响？vLLM 为什么用多进程（engine core 进程 + API server 进程 + ZMQ 通信）？
- tokenizer encode/decode 的开销？incremental detokenization 为什么必要？
- pybind11 / C++ extension 怎么把自定义 kernel 接进 PyTorch？
- 进程间怎么传 tensor？（shared memory / CUDA IPC）
- 你的压测脚本怎么写的？异步并发怎么控制？百分位延迟怎么算？

## 开放题（靠整体理解，无法靠背）

- 面试官说"线上 TTFT 突然变高，你怎么排查"？（队列深度、prefix 命中率、KV 使用率、长 prompt 占比、chunked prefill 配置……）
- "同一个模型，怎么在吞吐和延迟之间找平衡"？（batch 上限、max_num_seqs、SLO 驱动调度）
- "你的 mini-serving 和 vLLM 差在哪"？（诚实回答：无 prefix cache、无 cuda graph、kernel 未优化、无抢占——正好展示你懂全貌）
