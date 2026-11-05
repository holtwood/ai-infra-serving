# 实验设计 backlog（机器空闲时可按序执行）

环境：远端 `/data/ai-learn`，venv `.venv/bin/python`，仓库本地
`/home/shane/github/ai-learn`。每项含：目的 / 前置 / 步骤 / 预期产出 /
面试用法。**注意：执行前先 `nvidia-smi` 确认无他人在用 GPU。**

---

## E1. ngram 自投机解码 × 内容熵（半成品，脚本已就位）

**目的**：投机解码的接受率依赖输出熵——用 vLLM 原生 ngram 方法
（无 draft 模型）实测三档内容的接受率差异。

**步骤**：
```bash
.venv/bin/vllm serve /data/ai-learn/models/Qwen2.5-7B-Instruct \
  --served-model-name qwen7b --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":1}'
# 就绪后：
.venv/bin/python benchmarks/ngram_spec_content.py qwen7b
curl -s http://127.0.0.1:8000/metrics | grep spec_decode
# 对照组：去掉 --speculative-config 重跑同脚本
```
**产出**：repetitive/code/prose 三档 tok/s + spec_decode 接受计数。
**面试用法**：「draft 命中率∝输出可预测性；复读类 ~3-4 tok/step，
散文 ~1.x——所以 prod 里 spec-decode 收益看业务分布」。

## E2. AWQ-32B vs 7B-bf16 量化三角（已完成一半）

**已有**（2025-11-05 实测）：AWQ-32B 上 4090-48G：KV 池 85,120 tok；
conc8 185tok/s ITL p50 41.8ms；conc32 631tok/s ITL p50 49.8ms；
TTFT ~632ms；bf16-32B（~65GB）根本装不进 48G——**AWQ 是使能技术
不是加速技术**，TPOT~45ms ≈ 19.4GB权重/939GB/s + dequant。
**待补**：质量对照——同 5 道题让 7B-bf16 vs 32B-AWQ 作答，人工粗评
或跑 perplexity（wikitext 切片）量化质量差。

## E3. KV swap-to-CPU（抢占替代方案）

**目的**：抢占笔记只测了 recompute 代价；造 swap 路径测 trade-off。
**步骤**：`mini-serving/engine_batched.py` 加 `swap_out(req)`——KV
`cache.to('cpu', non_blocking)` 到 pinned buffer，`req.status=
SWAPPED`；`resume` 时拷回。挤爆条件同 preemption 实验
（小 max_len + 64 并发长输出）。测：swap 往返耗时 vs recompute
重跑 prefill 耗时（KV 池 99% 时后者 ~TTFT 量级）。
**产出**：swap_us vs recompute_ms 对照 + 何时 swap 亏（PCIe 带宽
~25GB/s，56KiB/tok × 大 ctx 时）。
**面试用法**：「vLLM V1 只实现了 recompute，SGLang 有 radix→CPU；
我实测 swap 在 ctx>8k 时优于重算」。

## E4. torch.compile vs 手写 graph

**目的**：「框架白给的图」vs「我手写的图」差多少。
**步骤**：`engine_batched.py` 的 decode forward 包
`torch.compile(mode="reduce-overhead")`（内部即 CUDA graph），
同 max_len 256/512 跑 profile_decode_step.py 对照手写 graph
的 5.46ms。
**产出**：compile 版 step_ms vs 手写 5.46ms vs 无图 13.75ms。
**面试用法**：「reduce-overhead 拿掉了大部分 launch gap 但仍有
guard 检查 + graph 按 shape 分桶重录；手写图对 KV 指针控制更细」。

## E5. paged kernel v2：split-K + engine 接入（旗舰延伸）

**目的**：05_paged_attention.py 单 program 串行扫块，长 ctx 吃亏；
vLLM v2 用 split-K（多 program 分段 + combine kernel）。
**步骤**：kernel 加 `SPLIT` 维度 program_id(2)，每段算部分
(m,l,acc)，二段 kernel 归并；bench ctx=4k/16k 对照 v1。
进阶：把它接进 engine_graph.py——自写 attention 替换 HF SDPA
需要自管 cache（正是 vLLM 自研 attention 后端的原因，可做成
`engine_paged.py` 最小版：只替换 decode 步 attention）。
**产出**：ctx 长度扫描曲线 v1 vs v2；若接入则 graph+paged 联合
实测——**「我造了 vLLM 的三件套：paged KV + continuous batch +
CUDA graph」** 的完整说法。

## E6. mini-serving prefix cache（radix block 复用）

**目的**：路由层 APC 已有数据；引擎内实现是自己的证据。
**步骤**：`allocator.py` 块按 (parent_hash, token_ids) 哈希存
radix 树；prefill 时查公共前缀跳过计算（KV 直接引用）。
测同前缀 8 条请求 TTFT2/TTFT1（对标 vLLM APC 的 4x）。
**产出**：自实现 APC 命中收益 + 哈希冲突/驱逐策略笔记。

## E7. router v2：负载感知回退

**目的**：现 prefix_router 只 sticky-hash；生产要「亲和但别压垮」。
**步骤**：`benchmarks/prefix_router.py` 查后端
`/metrics`（或轮询 running 数），目标实例 running > 阈值则回退
最空实例；双实例 + 前缀重复 + 突发流量验证。
**产出**：降级曲线 + 「cache 亲和 vs 队列均衡」权衡笔记。

---

执行顺序建议：E4/E2-补质量（最便宜）→ E1 → E3 → E6 → E5。
每项完成后写对应 notes/*.md，commit 进 RESULTS.md 总表。
