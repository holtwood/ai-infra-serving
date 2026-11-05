# 抢占（Preemption）生产现场实测

日期：2025-11-05 · vLLM 0.11.0 · Qwen2.5-7B · RTX 4090
设置：`--gpu-memory-utilization 0.55`（KV 池仅 195,392 token），
64 并发 × 512in/4000out（潜在需求 ~289k token > 池子）

## 日志时间线（逐 10s）

| 时刻 | Running | Waiting | KV 用量 | gen tok/s |
|---|---|---|---|---|
| 10:34:31 | 64 | 0 | 31.6% | 2815 |
| 10:35:11 | 64 | 0 | 82.1% | 2270 |
| 10:35:21 | 64 | 0 | **93.2%** | 2168 |
| 10:35:31 | 61 | 3 | **99.0%** | 2058 |
| 10:35:51 | 51 | 13 | 99.7% | 1720 |
| 10:36:01 | 47 | 21 | 99.7% | 1579 |
| 10:36:11 | 43 | 21 | 98.3% | 1453 |
| 10:36:21 | 21 | 0 | 39.8% | **301** |

bench 总账：TTFT mean 1220ms（排队）、TPOT p99 36.9ms、
**vllm:num_preemptions_total = 22**。

## 机制（源码对照）

`scheduler.py` 每步先给 running 分块；分配失败时从 running **尾部**
pop 逐出，`waiting.prepend_request` **塞回队首**（行 264-281）。
V1 默认 **recompute**：丢 KV、回头重算 prefill——所以抢占直接烧
prefill 算力，吞吐崩塌比并发数下降更快（2815→301，远超 3x）。

## 面试标准答案升级版（带实测）

"线上看四个信号：KV usage 顶到 ~100%、running 数下降且 waiting
反升、gen 吞吐坍缩、`num_preemptions_total` 增长；median TPOT
可能还正常，恶化先在尾部。修复：加大 KV 池（gpu_memory_utilization）、
降 max_num_seqs、缩短输出上限，或加副本分流。"

## 意外发现（标准答案里没有的）

抢占期间 Prefix cache hit rate 从 1.5% 飙到 ~49%——
**被逐请求 recompute 时命中了自己被逐前留下的前缀 KV**，
APC 部分赎回了 recompute 代价。V1 里 prefix caching + recompute
抢占是一对隐性协同。
