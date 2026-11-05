# FP8 KV Cache + 长上下文膨胀实验（M5 补）

日期：2025-11-05 · vLLM 0.11.0 · Qwen2.5-7B · RTX 4090（sm_89，无 FP8 tensor core）
脚本：`benchmarks/kv_context_sweep.py`

## KV 容量（vLLM 启动日志实测）

| dtype | KV 池容量 | 32k 上下文并发上限 |
|---|---|---|
| bf16 | 506,640 tokens | 15.46x |
| **fp8** | **1,013,296 tokens（精确 2.0x）** | **30.92x** |

单 token KV = 2(KV)×28层×4kv_head×128×2B = 56KiB（bf16），fp8 减半 28KiB。
27.06GiB 池 / 56KiB = 506k ✓ 公式与实测吻合。

## TTFT/TPOT 随上下文长度（同方法配对实测）

| prompt 长度 | bf16 TTFT | fp8 TTFT | bf16 TPOT | fp8 TPOT |
|---|---|---|---|---|
| ~2000 | 231ms | (1026ms 含首请求 warmup，剔除) | 7.8ms | 7.6ms |
| ~8000 | 418ms | 460ms（+10%） | 7.9ms | 7.7ms |
| ~16000 | 640ms | 759ms（+19%） | 8.0ms | 7.8ms |
| ~30000 | 1299ms | 1641ms（+26%） | 8.2ms | 8.1ms |

## 解读

- **TTFT 随长度近似线性增长**（prefill compute-bound：30k≈1.3s，和算力账对得上）。
- **fp8 KV 在 4090 上是「容量翻倍 + TTFT 变差 10-26%」的 trade-off**：
  sm_89 没有 FP8 tensor core，attention 走 dequant 路径，prefill 越长越亏。
  TPOT 基本不变（decode 被 15GB 权重读取主导，KV dtype 影响被淹没）。
- **TPOT 几乎不随上下文退化**（7.8→8.2ms）：30k KV=1.6GB，读一遍 +1.7ms
  @939GB/s，相对 ~8ms 的权重读取是小头——量化解释了「长上下文主要贵
  在 prefill 和容量，不在 decode 带宽」。
- 面试口径：fp8 KV 适合**容量受限的长上下文 decode 集群**（H100/B200 有
  FP8 算力时连 TTFT 代价都没有）；4090 上仅当显存装不下时才值得。

## 坑

- 每次新开实例的首个请求被 warmup/CUDA graph 编译污染（fp8 的 2000len
  出现 1026ms 假值）——扫测首点要预跑或剔除。
