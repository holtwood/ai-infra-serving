# M3 小结：vLLM 生产引擎实测（含 prefix caching 消融 + AWQ 量化对照）

日期：2026-11-05 | vllm 0.11.0 (V1 engine) | RTX 4090 48GB | transformers 4.56.2

## 一、并发扫描（benchmarks/vllm_bench.py）

### Qwen2.5-7B-Instruct (bf16, ~15GB 权重)
| 并发 | TTFT p50 | TPOT p50 | 单流 tok/s | 总吞吐 |
|---|---|---|---|---|
| 1 | 52.5ms | 16.1ms | 60.2 | 60.2 |
| 4 | 125.0ms | 16.4ms | 56.8 | 227.2 |
| 8 | 166.5ms | 16.5ms | 55.5 | 429.5 |
| 16 | 251.0ms | 16.9ms | 52.3 | 832.4 |
| 32 | 420.7ms | 18.8ms | 44.6 | 1415.6 |
| 64 | 624.8ms | 20.5ms | 38.5 | **2399.5** |

### Qwen2.5-0.5B (bf16, ~1GB)
| 并发 | TTFT p50 | TPOT p50 | 总吞吐 |
|---|---|---|---|
| 1 | 34.7ms | 2.6ms | 316.2 |
| 32 | 420.4ms | 4.4ms | 2762.3 |

### Qwen2.5-32B-AWQ (INT4, awq_marlin, ~19GB)
| 并发 | TTFT p50 | TPOT p50 | 总吞吐 |
|---|---|---|---|
| 1 | 59.6ms | 21.5ms | 42.8 |
| 8 | 135.5ms | 21.8ms | 318.3 |
| 32 | 448.4ms | 26.0ms | 895.0 |

## 二、核心发现（每条都是面试弹药）

**1. TPOT ≈ 权重大小/显存带宽 —— memory-bound 的教科书验证**
- 7B bf16：15GB ÷ 939GB/s ≈ 16ms → 实测 **16.1ms** ✓
- 0.5B：~1GB ÷ 939GB/s ≈ 1ms → 实测 2.6ms（+Python/scheduling 固定开销）
- 32B-AWQ：~19GB ÷ 939GB/s ≈ 20ms → 实测 21.5ms（+AWQ 反量化开销）

**2. continuous batching 的摊薄效应**
- 7B：64 并发时 TPOT 仅从 16.1→20.5ms（+27%），总吞吐却 40x
- 物理含义：64 个请求共享同一次权重读取；decode 步 GPU 主要等
  权重从显存搬进来，batch 里多 63 个请求几乎"免费"
- **面试话术**："decode 是 memory-bound，所以 batch 越大单次搬运的
  边际成本越低——直到 KV cache 读带宽或算力变成新瓶颈"

**3. TTFT vs TPOT 的分离行为**
- TTFT 随并发近线性增长（52→625ms）：prefill 是 compute-bound，
  并发请求排队等 GEMM
- TPOT 几乎平坦：这正是 chunked prefill + iteration-level
  scheduling 要保护的指标

**4. Prefix caching 消融（benchmarks/prefix_cache_test.py，2048-token 公共前缀）**
| 配置 | 首请求 TTFT | 后续同前缀 TTFT | 32并发总吞吐 |
|---|---|---|---|
| enable_prefix_caching=True | 234.5ms | **53-60ms (4x↓)** | 1415.6 tok/s |
| 关闭 | 578.1ms | 215-222ms（无复用）| 1287.2 tok/s |
- 服务端 `Prefix cache hit rate: 95.5%`（32并发混合负载）
- 机制：KV block 按 token hash 入池，同前缀直接复用 → 只算增量

**5. AWQ 量化实测（32B-AWQ vs 7B-bf16）**
- vLLM 自动选 `awq_marlin` kernel（启动日志实锤）
- 同显存下能跑 32B：43.5GB 占用 vs bf16 需 ~65GB 放不进
- TPOT 21.5ms vs 7B 16.1ms：字节数和 7B-bf16 接近，但每层多了
  dequant；32B 层数多 → kernel 数多 → 固定开销略大
- **量化收益本质**：decode 阶段权重搬运量 -75%（INT4 vs FP16），
  代价是 dequant 计算——memory-bound 阶段是净赚

## 三、vLLM V1 默认配置观察（启动日志）

`enable_prefix_caching=True, chunked_prefill_enabled=True,
use_cudagraph=true, cudagraph_mode=[FULL_AND_PIECEWISE],
max_capture_size=512, quantization=auto→awq_marlin`
→ V1 默认把面试点全开：APC + chunked prefill + piecewise cudagraph。

## 四、踩坑记录

- transformers 5.x 移除了 `all_special_tokens_extended` → vllm 0.11
  tokenizer 初始化崩 → 锁定 `transformers==4.56.2`
- vllm 0.33 会拉 torch 2.14+cu13（驱动 550 不支持）→ 锁
  `torch==2.8.0 + vllm==0.11.0`
- `gpu_memory_utilization=0.85` 是 KV pool 预占：0.5B 小模型也会
  吃 42GB——不是"用掉"，是"预留"（面试可展开讲 KV block 池化）

## 五、待补

- SGLang：apache-tvm-ffi 镜像缺包，加官方源重试中；失败则按预案
  做 RadixAttention 源码精读笔记
- P/D 分离：单卡限制下放到 M4 用双实例/模拟路由演示
