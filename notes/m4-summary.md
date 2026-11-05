# M4 小结：mini-serving 实测 —— 为什么"逐 seq forward"撑不住并发

日期：2026-11-05 | mini-serving/（535 行：engine.py + server.py + bench_client.py）
模型：Qwen2.5-0.5B | 对照：同模型 vLLM 0.11.0

## 一、mini-serving 实现了什么

- `KVBlockPool`：分页 KV 记账（block=16tok）、按需分配、**池满触发
  recompute 抢占**（evict → 重算 KV，vLLM 同款策略）
- `Scheduler`：FCFS + token 预算 + KV 准入控制
- `EngineCore.step()`：waiting→running→done 全生命周期
- `server.py`：uvicorn + SSE 流式 + OpenAI 兼容形态
- `bench_client.py`：并发/泊松到达压测 + P50/P95 + 散点图

## 二、压测实测（64 tokens/req）

| 并发 | TTFT P50 | TPOT P50 | 总吞吐 | vs vLLM(同模型) |
|---|---|---|---|---|
| 1 | 16ms | 13ms | 68 tok/s | 316 |
| 4 | 96ms | 51ms | 77 | 882 |
| 8 | 163ms | 100ms | 79 | 1606 |
| 16 | 258ms | **202ms** | **79（饱和）** | 2597 |

## 三、35x 差距的根因（面试核心）

**TPOT 与并发线性正相关**（13→51→100→202ms）：per-seq forward 意味着
batch 中每加一条请求，每个 decode 步都要多搬一遍全量权重。
总吞吐在 ~79 tok/s 封顶——带宽全被重复权重读取吃掉了。

vLLM 把 N 条 seq flatten 成一次 forward + PagedAttention 按
block table 寻址 → 权重只搬一遍 → 同硬件同模型 32 并发 2762 tok/s。

**面试话术**："我写过 mini-serving，实测逐 seq forward 的 TPOT
随并发线性退化、总吞吐 ~80 tok/s 封顶；这就是 PagedAttention
+ 单 batch forward 存在的意义——不是省显存，是省权重搬运。"

## 四、附带收获：CUDA Graph 消融（vLLM, 0.5B）

| 模式 | TPOT c=1 | TPOT c=32 | 总吞吐@32 |
|---|---|---|---|
| 默认（cudagraph FULL_AND_PIECEWISE） | 2.6ms | 4.4ms | 2762 |
| --enforce-eager | 7.2ms | 8.8ms | 2149 |

小模型 decode 每步 24 层×若干 kernel，launch 开销 ~5ms 与 GPU
实际算力同量级 → graph 化 TPOT -64%。**模型越小 cudagraph 收益越大**
（大模型权重搬运时间把 launch 开销掩盖了）。

## 五、mini-serving vs vLLM 的诚实差距（面试要主动讲）

已做：准入控制、抢占-重算、流式 SSE、P95 统计、泊松到达模拟
未做：batched/paged attention、prefix caching、KV 物理分页显存、
chunked prefill、CUDA graph、多进程 engine core 隔离
→ 下一步把 batched forward 补上，预计 TPOT 曲线能变平

## 六、续：batched forward 实测（engine_batched.py，2026-11-05）

把 decode 步改成"全体 running 行一次 forward"（共享 DynamicCache，
K/V 左 pad 到统一 L + attention_mask + 按行 position_ids），
其余调度/抢占/准入完全没动——**单一变量实验**：

| 并发 | per-seq TPOT | batched TPOT | per-seq 吞吐 | batched 吞吐 | 提升 |
|---|---|---|---|---|---|
| 1 | 13ms | 13ms | 68 | 64 | ~1x |
| 4 | 51ms | **19ms** | 77 | 196 | 2.6x |
| 8 | 100ms | **19ms** | 79 | 368 | 4.7x |
| 16 | 202ms | **20ms** | 79 | 657 | 8.3x |
| 32 | — | 26ms | — | **973.6** | — |

**结论**：TPOT 曲线被拉平 = decode 瓶颈确实是"权重重复搬运"，
batch 化后权重一次读取服务全 batch——vLLM 快的核心机制被
亲手复现。剩余 vs vLLM 差距（973 vs 2762）来自：pad KV 浪费、
无 CUDA graph（13ms 里 Python/launch 占大头）、逐 seq prefill、
HF 模型非最优 kernel——正好对应下一次迭代的四个优化点。

**正确性验证**（test_batched_equiv.py）：4 条 prompt 两引擎
greedy 对比——3 条逐 token IDENTICAL，1 条在第 17 token 分叉
（bf16 数值差异，pad 位参与计算的浮点序不同）——顺带实测了
"batch-invariance 为什么难"这个热门话题。

## 七、续2：手写 CUDA Graph（engine_graph.py，2025-11-05）

decode 步录图回放：max_len=256 时 13.75→**5.46ms/step（2.5x）**，
消灭 profiler 实测的 69% Python/launch 空转。
但定长图强制 attention 全扫 max_len → max_len=2048 时反变慢 1.3x，
**亲手撞出了 PagedAttention 变长寻址的存在理由**。
坑见 notes/cuda-graph-engine.md（StaticCache batch 锁定、
共享 cache_position 逼右对齐、两个 off-by-one）。

## T1 补充：分叉根因判定（2026-11-05 通宵复测）

重跑 test_batched_equiv.py：仍 3/4 identical、r3 diverge@17。
核查 engine_batched.masks_positions：pos = L-p 是新 token 的
正确序列 index（新 token 为第 ctx 位，0-indexed=ctx），
与 engine_graph 的 `li-1` 约定不同（后者 li 含当前 token）——
batched 引擎无 off-by-one。**判定：r3 分叉确认为 bf16
batch-invariance**（左 padding 改变浮点累加序，logit 近距翻转）。
