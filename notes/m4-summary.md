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
