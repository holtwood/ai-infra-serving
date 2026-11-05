# mini-serving：教学版迷你 LLM serving 引擎（M4 招牌项目）

对应面试原题：**"写一个最小 serving engine 的主循环"**。
目标是能用 ~400 行代码讲清 continuous batching / 调度 / KV cache 管理三件事。

## 文件

| 文件 | 内容 |
|---|---|
| `engine.py` | 引擎内核（纯逻辑）：`Request` / `KVBlockPool` / `Scheduler` / `EngineCore.step()` |
| `server.py` | FastAPI + asyncio：`POST /v1/completions`（SSE 流式）+ `GET /metrics` |
| `bench_client.py` | aiohttp 异步压测：TTFT/TPOT P50/P95、吞吐、散点图 |
| `benchmarks/` | 压测输出图 |

## 跑法（远端 /data/ai-learn）

```bash
python server.py            # 起 :8000，模型 Qwen2.5-0.5B-Instruct

# 流式请求
curl -N -X POST localhost:8000/v1/completions \
     -d '{"prompt":"用三句话解释 KV cache：","max_tokens":64,"stream":true}'

# 压测：固定并发 16 打 64 个请求
python bench_client.py --concurrency 16 --requests 64 --max-tokens 64
# 泊松到达（λ=4 req/s）——看 continuous batching 的真实收益
python bench_client.py --poisson --rate 4 --requests 100 --concurrency 32

# 观察调度行为
watch -n0.5 curl -s localhost:8000/metrics
```

## 架构

```
            ┌─────────────────────── server.py (asyncio) ───────────────────────┐
 client ──▶ │ POST /v1/completions ──▶ Request ──▶ engine.waiting (deque)        │
            │        ▲                                                           │
            │   SSE 流式返回 ◀── streams[req_id] : asyncio.Queue ◀──┐            │
            └────────┼─────────────────────────────────────────────┼────────────┘
                     │ engine_loop task: while True: step();       │
                     │   await asyncio.sleep(0)                    │ put(req)
            ┌────────▼─────────────── engine.py ───────────────────▼───────────┐
            │  step() 每迭代:                                                   │
            │   1. Scheduler.schedule()  FCFS + token预算 + KV准入 → admitted  │
            │   2. admitted: prefill ｜ running: decode 1 token （逐seq forward)│
            │   3. sample → append generated                                   │
            │   4. done → _release: 归还KV块 + 丢past_key_values               │
            │   5. KV池满 → preempt: 踢running队尾回waiting (recompute)        │
            │                                                                 │
            │  KVBlockPool: [blk0][blk1][blk2]...  free◀──alloc──▶ block_ids    │
            └─────────────────────────────────────────────────────────────────┘
```

## 对照 vLLM：我们简化了什么（面试必答）

| 维度 | vLLM V1 | 本版 |
|---|---|---|
| batch forward | N seq 拼 flattened batch，**一次** forward，权重搬一遍 | **逐 seq** forward，权重搬 N 遍（逻辑等价性能不等价） |
| KV 存储 | 物理分页显存 + block table，PagedAttention 按块寻址 | `KVBlockPool` 只记账；真 KV 在每请求的 HF `DynamicCache` |
| 调度 | FCFS + token budget + chunked prefill 切片 | FCFS + token budget，prompt 整条进/整条等 |
| 抢占 | recompute / swap-to-CPU，优先级+水位线防 thrash | 只有 recompute：踢 running 队尾回 waiting 队首 |
| 进程模型 | engine core 独立进程，ZMQ 与 API server 通信 | 单进程 asyncio task，`step()` 阻塞期间 loop 停转 |
| 其他 | prefix caching、CUDA graph、spec decode、P/D 分离 | 全无——这就是"还差什么"的标准答案 |

## 面试讲解要点（60 秒话术）

1. **主循环**：`step()` = 调度 → 混合批前向 → 采样 → 完成释放。
   iteration-level scheduling：每个 decode 步都重排 batch，
   完成即退、新请求即入——这就是 continuous batching。
2. **token 预算**：`max_batched_tokens` 限制单步前向总 token 数，
   running 每条占 1，prefill 按 prompt 长计 → 控制 TTFT/TPOT 平衡。
3. **KV 准入**：admit 前检查 `KVBlockPool.can_alloc`——显存是硬约束，
   排队优于 OOM；decode 中途池满 → 抢占队尾 recompute。
4. **指标**：TTFT 反映排队+prefill，TPOT 反映 decode 单步耗时；
   batch 越大吞吐越高（权重摊薄）直到撞 compute 墙/KV 墙。
5. **asyncio**：GPU forward 同步阻塞没法 await → 生产里 engine core
   独立进程 + ZMQ；教学版同 loop task + `sleep(0)` 让出。
6. **自己跑出的数据**（跑完填）：batch __ 时吞吐 __ tok/s，
   TPOT __ ms，KV 池 __% 时开始抢占——背下数字。

## 可追问的坑（提前想好）

- 逐 seq forward 为什么吞吐差？→ decode 是 memory-bound，权重每 seq 搬一遍；真 batch 只搬一遍。
- 抢占为什么会 thrash？→ 踢出去的 seq 重进又撞墙；要优先级/水位线/最小推进保证。
- prompt 超长怎么办？→ 本版判死；vLLM 是 chunked prefill 切片调度。
- 流式断连？→ generator `finally` 清通道，但请求还在 engine 里跑完（生产会 abort）。
