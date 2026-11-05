# vLLM V1 请求全链路走查（0.11.0 源码实证）

面试开场题："一个请求从 HTTP 到第一个 kernel launch 发生了什么"。
以下每一跳都在 `.venv/.../vllm/` 里核实过函数名。

## 进程拓扑（V1 = 两进程）

```
[API server 进程]                        [EngineCore 进程]
FastAPI + detokenizer  <-- ZMQ ipc -->  Scheduler + GPU worker
entrypoints/openai/                     v1/engine/core.py
api_server.py                           v1/worker/gpu_model_runner.py
```

## 逐跳链路

1. **HTTP 入口**：`api_server.py` `@router.post("/v1/completions"|"/v1/responses")`
   → serving_*.py 做 tokenize + 组装 `SamplingParams`。
   *此时套用 generation_config 默认值（rep_pen=1.05 坑就在这进）。*
2. **InputProcessor**（`v1/engine/processor.py`）：校验参数、
   `_validate_supported_sampling_params`（#31982 的校验点就在这）、
   多模态处理 → 打包成 EngineCoreRequest。
3. **ZMQ IPC**：`core_client.py` 跨进程发给 EngineCore——
   为什么 vLLM 要拆进程：Python GIL 下 HTTP/tokenize/反 detokenize
   与 engine 循环物理隔离，GPU 循环不被 API 开销拖累。
4. **EngineCore.add_request**（`core.py:217`）→ `scheduler.add_request`
   （`scheduler.py:1097`）进 `waiting` 队列
   （`create_request_queue(policy)`：fcfs / priority）。
5. **EngineCore.step()**（`core.py:272`）每步三件事：
   ```
   scheduler_output = self.scheduler.schedule()          # 组 batch
   model_output = model_executor.execute_model(sched)    # 真跑 forward
   outputs = scheduler.update_from_output(...)           # 写回/完成判定
   ```
6. **schedule()**（`scheduler.py:179`）核心循环：
   - 先遍历 `running`（行 210）：每请求按 token 预算分配
   - KV 不够 → **抢占**（行 264-281）：从 running 尾部 `pop()` 一个
     `preempted_req`，`waiting.prepend_request` **塞回队首**
     （V1 默认 recompute：丢 KV 重算，不 swap）
   - 再 `while waiting and token_budget`（行 336）准入新请求：
     chunked prefill 在这里切分——prefill 按
     `max_num_batched_tokens` 预算分块跨步
7. **execute_model**（`gpu_model_runner.py:2231`）：
   `InputBatch`（gpu_input_batch.py）把 running 组拼成张量
   （input_ids/positions/block_table/sampling 元数据），
   一次 forward；spec decode 的 `spec_decode_unsupported_reqs`
   集合也在这里维护。
8. **输出回写**：`update_from_output` 判 EOS/长度 → 完成请求释放
   KV block；增量 token 经 ZMQ 回 API 进程 → detokenizer
   （`detokenizer.py`）增量解码 → SSE chunk 推给客户端。

## 面试速答版（30 秒）

"API 进程 tokenize+校验后经 ZMQ 交给独立 EngineCore 进程；
每个 engine step 里 scheduler 先给 running 分 token 预算、KV 不够
就从尾抢占塞回队首，再按 chunked-prefill 预算准入 waiting；
InputBatch 拼张量一次 forward；输出增量走 detokenizer + SSE 回去。
进程拆分是为了隔离 GIL 下 HTTP/编解码开销和 GPU 循环。"

## 和我们 mini-serving 的对应

| vLLM | mini-serving | 我们实测的差距 |
|---|---|---|
| scheduler.schedule | `Scheduler.schedule` | 同构（FCFS+抢占回队首） |
| InputBatch 拼批 | `engine_batched` 左 pad | 我们 pad 浪费、它们 block table 零浪费 |
| recompute 抢占 | `KVBlockPool` 记账+重算 | 行为一致，物理层不同 |
| cudagraph | 无 | profiler 实测我们 69% GPU 空转 |
