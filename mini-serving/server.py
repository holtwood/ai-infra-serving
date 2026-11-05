"""M4: FastAPI + asyncio 封装 —— OpenAI /v1/completions 子集 + /metrics

架构（面试画图题直接画这个）：

    client ──POST──▶ FastAPI handler ──▶ engine.waiting ──▶ Scheduler.step()
                          ▲                                    │
                       SSE ◀── per-request asyncio.Queue ◀── token

关键取舍：model() 是同步阻塞的 CPU→GPU 调用，没法 await。
  * 本版：engine_loop 跑在同一事件循环的 task 里，step() 阻塞期间
    整个 loop 停转（含 HTTP I/O）——教学可接受，压出的 TTFT 会偏高。
  * 生产（vLLM V1）：engine core 独立进程，API server 另一进程，
    ZMQ/IPC 搬 token——既解阻塞又绕开 GIL。"serving 为什么多进程"
    的标准答案就在这里。运行：python server.py（起在 :8000）
"""
import asyncio, json, os, uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from engine import EngineCore, Request, ReqStatus

MODEL_PATH = os.environ.get("MINI_MODEL", "/data/ai-learn/models/Qwen2.5-0.5B-Instruct")
DEVICE = os.environ.get("MINI_DEVICE", "cuda")
KV_BLOCKS = int(os.environ.get("MINI_KV_BLOCKS", "256"))  # 256×16=4096 token 容量
ENGINE_KIND = os.environ.get("MINI_ENGINE", "perseq")     # perseq | batched | graph

engine: EngineCore          # lifespan 里初始化
tokenizer = None
_wakeup: asyncio.Event      # 请求到达时叫醒 engine loop，不空转轮询
streams: dict[str, asyncio.Queue] = {}   # req_id → 该请求的 token 通道


async def engine_loop() -> None:
    """引擎主循环：有活就同步 step()，每步后 sleep(0) 让出事件循环。

    注意 step() 是阻塞调用：GPU forward 期间 loop 无法处理 HTTP；
    且 step 抛异常会直接杀掉本 task（fail-fast，生产里 engine core
    进程有 supervisor 重启）。思考题：to_thread / 独立进程差别？
    """
    while True:
        if not engine.has_work():
            await _wakeup.wait()   # 睡到下一个请求到达
            _wakeup.clear()
            continue
        for req in engine.step():  # step 返回本步有新 token 的请求
            q = streams.get(req.req_id)
            if q is not None:
                q.put_nowait(req)  # 推对象引用；消费方读 generated[-1]
        await asyncio.sleep(0)     # 让出 loop 处理 I/O（关键！）


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, tokenizer, _wakeup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if ENGINE_KIND == "graph":
        from engine_graph import EngineCoreGraph
        engine = EngineCoreGraph(MODEL_PATH, DEVICE, num_kv_blocks=KV_BLOCKS,
                                 max_len=int(os.environ.get("MINI_MAX_LEN", "512")))
    elif ENGINE_KIND == "batched":
        from engine_batched import EngineCoreBatched
        engine = EngineCoreBatched(MODEL_PATH, DEVICE, num_kv_blocks=KV_BLOCKS)
    else:
        engine = EngineCore(MODEL_PATH, DEVICE, num_kv_blocks=KV_BLOCKS)
    _wakeup = asyncio.Event()
    task = asyncio.create_task(engine_loop())  # engine 与 API 同 loop
    yield
    task.cancel()


app = FastAPI(title="mini-serving", lifespan=lifespan)


class CompletionReq(BaseModel):
    prompt: str
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False


async def iter_deltas(req: Request, q: asyncio.Queue):
    """token 事件 → 增量文本， yields (delta, finish_reason|None)。

    增量 detokenize：每次 decode 全量 ids 再取差分，而不是 decode
    单个 token——UTF-8 多字节字符可跨 token 边界，单 token decode
    会切出 `�`。vLLM 在独立 detokenizer 进程做同样的事。
    """
    sent = ""
    try:
        while True:
            cur: Request = await q.get()        # 同一 Request 对象，状态在涨
            done = cur.status is ReqStatus.DONE
            delta = ""
            if cur.generated:
                text = tokenizer.decode(cur.generated)
                delta, sent = text[len(sent):], text
            yield delta, (cur.finish_reason if done else None)
            if done:
                return
    finally:
        streams.pop(req.req_id, None)           # 客户端断连也清通道


def sse_chunk(req: Request, delta: str, reason) -> str:
    chunk = {"id": f"cmpl-{req.req_id}", "object": "text_completion",
             "created": int(req.arrival_time),
             "choices": [{"index": 0, "text": delta, "finish_reason": reason}]}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def sse_stream(req: Request, q: asyncio.Queue):
    async for delta, reason in iter_deltas(req, q):
        yield sse_chunk(req, delta, reason)
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(body: CompletionReq):
    ids = tokenizer(body.prompt).input_ids
    if not ids:  # 空 prompt 会让 forward 崩掉整个 engine_loop，先挡掉
        return {"error": "empty prompt"}
    req = Request(req_id=uuid.uuid4().hex[:12], prompt_ids=ids,
                  max_new_tokens=body.max_tokens,
                  temperature=body.temperature, top_p=body.top_p)
    q: asyncio.Queue = asyncio.Queue()
    streams[req.req_id] = q
    engine.add_request(req)   # 进 waiting 队列，等 scheduler 准入
    _wakeup.set()             # 叫醒 engine_loop
    if body.stream:
        return StreamingResponse(sse_stream(req, q),
                                 media_type="text/event-stream")
    # 非流式：攒齐所有 delta 一次性返回
    parts, reason = [], None
    async for delta, r in iter_deltas(req, q):
        parts.append(delta)
        reason = r or reason
    return {"id": f"cmpl-{req.req_id}", "object": "text_completion",
            "choices": [{"index": 0, "text": "".join(parts),
                         "finish_reason": reason}],
            "usage": {"prompt_tokens": len(req.prompt_ids),
                      "completion_tokens": len(req.generated)}}


@app.get("/metrics")
async def metrics():
    return engine.stats()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
