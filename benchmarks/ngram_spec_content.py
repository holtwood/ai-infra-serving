"""ngram spec decode：按输出内容熵测接受率。

三种内容类型各发 4 条流式请求，记录 tok/s；
随后抓 /metrics 的 spec_decode 计数器算平均接受长度。
"""
import asyncio, aiohttp, time, sys, statistics

URL = "http://127.0.0.1:8000/v1/chat/completions"

PROMPTS = {
    "repetitive": "Repeat the exact sentence 'The quick brown fox jumps over the lazy dog.' 50 times, numbered.",
    "code": "Write a Python class implementing a circular buffer with put(), get(), is_full(), is_empty(), plus docstrings and a usage example.",
    "prose": "Write a reflective essay about the meaning of solitude in modern urban life, with original metaphors.",
}


async def one(session, prompt, model, max_tokens):
    t0 = time.perf_counter(); n = 0
    async with session.post(URL, json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
    }) as r:
        async for line in r.content:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                n += 1
    dt = time.perf_counter() - t0
    return n, n / dt


async def main():
    model = sys.argv[1]
    async with aiohttp.ClientSession() as s:
        for kind, p in PROMPTS.items():
            rs = await asyncio.gather(*[one(s, p, model, 400) for _ in range(4)])
            tps = statistics.mean(r[1] for r in rs)
            print(f"{kind:11s} mean_tok/s={tps:.1f}  out={rs[0][0]}")


asyncio.run(main())
