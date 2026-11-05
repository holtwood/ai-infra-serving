"""轻量 TTFT/ITL/吞吐压测：N 并发流式请求，逐 token 计时。"""
import asyncio, aiohttp, time, sys, json, statistics

URL = "http://127.0.0.1:8000/v1/chat/completions"


async def one(session, prompt, model, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    itls = []
    last = t0
    n = 0
    async with session.post(URL, json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
    }) as r:
        async for line in r.content:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - last)
                last = now
                n += 1
    return ttft or 0, itls, n


async def main():
    model, conc, max_tokens, prompt_file = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    prompts = json.load(open(prompt_file)) if prompt_file != "-" else [sys.stdin.read()]
    prompts = (prompts * ((conc // len(prompts)) + 1))[:conc]
    async with aiohttp.ClientSession() as s:
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[one(s, p, model, max_tokens) for p in prompts])
        wall = time.perf_counter() - t0
    ttfts = [r[0] for r in rs]
    itls = [x for r in rs for x in r[1]]
    toks = sum(r[2] for r in rs)
    itl_s = sorted(itls)
    print(f"conc={conc} wall={wall:.1f}s out_tok={toks} throughput={toks/wall:.0f} tok/s")
    print(f"TTFT mean={statistics.mean(ttfts)*1000:.0f}ms max={max(ttfts)*1000:.0f}ms")
    print(f"ITL  p50={itl_s[len(itl_s)//2]*1000:.1f}ms "
          f"p95={itl_s[int(len(itl_s)*0.95)]*1000:.1f}ms "
          f"p99={itl_s[int(len(itl_s)*0.99)]*1000:.1f}ms "
          f"max={itl_s[-1]*1000:.1f}ms")


asyncio.run(main())
