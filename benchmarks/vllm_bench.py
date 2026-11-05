"""vLLM serving 压测：TTFT / TPOT / 端到端吞吐 vs 并发 —— M3 核心数据。

走 OpenAI 兼容接口 /v1/chat/completions（stream=True），逐 chunk 计时：
- TTFT = 首 chunk 到达时间
- TPOT = 相邻 chunk 间隔中位数（近似 per-token 延迟）
- 吞吐 = 总输出 token / 墙钟时间

用法：
    python vllm_bench.py --url http://127.0.0.1:8000 --model qwen0.5b \
        --concurrency 1 4 8 16 --prompt-tokens 128 --output-tokens 64
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp


def make_prompt(n_tokens: int) -> str:
    """凑一个近似 n_tokens 的中文 prompt（中文 ~1字/token）。"""
    base = "请用简洁的语言回答以下问题：什么是 KV cache？它的作用是什么？"
    pad = "背景知识：自注意力机制中每个 token 会产生 key 和 value 向量。" * (
        n_tokens // len("背景知识：自注意力机制中每个 token 会产生 key 和 value 向量。"))
    return base + pad


async def one_request(session, url, model, prompt, max_tokens, rid):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    itls = []
    last_t = None
    n_out = 0
    async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            now = time.perf_counter()
            data = json.loads(line[5:].strip())
            delta = data["choices"][0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - last_t)
                last_t = now
                n_out += 1
    e2e = time.perf_counter() - t0
    return {"rid": rid, "ttft": ttft or 0, "e2e": e2e,
            "tpot": statistics.median(itls) if itls else 0, "n_out": n_out}


async def run_batch(url, model, prompt, max_tokens, concurrency):
    conn = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=conn,
                                     timeout=aiohttp.ClientTimeout(total=600)) as s:
        # warmup 1 发，预热 CUDA graph / kernel cache
        await one_request(s, url, model, prompt, 8, -1)
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[
            one_request(s, url, model, prompt, max_tokens, i)
            for i in range(concurrency)])
        wall = time.perf_counter() - t0
    return rs, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--output-tokens", type=int, default=64)
    args = ap.parse_args()

    prompt = make_prompt(args.prompt_tokens)
    print(f"{'conc':>5} | {'TTFT p50':>9} | {'TPOT p50':>9} | {'tok/s/req':>9} | "
          f"{'总吞吐':>8} | wall")
    print("-" * 70)
    for c in args.concurrency:
        rs, wall = asyncio.run(run_batch(args.url, args.model, prompt,
                                         args.output_tokens, c))
        ttfts = [r["ttft"] for r in rs]
        tpots = [r["tpot"] for r in rs if r["tpot"] > 0]
        total_out = sum(r["n_out"] for r in rs)
        tps_total = total_out / wall
        tps_req = statistics.median([r["n_out"] / r["e2e"] for r in rs])
        print(f"{c:>5} | {statistics.median(ttfts)*1e3:>8.1f}ms | "
              f"{statistics.median(tpots)*1e3 if tpots else 0:>8.1f}ms | "
              f"{tps_req:>8.1f} | {tps_total:>7.1f} | {wall:.1f}s")


if __name__ == "__main__":
    main()
