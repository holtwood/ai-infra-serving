"""M4: 异步压测客户端 —— 并发/泊松到达，量 TTFT·TPOT·吞吐，画散点图。

压测方法论（面试常考"怎么压测一个 serving 引擎"）：
  * 固定 output 长度（--max-tokens）扫并发：排除生成长度噪声，
    观察吞吐随 batch 增大而升、撞墙后饱和的曲线。
  * --poisson 用指数分布间隔模拟真实到达：更能暴露 continuous
    batching 的价值——到达是随机的，static batching 会大量空等。
  * 指标：TTFT（prefill+排队延迟）、TPOT（decode 每 token 间隔）、
    端到端延迟、总吞吐 tok/s。

用法（先起 server.py）：
    python bench_client.py --concurrency 16 --requests 64 --max-tokens 64
    python bench_client.py --poisson --rate 4 --requests 100
"""
import argparse
import asyncio
import json
import os
import random
import time

import aiohttp
import matplotlib
matplotlib.use("Agg")          # 无显示环境（远端服务器）也能存图
import matplotlib.pyplot as plt

PROMPTS = [
    "用三句话解释什么是 KV cache：",
    "写一段 Python 快速排序并说明时间复杂度：",
    "什么是 continuous batching？解决了什么问题？",
    "编一个关于机器人学做饭的小故事：",
    "解释 PagedAttention 和普通 KV cache 的区别：",
]


def pct(xs: list[float], p: float) -> float:
    """线性插值百分位（手写版——"P95 怎么算"是面试常问）。"""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    return xs[lo] if lo + 1 >= len(xs) else xs[lo] + (xs[lo + 1] - xs[lo]) * (k - lo)


async def one_req(session: aiohttp.ClientSession, url: str, args,
                  sem: asyncio.Semaphore, out: list) -> None:
    """发一条流式请求，记录 TTFT / 每 token 间隔 / 端到端延迟。"""
    async with sem:   # 信号量控并发：模拟固定 in-flight 请求数
        payload = {"prompt": random.choice(PROMPTS),
                   "max_tokens": args.max_tokens,
                   "temperature": args.temperature, "stream": True}
        ttft = t_last = None
        itl, n_tok = [], 0
        t0 = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                buf = b""
                async for chunk in resp.content.iter_any():
                    buf += chunk
                    while b"\n" in buf:               # SSE：按行切 data: 帧
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        now = time.perf_counter()
                        if line == b"data: [DONE]":
                            break
                        if ttft is None:
                            ttft = now - t0           # 首帧到达 = TTFT
                        else:
                            itl.append(now - t_last)  # 相邻帧间隔 ≈ TPOT 样本
                        t_last, n_tok = now, n_tok + 1
        except Exception:
            out.append({"ok": False})
            return
        e2e = time.perf_counter() - t0
        out.append({"ok": True, "ttft": ttft or e2e, "e2e": e2e, "tok": n_tok,
                    "tpot": sum(itl) / len(itl) if itl else 0.0})


async def run(args) -> None:
    sem = asyncio.Semaphore(args.concurrency)
    out: list = []
    url = f"{args.url}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=900)
    conn = aiohttp.TCPConnector(limit=0)  # 不限制连接池，并发由 sem 管
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        t_start = time.perf_counter()
        tasks = []
        for _ in range(args.requests):
            tasks.append(asyncio.create_task(one_req(s, url, args, sem, out)))
            if args.poisson:
                # 泊松到达：间隔 ~ Exp(λ)，λ=rate req/s；不服从则齐射
                await asyncio.sleep(random.expovariate(args.rate))
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start

    ok = [r for r in out if r["ok"]]
    ttfts = [r["ttft"] for r in ok]
    tpots = [r["tpot"] for r in ok if r["tpot"] > 0]
    e2es = [r["e2e"] for r in ok]
    toks = sum(r["tok"] for r in ok)
    print(f"\n== mini-serving bench ==")
    print(f"成功 {len(ok)}/{args.requests} | wall {wall:.1f}s | "
          f"总吞吐 {toks / wall:.1f} tok/s")
    print(f"TTFT  P50 {pct(ttfts, 50) * 1e3:6.0f}ms  P95 {pct(ttfts, 95) * 1e3:6.0f}ms")
    print(f"TPOT  P50 {pct(tpots, 50) * 1e3:6.0f}ms  P95 {pct(tpots, 95) * 1e3:6.0f}ms")
    print(f"E2E   P50 {pct(e2es, 50):6.2f}s   P95 {pct(e2es, 95):6.2f}s")

    # 吞吐-延迟散点：每请求一点，x=端到端延迟，y=该请求 decode 吞吐
    os.makedirs("benchmarks", exist_ok=True)
    path = f"benchmarks/tput_lat_{int(time.time())}.png"
    xs = [r["e2e"] for r in ok if r["e2e"] > 0]
    ys = [r["tok"] / r["e2e"] for r in ok if r["e2e"] > 0]
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, alpha=0.6, edgecolors="none")
    plt.xlabel("e2e latency (s)")
    plt.ylabel("per-request throughput (tok/s)")
    plt.title(f"mini-serving  c={args.concurrency} M={args.requests} "
              f"poisson={args.poisson}")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"散点图已存 {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="mini-serving 压测客户端")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=8, help="在途请求上限")
    ap.add_argument("--requests", type=int, default=32, help="总请求数 M")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--poisson", action="store_true", help="启用泊松到达")
    ap.add_argument("--rate", type=float, default=4.0, help="泊松到达率 req/s")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
