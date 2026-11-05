"""Prefix-aware routing 实验：round-robin vs 前缀亲和派发。

两个 vLLM 实例（0.5B，:8000 / :8001）。构造 K 个共享前缀的 prompt 族，
每族发 R 条（后缀不同、共 reps 轮）。两种派发策略：

  - rr     : round-robin，同族请求被打散到两个实例 → 二次到达约 50% miss
  - sticky : 按前缀 hash 固定实例 → 同族总命中同实例 → 二次到达必 hit

对比指标：整体 mean/p50 TTFT + 每实例均值。
"""
import argparse, asyncio, time, hashlib, statistics as st
import aiohttp

BASE = "http://127.0.0.1:{}"
MODEL = "qwen05"

TOPICS = ["CUDA kernel", "调度器", "量化", "NCCL", "KV cache",
          "投机解码", "CUDA Graph", "Triton", "prefix cache",
          "tensor parallel", "paged attention", "continuous batching"]
# 12 个不同前缀（>> 2 个后端），保证 rr 下二次到达有明显 miss 概率
PREFIXES = [f"你是专家。以下是关于{t}的技术文档。" + f"{t} 的机制涉及显存管理、调度与 kernel 融合。" * 20
            for t in TOPICS]
SUFFIXES = ["请总结第%d点。" % i for i in range(4)]


def port_for(mode, prompt, seq, n):
    if mode == "rr":
        return 8000 + (seq % n)
    h = int(hashlib.md5(prompt[:64].encode()).hexdigest(), 16)
    return 8000 + (h % n)


async def one(session, mode, prompt, suffix, seq, n, results):
    port = port_for(mode, prompt, seq, n)
    payload = {"model": MODEL, "prompt": prompt + suffix,
               "max_tokens": 8, "temperature": 0, "stream": True}
    t0 = time.perf_counter()
    ttft = None
    async with session.post(BASE.format(port) + "/v1/completions",
                            json=payload) as r:
        async for _ in r.content:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
    results.append((port, ttft))


async def run(mode, n_backends, reps):
    results = []
    async with aiohttp.ClientSession() as s:
        seq = 0
        for rep in range(reps):
            tasks = []
            for p in PREFIXES:
                for suf in SUFFIXES:
                    tasks.append(one(s, mode, p, suf, seq, n_backends, results))
                    seq += 1
            await asyncio.gather(*tasks)   # 每轮内部并发、轮次间串行，保证缓存代际清晰
            sub = results[-len(PREFIXES) * len(SUFFIXES):]
            print(f"  rep{rep}: mean_TTFT={st.mean(t for _, t in sub):.1f}ms "
                  f"p50={st.median(t for _, t in sub):.1f}ms")
    t8k = [t for p, t in results if p == 8000]
    t8k1 = [t for p, t in results if p == 8001]
    allt = [t for _, t in results]
    print(f"mode={mode}: n={len(allt)} mean={st.mean(allt):.1f}ms "
          f"p50={st.median(allt):.1f}ms | :8000 mean={st.mean(t8k):.1f}ms "
          f":8001 mean={st.mean(t8k1):.1f}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rr", "sticky"], required=True)
    ap.add_argument("--backends", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    asyncio.run(run(a.mode, a.backends, a.reps))
