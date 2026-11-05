"""chunked prefill / ITL 干扰实验 —— P/D 分离动机的实测证据。

场景：8 条短请求持续 decode（受害者），期间注入 1 条 6000-token
长 prompt（干扰者）。
- chunked prefill ON：长 prefill 被切成 ≤max_num_batched_tokens 的
  片，与 decode 步交错 → 受害者 ITL 仅温和上升
- OFF：长 prefill 独占一个 iteration → 受害者出现一整段 ITL 尖刺
  （head-of-line blocking）——这就是线上 ITL p99 恶化和 P/D 分离
  存在的直接原因

输出：受害者 ITL p50/p99/max、长 prompt 请求的 TTFT。
用法：python chunked_prefill_itl.py --model qwen7b [--long-tokens 6000]
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp

SHORT_PROMPT = "请简述 Transformer 的自注意力机制及其复杂度。"
LONG_PREFIX = ("以下是一段技术文档。KV cache 是大模型推理的关键机制，"
               "它缓存历史 token 的 key 和 value，避免每步重复计算。")


def make_long_prompt(n_tokens: int) -> str:
    # 中文约 1字/token 量级；重复填充到目标长度。
    # 加随机盐：否则 APC 会命中上一次的长 prompt，prefill 免费、实验失效
    import random
    salt = "".join(random.choices("0123456789abcdefghijklmnopqrstuvwxyz", k=16))
    unit = LONG_PREFIX + salt
    return ("请阅读并总结：" + salt + unit * (n_tokens // len(unit)) +
            "请用一句话总结核心观点。")


async def stream_req(url, model, prompt, max_tokens, tag, results):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    t0 = time.perf_counter()
    ttft = None
    itls = []
    last = None
    try:
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)) as s:
            async with s.post(f"{url}/v1/chat/completions",
                              json=payload) as r:
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    now = time.perf_counter()
                    d = json.loads(line[5:].strip())
                    if d["choices"][0].get("delta", {}).get("content"):
                        if ttft is None:
                            ttft = now - t0
                        else:
                            itls.append(now - last)
                        last = now
    except Exception as e:
        results.append({"tag": tag, "error": str(e)})
        return
    results.append({"tag": tag, "ttft": ttft or 0, "itls": itls,
                    "e2e": time.perf_counter() - t0})


async def run(url, model, n_bg, long_tokens, bg_tokens, bg_rounds):
    results = []
    # 干扰者：等背景跑热后注入
    long_prompt = make_long_prompt(long_tokens)

    async def interferer():
        await asyncio.sleep(1.0)               # 让背景先进入 decode
        await stream_req(url, model, long_prompt, 48, "LONG", results)

    tasks = [stream_req(url, model, SHORT_PROMPT + f"（第{i}个）",
                        bg_tokens, f"BG{i}", results)
             for i in range(n_bg)]
    tasks += [interferer()]
    await asyncio.gather(*tasks)
    return results


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-bg", type=int, default=8)
    ap.add_argument("--long-tokens", type=int, default=6000)
    ap.add_argument("--bg-tokens", type=int, default=96)
    args = ap.parse_args()

    rs = asyncio.run(run(args.url, args.model, args.n_bg,
                         args.long_tokens, args.bg_tokens, None))
    bg_itls = [x for r in rs if r["tag"].startswith("BG")
               for x in r.get("itls", [])]
    long_r = next((r for r in rs if r["tag"] == "LONG"), {})

    print(f"背景 {args.n_bg} 条短请求 ITL（共 {len(bg_itls)} 个间隔）：")
    if bg_itls:
        print(f"  p50={pct(bg_itls,50)*1e3:.1f}ms  "
          f"p95={pct(bg_itls,95)*1e3:.1f}ms  "
          f"p99={pct(bg_itls,99)*1e3:.1f}ms  "
          f"max={max(bg_itls)*1e3:.1f}ms")
    if long_r:
        print(f"长 prompt（{args.long_tokens}tok）TTFT="
              f"{long_r.get('ttft',0)*1e3:.0f}ms  e2e={long_r.get('e2e',0):.2f}s")


if __name__ == "__main__":
    main()
