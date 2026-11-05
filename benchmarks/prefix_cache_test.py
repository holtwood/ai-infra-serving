"""prefix caching 消融：同一长前缀连发 N 次，看 TTFT 是否骤降。

vLLM V1 默认 enable_prefix_caching=True：KV block 按 token hash 复用，
同前缀请求命中后只需算增量部分 → TTFT 应显著下降。

用法：python prefix_cache_test.py --model qwen7b --prefix-tokens 2048 --repeat 5
"""
import argparse
import json
import time

import aiohttp
import asyncio


async def send(url, model, prompt, max_tokens=8):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    t0 = time.perf_counter()
    ttft = None
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)) as s:
        async with s.post(f"{url}/v1/chat/completions", json=payload) as r:
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    d = json.loads(line[5:].strip())
                    if d["choices"][0].get("delta", {}).get("content"):
                        pass
    return ttft or (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-tokens", type=int, default=2048)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    # 构造 ~prefix_tokens 的公共前缀 + 每轮不同的尾缀（保证前缀命中、后缀不同）
    base = "下面是一段技术文档，请阅读后总结要点：" + \
        "KV cache 是 LLM 推理中缓存历史 key/value 向量的机制。" * (
            args.prefix_tokens // 22)
    for i in range(args.repeat):
        prompt = base + f"（第{i}次提问：请用一句话概括。）"
        ttft = asyncio.run(send(args.url, args.model, prompt))
        print(f"req {i}: TTFT={ttft*1e3:.1f}ms")


if __name__ == "__main__":
    main()
