"""长上下文 KV 膨胀实验：扫 2k/8k/16k/30k prompt，测 TTFT 与 KV 占用。

vLLM 启动日志给出总 KV cache 池大小；单请求 KV 用量按公式算：
  bytes = 2(K+V) × n_layer × n_kv_head × head_dim × dtype × seq_len
Qwen2.5-7B: 28层 GQA(4 kv head, head_dim128) → bf16 每 tok = 2*28*4*128*2 = 57344B
"""
import argparse, asyncio, time, statistics as st
import aiohttp, random

URL = "http://127.0.0.1:8000/v1/completions"


async def one(s, model, prompt, out_tok=32):
    payload = {"model": model, "prompt": prompt, "max_tokens": out_tok,
               "temperature": 0, "stream": True}
    t0 = time.perf_counter()
    ttft = None
    n = 0
    async with s.post(URL, json=payload) as r:
        async for _ in r.content:
            n += 1
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
    e2e = (time.perf_counter() - t0) * 1000
    tpot = (e2e - ttft) / max(n - 1, 1)
    return ttft, tpot, e2e


async def main(model, lens):
    salt = "".join(random.choices("0123456789abcdef", k=16))
    unit = "KV cache 机制与推理调度涉及显存分块与连续批处理。" + salt
    async with aiohttp.ClientSession() as s:
        for L in lens:
            prompt = ("请总结：" + salt + unit * (L // len(unit)))
            ttft, tpot, e2e = await one(s, model, prompt)
            kv_mib = L * 57344 / 2**20   # bf16 公式值，fp8 减半
            print(f"len≈{L}: TTFT={ttft:.0f}ms TPOT={tpot:.1f}ms "
                  f"e2e={e2e/1000:.2f}s KV公式≈{kv_mib:.0f}MiB(bf16)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--lens", type=int, nargs="+",
                    default=[2000, 8000, 16000, 30000])
    a = ap.parse_args()
    asyncio.run(main(a.model, a.lens))
