"""spec-decode 接受率 vs 输出内容类型 + 质量对照题。

用法：HF_HUB_OFFLINE=1 python spec_content_ablation.py
三类 prompt（代码/散文/重复），每类发一条 rep_pen=1.0 请求，
通过 /metrics 的 spec_decode_num_drafts/accepted_tokens 增量算接受长度。
"""
import time, json, urllib.request

def metrics():
    txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics").read().decode()
    d = a = 0
    for l in txt.splitlines():
        if l.startswith("vllm:spec_decode_num_drafts_total"):
            d = int(float(l.rsplit(" ", 1)[-1]))
        if l.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a = int(float(l.rsplit(" ", 1)[-1]))
    return d, a

CASES = {
    "重复文本": "请把『好好学习天天向上』重复写100遍。",
    "代码":    "写一个完整的Python快排实现，带详细注释和测试用例。",
    "散文":    "写一篇800字散文，描写秋天江南的雨巷。",
}

def gen(prompt, n=200):
    p = {"model": "qwen7b", "prompt": prompt, "max_tokens": n,
         "temperature": 0, "repetition_penalty": 1.0}
    t0 = time.perf_counter()
    r = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
        data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r).read())
    dt = time.perf_counter() - t0
    return d["usage"]["completion_tokens"], dt

for name, prompt in CASES.items():
    d0, a0 = metrics()
    n, dt = gen(prompt)
    d1, a1 = metrics()
    drafts, acc = d1 - d0, a1 - a0
    per = acc / drafts if drafts else 0
    print(f"{name}: {n}tok {dt:.2f}s={n/dt:.0f}tok/s | "
          f"drafts={drafts} accepted={acc} 接受长度={per:.2f}/5")
