"""batched vs per-seq 引擎等价性验证（greedy 应逐 token 一致）。

并发塞 4 条不同长度 prompt，跑 16 步，对比两引擎每条 generated：
- ids 完全一致 = pad/mask/position 全对
- 前缀一致后分叉 = bf16 数值差（正常，报告分叉位置）
- 完全不一致 = bug
"""
import sys
import torch
from transformers import AutoTokenizer

from engine import EngineCore, Request
from engine_batched import EngineCoreBatched

MODEL = "/data/ai-learn/models/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "用三句话解释什么是 KV cache：",
    "写一段 Python 快速排序：",
    "什么是 continuous batching？",
    "讲一个短笑话：",
]
MAX_NEW = 24


def run(engine, tok):
    reqs = []
    for i, p in enumerate(PROMPTS):
        r = Request(req_id=f"r{i}", prompt_ids=tok(p).input_ids,
                    max_new_tokens=MAX_NEW)
        engine.add_request(r)
        reqs.append(r)
    while engine.has_work():
        engine.step()
    return reqs


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    torch.manual_seed(0)
    e1 = EngineCore(MODEL, "cuda", num_kv_blocks=4096)
    r1 = run(e1, tok)
    del e1
    torch.cuda.empty_cache()

    torch.manual_seed(0)
    e2 = EngineCoreBatched(MODEL, "cuda", num_kv_blocks=4096)
    r2 = run(e2, tok)

    all_ok = True
    for a, b in zip(r1, r2):
        ga, gb = a.generated, b.generated
        same = ga == gb
        # 找分叉点
        diverge = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y),
                       min(len(ga), len(gb)))
        print(f"{a.req_id}: len {len(ga)} vs {len(gb)}, "
              f"{'IDENTICAL' if same else f'diverge@{diverge}'}")
        print(f"  A: {tok.decode(ga)[:60]}")
        if not same:
            all_ok = False
            print(f"  B: {tok.decode(gb)[:60]}")
    print("\n" + ("EQUIV PASS" if all_ok else "DIVERGED (检查是数值差还是bug)"))


if __name__ == "__main__":
    main()
