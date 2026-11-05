"""torch.profiler 抓 mini-serving batched decode 单步时间线。

用法：.venv/bin/python benchmarks/profile_decode_step.py
输出：CPU/CUDA 耗时 top 表 + chrome trace（可拖入 ui.perfetto.dev 看图）
目的：量化 decode step 中 kernel 间隙（Python/launch 开销）占比——
      即 CUDA Graph 要消掉的那部分时间。
"""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mini-serving"))
from engine_batched import EngineCoreBatched   # noqa
from engine import Request                     # noqa
from transformers import AutoTokenizer

MODEL = "/data/ai-learn/models/Qwen2.5-0.5B-Instruct"
PROMPTS = ["你好，讲一个笑话", "解释一下KV cache", "写一首关于GPU的诗",
           "CUDA是什么", "什么是投机解码", "介绍transformer", "讲个故事", "什么是RoPE"]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    eng = EngineCoreBatched(model_path=MODEL, device="cuda",
                            num_kv_blocks=256)
    for i, p in enumerate(PROMPTS):
        eng.add_request(Request(req_id=str(i),
                                prompt_ids=tok(p).input_ids,
                                max_new_tokens=64))
    for _ in range(4):
        eng.step()                       # 预热 + 完成 prefill

    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):              # 抓 10 个 batched decode step
            eng.step()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    prof.export_chrome_trace("/data/ai-learn/logs/decode_trace.json")

    cuda_total = sum(e.self_device_time_total for e in prof.key_averages()) / 1e3
    t0 = time.perf_counter()
    for _ in range(10):
        eng.step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000
    print(f"\n10 steps wall={wall:.1f}ms ({wall/10:.2f}ms/step), "
          f"GPU busy={cuda_total:.1f}ms, 间隙≈{(1 - cuda_total/wall)*100:.0f}%")


if __name__ == "__main__":
    main()
