"""AWQ-32B vs 7B-bf16 质量对照（vLLM 离线 API，无 autoawq 依赖）。

PPL：prompt_logprobs=0 求平均 logprob 取 exp。
QA：greedy 生成存 json 供人工对评。
用法：.venv/bin/python benchmarks/ppl_compare.py
"""
import glob, json, math
from vllm import LLM, SamplingParams

MODELS = {
    "qwen7b_bf16": "/data/ai-learn/models/Qwen2.5-7B-Instruct",
    "awq32b": "/model/vllm/Qwen2.5-32B-Instruct-AWQ",
}
QA = [
    "If a train travels 60km in 45 minutes, what is its speed in km/h? Show reasoning.",
    "Explain the difference between TCP and UDP in two sentences.",
    "Write a Python function to check if a string is a palindrome, ignoring spaces and case.",
    "Write a SQL query to find the second highest salary from an Employee table.",
    "Translate to English: '大模型推理的核心瓶颈是显存带宽而不是算力'",
]
PROMPT_FMT = "Question: {q}\nAnswer:"


def run(path):
    text = "\n".join(open(f).read() for f in
                     sorted(glob.glob("/data/ai-learn/notes/*.md"))[:8])[:8000]
    llm = LLM(model=path, max_model_len=8192,
              gpu_memory_utilization=0.85, enforce_eager=True)
    o = llm.generate([text], SamplingParams(prompt_logprobs=0, max_tokens=1))
    lps = [list(d.values())[0].logprob for d in o[0].prompt_logprobs if d]
    ppl = math.exp(-sum(lps) / len(lps))
    outs = llm.generate([PROMPT_FMT.format(q=q) for q in QA],
                        SamplingParams(temperature=0, max_tokens=200))
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return ppl, [o.outputs[0].text for o in outs]


def main():
    res = {}
    for name, path in MODELS.items():
        ppl, qa = run(path)
        res[name] = {"ppl": ppl, "qa": qa}
        print(f"{name}: PPL={ppl:.2f}")
        for i, a in enumerate(qa):
            print(f"  Q{i}: {a[:80].replace(chr(10), ' ')}")
    json.dump(res, open("/data/ai-learn/logs/ppl_compare.json", "w"),
              ensure_ascii=False, indent=1)


main()
