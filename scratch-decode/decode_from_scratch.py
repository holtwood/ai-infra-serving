"""M1: 手写带 KV cache 的 decode loop —— 不用 model.generate

学习目标（对应 interview-map.md M1 部分）：
1. prefill 阶段：一次 forward 吃掉整个 prompt，产出全部 KV cache
   - 计算形态是大 GEMM（seq_len × hidden），compute-bound
2. decode 阶段：每步只喂 1 个 token，靠 past_key_values 复用历史 K/V
   - 计算形态是 GEMV（1 × hidden），memory-bound —— 每步都要把全量权重+KV 从显存搬一遍
3. 采样：greedy / temperature / top-p 的实现差异

用法（在远端机器 /data/ai-learn 下运行）：
    python decode_from_scratch.py
"""
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data/ai-learn/models/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    return model, tokenizer


def sample_next_token(logits: torch.Tensor, temperature: float = 0.0,
                      top_p: float = 1.0) -> int:
    """从最后一个位置的 logits 采样下一个 token。

    temperature=0 → greedy argmax；否则 softmax(logits/T) 后按 top-p 截断采样。
    """
    if temperature <= 0:
        return int(logits.argmax().item())

    logits = logits / temperature
    if top_p < 1.0:
        # top-p（nucleus）：只保留累计概率达到 p 的最小 token 集合
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        # 把累计概率超过 p 之后的 token 全部屏蔽
        cutoff = (cumprobs - probs) > top_p
        sorted_logits[cutoff] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_manual(model, input_ids: torch.Tensor, max_new_tokens: int = 64,
                    temperature: float = 0.0, top_p: float = 1.0):
    """手写 generate：prefill 一次 + 逐 token decode。

    返回 (generated_ids, prefill_ms, per_decode_ms)。
    """
    past_key_values = None
    generated = []

    # ---------- prefill：整个 prompt 一次前向 ----------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3

    past_key_values = out.past_key_values
    next_logits = out.logits[0, -1]                    # 最后一个位置的分布
    next_id = sample_next_token(next_logits, temperature, top_p)
    generated.append(next_id)

    # ---------- decode：每步只输入刚生成的 1 个 token ----------
    decode_times = []
    for _ in range(max_new_tokens - 1):
        cur = torch.tensor([[next_id]], device=input_ids.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=cur, past_key_values=past_key_values, use_cache=True)
        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - t0)

        past_key_values = out.past_key_values         # 每步 append 新 K/V
        next_id = sample_next_token(out.logits[0, -1], temperature, top_p)
        generated.append(next_id)
        if next_id == model.config.eos_token_id:
            break

    per_decode_ms = sum(decode_times) / len(decode_times) * 1e3
    return generated, prefill_ms, per_decode_ms


def kv_cache_report(model, seq_len: int):
    """面试题：算一下这个模型 seq_len 下的 KV cache 显存。"""
    cfg = model.config
    bytes_per_elem = 2  # bf16
    total = (cfg.num_hidden_layers * 2 * seq_len
             * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
             * bytes_per_elem)
    print(f"KV cache @ {seq_len} tokens: {total / 2**20:.1f} MiB "
          f"({cfg.num_hidden_layers} layers x 2 x {cfg.num_key_value_heads} kvh "
          f"x {cfg.hidden_size // cfg.num_attention_heads} dim x 2B)")


def main():
    model, tokenizer = load_model()
    kv_cache_report(model, seq_len=2048)

    prompt = "用三句话解释什么是 KV cache："
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"prompt tokens: {input_ids.shape[1]}")

    ids, prefill_ms, decode_ms = generate_manual(model, input_ids, 64)
    print(f"\n--- 生成结果 ---\n{tokenizer.decode(ids)}")
    print(f"\nprefill: {prefill_ms:.1f} ms ({input_ids.shape[1]} tokens)")
    print(f"decode : {decode_ms:.2f} ms/token -> {1000/decode_ms:.1f} tok/s")

    # 对照组：HF 官方 generate，输出应一致（greedy）
    ref = model.generate(input_ids, max_new_tokens=64, do_sample=False)
    ref_text = tokenizer.decode(ref[0, input_ids.shape[1]:])
    print(f"\n--- model.generate 对照 ---\n{ref_text}")


if __name__ == "__main__":
    main()
