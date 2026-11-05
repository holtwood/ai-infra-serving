"""手撕系列 01：MHA / GQA / KV cache decode step（纯张量，禁高层 attention API）

面试共识：这三段是入场券。必须显式写 QK^T → scale → mask → softmax → AV，
注释标清每步 shape (B,H,S,D)。F.scaled_dot_product_attention 只作对照。

约定：q (B,Hq,S,D)，k/v (B,Hkv,S,D)；GQA 中 Hq = G·Hkv（G=group size）。
运行：python 01_mha_gqa_kvcache.py
"""
import math
import torch
import torch.nn.functional as F


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """减 max 防 exp 溢出（数学不变：softmax(x-c)=softmax(x)）。详见 03。"""
    e = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return e / e.sum(dim=dim, keepdim=True)


def mha_forward(q, k, v, causal: bool = True):
    """(a) 标准 MHA。q,k,v: (B,H,S,D) → out (B,H,S,D)。"""
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = q @ k.transpose(-2, -1) * scale       # (B,H,S,S)：行=query，列=key
    if causal:
        # 上三角（j>i）置 -inf → softmax 后权重=0；triu(...,1) 不含对角
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = stable_softmax(scores, dim=-1)          # (B,H,S,S)，行和=1
    return attn @ v                                 # (B,H,S,S)@(B,H,S,D) → (B,H,S,D)


def expand_kv(x, n_rep: int):
    """GQA 关键步：把 Hkv 复制 G 份对齐 Hq。x:(B,Hkv,S,D) → (B,Hq,S,D)

    两种等价写法（面试都要会讲）：
    (a) x.repeat_interleave(n_rep, dim=1) —— 一步到位，直接拷贝
    (b) 下面：unsqueeze+expand 先广播（视图零拷贝），reshape 时才物化
    注意：省显存的是【存储侧】（cache/权重投影），算的时候仍要展开成 Hq。
    """
    if n_rep == 1:
        return x
    B, Hkv, S, D = x.shape
    # (B,Hkv,S,D) → (B,Hkv,1,S,D) → (B,Hkv,G,S,D) → (B,Hq,S,D)
    return x.unsqueeze(2).expand(B, Hkv, n_rep, S, D).reshape(B, Hkv * n_rep, S, D)


def gqa_forward(q, k, v, causal: bool = True):
    """(b) GQA：Hkv < Hq，每 G 个 q head 共享一组 kv。MQA = G=Hq 的特例。"""
    Hq, Hkv = q.shape[1], k.shape[1]
    assert Hq % Hkv == 0, f"Hq={Hq} 必须整除 Hkv={Hkv}"
    return mha_forward(q, expand_kv(k, Hq // Hkv),
                       expand_kv(v, Hq // Hkv), causal)


def decode_step(q_1tok, k_1tok, v_1tok, k_cache, v_cache):
    """(c) 带 KV cache 的单步 decode。

    签名对照面试口径 step(q_1tok, k_cache, v_cache)：物理上本步新 token 的
    K/V 投影必须先算出来（q 推不出 k/v），所以显式传 k_1tok/v_1tok。
    q_1tok: (B,Hq,1,D)；k_1tok/v_1tok: (B,Hkv,1,D)；cache: (B,Hkv,Spast,D)
    返回 (out (B,Hq,1,D), k_cache', v_cache')。
    """
    k_cache = torch.cat([k_cache, k_1tok], dim=2)   # (B,Hkv,Spast+1,D)：append
    v_cache = torch.cat([v_cache, v_1tok], dim=2)   # cat 是 O(S) 拷贝 → 工程上预分配
    k_all = expand_kv(k_cache, q_1tok.shape[1] // k_cache.shape[1])  # (B,Hq,Sall,D)
    v_all = expand_kv(v_cache, q_1tok.shape[1] // v_cache.shape[1])
    scores = q_1tok @ k_all.transpose(-2, -1) / math.sqrt(q_1tok.shape[-1])
    attn = stable_softmax(scores, dim=-1)           # (B,Hq,1,Sall)
    # 单 query 是最后一个位置，可见全部历史 → 不需要 causal mask
    return attn @ v_all, k_cache, v_cache           # (B,Hq,1,D)


def check(name, a, b, atol=2e-2):
    err = (a.float() - b.float()).abs().max().item()
    assert err < atol, f"{name} err={err:.3e}"
    print(f"{name:<22} max_err={err:.3e}  PASS")


if __name__ == "__main__":
    torch.manual_seed(0)
    dt, dev = torch.bfloat16, "cpu"          # 有 GPU 换 "cuda"（CPU 走 SDPA math 路径）
    B, Hq, Hkv, S, D, G = 2, 8, 2, 32, 64, 4

    # (a) MHA：手写 vs SDPA
    q, k, v = (torch.randn(B, Hq, S, D, dtype=dt, device=dev) for _ in range(3))
    check("MHA vs SDPA", mha_forward(q, k, v),
          F.scaled_dot_product_attention(q, k, v, is_causal=True))

    # (b) GQA：参考=先把 kv expand 成 MHA 再 SDPA
    qg = torch.randn(B, Hq, S, D, dtype=dt, device=dev)
    kg, vg = (torch.randn(B, Hkv, S, D, dtype=dt, device=dev) for _ in range(2))
    ref = F.scaled_dot_product_attention(
        qg, expand_kv(kg, G), expand_kv(vg, G), is_causal=True)
    check("GQA vs SDPA(expand)", gqa_forward(qg, kg, vg), ref)

    # (c) decode：逐步 append，每步对照 SDPA。注意 is_causal 对非方阵是
    # 左上对齐——S_q=1 时开 causal 只会 attend 到 key0，必须用无 mask 版。
    kc = torch.zeros(B, Hkv, 0, D, dtype=dt, device=dev)   # 空 cache 起步
    vc = torch.zeros(B, Hkv, 0, D, dtype=dt, device=dev)
    max_err = 0.0
    for t in range(5):
        qt = torch.randn(B, Hq, 1, D, dtype=dt, device=dev)
        kt, vt = (torch.randn(B, Hkv, 1, D, dtype=dt, device=dev) for _ in range(2))
        out, kc, vc = decode_step(qt, kt, vt, kc, vc)
        ref_t = F.scaled_dot_product_attention(qt, expand_kv(kc, G), expand_kv(vc, G))
        max_err = max(max_err, (out.float() - ref_t.float()).abs().max().item())
    assert max_err < 2e-2, max_err
    print(f"decode_step x5       max_err={max_err:.3e}  cache={tuple(kc.shape)}  PASS")

    # KV cache 显存公式（⭐ 必背，推导见 notes/inference-math.md §3）：
    #   bytes/token = 2(K,V) × n_layers × Hkv × D × dtype_bytes
    #   总 KV       = bytes/token × seq_len × batch
    # 例 Qwen2.5-7B：2×28×4×128×2B ≈ 56KB/token → 4k ctx ≈ 224MB/seq。
    # GQA 省的就是 Hkv（Hq→Hkv 缩几倍 cache 小几倍），算力几乎不变。
