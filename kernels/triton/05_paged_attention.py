"""Kernel 05：Triton paged-attention decode kernel（block table 寻址）

故事：graph 实验发现定长 StaticCache 让 attention 白扫 max_len 的
padding（2048 时反超无图 1.3x）。vLLM 的答案是 block table：KV 存进
全局物理块池，每 seq 一张 block_table，kernel 按表取块——变长 KV
不再为定长窗口付税。

实现：flash-decoding 风格在线 softmax。每个 program 负责 (batch, q头)，
沿 block_table 迭代物理块。支持 GQA（Hq = G*Hkv）。

对比：gather+SDPA 参考实现逐元素验证；perf 对照「padding 到
max_len 的定长 SDPA」展示变长寻址省掉多少。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_decode(
    Q, K, V, BT, CTX, OUT,
    sqb, sqh,
    skb, sks, skh,
    svb, svs, svh,
    sbtb, sbtn,
    sob, soh,
    scale,
    G: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    MAX_NB: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // G
    d = tl.arange(0, D)
    slots = tl.arange(0, BS)

    q = tl.load(Q + b * sqb + h * sqh + d)          # [D]
    ctx = tl.load(CTX + b)                          # 实际上下文长度
    n_blocks = (ctx + BS - 1) // BS

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for nb in range(0, MAX_NB):
        if nb < n_blocks:
            blk = tl.load(BT + b * sbtb + nb * sbtn)
            kp = K + blk * skb + slots[:, None] * sks + kvh * skh + d[None, :]
            k = tl.load(kp)                                    # [BS, D]
            s = tl.sum(k * q[None, :], axis=1) * scale         # [BS]
            valid = nb * BS + slots < ctx
            s = tl.where(valid, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            vp = V + blk * svb + slots[:, None] * svs + kvh * svh + d[None, :]
            v = tl.load(vp)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p)
            m_i = m_new

    out = acc / l_i
    tl.store(OUT + b * sob + h * soh + d, out.to(OUT.dtype.element_ty))


def paged_attention_decode(q, k_pool, v_pool, block_table, ctx_lens):
    """q [B,Hq,D] fp32/bf16; pools [NB,BS,Hkv,D]; block_table [B,MAX_NB] int32;
    ctx_lens [B] int32。返回 [B,Hq,D]。"""
    B, Hq, D = q.shape
    Hkv = k_pool.shape[2]
    G = Hq // Hkv
    BS = k_pool.shape[1]
    MAX_NB = block_table.shape[1]
    out = torch.empty_like(q)
    scale = 1.0 / (D ** 0.5)
    _paged_attn_decode[(B, Hq)](
        q, k_pool, v_pool, block_table, ctx_lens, out,
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1),
        scale, G=G, D=D, BS=BS, MAX_NB=MAX_NB,
    )
    return out


def ref_paged_attention(q, k_pool, v_pool, block_table, ctx_lens):
    """参考：按表 gather 后 per-seq SDPA。"""
    B, Hq, D = q.shape
    Hkv = k_pool.shape[2]
    G = Hq // Hkv
    outs = []
    for b in range(B):
        n = int(ctx_lens[b])
        nb = (n + k_pool.shape[1] - 1) // k_pool.shape[1]
        ids = block_table[b, :nb].long()
        k = k_pool[ids].reshape(-1, Hkv, D)[:n]   # [n,Hkv,D]
        v = v_pool[ids].reshape(-1, Hkv, D)[:n]
        k = k.permute(1, 0, 2).repeat_interleave(G, 0)  # [Hq,n,D]
        v = v.permute(1, 0, 2).repeat_interleave(G, 0)
        o = torch.nn.functional.scaled_dot_product_attention(
            q[b].unsqueeze(0).unsqueeze(2).float(),   # [1,Hq,1,D]
            k.unsqueeze(0).float(),                 # [1,Hq,n,D]
            v.unsqueeze(0).float())
        outs.append(o.squeeze(0).squeeze(1))          # [Hq,D]
    return torch.stack(outs)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    B, Hq, Hkv, D, BS = 8, 28, 4, 128, 16
    NB = 64
    MAX_NB = 32          # 表容量（≤512 tok）
    k_pool = torch.randn(NB, BS, Hkv, D, device=dev)
    v_pool = torch.randn(NB, BS, Hkv, D, device=dev)
    q = torch.randn(B, Hq, D, device=dev)
    # 变长上下文：1~512 随机
    ctx = torch.randint(1, MAX_NB * BS + 1, (B,), device=dev, dtype=torch.int32)
    block_table = torch.stack([
        torch.randperm(NB, device=dev)[:MAX_NB] for _ in range(B)
    ]).int().contiguous()

    got = paged_attention_decode(q, k_pool, v_pool, block_table, ctx)
    want = ref_paged_attention(q, k_pool, v_pool, block_table, ctx)
    err = (got - want).abs().max().item()
    rel = err / want.abs().max().item()
    print(f"correctness: max_abs_err={err:.3e} rel={rel:.3e} ctx={ctx.tolist()}")
    assert rel < 1e-2, "FAILED"
    print("PASS")

    # perf：变长 paged kernel vs 定长窗口 SDPA（graph 实验的 padding 税）
    import time
    ctx2 = torch.full((B,), 64, device=dev, dtype=torch.int32)  # 真实 ctx=64
    def bench(fn, n=200):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1e6
    t_paged = bench(lambda: paged_attention_decode(
        q, k_pool, v_pool, block_table, ctx2))
    # 定长对照：gather 512 再 SDPA（模拟定长窗口扫描的开销量级）
    ids = block_table.long()
    k_full = k_pool[ids].reshape(B, MAX_NB * BS, Hkv, D)
    v_full = v_pool[ids].reshape(B, MAX_NB * BS, Hkv, D)
    G = Hq // Hkv
    kf = k_full.permute(0, 2, 1, 3).repeat_interleave(G, 1)
    vf = v_full.permute(0, 2, 1, 3).repeat_interleave(G, 1)
    t_fixed = bench(lambda: torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(2), kf, vf))
    print(f"paged kernel (ctx=64): {t_paged:.1f}us")
    print(f"fixed-window SDPA (512, maskless): {t_fixed:.1f}us")
    print(f"padding 税倍数: {t_fixed/t_paged:.1f}x")


if __name__ == "__main__":
    main()
