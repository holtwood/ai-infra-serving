"""M2-Triton 第 4 课：FlashAttention 前向（教学简化版）

面试考点（⭐ 必考：online softmax 怎么做、为什么省显存）：
- 标准 attention 的痛点：S_matrix = QK^T 是 S×S，seq 一长显存/带宽全爆
  （S=128k 时单 head 的 attention matrix 就是 64GB fp16）
- online softmax 思想：softmax 需要整行的 max 和 sum，但其实可以"边走边算"——
  按块扫 K/V，只维护三个量：
      m_i = 到目前见过的行最大值
      l_i = 以 m_i 为基准的 exp 累加和（即 sum(exp(qk - m_i))）
      acc = 以 m_i 为基准的加权值累加（即 sum(exp(qk - m_i) * v)）
  每来一个新块，qk_new 算出后：
      m_new = max(m_i, rowmax(qk))
      alpha = exp(m_i - m_new)          ← 旧累计量折算到新基准的修正系数
      l_i  = l_i * alpha + rowsum(exp(qk - m_new))
      acc  = acc * alpha + exp(qk - m_new) @ V
  数学上与一次性 softmax 完全等价（exp 基准平移不变性），
  但 S×S 矩阵从头到尾不落显存 → 显存 O(S·D)，访存是流式扫 K/V
- 本教学版简化点（面试要主动交代）：non-causal、QKV 等长、batch*heads 展平、
  head_dim 一块放得下（64/128）、无 dropout、无 alibi
  生产版（FA2/FA3、vLLM）还要处理：causal mask 的块跳过、GQA/MQA、
  paged KV、变长 seq 的 cu_seqlens、warp 级流水线
- 追问"Q 的 tile 为什么只读一次"：grid 第 0 维按 Q 的块划分，
  每个 program 持有一块 Q 循环扫全部 K/V —— 这正是"outer loop over KV,
  inner keep Q in SRAM"的结构

运行：python 04_flash_attention_fwd.py
"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(Q, K, V, O,
                     sqb, sqm, sqd,
                     skb, skn, skd,
                     svb, svn, svd,
                     sob, som, sod,
                     S, SCALE,
                     D: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """grid = (cdiv(S, BLOCK_M), batch*heads)。
    每个 program：固定一块 Q（BLOCK_M×D），循环扫过所有 K/V 块做 online softmax。
    """
    pid_m = tl.program_id(0)     # 第几个 Q 块
    pid_bh = tl.program_id(1)    # 第几个 (batch, head)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # ---- Q 块只读一次，常驻寄存器/smem（面试点：这就是为什么按 Q 分块）----
    q = tl.load(Q + pid_bh * sqb + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                mask=offs_m[:, None] < S, other=0.0)

    # ---- online softmax 的三个运行状态（全在寄存器里，O(BLOCK_M) 大小）----
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)  # 行最大值
    l_i = tl.zeros((BLOCK_M,), tl.float32)                # exp 累加和
    acc = tl.zeros((BLOCK_M, D), tl.float32)              # 加权和累加器

    base_k = K + pid_bh * skb
    base_v = V + pid_bh * svb

    # ---- 沿 KV 方向分块扫：每轮把旧累计"折算"到新的 max 基准 ----
    for n0 in range(0, S, BLOCK_N):
        n_mask = offs_n + n0 < S

        # 载入 K^T 的一个块，形状 (D, BLOCK_N)——直接用指针摆成转置，
        # 这样 tl.dot(q, k) 出来的就是 (BLOCK_M, BLOCK_N) 的 qk 分数
        k = tl.load(base_k + offs_d[:, None] * skd + (offs_n[None, :] + n0) * skn,
                    mask=n_mask[None, :], other=0.0)

        qk = tl.dot(q, k) * SCALE                       # fp32 分数块
        # 越界列置 -inf：既不影响本块 max，也让 exp 贡献为 0
        qk = tl.where(n_mask[None, :], qk, -float("inf"))

        # === online softmax 更新（本 kernel 的灵魂，背也要背下来）===
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))     # 新基准 = 新旧 max 取大
        alpha = tl.exp(m_i - m_new)                     # 旧累计的折算系数
        p = tl.exp(qk - m_new[:, None])                 # 以新基准算 exp
        l_i = l_i * alpha + tl.sum(p, axis=1)           # 累加和先折算再累加

        v = tl.load(base_v + (offs_n[:, None] + n0) * svn + offs_d[None, :] * svd,
                    mask=n_mask[:, None], other=0.0)
        # acc 先折算，再加上本块贡献；p 降到 bf16 让第二个 dot 也走 tensor core
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        # ===============================================

    # 收尾：除以真正的 sum(exp(qk - max))，等价于 softmax 归一化
    acc = acc / l_i[:, None]
    tl.store(O + pid_bh * sob + offs_m[:, None] * som + offs_d[None, :] * sod,
             acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < S)


def triton_flash(q, k, v):
    """q/k/v: (B*H, S, D) 连续 bf16；返回同形状输出。"""
    BH, S, D = q.shape
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64      # 教学版固定值；生产版会按 D/硬件 autotune
    scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(S, BLOCK_M), BH)
    flash_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        S, scale,
        D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return o


def bench(fn, flops, iters=50):
    """测 zero-arg callable，返回 (ms, TFLOPS)。"""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return ms, flops / (ms / 1e3) / 1e12


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H = 4, 8

    for S in [1024, 2048, 4096]:
        for D in [64, 128]:
            # (B, H, S, D) 建好后展平成 (B*H, S, D) 喂给 kernel
            q4 = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
            k4 = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
            v4 = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
            q3, k3, v3 = (t.reshape(B * H, S, D) for t in (q4, k4, v4))

            # 参考：torch SDPA（内部会走它的 fused attention 后端）
            ref = F.scaled_dot_product_attention(q4, k4, v4)
            out = triton_flash(q3, k3, v3).view(B, H, S, D)

            # bf16 + 分块归约顺序不同 → 误差放宽到 2e-2 量级
            err = (out.float() - ref.float()).abs().max().item()
            ok = torch.allclose(out, ref, rtol=2e-2, atol=2e-2)

            # fwd FLOPs ≈ QK^T + softmax@PV 两个 matmul：4·B·H·S²·D
            flops = 4 * B * H * S * S * D
            t_ref, f_ref = bench(lambda: F.scaled_dot_product_attention(q4, k4, v4), flops)
            t_tri, f_tri = bench(lambda: triton_flash(q3, k3, v3), flops)

            print(f"S={S:>5} D={D:>3} | err={err:.2e} "
                  f"{'PASS' if ok else 'FAIL':<4} | "
                  f"sdpa {t_ref:.3f}ms {f_ref:5.1f}TF | "
                  f"triton {t_tri:.3f}ms {f_tri:5.1f}TF")
