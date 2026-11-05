"""M2-Triton 第 2 课：LayerNorm 与 RMSNorm

面试考点：
- LayerNorm：y = (x - mean) / sqrt(var + eps) * w + b（减均值、除标准差、有 bias）
  RMSNorm：   y = x / sqrt(mean(x^2) + eps) * w（不减均值、无 bias）
  RMSNorm 少算一次均值归约、少一次减法、bias 也没有 → LLM（Llama/Qwen 等）全用它
- norm 类 kernel 是典型 memory-bound：每元素只做 O(1) 计算，性能几乎完全
  由 HBM 带宽决定 → benchmark 用 GB/s 而不是 FLOPS 衡量
- "一行一个 program"：把整行放进一个 BLOCK，mean/var 归约（tl.sum）在寄存器/
  warp shuffle / shared memory 内完成，x 只从 HBM 读一次
- 方差用"两遍法"：先 mean 再 sum((x-mean)^2)。两遍都发生在寄存器里，
  不增加显存往返；对比 torch 手写版，每个中间结果都落显存（≥5 次往返），
  这就是 kernel fusion 省时间的本质：省的是访存，不是计算
- 面试追问：BLOCK 怎么选？→ next_power_of_2(N_cols)，一行放得下就不用循环；
  num_warps 怎么选？→ 行越长 warps 越多（这里统一 8，够用）

运行：python 02_layernorm_rmsnorm.py
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(X, Y, W, B,
                     N_cols: tl.constexpr, EPS,
                     BLOCK: tl.constexpr):
    """一个 program 处理一行。

    访存次数：读 x 一次 + 读 w/b 各一次（per program）+ 写 y 一次。
    mean/var 全部在寄存器里算出来，x 不回读。
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_cols

    # ---- 读入整行（fp32 累加保精度；生产环境输入多是 bf16，也建议升 fp32 再归约）
    # masked 位置填 0.0：对 sum 无影响。注意这里不能填 -inf（那是给 max 用的）
    x = tl.load(X + row * N_cols + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- 两遍法第 1 遍：均值（寄存器内归约）----
    mean = tl.sum(x, axis=0) / N_cols

    # ---- 两遍法第 2 遍：方差（仍在寄存器内）----
    # 掩码外位置必须清零：否则 (0 - mean)^2 会混进 var
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N_cols
    rstd = 1.0 / tl.sqrt(var + EPS)   # rstd = rsqrt(var+eps)，工程实现会缓存它做 backward

    # ---- 归一化 + affine，写回（写 y 一次）----
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b
    tl.store(Y + row * N_cols + cols, y, mask=mask)


@triton.jit
def rmsnorm_kernel(X, Y, W,
                   N_cols: tl.constexpr, EPS,
                   BLOCK: tl.constexpr):
    """RMSNorm：y = x * rsqrt(mean(x^2) + eps) * w

    和 LayerNorm 的差异（面试爱问）：
    - 不求 mean、不减 mean → 少一次归约、少一次减法
    - 没有 bias → 少读一块权重、少一次加法
    - 因为不中心化，对输入分布漂移更鲁棒的争议点这里不展开；
      工程上选它纯粹因为"够好且更便宜"
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_cols

    x = tl.load(X + row * N_cols + cols, mask=mask, other=0.0).to(tl.float32)

    # 只需 mean(x^2) 一次归约，比 LayerNorm 少一遍
    ms = tl.sum(x * x, axis=0) / N_cols
    rstd = 1.0 / tl.sqrt(ms + EPS)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N_cols + cols, x * rstd * w, mask=mask)


def triton_layernorm(x, w, b, eps=1e-5):
    rows, cols = x.shape
    BLOCK = triton.next_power_of_2(cols)   # 一行一个 BLOCK；本文件 shape 下 BLOCK == cols
    y = torch.empty_like(x)
    layernorm_kernel[(rows,)](x, y, w, b, cols, eps, BLOCK=BLOCK, num_warps=8)
    return y


def triton_rmsnorm(x, w, eps=1e-5):
    rows, cols = x.shape
    BLOCK = triton.next_power_of_2(cols)
    y = torch.empty_like(x)
    rmsnorm_kernel[(rows,)](x, y, w, cols, eps, BLOCK=BLOCK, num_warps=8)
    return y


def torch_rmsnorm(x, w, eps=1e-5):
    """手写 torch 版：每一步都是独立 kernel，中间结果全部落显存。

    数一下访存：x.pow(2) 读写一次 → mean 读写一次 → rsqrt 一次 →
    x*rsqrt 读写一次 → *w 再读写一次 ≈ 7 次往返。
    对照 Triton 版的 1 读 1 写，就知道融合版快在哪了。
    """
    ms = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * w


def bench(fn, nbytes, iters=100):
    """测 zero-arg callable，返回 (ms, GB/s)。GB/s 按请求字节数 / 时间算。"""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return ms, nbytes / (ms / 1e3) / 1e9


if __name__ == "__main__":
    torch.manual_seed(0)
    esize = 4  # fp32

    for rows, cols in [(4096, 4096), (32768, 1024)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        w = torch.randn(cols, device="cuda", dtype=torch.float32)
        b = torch.randn(cols, device="cuda", dtype=torch.float32)

        # ---- 正确性校验（fp32，误差应 ~1e-6）----
        ref_ln = F.layer_norm(x, (cols,), w, b, eps=1e-5)
        out_ln = triton_layernorm(x, w, b)
        err_ln = (out_ln - ref_ln).abs().max().item()

        ref_rms = torch_rmsnorm(x, w)
        out_rms = triton_rmsnorm(x, w)
        err_rms = (out_rms - ref_rms).abs().max().item()

        # ---- 有效字节数：x 读 + y 写 + 权重按行重复读 ----
        # 每个 program 都读整条 W（和 B），所以权重请求流量 = rows*cols 每份权重；
        # 实际上权重被 L2 大量缓存，真实 DRAM 流量低于此——GB/s 是"请求带宽"口径
        ln_bytes = (2 * rows * cols + 2 * rows * cols) * esize   # x,y + w,b
        rms_bytes = (2 * rows * cols + rows * cols) * esize      # x,y + w

        t_ln_torch, b_ln_torch = bench(lambda: F.layer_norm(x, (cols,), w, b, eps=1e-5), ln_bytes)
        t_ln_tri,   b_ln_tri   = bench(lambda: triton_layernorm(x, w, b), ln_bytes)
        t_rms_torch, b_rms_torch = bench(lambda: torch_rmsnorm(x, w), rms_bytes)
        t_rms_tri,   b_rms_tri   = bench(lambda: triton_rmsnorm(x, w), rms_bytes)

        print(f"shape=({rows}, {cols})")
        print(f"  layernorm | err={err_ln:.2e} | "
              f"torch {t_ln_torch:.3f}ms {b_ln_torch:6.0f}GB/s | "
              f"triton {t_ln_tri:.3f}ms {b_ln_tri:6.0f}GB/s")
        print(f"  rmsnorm   | err={err_rms:.2e} | "
              f"torch {t_rms_torch:.3f}ms {b_rms_torch:6.0f}GB/s | "
              f"triton {t_rms_tri:.3f}ms {b_rms_tri:6.0f}GB/s")
