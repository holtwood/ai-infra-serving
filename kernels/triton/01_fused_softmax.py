"""M2-Triton 第 1 课：fused softmax

面试考点：
- softmax 为什么是 memory-bound：3 次访存（读 max、读 exp-sum、读写归一化）
- fuse 成一次遍历后只做 1 次读 + 1 次写 → 带宽利用率直接体现优化效果
- online softmax（一次遍历同时求 max 和 sum）是 FlashAttention 的思想源头

运行：python 01_fused_softmax.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, Y, N_cols: tl.constexpr, BLOCK: tl.constexpr):
    # 一个 program 处理一行：grid = (n_rows,)
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_cols

    x = tl.load(X + row * N_cols + cols, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)                       # 数值稳定：先减行最大值
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(Y + row * N_cols + cols, e / s, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    BLOCK = triton.next_power_of_2(cols)        # 每个 program 的 tile 大小
    y = torch.empty_like(x)
    softmax_kernel[(rows,)](x, y, cols, BLOCK=BLOCK, num_warps=8)
    return y


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """教学版：分三步，访存 3 次。"""
    m = x.max(dim=-1, keepdim=True).values      # pass 1: 读 x 求 max
    e = torch.exp(x - m)                        # pass 2: 读 x 读 m 写 e
    return e / e.sum(dim=-1, keepdim=True)      # pass 3: 读 e 求 sum 再读写


def bench(fn, x, iters=200):
    for _ in range(20): fn(x)
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters): fn(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    # 有效带宽 = 读一遍 + 写一遍
    gbs = x.numel() * x.element_size() * 2 / (ms / 1e3) / 1e9
    return ms, gbs


if __name__ == "__main__":
    torch.manual_seed(0)
    for shape in [(4096, 4096), (8192, 8192), (32768, 1024)]:
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)
        out = triton_softmax(x)
        err = (out - ref).abs().max().item()

        t_naive, b_naive = bench(naive_softmax, x)
        t_torch, b_torch = bench(lambda v: torch.softmax(v, dim=-1), x)
        t_triton, b_triton = bench(triton_softmax, x)
        print(f"{str(shape):>16} | err={err:.2e} | "
              f"naive {t_naive:.3f}ms {b_naive:6.0f}GB/s | "
              f"torch {t_torch:.3f}ms {b_torch:6.0f}GB/s | "
              f"triton {t_triton:.3f}ms {b_triton:6.0f}GB/s")
