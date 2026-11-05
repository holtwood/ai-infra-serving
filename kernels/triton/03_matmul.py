"""M2-Triton 第 3 课：matmul 三版本演进（naive → swizzle → autotune）

面试考点（⭐ "GEMM 怎么优化"必答题）：
- 演进路线：naive → shared memory tiling → register tiling → tensor core。
  Triton 里 smem/register tiling 由编译器按 BLOCK_* 自动完成，你只需指定 tile 尺寸；
  tl.dot 自动走 tensor core（bf16 输入、fp32 累加），要求 block 三维都 ≥16
- 为什么 tiling：每个 program 只算 C 的一个 BLOCK_M×BLOCK_N 小块，
  A/B 的对应条带进 shared memory 后被复用 BLOCK_N / BLOCK_M 次
  → 同一数据从 HBM 只搬一次，AI（arithmetic intensity）×BLOCK
- 为什么 swizzle（GROUP_M 分组重排）：同一时刻在 SM 上并发运行的 CTA 只有
  一个 wave（4090 上约一两百个）。若 wave 内的 tile 排成细长条（naive 行优先），
  每个 CTA 都 miss 不同的 A 行 / B 列 → L2 命中率低；
  把 wave 内 tile 重排成接近方形（GROUP_M×GROUP_N），
  A 的行块被同 group 内整行 CTA 复用、B 的列块被整列复用 → L2 命中↑
- @triton.autotune 原理：对 key 里指定的参数（M/N/K）的每种新组合，
  把 configs 里每个 config 真跑一遍取最快并缓存；num_warps/num_stages
  （pipeline 级数，即 smem 里叠几块 A/B tile 做预取）也随 config 一起调
- 判定 bound：实测 TFLOPS 接近 bf16 峰值（4090 约 165 TFLOPS dense）
  就是 compute-bound；离得远说明被访存/调度卡住

运行：python 03_matmul.py
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 公共部分：算一个 BLOCK_M×BLOCK_N 的 C tile（三个版本复用）
#   循环沿 K 方向分块推进：每次把 A 的 (BLOCK_M, BLOCK_K) 和 B 的 (BLOCK_K, BLOCK_N)
#   搬进 shared memory（编译器自动做），tl.dot 在 tensor core 上累加进 fp32 acc。
#   Triton 看不到 shared memory 关键字——写 BLOCK_* + tl.dot 就是"已做 smem tiling"
# ---------------------------------------------------------------------------
@triton.jit
def _matmul_tile(A, B, C, M, N, K,
                 sam, sak, sbk, sbn, scm, scn,
                 pid_m, pid_n,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr):
    # 本 tile 负责的 C 的 row/col 范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 行主序：A[m,k] = A + m*sam + k*sak；B[k,n] = B + k*sbk + n*sbn
    a_ptrs = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    # fp32 累加器：tensor core 的 bf16 mma 也是 fp32 累加，这里对齐
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # K 不整除 BLOCK_K 时，最后一块要按剩余长度 mask（其余位置填 0 不影响乘加）
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N),
                    other=0.0)
        # tl.dot(a, b, acc) ≡ acc += a @ b，走 tensor core
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk

    c = acc.to(C.dtype.element_ty)   # fp32 结果降回 bf16 写出
    c_ptrs = C + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# (a) 朴素版：一维 grid 按行优先直接映射到 tile 坐标
#     连续的 pid → 同一 pid_m 下的不同 pid_n → 一个 wave 是"一横条"：
#     A 的行块被复用，但每个 CTA 都要拉不同的 B 列块，B 的 L2 miss 多
# ---------------------------------------------------------------------------
@triton.jit
def matmul_naive_kernel(A, B, C, M, N, K,
                        sam, sak, sbk, sbn, scm, scn,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n          # 行优先展开：pid = pid_m * num_pid_n + pid_n
    pid_n = pid % num_pid_n
    _matmul_tile(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
                 pid_m, pid_n, BLOCK_M, BLOCK_N, BLOCK_K)


# ---------------------------------------------------------------------------
# (b) swizzle 版：同样的 tile 计算，只是把 pid → (pid_m, pid_n) 换成"方块化"映射
#     每 GROUP_M 行 tile 划为一组，组内按列优先编号 → 一个 wave 覆盖的是
#     接近方形的 tile 块，A 行块和 B 列块在 wave 内部就能互相命中 L2
# ---------------------------------------------------------------------------
@triton.jit
def matmul_swizzle_kernel(A, B, C, M, N, K,
                          sam, sak, sbk, sbn, scm, scn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_in_group = GROUP_M * num_pid_n      # 每组含 GROUP_M 行 × 全部列 的 tile
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP_M
    # 最后一组可能不满 GROUP_M 行，取真实行数
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    # 组内按列优先展开：同 pid_n 的 GROUP_M 个 tile 连续 → wave 内 B 列块复用
    pid_m = first_pid_m + (pid % num_in_group) % group_size_m
    pid_n = (pid % num_in_group) // group_size_m

    _matmul_tile(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
                 pid_m, pid_n, BLOCK_M, BLOCK_N, BLOCK_K)


# ---------------------------------------------------------------------------
# (c) autotune 版：swizzle kernel + 一组候选配置
#     key=['M','N','K']：每个新的 shape 组合首次调用时把所有 config
#     实测一遍，缓存最优；注意选 config 时留意 shared memory 预算
#     （smem/stage ≈ (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * dtype_size，
#       ×num_stages 不能超过 SM 的 smem 上限，超了的 config 会被 autotune 判负）
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=8, num_stages=3),   # ~72KB smem
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=8, num_stages=4),   # ~64KB smem
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=4, num_stages=4),   # 兜底小 tile
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotune_kernel(A, B, C, M, N, K,
                           sam, sak, sbk, sbn, scm, scn,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    # 与 (b) 完全相同的 swizzle 映射；autotune 只负责替我们挑 BLOCK/warps/stages
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_in_group) % group_size_m
    pid_n = (pid % num_in_group) // group_size_m
    _matmul_tile(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
                 pid_m, pid_n, BLOCK_M, BLOCK_N, BLOCK_K)


def triton_matmul(a, b, version="autotune"):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    strides = (a.stride(0), a.stride(1), b.stride(0), b.stride(1),
               c.stride(0), c.stride(1))
    if version == "naive":
        grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
        matmul_naive_kernel[grid](a, b, c, M, N, K, *strides,
                                  BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                                  num_warps=4, num_stages=3)
    elif version == "swizzle":
        grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
        matmul_swizzle_kernel[grid](a, b, c, M, N, K, *strides,
                                    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                                    GROUP_M=8, num_warps=8, num_stages=4)
    else:  # autotune：grid 必须是 META 的函数，因为 BLOCK_M/N 由 config 决定
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M'])
                             * triton.cdiv(N, META['BLOCK_N']),)
        matmul_autotune_kernel[grid](a, b, c, M, N, K, *strides)
    return c


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

    for n in [2048, 4096]:
        M = N = K = n
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        flops = 2 * M * N * K          # matmul 的 FLOPs = 2*M*N*K（乘+加各一次）

        ref = torch.matmul(a, b)
        outs = {v: triton_matmul(a, b, v) for v in ("naive", "swizzle", "autotune")}
        # Frobenius 相对误差：bf16 输出元素量级 ~sqrt(K)，比 max-abs 更稳
        errs = {v: ((o.float() - ref.float()).norm() / ref.float().norm()).item()
                for v, o in outs.items()}

        t_nv, f_nv = bench(lambda: triton_matmul(a, b, "naive"), flops)
        t_sw, f_sw = bench(lambda: triton_matmul(a, b, "swizzle"), flops)
        t_at, f_at = bench(lambda: triton_matmul(a, b, "autotune"), flops)
        t_tc, f_tc = bench(lambda: torch.matmul(a, b), flops)

        print(f"{M}×{N}×{K} | "
              f"err naive={errs['naive']:.2e} swizzle={errs['swizzle']:.2e} "
              f"autotune={errs['autotune']:.2e}")
        print(f"  naive    {t_nv:.3f}ms {f_nv:6.1f} TFLOPS | "
              f"swizzle  {t_sw:.3f}ms {f_sw:6.1f} TFLOPS")
        print(f"  autotune {t_at:.3f}ms {f_at:6.1f} TFLOPS | "
              f"torch    {t_tc:.3f}ms {f_tc:6.1f} TFLOPS")
