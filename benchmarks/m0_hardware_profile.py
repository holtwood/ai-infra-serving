"""M0 硬件画像：把机器"能力上限"变成实测数字，后续所有实验以此为基线。

测量项：
1. GPU 基本信息（名称、显存、SM 数、时钟）
2. H2D / D2H / D2D 带宽（pinned vs pageable 对比 —— pinned 快多少是面试题）
3. BF16 GEMM 实测 TFLOPS vs 理论峰值（4090 BF16 tensor-core dense ≈ 165 TFLOPS）
4. FP32 GEMM（4090 FP32 CUDA-core ≈ 82.6 TFLOPS / TF32 dense ≈ 82.6）
5. 显存带宽实测（memcpy D2D 大 buffer，4090 GDDR6X 理论 ~1.0 TB/s）

运行：python benchmarks/m0_hardware_profile.py
"""
import time

import torch


def bench(fn, warmup=5, iters=20):
    """CUDA event 计时 —— 比 time.perf_counter 准（GPU 是异步的）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    dev = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(dev)
    print("=" * 64)
    print("GPU 基本信息")
    print("=" * 64)
    print(f"name           : {p.name}")
    print(f"total_memory   : {p.total_memory / 2**30:.1f} GiB")
    print(f"SM count       : {p.multi_processor_count}")
    print(f"major.minor    : {p.major}.{p.minor}")
    print(f"torch          : {torch.__version__}  cuda={torch.version.cuda}")
    try:
        print(f"clock          : {torch.cuda.clock_rate(dev)/1e6:.2f} GHz (current)")
    except Exception:
        pass

    print("\n" + "=" * 64)
    print("PCIe 带宽：H2D / D2H（pinned vs pageable）")
    print("=" * 64)
    n = 256 * 2**20  # 256 MiB
    dst_gpu = torch.empty(n, dtype=torch.uint8, device="cuda")

    for pinned in (False, True):
        src = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
        ms = bench(lambda: dst_gpu.copy_(src, non_blocking=True), iters=10)
        print(f"H2D pinned={pinned!s:5} : {n / 2**20:.0f} MiB in {ms:.2f} ms  →  {n/ms/1e6:.1f} GB/s")
        dst_cpu = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
        ms = bench(lambda: dst_cpu.copy_(dst_gpu, non_blocking=True), iters=10)
        print(f"D2H pinned={pinned!s:5} : {n / 2**20:.0f} MiB in {ms:.2f} ms  →  {n/ms/1e6:.1f} GB/s")

    print("\n" + "=" * 64)
    print("显存带宽：D2D copy（理论 ~1 TB/s）")
    print("=" * 64)
    a = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a), iters=20)
    # copy 读+写各一遍 → 实际搬运 2×
    print(f"D2D 512MiB        : {ms:.2f} ms  →  {2*a.numel()/ms/1e6:.0f} GB/s")

    print("\n" + "=" * 64)
    print("GEMM 实测 TFLOPS（MNK=8192）")
    print("=" * 64)
    M = N = K = 8192
    flops = 2.0 * M * N * K
    for dtype, name in ((torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")):
        x = torch.randn(M, K, dtype=dtype, device="cuda")
        y = torch.randn(K, N, dtype=dtype, device="cuda")
        ms = bench(lambda: x @ y, iters=10)
        print(f"{name:5} @ 8192³ : {ms:.2f} ms  →  {flops/ms/1e9:.1f} TFLOPS")

    print("\n提示：bf16 实测 / 理论 165 TFLOPS ≥85% 即正常；FP32 走 CUDA-core ≈82.6 或 TF32。")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time()-t0:.1f}s")
