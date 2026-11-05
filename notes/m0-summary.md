# M0 小结：环境与硬件画像（实测）

日期：2026-11-05 | 机器：RTX 4090 48GB 改版 / CUDA 12.8 / torch 2.8.0+cu128 / vllm 0.11.0

## 硬件画像实测（benchmarks/m0_hardware_profile.py）

| 项目 | 实测 | 理论/参考 | 结论 |
|---|---|---|---|
| 显存 | 47.5 GiB | 48GB 标称 | 改版卡可用 |
| SM 数 | 128 | — | sm_89 Ada |
| H2D pinned | 25.1 GB/s | PCIe4 x16 ≈ 32 GB/s | ~78%，正常 |
| H2D pageable | 22.0 GB/s | — | pinned 快 ~14% |
| D2H pinned | 26.3 GB/s | — | pageable 只有 13.6 → pinned 快近 2x |
| D2D copy | 939 GB/s | GDDR6X ≈ 1008 GB/s | 93%，物理上限附近 |
| BF16 GEMM 8192³ | **161.8 TFLOPS** | tensor-core dense ≈165 | **98%** |
| FP16 GEMM | 150.2 TFLOPS | ≈165 | 91% |
| FP32 GEMM | 52.8 TFLOPS | FP32 cuda-core ≈82.6 | 64%（或走 TF32 部分） |

**面试点**：D2H pageable(13.6) vs pinned(26.3) 差 ~2x——
pageable 要走 staging buffer 拷贝；pinned 可 DMA 直写 + 真异步。
KV offload / CPU offload 场景必须用 pinned + non_blocking。

## 环境踩坑实录（详见 notes/debug-log.md）

1. **`ERR_NVGPUCTRPERM`**：宿主机 `RmProfilingAdminOnly=1`，容器内
   无性能计数器权限 → **ncu 不可用**。替代方案：CUDA events 计时 +
   `nvcc --ptxas-options=-v`（寄存器/smem/spill）+ compute-sanitizer
   （可用，0 errors；副作用是 kernel 慢 ~20x）+ torch.profiler。
2. **Error 804 forward compatibility**：镜像的
   `/etc/ld.so.conf.d/00-compat-*.conf` 把 `/usr/local/cuda-12.8/compat`
   排在 ldconfig 首位 → torch 加载 compat 版 libcuda.so.570 而非真实
   驱动 550 → 报 "forward compatibility on non supported HW"。
   **修复**：重命名该 conf + `ldconfig`，让 libcuda 解析回
   `/usr/lib/x86_64-linux-gnu/libcuda.so.550.107.02`。
3. **vllm 0.33 默认拉 torch 2.14+cu13**：与驱动 550/CUDA12.8 不匹配
   → 锁定 `torch==2.8.0 + vllm==0.11.0`（官方配套）。
4. **网络**：hf-mirror 856B/s 不可用；阿里 PyPI 源 ~6MB/s 最优；
   `hf`/pip 均被环境中损坏的 brotli/zstd 解码库搞崩
   （`process() takes no keyword arguments`）→ 大文件一律
   `curl -L -C -` 断点续传。

## 待办/风险

- ncu 不可用 → M2 的"ncu 分析"用 ptxas+events+sanitizer 组合替代，
  notes 里如实说明；若后面需要真 ncu 数据，可换能开 perf counter 的实例。
- 7B 模型只下了 ~2GB（共享盘有 4B/27B/32B 现成模型兜底）。
