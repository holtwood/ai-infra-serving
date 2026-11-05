# M2 小结：Kernel 实战（Triton + CUDA C++，全部远端实测）

日期：2026-11-05 | RTX 4090 (sm_89, 128 SM, D2D 实测 939 GB/s, BF16 peak ~165 TFLOPS)
注：ncu 被宿主机禁用（见 m0-summary），下表性能均来自 CUDA events 实测。

## Triton（kernels/triton/）

### 01 fused softmax（vs naive 三段式 / torch.softmax）
| shape | naive | torch | triton | 结论 |
|---|---|---|---|---|
| 4096×4096 | 0.480ms/279GB/s | 0.143ms/936GB/s | **0.143ms/942GB/s** | 打平 torch |
| 8192×8192 | 2.261ms/237GB/s | 0.568ms/945GB/s | **0.570ms/942GB/s** | naive 慢 4x |
| 32768×1024 | 1.136ms/236GB/s | 0.290ms/924GB/s | **0.285ms/942GB/s** | 94% 带宽 |

### 02 LayerNorm / RMSNorm
| shape | torch LN | triton LN | torch RMSN* | triton RMSN |
|---|---|---|---|---|
| 4096×4096 | 0.154ms/1747GB/s | **0.143ms/1875GB/s** | 0.449ms/448GB/s | **0.143ms/1406GB/s** |
| 32768×1024 | 0.286ms/1878GB/s | **0.285ms/1883GB/s** | 0.996ms/404GB/s | **0.285ms/1414GB/s** |

*torch 无 fused RMSNorm（拆成 mul+pow+mean+rsqrt+mul 多个 kernel）→
**Triton 快 3~7x**。这是"为什么 LLM 框架全用 RMSNorm+融合 kernel"的直接证据。

### 03 matmul（naive / L2-swizzle / autotune vs cuBLAS）
| MNK | naive | swizzle | autotune | torch(cuBLAS) |
|---|---|---|---|---|
| 2048³ | 143.3 TF | 148.3 TF | 155.5 TF | 165.8 TF |
| 4096³ | 142.2 TF | 147.3 TF | **160.8 TF** | 148.9 TF |

autotune 版在 4096³ **反超 cuBLAS**；swizzle 稳定 +3~5%（L2 局部性）。

### 04 FlashAttention forward（vs SDPA=FA2）
| S×D | SDPA | triton | 达成率 |
|---|---|---|---|
| 1024×64 | 135.8 TF | 118.4 TF | 87% |
| 1024×128 | 140.5 TF | 82.0 TF | 58% |
| 2048×64 | 145.7 TF | 124.9 TF | 86% |
| 4096×64 | 155.0 TF | 134.6 TF | 87% |
| 4096×128 | 157.8 TF | 88.2 TF | 56% |

D=128 差距大：tutorial 版没有 warp-specialization/pipelining，
headdim=128 时寄存器压力吃紧。全部精度 PASS（err≤2e-3）。

## CUDA C++（kernels/cuda/，nvcc -O3 -arch=sm_89 -lineinfo）

### 01 vector_add / saxpy —— 打到 95% 带宽
block 128~1024 全部 ~950 GB/s / grid-stride 915 / saxpy 920
→ memory-bound 天花板；compute-sanitizer memcheck 0 errors（代价：慢 ~20x）

### 02 reduction 四版演进（n=16M floats）
| 版本 | ms | 有效 GB/s | 优化点 |
|---|---|---|---|
| v0 gmem 相邻配对 | 0.143 | 468 | 取模发散 + 多次全局读写 |
| v1 gmem 顺序寻址 | 0.086 | 779 | coalescing |
| v2 shared memory | 0.103 | 651 | smem 减少 gmem 往返 |
| **v3 smem+warp shuffle** | **0.053** | **1259** | 免 syncthreads 末段归约 |

> v3 "有效带宽"超理论 1008 GB/s：n=64MB 可部分驻留 72MB L2，
> 多次迭代命中率抬高——本身是"L2 如何影响 benchmark"的好案例。

ptxas -v：全部 kernel 12~13 寄存器、0 spill、1KB smem。

### 03 GEMM（n=2048）
| 版本 | GFLOPS | vs cuBLAS |
|---|---|---|
| naive | 5096 | 9% |
| smem tile 32×32 | 6724 | 12% |
| cuBLAS sgemm | **54571** | 100% |

只加 smem tiling 到不了 cuBLAS 的 1/8——差在 register tiling、
double-buffering、向量化 ld、tensor core。教学结论：
**"会用 tensor core 的库"和"写得正确的 kernel"之间隔着整个优化栈。**

## M2 面试弹药

- "softmax/LN/RMSNorm 为什么值得融合"：RMSNorm 3~7x 实测
- "Triton vs cuBLAS"：matmul 97%~反超；flash-attn 55~87%
- "memory-bound 怎么证"：vector_add 95% 带宽、softmax 94%
- "reduction 优化路径"：468→1259 GB/s 的四个台阶
- "没有 ncu 怎么 profile"：events + ptxas -v + sanitizer + L2 分析
