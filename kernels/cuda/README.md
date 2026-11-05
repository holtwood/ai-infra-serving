# kernels/cuda —— CUDA C++ 手写 kernel（面试教学版）

配套 `../triton/`：Triton 偏"快速写出可用 kernel"，这里偏"面试官追问的底层细节"
（block/smem/warp/访存合并/ncu 指标）。每个文件都自带 CPU/cuBLAS 数值校验 + benchmark。

| 文件 | 内容 | 一句话 |
|---|---|---|
| `bench_utils.cuh` | 公共头文件 | CUDA_CHECK 宏、cudaEvent 计时器（RAII）、cudaMalloc 的 RAII 封装 `DeviceBuffer<T>`、表格打印、理论带宽计算 |
| `01_vector_add.cu` | vector add + saxpy | grid-stride loop；block size 128/256/512/1024 扫描；实测 GB/s vs 理论带宽 |
| `02_reduction.cu` | ⭐ sum reduction 四级演进 | v0 gmem 相邻配对 → v1 gmem 顺序寻址 → v2 shared memory → v3 warp shuffle；块内归约 + atomicAdd 汇总 |
| `03_gemm_tiled.cu` | GEMM naive vs smem tiling vs cuBLAS | 32×32 tile，fp32；列主序 trick 调 cublasSgemm；各自 GFLOPS |

## 编译运行

### 方式一：nvcc 直接编译（推荐——单文件、零配置）

```bash
cd kernels/cuda
nvcc -O2 -arch=sm_89 -lineinfo 01_vector_add.cu -o 01_vector_add
nvcc -O2 -arch=sm_89 -lineinfo 02_reduction.cu  -o 02_reduction
nvcc -O2 -arch=sm_89 -lineinfo 03_gemm_tiled.cu -o 03_gemm_tiled -lcublas

./01_vector_add
./02_reduction
./03_gemm_tiled
```

说明：
- `-arch=sm_89`：RTX 4090（Ada）。其它卡查 `nvidia-smi --query-gpu=compute_cap --format=csv`
- `-lineinfo`：给 ncu 的源码关联用，不加也行，不影响性能
- `-lcublas` 只有 03 需要

### 方式二：CMake

```bash
cd kernels/cuda
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/01_vector_add && ./build/02_reduction && ./build/03_gemm_tiled
```

## ncu / nsys 速查

```bash
# 最常用：全量分析（信息量最大，跑一遍看每个 kernel 的瓶颈报告）
ncu --set full ./02_reduction

# 只看显存侧：DRAM 吞吐、L1/L2 命中、每 kernel 实际读写字节数
ncu --section MemoryWorkloadAnalysis ./02_reduction

# 只看计算侧：SM busy、warp 占用、指令分布
ncu --section ComputeWorkloadAnalysis ./03_gemm_tiled

# 精确抓某个 kernel（一个程序里有多个 kernel 时必用）
ncu --kernel-name regex:"reduce_v" ./02_reduction
ncu --kernel-name gemm_tiled --launch-count 3 ./03_gemm_tiled

# 直接点名指标：验证"归约的必要访存量就是读一遍输入"
ncu --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed ./02_reduction

# shared memory bank conflict / warp stall 原因
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./02_reduction
ncu --section SchedulerStats --section WarpStateStats ./02_reduction
# 指标名各版本有差异，先查再点：ncu --query-metrics | grep stall

# 时间线级（多 kernel / memcpy 交叠看这里，不是 ncu 是 nsys）
nsys profile -o reduction_prof ./02_reduction && nsys stats reduction_prof.nsys-rep
```

ncu 里重点看的指标（面试会问"你 profile 看什么"）：

| 指标 | 含义 |
|---|---|
| `DRAM Throughput` / `dram__bytes` | memory-bound kernel 的主指标：实际流量 vs 理论流量 |
| `Compute (SM) Throughput` | SM 利用率；和 DRAM 一起看判断 bound 类型 |
| `Achieved Occupancy` | 实际常驻 warp / 上限；被寄存器/smem 用量压低 |
| `Warp Stall Reasons` | warp 卡在等什么（long scoreboard = 等 gmem；barrier = 等 __syncthreads） |
| `L2 Hit Rate` | tiling/fusion 有没有效 |
| bank conflict 计数 | reduction/GEMM 的 smem 访问是否设计对了 |

## 面试考点对照表

### 01_vector_add
- 怎么判断 kernel 是 memory-bound 还是 compute-bound？（算 AI，对照 roofline）
- grid-stride loop 为什么比"一线程一元素"更通用？（线程数解耦、occupancy 固定可调）
- block size 怎么选？为什么 256 是常见甜点？（occupancy 与资源占用平衡；128 以下调度粒度太碎，1024 占满 SM 但无调度余地）
- 实测带宽到理论值的百分之多少算"优化到位"？（~80%+）

### 02_reduction ⭐
- 手写 sum reduction——四级演进能脱口而出：相邻配对（模运算 + warp 内发散 + 非合并访存；同模式进 smem = bank conflict）→ 顺序寻址 → smem → warp shuffle
- `__shfl_down_sync(0xffffffff, v, off)` 语义？为什么最后 32 个元素免 __syncthreads？（warp 内锁步执行）
- 为什么每轮都要 `__syncthreads()`？写在 `if` 里面会发生什么？（死锁/未定义行为——同步原语必须全 block 到达）
- 跨 block 怎么汇总？atomicAdd vs 二级归约 kernel 的取舍（部分和数量、原子操作开销）
- 归约的访存量下限是多少？为什么最终版逼近显存带宽？
- 继续优化还能做什么？（每线程预加 2~4 元素再进 smem、block 展开、`#pragma unroll`、float4 加载）

### 03_gemm_tiled
- GEMM 优化演进路线：naive → smem tiling → register tiling → tensor core（每步消除什么）
- 32×32 tile 把全局访存降了多少倍？为什么？（每元素复用 32 次：读 n 次 → n/32 次）
- smem 上 `__syncthreads()` 为什么出现两次？（搬完才能用 / 用完才能覆盖）
- bank conflict：为什么 `As[ty][k]` 和 `Bs[k][tx]` 无冲突？（broadcast + 连续 bank）
- 为什么手写还差 cuBLAS 很远？（register tiling、double buffering、向量化、tensor core；以及分块形状 autotune）
- cuBLAS 列主序怎么调行主序矩阵？（交换 A/B 指针的转置 trick）
- TF32/tensor core 怎么开？（`cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)`，精度换速度）

### bench_utils.cuh（工程素养题）
- kernel 计时为什么用 cudaEvent 而不是 std::chrono？（GPU 时间线 vs CPU 墙钟 + launch 异步性）
- RAII 封装显存的好处？（异常安全、不漏 cudaFree；面试官爱看 C++ 功底）
