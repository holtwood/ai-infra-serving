// M2-CUDA 第 2 课：sum reduction 四级演进 ⭐（CUDA 手写题出现频率最高）
//
// 面试考点：
// - 归约演进路线（每版消除了什么瓶颈要能脱口而出）：
//   v0 全局内存相邻配对：模运算判活 + warp 内发散 + 非合并访存，
//        且 log2(BLOCK) 轮全部走 gmem 往返（同一访存模式放进 smem 就是
//        教科书级 bank conflict 案例）
//   v1 顺序寻址：活跃线程变成连续前缀，发散/访存合并/取模开销一起解决
//   v2 shared memory：数据只从 gmem 读一遍，树形归约全程在 smem 完成
//   v3 warp shuffle：最后 32 个元素用 __shfl_down_sync 在寄存器里归约，
//        免 __syncthreads、免 smem 访问
// - 跨 block 汇总：每块累出一个部分和 → atomicAdd 到全局标量；
//   追问点：部分和太多时的替代方案 = 把部分和写数组、再起一个小 kernel 归约
// - 为什么最终版应逼近显存带宽：归约的必要访存量就是"读一遍输入"，
//   DRAM throughput 是唯一瓶颈（ncu 里看 Memory Workload Analysis）
//
// 编译运行：nvcc -O2 -arch=sm_89 02_reduction.cu -o 02_reduction && ./02_reduction
// ncu 观察：ncu --set full ./02_reduction（对比四版 dram__bytes / warp stall）
#include <cstdio>
#include <cmath>
#include <vector>
#include "bench_utils.cuh"

constexpr int N = 1 << 24;   // 16M float = 64MB
constexpr int BLOCK = 256;   // 归约常用 128/256/512；256 = 8 个 warp
constexpr int GRID = 2048;   // 每 block 顺序处理 N/(BLOCK*GRID)=32 个 chunk，
                             // 每块只发一次 atomicAdd（2048 个部分和，开销可忽略）

// 教学版统一要求 n % BLOCK == 0（真实 kernel 的边界 mask 是常见追问）
static_assert(N % BLOCK == 0, "N must be divisible by BLOCK");

// ---------------------------------------------------------------------------
// v0 朴素相邻配对（全局内存，in-place —— 会破坏输入数据！）
// 每轮 stride 翻倍，只有 tid % (2*stride) == 0 的线程干活：
//   * 取模本身贵，且活跃线程在 warp 内跳着分布（stride 小时一个 warp 里
//     只有 32/(2*stride) 个 lane 干活）→ 教科书级 warp divergence
//   * 访问地址间隔 2*stride → gmem 上非合并访存（uncoalesced）；
//     同样的模式放进 shared memory 就是 bank conflict 的经典案例
//   * log2(BLOCK)=8 轮全部读写全局内存，高延迟完全暴露
// ---------------------------------------------------------------------------
__global__ void reduce_v0_gmem_pairwise(float* buf, float* out, int n) {
    int tid = threadIdx.x;
    float acc = 0.f;                        // 每块一个部分和，最后统一 atomicAdd
    for (int base = blockIdx.x * BLOCK; base < n; base += gridDim.x * BLOCK) {
        for (int stride = 1; stride < BLOCK; stride <<= 1) {
            if (tid % (2 * stride) == 0)
                buf[base + tid] += buf[base + tid + stride];
            __syncthreads();                // 每轮必须等全块写完才能进下一轮
        }
        if (tid == 0) acc += buf[base];
        __syncthreads();                    // 保险：等 tid0 读完再进下一 chunk
    }
    if (tid == 0) atomicAdd(out, acc);
}

// ---------------------------------------------------------------------------
// v1 顺序寻址（仍是全局内存 in-place）
// 活跃线程改为连续前缀 tid < stride：
//   * 消灭取模；warp 要么全活跃要么全不活跃 → 发散基本消除
//   * 每轮读写都是一段连续地址 → 访存合并（coalesced）
//   * 若换成 smem 存储，同一改动同时消灭 bank conflict
// 剩下的瓶颈：树形归约的每一轮仍要全局内存往返。
// ---------------------------------------------------------------------------
__global__ void reduce_v1_gmem_sequential(float* buf, float* out, int n) {
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int base = blockIdx.x * BLOCK; base < n; base += gridDim.x * BLOCK) {
        for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                buf[base + tid] += buf[base + tid + stride];
            __syncthreads();
        }
        if (tid == 0) acc += buf[base];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, acc);
}

// ---------------------------------------------------------------------------
// v2 shared memory 版：把本块数据搬进 smem，log2 轮全部在 smem 内完成
//   * gmem 只被读一遍——这是归约的必要访存量下限，带宽花在刀刃上
//   * smem 延迟 ~20-30 cycle vs gmem ~400-800 cycle，树形阶段延迟大降
//   * 附带好处：不再原地改写输入（v0/v1 需要 scratch 副本的原因）
// ---------------------------------------------------------------------------
__global__ void reduce_v2_smem(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int base = blockIdx.x * BLOCK; base < n; base += gridDim.x * BLOCK) {
        sdata[tid] = in[base + tid];
        __syncthreads();
        for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }
        if (tid == 0) acc += sdata[0];
        __syncthreads();                    // 保险：读完才能覆盖下一 chunk
    }
    if (tid == 0) atomicAdd(out, acc);
}

// ---------------------------------------------------------------------------
// v3 warp shuffle 版：smem 树形归约只做到 32 个元素，
// 最后 32 个交给 warp 内 __shfl_down_sync 寄存器归约：
//   * warp 内 32 个 lane 天然锁步执行 → 免 __syncthreads
//   * 寄存器间直接传值 → 免 smem 读写
// 省掉的是 log2(32)=5 轮 smem 往返 + 块级同步；到这里 kernel 已完全
// memory-bound，再优化只剩"每线程预读多元素/向量化加载"这类访存侧手段
// ---------------------------------------------------------------------------
__global__ void reduce_v3_shuffle(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int base = blockIdx.x * BLOCK; base < n; base += gridDim.x * BLOCK) {
        sdata[tid] = in[base + tid];
        __syncthreads();
        for (int stride = BLOCK / 2; stride >= 32; stride >>= 1) {   // 归约到 32 个为止
            if (tid < stride)
                sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }
        // 此刻 sdata[0..31] 就是本 chunk 的 32 个部分和；只有 warp0 有活干，
        // __shfl_down_sync(mask, v, off)：lane i 拿到 lane i+off 的 v，
        // off 从 16 折半到 1，5 步后 lane0 持有 32 个值的总和
        if (tid < 32) {
            float val = sdata[tid];
            for (int off = 16; off > 0; off >>= 1)
                val += __shfl_down_sync(0xffffffff, val, off);
            if (tid == 0) acc += val;
        }
        __syncthreads();                    // 保险：warp0 读完再覆盖 sdata
    }
    if (tid == 0) atomicAdd(out, acc);
}

// ---------------------------------------------------------------------------
// 归约专用计时：每轮 kernel 之前有"清零输出 / 恢复被 v0/v1 破坏的输入副本"
// 等准备工作（不该计入耗时），所以每轮用一对 event 只包住 kernel 本身。
// ---------------------------------------------------------------------------
template <typename Prep, typename Launch>
float bench_reduce(Prep&& prep, Launch&& launch, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) { prep(); launch(); }
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    float total = 0.f;
    for (int i = 0; i < iters; ++i) {
        prep();
        t.start();
        launch();
        t.stop();
        total += t.elapsed_ms();
    }
    return total / iters;
}

// 跑一遍取回输出标量（校验用）
template <typename Prep, typename Launch>
float run_and_get(Prep&& prep, Launch&& launch, DeviceBuffer<float>& out) {
    prep();
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    float v;
    out.copy_to_host(&v);
    return v;
}

int main() {
    print_device_info();
    printf("n = %d floats (%.0f MB), block = %d, grid = %d（每 block 处理 %d chunk）\n\n",
           N, N * sizeof(float) / 1e6, BLOCK, GRID, N / BLOCK / GRID);

    std::vector<float> h(N);
    fill_uniform(h.data(), N, 7u, 0.f, 1.f);       // [0,1)：sum≈N/2，远离 0
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) cpu_sum += h[i];   // double 参考值

    DeviceBuffer<float> d_in(N), d_work(N), d_out(1);
    d_in.copy_from_host(h.data());

    const double eff_bytes = (double)N * sizeof(float);  // 必要访存量：读一遍输入

    // 每个版本一组 (prep, launch)：prep 不计入计时
    auto prep_common = [&] {                            // v2/v3：只清输出
        CUDA_CHECK(cudaMemsetAsync(d_out.get(), 0, sizeof(float)));
    };
    auto prep_with_copy = [&] {                         // v0/v1：先恢复输入副本
        d_work.copy_from_device(d_in.get());            //   in-place 会破坏数据
        CUDA_CHECK(cudaMemsetAsync(d_out.get(), 0, sizeof(float)));
    };
    // 四版的 launch lambda 先统一定义，校验和计时复用同一份
    auto launch_v0 = [&] {
        reduce_v0_gmem_pairwise<<<GRID, BLOCK>>>(d_work.get(), d_out.get(), N);
    };
    auto launch_v1 = [&] {
        reduce_v1_gmem_sequential<<<GRID, BLOCK>>>(d_work.get(), d_out.get(), N);
    };
    auto launch_v2 = [&] {
        reduce_v2_smem<<<GRID, BLOCK>>>(d_in.get(), d_out.get(), N);
    };
    auto launch_v3 = [&] {
        reduce_v3_shuffle<<<GRID, BLOCK>>>(d_in.get(), d_out.get(), N);
    };

    auto check = [&](const char* tag, float got) {
        double rel = fabs(got - cpu_sum) / fabs(cpu_sum);
        printf("%-24s 校验: sum=%.1f cpu=%.1f rel_err=%.2e %s\n",
               tag, (double)got, cpu_sum, rel, rel < 2e-4 ? "PASS" : "FAIL");
    };

    // -------------------- 正确性校验（CPU double 参考） --------------------
    check("v0 gmem 相邻配对", run_and_get(prep_with_copy, launch_v0, d_out));
    check("v1 gmem 顺序寻址", run_and_get(prep_with_copy, launch_v1, d_out));
    check("v2 shared memory", run_and_get(prep_common, launch_v2, d_out));
    check("v3 warp shuffle", run_and_get(prep_common, launch_v3, d_out));
    printf("\n");

    // -------------------- 计时 --------------------
    // GB/s(有效) = N*4B / 耗时；bytes/ms/1e6 即 bytes/(ms*1e-3)/1e9
    print_header("GB/s(有效)");
    float ms;
    ms = bench_reduce(prep_with_copy, launch_v0);
    print_row("v0 gmem 相邻配对(模+发散)", ms, eff_bytes / ms / 1e6);
    ms = bench_reduce(prep_with_copy, launch_v1);
    print_row("v1 gmem 顺序寻址", ms, eff_bytes / ms / 1e6);
    ms = bench_reduce(prep_common, launch_v2);
    print_row("v2 shared memory", ms, eff_bytes / ms / 1e6);
    ms = bench_reduce(prep_common, launch_v3);
    print_row("v3 smem + warp shuffle", ms, eff_bytes / ms / 1e6);

    printf("\n说明：GB/s 按必要访存量 N*4B 折算（有效带宽）。v0/v1 实际 DRAM 流量\n");
    printf("      约为此值的 ~%d 倍（log2(%d) 轮读写），可用\n", 8, BLOCK);
    printf("        ncu --metrics dram__bytes.sum ./02_reduction\n");
    printf("      验证；v2/v3 的 GB/s 应接近理论带宽。追问：不用 atomicAdd 就\n");
    printf("      把 %d 个部分和写进数组，再起一个小 kernel（或 CPU）收尾。\n", GRID);
    return 0;
}
