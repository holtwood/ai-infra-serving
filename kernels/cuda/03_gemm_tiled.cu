// M2-CUDA 第 3 课：GEMM naive → shared memory tiling（对照 cuBLAS）
//
// 面试考点：
// - naive GEMM 的算术强度：每个线程读 2n 个 float、做 n 次 FMA，
//   AI ≈ 0.25 FLOP/B → 1TB/s 带宽下理论上限只有 ~250 GFLOPS（memory-bound）
// - tiling 为什么省全局访存：32×32 tile 让 A/B 每个元素从"被读 n 次"降为
//   "被读 n/32 次"，gmem 流量缩 32× → AI 提升 32×；
//   shared memory 本质就是"程序员手动管理的 cache"
// - 为什么还差 cuBLAS 一个数量级：register tiling（每线程算 8×8 子块）、
//   double buffering、float4 向量加载、tensor core(MMA)——本课只到 smem 层，
//   但演进路线要会背：naive → smem tile → register tile → tensor core
// - cuBLAS 是列主序！行主序 C=A·B ⇔ 列主序算 B·A（(A·B)^T = B^T·A^T），
//   所以调用时交换 A/B 两个指针即可
//
// 编译运行：
//   nvcc -O2 -arch=sm_89 03_gemm_tiled.cu -o 03_gemm_tiled -lcublas && ./03_gemm_tiled
#include <cstdio>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include "bench_utils.cuh"

// cuBLAS 有自己的错误类型 cublasStatus_t，单独一个检查宏
#define CUBLAS_CHECK(call)                                           \
    do {                                                             \
        cublasStatus_t _st = (call);                                 \
        if (_st != CUBLAS_STATUS_SUCCESS) {                          \
            fprintf(stderr, "[CUBLAS_ERROR] %s:%d  %s -> %d\n",      \
                    __FILE__, __LINE__, #call, (int)_st);            \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

constexpr int TILE = 32;   // 32×32 tile：一个 block 1024 线程，smem 用量 2*32*32*4=8KB

// ---------------------------------------------------------------------------
// naive：一线程算一个 C 元素，沿 k 扫整行 A、整列 B。
// A[row,:] 会被同一行的 32 个线程各读一遍（好在 L2/L1 能兜一部分），
// B[:,col] 每次访问都跨 n 个 float → 完全不合并，是主要瓶颈。
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0.f;
        for (int k = 0; k < n; ++k)
            acc += A[row * n + k] * B[k * n + col];
        C[row * n + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// smem tiling：block 负责 C 的一个 32×32 子块；
// k 维分 tile 前进，每步全 block 协作把一对 32×32 的 A/B tile 搬进 smem，
// 然后 32 次 FMA 全部命中 smem（免 gmem 往返）。
// 访存细节：As[ty][k] 全 warp 读同一地址（broadcast，无 bank conflict），
//          Bs[k][tx] warp 内地址连续（每个 lane 一个 bank，无冲突）。
// 教学版假设 n % TILE == 0（真实 kernel 要做边界 mask——常见追问）。
// ---------------------------------------------------------------------------
__global__ void gemm_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.f;

    for (int t = 0; t < n / TILE; ++t) {
        As[ty][tx] = A[row * n + t * TILE + tx];   // 全员协作搬一块 A tile
        Bs[ty][tx] = B[(t * TILE + ty) * n + col]; // 和一块 B tile
        __syncthreads();                           // 搬完才能用
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();                           // 用完才能覆盖下一 tile
    }
    C[row * n + col] = acc;
}

// host 端三重循环参考（只在小 n 上跑）
static void gemm_cpu_ref(const std::vector<float>& A, const std::vector<float>& B,
                         std::vector<float>& C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int k = 0; k < n; ++k) acc += A[i * n + k] * B[k * n + j];
            C[i * n + j] = acc;
        }
}

// 两组输出求最大相对误差
static double max_rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        e = fmax(e, fabs(a[i] - b[i]) / (fabs(b[i]) + 1e-3));
    return e;
}

int main() {
    print_device_info();

    // -------------------- 小规模：对 CPU 参考做正确性校验 --------------------
    {
        const int n = 256;                        // 256 % 32 == 0
        std::vector<float> hA(n * n), hB(n * n), hC(n * n), hRef(n * n);
        fill_uniform(hA.data(), n * n, 3u);
        fill_uniform(hB.data(), n * n, 4u);
        DeviceBuffer<float> dA(n * n), dB(n * n), dC(n * n);
        dA.copy_from_host(hA.data());
        dB.copy_from_host(hB.data());

        gemm_cpu_ref(hA, hB, hRef, n);

        dim3 nb(16, 16), ng((n + 15) / 16, (n + 15) / 16);
        gemm_naive<<<ng, nb>>>(dA.get(), dB.get(), dC.get(), n);
        dC.copy_to_host(hC.data());
        double e1 = max_rel_err(hC, hRef);
        printf("n=%d naive  校验: max_rel_err=%.2e %s\n", n, e1,
               e1 < 1e-4 ? "PASS" : "FAIL");

        dim3 tb(TILE, TILE), tg(n / TILE, n / TILE);
        gemm_tiled<<<tg, tb>>>(dA.get(), dB.get(), dC.get(), n);
        dC.copy_to_host(hC.data());
        double e2 = max_rel_err(hC, hRef);
        printf("n=%d tiled  校验: max_rel_err=%.2e %s\n\n", n, e2,
               e2 < 1e-4 ? "PASS" : "FAIL");
    }

    // -------------------- 性能对比：n = 2048 --------------------
    const int n = 2048;
    std::vector<float> hA(n * n), hB(n * n), hC(n * n), hRef(n * n);
    fill_uniform(hA.data(), n * n, 5u);
    fill_uniform(hB.data(), n * n, 6u);
    DeviceBuffer<float> dA(n * n), dB(n * n), dC(n * n), dBlas(n * n);
    dA.copy_from_host(hA.data());
    dB.copy_from_host(hB.data());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const double flops = 2.0 * n * n * n;         // C=A·B 方阵：2n³ FLOP
    print_header("GFLOPS");
    float ms;

    {   // naive：16×16 block（它最慢，少跑几轮计时省时间）
        dim3 nb(16, 16), ng(n / 16, n / 16);
        ms = bench_ms([&] {
            gemm_naive<<<ng, nb>>>(dA.get(), dB.get(), dC.get(), n);
        }, 3, 20);
        print_row("gemm naive", ms, flops / ms / 1e6);
    }
    {   // smem tiling：32×32 block = 1024 线程
        dim3 tb(TILE, TILE), tg(n / TILE, n / TILE);
        ms = bench_ms([&] {
            gemm_tiled<<<tg, tb>>>(dA.get(), dB.get(), dC.get(), n);
        });
        print_row("gemm smem tile 32x32", ms, flops / ms / 1e6);
    }
    {   // cuBLAS 对照。
        // 行主序 C = A·B 等价于列主序 C^T = B^T·A^T，而"行主序存的 M"
        // 在列主序眼里就是 M^T——所以直接把 (B, A) 传给 sgemm 即可。
        float alpha = 1.f, beta = 0.f;
        auto sgemm = [&] {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, n, n, &alpha,
                                     dB.get(), n, dA.get(), n,
                                     &beta, dBlas.get(), n));
        };
        ms = bench_ms(sgemm);
        print_row("cuBLAS sgemm", ms, flops / ms / 1e6);
    }

    // 以 cuBLAS 输出为参考，复核 n=2048 下手写 kernel 的数值
    dBlas.copy_to_host(hRef.data());
    dim3 tb(TILE, TILE), tg(n / TILE, n / TILE);
    gemm_tiled<<<tg, tb>>>(dA.get(), dB.get(), dC.get(), n);
    dC.copy_to_host(hC.data());
    double e3 = max_rel_err(hC, hRef);
    printf("\ntiled vs cublas (n=%d): max_rel_err=%.2e %s\n", n, e3,
           e3 < 1e-3 ? "PASS" : "FAIL");

    CUBLAS_CHECK(cublasDestroy(handle));

    printf("\n说明：GFLOPS = 2n^3 / 耗时。追问可提：cublasSetMathMode(\n");
    printf("      CUBLAS_TF32_TENSOR_OP_MATH) 能让 fp32 GEMM 走 tensor core，\n");
    printf("      4090 上再快数倍——精度换速度，和量化是同一类 trade-off。\n");
    return 0;
}
