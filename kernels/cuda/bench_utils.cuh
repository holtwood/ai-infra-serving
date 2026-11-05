// bench_utils.cuh —— CUDA 教学公共工具
//
// 面试考点：
// - CUDA_CHECK 宏：面试手写 kernel 时主动做错误检查是加分项
//   （cudaGetErrorString + __FILE__/__LINE__ 定位错误现场）
// - RAII 封装 cudaMalloc / cudaEvent：体现 C++ 工程素养，面试官爱看；
//   析构自动释放，中途 return / 抛异常也不会泄漏显存
// - cudaEvent 计时：GPU kernel 计时的标准做法——std::chrono 测的是 CPU 墙钟
//   时间（含 launch 开销），event 打在 GPU 时间线上才反映 kernel 真实耗时
#pragma once

#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// 错误检查宏：任何返回 cudaError_t 的调用都该包一层
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA_ERROR] %s:%d  %s -> %s\n", __FILE__,       \
                    __LINE__, #call, cudaGetErrorString(_err));               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// GPU 计时器（RAII）：构造时建一对 event，析构时自动销毁
// 用法：GpuTimer t; t.start(); kernel<<<...>>>(); t.stop(); t.elapsed_ms();
// ---------------------------------------------------------------------------
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {                                  // 析构不 CHECK：失败也无从补救
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;            // event 是独占资源，禁止拷贝
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
    void stop(cudaStream_t s = 0)  { CUDA_CHECK(cudaEventRecord(stop_, s)); }

    // 阻塞直到 stop event 完成，返回 start → stop 之间的毫秒数
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// ---------------------------------------------------------------------------
// 显存 RAII 封装：DeviceBuffer<T> 构造即 cudaMalloc，析构 cudaFree
// ---------------------------------------------------------------------------
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) : n_(n) {
        CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return n_; }

    void copy_from_host(const T* h) {
        CUDA_CHECK(cudaMemcpy(ptr_, h, n_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_to_host(T* h) const {
        CUDA_CHECK(cudaMemcpy(h, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
    // D2D 拷贝：恢复被 kernel 原地改写（破坏）的输入数据时用
    void copy_from_device(const T* d) {
        CUDA_CHECK(cudaMemcpy(ptr_, d, n_ * sizeof(T), cudaMemcpyDeviceToDevice));
    }

private:
    T* ptr_ = nullptr;
    size_t n_;
};

// ---------------------------------------------------------------------------
// 常用计时入口：warmup 后用一对 event 包住 iters 次调用取平均
// 注意：f 里若含"每轮的准备工作"（如重置输出 buffer、恢复输入），不应计入
//       耗时——那种情况改用 GpuTimer 手动按轮计时（见 02_reduction.cu 的
//       bench_reduce）
// ---------------------------------------------------------------------------
template <typename F>
float bench_ms(F&& f, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) f();          // warmup：预热时钟 / L2 / TLB
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) f();
    t.stop();
    return t.elapsed_ms() / iters;
}

// ---------------------------------------------------------------------------
// 设备信息小工具
// ---------------------------------------------------------------------------
inline int get_sm_count() {
    int dev = 0, v = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
    return v;
}

// 理论峰值带宽（cudaSamples bandwidthTest 同款公式）：
// 显存有效时钟 = memoryClockRate × 2 (DDR)，字节宽 = busWidth / 8
inline double peak_bw_gbs() {
    int dev = 0, clk_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    return 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

inline void print_device_info() {
    int dev = 0;
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("== %s | SM x%d | 显存 %.1f GB | 理论带宽 %.0f GB/s ==\n",
           p.name, p.multiProcessorCount,
           p.totalGlobalMem / 1e9, peak_bw_gbs());
}

// host 侧填均匀分布随机数。归约/校验场景用它比 randn 稳：sum 远离 0，
// 相对误差不至于数值上爆炸
inline void fill_uniform(float* p, size_t n, unsigned seed,
                         float lo = 0.f, float hi = 1.f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (size_t i = 0; i < n; ++i) p[i] = d(rng);
}

// ---------------------------------------------------------------------------
// 表格打印：统一全课程的输出格式
// ---------------------------------------------------------------------------
inline void print_header(const char* perf_unit) {
    printf("%-34s | %10s | %12s\n", "version", "time(ms)", perf_unit);
    printf("-----------------------------------+------------+-------------\n");
}

inline void print_row(const char* name, double ms, double perf) {
    printf("%-34s | %10.3f | %12.1f\n", name, ms, perf);
}
