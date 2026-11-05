// M2-CUDA 第 1 课：vector add / saxpy —— 显存带宽实测
//
// 面试考点：
// - 逐元素 kernel 是纯 memory-bound：性能指标 = 有效访存字节数 / 时间，
//   拿"实测 GB/s vs 理论带宽"当优化标尺（roofline 模型的 byte 轴）
// - grid-stride loop：线程总数与数据量解耦，grid 开多大都能扫完全数组，
//   是 CUDA kernel 通用化 / occupancy 调优的标准写法（triton 里对应 masked load）
// - block size 怎么选：对 128/256/512/1024 扫描——甜点通常在 256 附近，
//   背后是 occupancy（每 SM 常驻 warp 数）与寄存器/smem 资源占用的平衡
// - saxpy（y = a*x + y）是 BLAS-1 原型：每元素 2 FLOP / 12B 访存，
//   算术强度极低，4090 上再优化也只能贴着 ~1TB/s 带宽上限跑
//
// 编译运行：nvcc -O2 -arch=sm_89 01_vector_add.cu -o 01_vector_add && ./01_vector_add
#include <cstdio>
#include <cmath>
#include <vector>
#include "bench_utils.cuh"

// ---------------------------------------------------------------------------
// grid-stride loop：
//   idx 起点 = 全局线程号，步长 = 全 grid 线程总数。
//   不管 grid 开多大，每个元素恰好被一个线程处理一次——
//   所以"grid 设多少"从正确性问题降维成纯性能问题（够填满所有 SM 即可）。
// ---------------------------------------------------------------------------
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];
}

// saxpy: y = alpha*x + y —— in-place 读写 y，有效访存量仍是 3 个数组的量
__global__ void saxpy(float alpha, const float* x, float* y, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        y[i] = alpha * x[i] + y[i];
}

int main() {
    const int n = 1 << 26;                          // 64M float，单数组 256MB
    const double bytes = 3.0 * n * sizeof(float);   // 2 读 + 1 写
    print_device_info();

    std::vector<float> ha(n), hb(n), hc(n), hy(n);
    fill_uniform(ha.data(), n, 1u, -1.f, 1.f);
    fill_uniform(hb.data(), n, 2u, -1.f, 1.f);

    DeviceBuffer<float> da(n), db(n), dc(n), dy(n);
    da.copy_from_host(ha.data());
    db.copy_from_host(hb.data());
    dy.copy_from_host(hb.data());                   // dy 初值 = b，用于 saxpy

    // -------------------- 正确性校验（CPU 参考） --------------------
    // 故意只开 4*SM 个 block，逼每个线程走多次 grid-stride 迭代，
    // 顺便验证 grid-stride 的正确性
    int grid = get_sm_count() * 4;
    vector_add<<<grid, 256>>>(da.get(), db.get(), dc.get(), n);
    dc.copy_to_host(hc.data());
    double max_err = 0.0;
    for (int i = 0; i < n; ++i)
        max_err = fmax(max_err, (double)fabs(hc[i] - (ha[i] + hb[i])));
    printf("vector_add 校验: max_err=%.2e  %s\n",
           max_err, max_err < 1e-6 ? "PASS" : "FAIL");

    const float alpha = 2.5f;
    saxpy<<<grid, 256>>>(alpha, da.get(), dy.get(), n);
    dy.copy_to_host(hy.data());
    max_err = 0.0;
    for (int i = 0; i < n; ++i)
        max_err = fmax(max_err, (double)fabs(hy[i] - (alpha * ha[i] + hb[i])));
    // GPU 端 a*x+y 会被编译成单条 FMA（一次舍入），CPU 侧乘加各舍入一次，
    // 差值在 1 ulp 量级，容忍 1e-5
    printf("saxpy      校验: max_err=%.2e  %s\n\n",
           max_err, max_err < 1e-5 ? "PASS" : "FAIL");

    // -------------------- 带宽实测 --------------------
    // GB/s = 有效字节数 / 秒；bytes/ms/1e6 即 bytes/(ms*1e-3)/1e9
    print_header("GB/s");
    char name[64];

    // (a) 一线程一元素的经典开法：扫 block size
    for (int bs : {128, 256, 512, 1024}) {
        int g = (n + bs - 1) / bs;
        snprintf(name, sizeof(name), "vector_add block=%d (1elem/thread)", bs);
        float ms = bench_ms(
            [&] { vector_add<<<g, bs>>>(da.get(), db.get(), dc.get(), n); });
        print_row(name, ms, bytes / ms / 1e6);
    }

    // (b) 对照：固定小 grid + grid-stride，每线程处理多个元素。
    //     大 n 下带宽与 (a) 几乎一致——这才是该记住的通用写法
    snprintf(name, sizeof(name), "vector_add block=256 grid-stride");
    {
        float ms = bench_ms(
            [&] { vector_add<<<grid, 256>>>(da.get(), db.get(), dc.get(), n); });
        print_row(name, ms, bytes / ms / 1e6);
    }

    // (c) saxpy in-place：读 x + 读写 y，同样 3 倍数组流量
    {
        float ms = bench_ms(
            [&] { saxpy<<<grid, 256>>>(alpha, da.get(), dy.get(), n); });
        print_row("saxpy in-place block=256", ms, bytes / ms / 1e6);
    }

    printf("\n说明：GB/s = 3*N*4B / 耗时；与上面理论带宽的比值就是带宽利用率，\n");
    printf("      memory-bound kernel 优化到 ~80%%+ 即接近物理上限。\n");
    return 0;
}
