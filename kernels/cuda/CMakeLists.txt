# M2-CUDA 教学工程：RTX 4090 (sm_89, Ada Lovelace) + CUDA 12.x
cmake_minimum_required(VERSION 3.18)
project(cuda_kernels LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)          # 默认 Release（-O3 + NDEBUG）
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)           # RTX 4090 = sm_89

# -lineinfo：生成行号信息，ncu 的 Source 页能把指标对应回 .cu 行号；
# 与 -G（全 debug）不同，不改变优化、零运行时开销
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

find_package(CUDAToolkit REQUIRED)

add_executable(01_vector_add 01_vector_add.cu)
add_executable(02_reduction  02_reduction.cu)
add_executable(03_gemm_tiled 03_gemm_tiled.cu)

# 只有 03 需要 cuBLAS
target_link_libraries(03_gemm_tiled PRIVATE CUDA::cublas)
