# AI Infra（LLM Serving 方向）70 小时集训计划

> 目标：面试 **AI Infra / 大模型推理服务工程师**（Python/PyTorch 技术栈为主）。
> 硬件：RTX 4090 48GB（48G 显存，~1TB/s 显存带宽，CUDA 12.8）+ 128 核 CPU + 1TB 内存 + `/data` 500G 数据盘。
> 使用方式：集中几天（约 6 天）用完。

## 一、岗位画像：面试官实际考什么

通用 serving 方向的 AI Infra 面试通常由四块组成，本计划每个模块都对应到具体题目：

| 考察块 | 典型问题 | 覆盖模块 |
|---|---|---|
| 推理引擎原理 | PagedAttention/RadixAttention、continuous batching、KV cache 管理、prefill/decode 特性、chunked prefill、prefix caching、P/D 分离 | M1, M3, M4 |
| Kernel 与性能 | 手写/读懂 Triton kernel；CUDA 基础（occupancy、coalescing、ncu 指标）；kernel fusion；CUDA Graph | M2, M5 |
| Serving 系统设计 | 设计 LLM 推理服务、TTFT/TPOT/SLO、调度与抢占、压测方法、speculative decoding | M4 |
| Python/系统工程 | asyncio 服务框架、流式输出、tokenizer/detokenize、量化与显存数学 | 贯穿全部 |

真题来源（自测用）：
- AI Infra 305 题面经: https://riddlego.github.io/ai-infra-interview-305/study.html
- 推理引擎 internals 题库: https://inferenceinterviews.com/c/serving-engines
- 推理框架 28 题（vLLM/SGLang/TRT-LLM）: https://ai-interview-guide.cdqyfdbymn.me/topics/19-inference-frameworks/

## 二、总路线图（66h 计划 + 4h 机动）

| 模块 | 小时 | 内容 | 产出物 |
|---|---|---|---|
| M0 环境与推理数学 | 3 | 装环境；推理算术（FLOPs/bytes/KV cache 公式） | 环境 + 硬件参数实测表 |
| M1 PyTorch 手写推理 | 6 | 不用 `model.generate`，手写带 KV cache 的 decode loop；测 prefill/decode 耗时构成 | `scratch-decode/` 可运行 + 笔记 |
| M2 Kernel 实战 | 14 | Triton 主（fused softmax/layernorm/matmul/flash-attn）+ CUDA C++ 基础（reduction/GEMM/ncu） | `kernels/` repo + benchmark 表 |
| M3 生产引擎深挖 | 16 | vLLM 部署+全特性 ablation+V1 源码精读；SGLang 对照（RadixAttention）；llama.cpp/TRT-LLM 速览 | 引擎对比 benchmark + 架构笔记 |
| M4 迷你 serving 引擎 | 12 | asyncio + scheduler + KV block pool + continuous batching + 流式输出，压测对照 vLLM | `mini-serving/` 项目 ⭐ |
| M5 量化与 profiling | 8 | 量化精度/吞吐实验；nsys/torch profiler/CUDA Graph | 量化对比表 + profile 截图集 |
| M6 面试冲刺 | 7 | 题库自测；整理 repo 与话术 | 面经笔记 + 60 秒话术 |

## 三、每日节奏（约 6 天，按需压缩）

- **Day 1（9h）**：M0 + M1。上午装环境、读推理算术；下午手写 PyTorch decode loop（这是 M4 的种子代码）。
- **Day 2（11h）**：M2 Triton 部分。fused softmax → layernorm → matmul → flash-attention 前向，全程对拍 torch。
- **Day 3（11h）**：M2 CUDA 部分 + M3 起步。reduction/GEMM 各写一版、ncu 上手；下午起 vLLM 部署压测。
- **Day 4（11h）**：M3 主体。vLLM 特性 ablation（prefix cache/chunked prefill/spec decode）+ V1 scheduler 源码；SGLang 对照实验。
- **Day 5（12h）**：M4。迷你 serving engine 一整天，能跑通 continuous batching + 流式输出 + 简单压测。
- **Day 6（12h）**：M5 + M6。量化与 profiling 实验，题库自测，整理产出。

## 四、资源清单（精选，按优先级）

**Kernel**
- GPU MODE Lectures: https://github.com/gpu-mode/lectures —— Lecture 1（profiling）、8（性能 checklist）、9（reductions）、Triton 相关几讲，1.5x
- Triton 官方 tutorials: https://triton-lang.org/main/getting-started/tutorials/ —— fused-softmax、layernorm、matmul、06-fused-attention 四篇照做
- PMPP《Programming Massively Parallel Processors》ch1-6 —— 不懂就查的工具书
- NVIDIA CUDA C++ Best Practices Guide / Nsight Compute 文档 —— 查指标用

**推理系统（主线）**
- LLM Inference Arithmetic (kipply): https://kipp.ly/blog/llm-inference-arithmetic/ —— M0 必读
- vLLM 论文（SOSP'23 PagedAttention）+ vLLM V1 架构文档/blog + 源码 `vllm/v1/` —— M3 主线
- SGLang blog（RadixAttention、cache-aware scheduling）+ 源码 `python/sglang/srt/` —— M3 对照
- SARATHI-Serve（chunked prefill）、Splitwise/DistServe（P/D 分离）、speculative decoding survey —— M4/M5 读
- llama.cpp（GGUF/量化格式）与 TensorRT-LLM 文档 —— M3 速览，应对"引擎对比"题

**中文补充**
- 猛猿知乎专栏（大模型推理系列）、DefTruth/Awesome-LLM-Inference、reed-lau 博客（LLM 推理源码解析）

## 五、产出物清单（= 简历作品集）

1. `scratch-decode/`：手写 KV cache decode loop，含 prefill/decode 耗时分解数据
2. `kernels/`：Triton + CUDA 两组 kernel，各带 benchmark 表和（CUDA 部分）ncu 分析
3. `mini-serving/`：⭐ 迷你 LLM serving engine——asyncio 请求队列 + 调度器 + KV block pool + continuous batching + SSE 流式输出 + 压测脚本
4. `benchmarks/`：vLLM vs SGLang vs llama.cpp 同模型 TTFT/TPOT/吞吐-并发曲线；vLLM 特性 ablation 表
5. `notes/`：vLLM V1 架构笔记、推理数学 cheatsheet、填满答案的 interview-map.md

## 六、删减预案（时间不够时按此顺序砍）

1. M3 中 llama.cpp/TRT-LLM 速览 → 只读文档不跑实验（省 2h）
2. M2 CUDA 部分只写 reduction + vector add，GEMM 只看 CUTLASS 概念不实现（省 3h）
3. M5 nsys/torch profiler 实操 → 只跑 CUDA Graph 对比（省 2h）
4. SGLang 部署 → 只读 RadixAttention 博客（省 2h）

**红线（绝对不能砍）**：M2 Triton kernel、M4 mini-serving、M6 复盘——这三个是面试硬通货。

## 七、环境备忘

- 工作目录统一放 `/data/ai-learn/`（系统盘只有 40G）
- conda 在 `/opt/conda`（Python 3.11）；CUDA toolkit 在 `/usr/local/cuda-12.8`（需加 PATH）
- ncu 在 `/opt/nvidia/nsight-compute`；HF 下载走 `HF_ENDPOINT=https://hf-mirror.com`
- 已存在的 `/model/Qwen3.8-27B-FP8` 可直接给 vLLM 用；`/model` 下别的脚本不要动
- 测试模型建议：Qwen2.5-0.5B（M1 手写）、Qwen2.5-7B-Instruct（M3-M5 主力）、现成的 27B-FP8（大模型体验）
