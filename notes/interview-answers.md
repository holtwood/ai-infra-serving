# 面试答案卡：每条都带实测数据（60 秒版）

用法：面试官追问时先给结论 → 给机制 → 给"我实测的数字"。
所有数字出自本 repo（RTX 4090 48GB / vllm 0.11.0 / torch 2.8.0+cu128）。

## Q: 为什么 decode 是 memory-bound、prefill 是 compute-bound？

A: Arithmetic intensity。decode 每步输入 1 个 token，权重读 N GB 只做
N×1 的 GEMV——计算/访存比 ≈ 1:1，远低于 tensor core 的拐点
（4090 上 ~176 FLOP/B）。prefill 一次过 S 个 token，同样的权重读取
做 S×N 的 GEMM，AI 随 S 线性升，S>~200 就进 compute-bound 区。
**实测**：7B bf16 decode TPOT 16.1ms ≈ 15GB 权重 ÷ 939GB/s 实测带宽，
理论命中误差 <5%。

## Q: vLLM 为什么快？（拆解贡献）

A: 四件套，我逐个做过消融：
- **continuous batching**：iteration 级调度，请求边来边插进 batch。
  实测 7B：并发 64 时 TPOT 仅从 16.1→20.5ms，总吞吐 60→2399 tok/s（40x）
- **PagedAttention**：KV 按 16-token block 分页，消内部碎片，
  vLLM 论文口径显存利用率 ~40%→96%；我手写过一个带
  freelist/refcount/COW/hash 前缀共享的 paged allocator（handwrite/04）
- **prefix caching**：block 按 token hash 复用。实测 2048-token
  公共前缀，TTFT 234ms→54ms（4x），服务端 hit rate 95.5%
- **CUDA Graph**：decode 步录成单 graph。实测 0.5B TPOT
  2.6ms vs eager 7.2ms（-64%）；模型越小收益越大（launch 开销占比高）

## Q: continuous batching 原理？我自己实现的 mini-serving 呢？

A: 不等 batch 凑齐，每个 iteration 结束就把完成的请求换下来、
waiting 里的补进去——SLO 驱动的 token budget 调度。
我的 mini-serving 实现了 FCFS+token 预算+KV 准入+recompute 抢占。
**关键实测**：我刻意用 per-seq forward（便于教学），TPOT 随并发
线性退化 13→202ms、总吞吐 79 tok/s 封顶——权重每 seq 搬一遍；
vLLM 同模型同卡 2762 tok/s（35x）。这个负面对照恰好证明了
batched forward + PagedAttention 的全部意义：不是省显存，是省权重搬运。

## Q: KV cache 显存怎么算？（笔算 + 实测）

公式：layers × 2(K,V) × kv_heads × head_dim × dtype字节 × seq_len × batch
- Qwen2.5-0.5B：24×2×2×64×2B×2048 = **24MiB**（脚本实测输出一致）
- 8B/32层/8KVh/128d/4k ctx/batch32：32×2×8×128×2×4096×32 = **34GB**
  → KV cache 比权重还大，这就是 serving 显存管理是一等公民的原因

## Q: 量化为什么加速？什么时候不加速？

A: decode 阶段权重搬运量 ∝ 权重大小：INT4 vs FP16 少搬 75%。
实测 32B-AWQ（awq_marlin kernel）：19GB 权重组装进 43.5GB 卡，
TPOT 21.5ms ≈ 19GB/939GB/s + dequant 开销——bytes 模型仍然成立。
**不加速的情况**：prefill-heavy（compute-bound，dequant 纯加算）、
batch 很大（摊薄后带宽不再是瓶颈）、activation 误差敏感场景。
weight-only 量化（AWQ/GPTQ）保 activation 精度就是为了这个。

## Q: 投机解码原理？我实测的坑？

A: 小 draft 猜 k 个 token，大模型一次 forward 并行 verify，
接受的直接收下 → 每步产出 1~k+1 个 token。数学上等价于
target model 分布（拒绝采样保证无损）。
**我的实测坑**（vllm 0.11 ngram）：Qwen generation_config 默认
repetition_penalty=1.05 → is_spec_decode_unsupported 判定全部
请求不兼容 → spec decode 静默零产出，metrics 全是 0、日志无提示；
显式 rep_pen=1.0 后 drafts=30/accepted=129、重复文本 e2e **4x 加速**。
这同时说明：ngram 只对可预测输出（重复/模板/代码）有效。

## Q: Triton vs CUDA C++ 怎么选？我的数据？

A: 我 Triton 写的 RMSNorm 比 torch 快 3-7x（torch 没 fuse，
拆成 mul+pow+mean+rsqrt+mul 一串 kernel）；Triton matmul
autotune 后在 4096³ 反超 cuBLAS（160.8 vs 148.9 TFLOPS）；
但 flash-attn forward 只到 SDPA(FA2) 的 56-87%——D=128 时寄存器
压力打不过手写 pipelining。**结论**：开发期/非热点 kernel 用
Triton；serving 引擎热路径 attention 仍用 CUDA（warp 级控制）。

## Q: GEMM 优化路径（我写过三版）

A: naive(5.1TF) → smem tile 32×32(6.7TF) → cuBLAS(54.6TF)。
只做 smem tiling 到不了库的 1/8——后面还差 register tiling、
double buffering、float4 向量化、tensor core mma。
面试别吹"我 GEMM 优化到 XX%"，要说清每一级解决什么瓶颈。

## Q: TTFT 突然变高怎么排查？（系统设计题）

A: 按 serving 链路自上而下：① 队列深度/到达率突增（接入层 metrics）
→ ② 长 prompt 占比变化（prefill 抢 decode 资源，chunked prefill
参数）→ ③ prefix cache 命中率掉了（我的实验里开关差 4x TTFT）
→ ④ KV pool 水位（抢占触发 recompute，TTFT 加倍）
→ ⑤ GPU 侧：clock 降/功耗墙/抢占共享。
**我压测里 32 并发 TTFT 52→420ms 就是纯排队效应**，先分清
"排队延迟"和"计算延迟"——vLLM 的 metrics 里 waiting/running 数
和 prompt throughput 能直接区分。

## Q: 10 万 QPS 怎么扛？

A: 单机极限先说清：7B 单卡 ~2400 tok/s 输出 ≈ 几十 req/s；
10 万 QPS 需要 ~千卡集群。设计：无状态 API 层水平扩展 +
prefix-aware 路由（同租户/同前缀打同一引擎，吃 APC）+
SLO 分级队列 + P/D 分离集群（prefill 节点算力型、decode 节点
带宽型，KV 经 RDMA/NVLink 传输）+ 按 goodput 弹性伸缩。
单实例内就是 vLLM 那套：continuous batching + APC + chunked prefill。

## 手撕题状态（全 PASS，可白板复现）

- MHA/GQA + KV cache decode：对拍 SDPA err 1.6e-2(bf16)
- RoPE 双版本 + 相对位置恒等式：err 4.8e-7
- stable softmax/log_softmax/top-k/top-p：e^1000 溢出 demo + 卡方验证
- paged allocator：freelist/block table/refcount/hash 共享/COW/OOM
- mini-serving 主循环：scheduler+KV 池+SSE（远端实测数据见 m4）
