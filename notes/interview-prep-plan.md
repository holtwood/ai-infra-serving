# AI Infra Serving 面试备战方案（2025-11 调研版）

来源：牛客智谱/美团北斗/字节面经实录、Blind/Exponent OpenAI+Anthropic
面经、Aman Karki 等真实录用简历模板。

## 面试结构

**国内**：八股连问 → 项目深挖 → 手撕（有时 C++）
**国外**：real-work coding（非 LC）→ system design → project
presentation/deep-dive → behavioral

## 高频原题 → 我们的弹药（逐题对位）

| 原题（出处） | 我们的实测数据 |
|---|---|
| prefill compute/decode memory-bound（美团） | TPOT 16.1ms≈15GB/939GB/s |
| continuous batching（智谱/美团） | per-seq 79→batched 973 tok/s |
| PagedAttention block 大小权衡（智谱） | allocator + 「页大→内部碎片，页小→表长+分配频繁」口径 |
| 抢占识别与代价（生产 triage） | KV99%→22 次逐出、吞吐 2815→301 |
| P/D 分离动机与判据 | ITL 税 195%；「ITL 税>15%、prompt:out 不对称、有闲带宽」三信号 |
| CUDA Graph 动机 | profiler 69% 空转 → 手写 graph 2.5x + 定长图边界 |
| 量化取舍 | AWQ 实测 + fp8KV 容量 2x/TTFT +10-26% |
| 成本意识（字节系统设计） | 416W、8.6tok/J、~¥0.05/百万 token |
| Prefix 路由 | sticky vs rr -22%，rep1 88 vs 243ms |

## 必补理论（不跑实验，写成笔记）

- **MLA vs MHA/GQA/MQA**：低秩 KV 联合压缩（c_kv 维度 ~4*head_dim
  vs GQA 每头全存），KV 量 = 层数×c_kv×2B —— 与 GQA 公式对比即可
- **TP/PP/DP/EP**：TP 切 attention head 维度+FFN 列/行（每层 2 次
  all-reduce）；PP 切层（气泡/流水线调度）；DP 数据并行；EP 是 MoE
  专家并行（dispatch/combine = all-to-all）
- **MoE**：top-k router、capacity factor、负载均衡 aux loss、
  dispatch/combine 通信

## 手撕补充（进 handwrite/）

- MoE dispatch：gating_scores [T,E] → top-k → 每专家 token_id 列表
  （智谱一面原题）
- 已有：MHA/GQA、RoPE、stable softmax、KV cache、top-p、allocator

## 表达训练（最高优先级，占准备时间 50%）

1. **presentation**：把 RESULTS.md 当 10 分钟演讲稿练——
   「背景→我造了什么→验证→数字→输在哪→下一步」
2. **深挖防御**：每个数字都要能答「为什么」——973 vs 2762 的
   四个原因、69% 空转的构成、抢占时 APC 命中率为什么升到 49%
3. **系统设计模板**（拿我们的数字当素材）：
   请求→路由(前缀亲和+负载)→prefill 池/decode 池(P/D 分离三信号)
   →引擎内(continuous batching+paged KV+chunked prefill)
   →监控(KV 占用/抢占计数/TTFT·TPOT p99/功耗成本)→伸缩(预热扩容)
4. **诚实边界**：单卡 4090 ≠ 生产规模；一律说"我在单机验证过
   机制，生产级差异在 X"

## 面试红线（调研共识）

- 背 flag/参数 = 零分（"memorized vLLM flags score nothing"）
- 说不清自己项目的 why = 负资产
- 有 bug 没发现/发现装没看见 > 承认局限
