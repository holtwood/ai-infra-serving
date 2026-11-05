# mini-serving CUDA Graph 实验：69% 空转的真相与边界

日期：2025-11-05 · `mini-serving/engine_graph.py` · Qwen2.5-0.5B · RTX 4090
（profiler 前置数据见 `notes/profiler-decode-timeline.md`：batched decode
步 GPU 忙时仅 31%，其余是 ~170 次/步的 Python+launch 开销）

## 实测（B=8 并发，decode 单步耗时）

| 引擎 | ms/step | 结论 |
|---|---|---|
| batched（无图） | 13.75 | 基线 |
| **graph max_len=256** | **5.46** | **2.5x**——launch 开销确实被砍 |
| graph max_len=512 | 7.05 | 1.9x |
| graph max_len=2048 | 17.83 | **1.3x 变慢**——pad 浪费反超 |

## 核心发现：定长图 × 变长 KV 的矛盾

- CUDA Graph 要求全部输入/中间张量 **shape 固定、指针稳定**。
  → StaticCache 预分配 [B, max_len]，attention mask 必须全宽
  → **SDPA 每一步扫满 max_len 个 KV 位**，实际有效才几十个
  → pad 浪费随 max_len 线性涨，盈亏点 ~512-1024
- **这就是 vLLM 必须自研 PagedAttention kernel 的物理原因**：
  block table 让 kernel 在定长图里只读真实块（变长寻址藏在
  kernel 内部），graph 只管 shape 不变。HF 通用模型没有这条路径，
  所以「eager HF + 手写 graph」存在天然天花板。
- vLLM 的 PIECEWISE graph 也只录 GEMM/norm 等定长段，
  attention 留在图外——我们全量录图反而证明了为什么要 piecewise。

## 实现过程中踩的真坑（面试可讲）

1. **StaticCache batch 维不可变**：batch 由首个 update 的
   key_states 决定 → decode 必须永远 pad 到 Bmax=32 行
   （dummy 行 = vLLM 的 padding slot）。
2. **cache_position 全 batch 共享单索引** → 所有行 KV 右对齐到
   同一 [0:L) 窗口；更长 prompt 准入要把全体 KV 右移
   （`_shift_right`）——vLLM block table 根本没这个痛。
3. **两个 off-by-one**（都靠真实输出抓出来的）：
   - attention_mask 窗口 `[L-li:L+1)` vs `[L+1-li:L+1)`——
     多放一格 pad 位 → 短 prompt 行输出乱码
   - `position_ids` 是**输入 token** 的绝对位置 = context_len-1，
     给错成 context_len → rope 偏移，短序列受害最大
   - （附注：engine_batched 的 pos 也有同一隐患，bf16 容忍度
     下 3/4 逐 token 一致——这正是那 1/4 分叉的疑似根因，待验）
4. graph capture 必须在**侧流 warmup 3 次**再录（官方 recipe），
   否则 cublas workspace 等懒初始化会被录进图。

## 完整弧线（面试一句话版）

per-seq forward 79 tok/s → batched forward 973（12.3x，消灭权重重复
搬运）→ +CUDA graph：短上下文 decode 步再快 2.5x（消灭 Python/launch
开销）→ 长上下文反而变慢，引出「为什么需要 PagedAttention kernel」
——每一层瓶颈都用数字指认了下一个优化点。
