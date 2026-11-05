# 实验：Triton paged-attention decode kernel（block table 寻址）

`kernels/triton/05_paged_attention.py`

## 动机

graph 实验的负结果：定长 StaticCache 强制 attention 扫满 max_len
窗口——`max_len=2048` 时图版 17.8ms 反而慢于无图 13.75ms。
vLLM 的答案是 block table：KV 进全局物理块池，kernel 按表取块，
**变长 KV 不再为定长窗口付税**。这个 kernel 就是把那堵墙翻过去。

## 实现

- 全局块池 `pool [NB, BS=16, Hkv, D]`，每 seq 一张 `block_table`
- flash-decoding 风格：每 program 负责 (batch, q头)，沿表迭代块，
  在线 softmax（m_i/l_i/acc），尾块用 `ctx` 掩掉无效 slot
- GQA 直接支持：`kvh = h // G`，无 repeat_interleave 物化
- 正确性：8 seq 变长 ctx（25~459），vs gather+SDPA 参考
  max_rel_err **8.2e-7**（fp32 级）

## 数字（B=8, Hq=28, Hkv=4, D=128, BS=16, 真实 ctx=64）

| 路径 | 耗时 |
|---|---|
| paged kernel（按表取块） | **53.4us** |
| 定长窗口 SDPA（扫满 512） | 195.9us |
| **padding 税** | **3.7x** |

这正是 graph+max_len=2048 变慢的微观证据：那 2048 窗口里
~96% 是 padding，全部进了 GEMM/GEMV 的计算量。

## 与 vLLM 真实实现的差距（诚实版）

- 无 split-K（长 ctx 时单 program 串行扫块，vLLM v2 有 flash-decode
  split + combine）
- block 内逐 slot `tl.load` 而非向量段加载
- 未接入 engine（HF StaticCache 所有权问题——这正是 vLLM 要自研
  整个 attention 后端的原因）

## 弧线闭环

per-seq 79 → batched 973 → graph 5.46ms（短 ctx 2.5x）→
长 ctx 撞 padding 墙 → **block-table kernel 实测省 3.7x 扫描税**
= vLLM PagedAttention 的存在理由，亲手从瓶颈走到了内核。
