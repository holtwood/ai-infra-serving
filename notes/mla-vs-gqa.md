# MLA vs MHA/GQA/MQA：KV 显存账（美团面经原题）

## 四种注意力的 KV 存储量（每 token 每层，bf16）

| 结构 | K/V 头数 | 每 tok·层字节 | 相对 MHA |
|---|---|---|---|
| MHA（32 头） | 32+32 | 2×32×128×2 = 32KiB | 1x |
| GQA（Qwen2.5-7B: 4 kv 头） | 4+4 | 2×4×128×2 = 4KiB | 1/8 |
| MQA（1 kv 头） | 1+1 | 2×1×128×2 = 1KiB | 1/32 |
| **MLA**（DeepSeek） | 低秩联合向量 | **c_kv≈512 维 → 2KiB**（另加 rope 分量） | ~1/16 |

## MLA 的原理（一句话版）

不为每个 head 各存 K、V，而是把整个 KV 压成**一条低秩联合潜向量
c_kv**（~512 维，RoPE 部分单独存 ~64 维）。推理时每个 head 的 K、V
由 c_kv 上采样展开——存的只是 c_kv，不是展开的 K/V。

- 数学：W_k ≈ W_uk × W_dk（低秩分解），KV cache 只存 W_dk·x = c_kv
- 收益：KV 显存 ~1/16 of MHA，且不丢精度（展开是确定的线性变换）
- 代价：decode 每步要展开 c_kv → K/V（可吸收进 Q/O 投影矩阵，
  推理期不实际展开——"absorbed MLA"），实现复杂度在权重吸收上

## 和我们实测的勾连

Qwen2.5-7B（28 层 GQA 4 头）每 token KV = **56KiB**；
若换 MLA（c_kv 512+rope 64）≈ 576×2B = **1.1KiB** → 每 token 省 ~98%。
同样 27GiB KV 池：GQA 506k tokens → MLA 模型能装 ~25M tokens——
**长上下文时代 GQA 是折中，MLA 是激进解法**，这就是 DeepSeek
上下文窗口能做长的显存基础。

## 面试追问预备

- "MLA 为什么 decode 也快？"→ 读 KV 变读 c_kv（1/16 字节），
  memory-bound 的 decode 直接受益；上采样计算可被 GEMM 吸收
- "GQA vs MLA 取舍？"→ GQA 零精度损失、实现简单、社区默认；
  MLA 更省但需要从头训（低秩分解学进权重里）、推理实现复杂
- "KIVI/INT4 KV 呢？"→ 另一条路：量化 KV 而非改结构，可叠加
