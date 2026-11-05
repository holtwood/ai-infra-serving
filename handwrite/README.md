# handwrite/ —— AI Infra 手撕白板代码集

> 国外大场（OpenAI/Anthropic/xAI/Meta 等）面经共识：从零手写 MHA / GQA /
> KV cache / RoPE / stable softmax 是**入场券**。本目录只用 torch 张量原语；
> `nn.MultiheadAttention`、`F.scaled_dot_product_attention` 仅出现在 assert
> 对照里，绝不当实现。每个文件独立运行、自带校验：
>
> ```bash
> python handwrite/01_mha_gqa_kvcache.py   # 全 CPU 可跑（bf16 + SDPA math 路径）
> ```
> 有 GPU 时把 `dev` 换成 `"cuda"` 即可。

## 文件清单

| 文件 | 白板题 | 自校验方式 |
|---|---|---|
| `01_mha_gqa_kvcache.py` | MHA forward / GQA / decode 单步带 KV cache | 对照 SDPA，bf16 atol=2e-2 |
| `02_rope.py` | RoPE 教科书版 + HF 工程版 | 对拍纯 Python naive 版 + 保范数 + 相对位置恒等式 |
| `03_stable_softmax_sampler.py` | 稳定 softmax / LSE / 温度+top-k+top-p 采样 | x=1000 反例 + 对拍 F.softmax + 10000 次采样分布 smoke test |
| `04_paged_kv_allocator.py` | vLLM 分页 KV 块分配器（纯 Python） | 分配/释放/共享/COW/缺页拒绝断言 |
| `05_backprop_derivation.md` | 手推 softmax+CE、LayerNorm、attention VJP | 讲题模板 + 形状检查表 |

## 每题必被追问的点

### 01 MHA → GQA → KV cache
- **GQA 省了什么**：KV cache 显存和 decode 读带宽按 Hkv 算（bytes/token =
  2·L·Hkv·D·dtype，见 `notes/inference-math.md` §3），省的是存储侧；
  **不省算力**——算 attention 时仍要 expand 回 Hq。MQA = G=Hq 的特例。
- **KV cache append 复杂度**：`torch.cat` 每步 O(S) 拷贝；工程做法是预分配
  max_len 缓冲区按位置写入（O(1)），vLLM 再加分页避免按 max_len 预留。
- **decode 为什么不用 mask**：单 query 是序列最后位置，可见全部历史。
  坑：SDPA `is_causal` 对非方阵是左上对齐，S_q=1 开 causal 会只 attend 到 key0。
- **scale 为什么是 1/√D**：qk 内积方差随 D 线性涨，不缩 softmax 进入饱和区梯度消失。

### 02 RoPE
- **为什么只旋 q/k**：只有 q·k 内积吃位置；旋完内积只依赖相对位置 n−m，v 不需要。
- **θ_i = base^{-2i/D} 的含义**：D/2 个二维平面从高频到低频的旋转谱；
  base=10000 决定最低频波长。
- **外推为什么挂**：训练见过的 pos·θ 范围内是插值，超出后高频维先混叠；
  NTK-aware 缩 base、YaRN 分频段插值——把外推变回插值。
- **两种配对约定**：交错 (2i,2i+1) vs half-split (i,i+D/2) 差一个维度置换，
  权重训练好后等价；写代码时必须说清自己用哪种。

### 03 softmax / 采样
- **为什么减 max 不是 clamp**：softmax 平移不变（分子分母同乘 e^{-c}）→ 减 max
  不改分布只防溢出；clamp 截断会改相对大小 = 改分布。
- **溢出阈值**：exp(x) 在 x≈88.7 撞 fp32/bf16 上限（e^88.7≈3.4e38=FLT_MAX）。
- **LSE 的意义**：CE 需要 log p 不是 p（防 log(0)）；也是 FlashAttention
  online softmax 的原语。
- **采样为什么用逆 CDF**：multinomial 本质即 inverse-CDF；top-p 语义是
  "去掉自己后累计仍 ≤ p 的保留"，首个超 p 的 token 不能砍。

### 04 分页 KV 分配器
- **block_size=16 的取舍**：块大 → 内部碎片浪费 KV；块小 → block table 元数据
  和寻址开销涨。vLLM 默认 16 是平衡点。
- **ref_cnt>1 为什么要 COW**：共享块被写会改到别的 seq 的前缀；必须先摘私有副本。
- **缺页怎么办**：真系统不直接报错——先拒入（waiting），再抢占（recompute 丢 KV
  / swap 到 CPU），见 `mini-serving/engine.py` 的 Scheduler.preempt。
- **还差的机制**：ref=0 块的 LRU 缓存（vLLM 命中才真省 prefill）、per-token slot。

### 05 反推
- 追问路径：`p−y` 结论 → softmax Jacobian 形状 → LN 的 dx 三条依赖路径 →
  attention VJP 里 scale/mask/FlashAttention 重算 P——讲法见文件内模板。
