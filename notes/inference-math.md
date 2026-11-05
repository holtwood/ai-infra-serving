# 推理数学速查表（M0 笔记）

> 面试数学题的根基：所有估算都要能手推。参考 https://kipp.ly/blog/llm-inference-arithmetic/

## 1. Prefill vs Decode：为什么形态不同

| | Prefill | Decode |
|---|---|---|
| 输入 | 整个 prompt（S 个 token 一次过） | 每步 1 个新 token |
| 矩阵形态 | GEMM（S×K × K×N），token 维大 | GEMV（1×K × K×N），token 维=1 |
| 瓶颈 | **compute-bound**（tensor core 忙） | **memory-bound**（算力闲置，卡在搬权重） |
| 衡量指标 | TTFT（首 token 延迟） | TPOT/ITL（每 token 间隔） |

**Arithmetic intensity（AI）= FLOPs / 访存字节数**。ridge point = 峰值算力 / 峰值带宽。
decode 一步：FLOPs ≈ 2·P（P=参数量），访存 ≈ P·bytes + KV cache。AI ≈ 2/bytes_per_param ≈ 1（fp16）——远低于 4090 的 ridge point（~160 TFLOPS / 1TB/s ≈ 160），所以 memory-bound。

## 2. 吞吐上限估算（decode）

```
理论 tok/s ≈ 显存带宽 / (模型权重字节数 + 每 token KV 字节数)
```

例：7B bf16 ≈ 14GB，4090 ~1TB/s → 上限 ~71 tok/s/sequence（batch=1）。
batch 增大时权重只搬一次 → **batch 是 decode 阶段最便宜的吞吐来源**，直到撞 compute 墙或显存墙。

## 3. KV cache 显存公式（⭐ 必背）

```
KV bytes/token = 2(K,V) × n_layers × n_kv_heads × head_dim × dtype_bytes
总 KV = KV bytes/token × seq_len × batch
```

例（Qwen2.5-7B：28 层，4 kv head，head_dim 128，bf16）：
2 × 28 × 4 × 128 × 2B = 57,344 B ≈ 56 KB/token → 4k context ≈ 224MB/seq。

## 4. 显存账本（ serving 时）

```
总占用 = 权重 + KV pool + activation/scratch + CUDA context/框架开销(≈1-3GB)
```
vLLM 的 `gpu_memory_utilization=0.9` 意思是：权重加载后，(0.9×总显存 − 权重 − 开销) 全划给 KV pool。

## 5. 量化换算

| 精度 | bytes/param | 7B 权重 | 32B 权重 |
|---|---|---|---|
| FP32 | 4 | 28 GB | 128 GB |
| FP16/BF16 | 2 | 14 GB | 64 GB |
| FP8/INT8 | 1 | 7 GB | 32 GB |
| INT4 | 0.5 | 3.5 GB | 16 GB |

注意 INT4 实际占用 > 0.5B/param：还有 per-group scale/zero 元数据（典型 group=128 → +3~4%）。

## 6. 4090 48G 实测锚点（跑完 hw_profile.py 填）

- D2D 带宽：___ GB/s（spec ~1008）
- H2D pinned：___ GB/s（PCIe 4.0 x16 理论 ~25）
- BF16 GEMM：___ TFLOPS
- ridge point = BF16 TFLOPS / D2D 带宽：___

## 7. 常用估算题模板

- "这个模型能开多大 batch？"→ (可用显存 − 权重 − 开销) / 每 token KV
- "4k 上下文 32 并发要多少 KV？"→ 每 token KV × 4k × 32
- "为什么 batch=64 后吞吐不涨？"→ 撞上 compute 墙（prefill 占比）或 KV 显存墙
