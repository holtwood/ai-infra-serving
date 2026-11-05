# M1 小结：手写 decode loop + 手撕题集（实测）

日期：2026-11-05 | 模型：Qwen2.5-0.5B-Instruct (bf16, /data/ai-learn/models)

## scratch-decode/decode_from_scratch.py 实测

- prompt 37 tokens：**prefill 295.0 ms**（含首轮 CUDA 初始化，偏高）
- decode：**13.54 ms/token → 73.9 tok/s**（greedy）
- KV cache 公式验证：24 layers × 2(K,V) × 2 KV-heads × 64 head_dim
  × 2B(bf16) × 2048 tok = **24.0 MiB** ✓
- 与 `model.generate` 输出对照：均通顺（greedy 下小模型输出有分叉，正常）

**关键数据解读（面试可直接讲）**：
decode 13.5ms >> 理论下限 ~1ms（0.5B bf16 权重 ~1GB ÷ 939GB/s 带宽）。
差距全部来自 Python 逐 token 驱动 + 每步数十次 kernel launch 的 CPU 开销
——这就是 vLLM 用 CUDA Graph 抓 decode 步、以及 continuous batching
摊薄 overhead 的动机。0.5B 小模型把这个 gap 放大得最清楚。

## handwrite/ 手撕题集 —— 远端 GPU 实测全 PASS

| 文件 | 内容 | 实测 |
|---|---|---|
| 01_mha_gqa_kvcache.py | MHA/GQA 手写 attention + KV cache decode | 对拍 SDPA max_err 1.6e-2(bf16)；decode 5 步 cache 形状正确 |
| 02_rope.py | RoPE 教科书交错版 + HF half-split 版 + naive 对拍 | max_err 4.8e-7；相对位置恒等式成立 |
| 03_stable_softmax_sampler.py | naive vs stable softmax、log_softmax、top-k/top-p/逆 CDF 采样 | e^1000 溢出 demo 成功；分布卡方误差 0.003 |
| 04_paged_kv_allocator.py | paged KV 池：freelist/block table/refcount/hash 前缀共享/COW/OOM | alloc/share/COW/OOM-reject/free 全 PASS |
| 05_backprop_derivation.md | softmax+CE、LayerNorm、attention VJP 推导 | 文档 |

**bug 记录**：02_rope.py 初版自检调用漏传 `pos_ids` ——
测试代码本身是考点的最好证明，修后 4.8e-7 精度。

## 下一步（M2 剩余 + M3）

- Triton/CUDA kernel 实测数据已出 → m2-summary.md
- vllm serve 起 0.5B 验证 serving 链路 → M3
