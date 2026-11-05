# torch.profiler decode 单步时间线（M5 补）

日期：2025-11-05 · mini-serving `engine_batched.py` · Qwen2.5-0.5B · RTX 4090
脚本：`benchmarks/profile_decode_step.py`，trace: `logs/decode_trace.json`
（可拖入 ui.perfetto.dev 看时间线）

## 关键数字

- 10 个 batched decode step（B=8 running）：墙钟 225.8ms（22.6ms/step）
- **GPU 实际忙时 69.9ms → 间隙 ≈ 69%**
- Self CPU 210.5ms ≈ 墙钟 → **纯 CPU-bound**：Python/aten dispatch 比 GPU 慢 3 倍
- 每步 169 个 aten::linear（24 层 × ~7，与模型结构吻合）、970 次 mm
- GPU 时间分布：cutlass bf16 GEMM 41% + aten::mm 38% ≈ 80% 在权重 GEMM
  （decode memory-bound 的直接证据：时间花在搬权重）

## 解读（面试口径）

1. **69% 间隙就是 CUDA Graph 的靶子**：每步 ~170 次 kernel launch，
   每次 ~10-20μs Python+dispatch 开销串行排队，GPU 大部分时间在等 CPU。
   vLLM `full_cudagraph` 把整个 decode 步录成一张图回放 → 这 69% 基本归零，
   与之前实测 vLLM eager→graph e2e -64% 互相印证。
2. **decode 的 GEMM 主导**（80% GPU 时间在 mm/linear）证实 memory-bound
   权重搬运模型：0.5B×8 batch 仍跑不满 tensor core，瓶颈是显存带宽。
3. mini-serving 13ms/token 地板价 ≈ Python 开销，不是模型极限——
   同一模型 vLLM 2.6ms 的差距基本全在 dispatch/graph/fused kernel。

## 已知局限

- profiler 本身有开销，22.6ms/step 高于非 profile 的 ~19ms；间隙比例仍可信。
- trace 为 10 步聚合；单步火焰图细节看 chrome trace 文件。
