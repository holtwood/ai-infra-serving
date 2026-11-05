# RESULTS —— RTX 4090 上 LLM Serving 实测汇总

全部数字来自真实运行（vLLM 0.11.0 / torch 2.8.0+cu128 / 单卡 RTX 4090 48G）。
每行都能用 `benchmarks/` 下脚本复现。详细过程见 `notes/`。

## 硬件基线（M0）

| 指标 | 实测 | 备注 |
|---|---|---|
| D2D 带宽 | 939 GB/s | 理论 ~1008 |
| BF16 GEMM | 161.8 TFLOPS | ~98% dense 峰值 |
| H2D pinned/pageable | 25.1 / 22.0 GB/s | pinned 价值 +14% |

## vLLM 生产口径（Qwen2.5-7B bf16）

| 场景 | 数字 | 出处 |
|---|---|---|
| 单流 TPOT | 16.1ms | ≈15GB 权重 / 939GB/s，memory-bound 验证 |
| 64 并发总吞吐 | 2399 tok/s | TPOT 仅退化到 20.5ms |
| Prefix caching | TTFT 234→54ms（4x） | hit rate 95.5% |
| **Poisson goodput**（TPOT p99<30ms） | **~20 req/s / 2600 out tok/s** | bench serve QPS 扫描 |
| 饱和吞吐 | 3461 out tok/s | QPS64 峰值并发 160 |
| Chunked prefill 尾延迟 | 受害者 max ITL 377→55ms | token 预算 8192→512 |
| FP8 KV cache | 容量 2x（506k→1013k tok） | TTFT +10~26%（sm_89 无 FP8 算力） |
| Prefix-aware routing | TTFT -22% vs round-robin | 双实例，rep1 88 vs 243ms |
| CUDA Graph | e2e -64% | 0.5B eager→graph |
| ngram spec decode | e2e 4x（重复输出） | rep_pen 冲突坑已记录 |
| guided_json 开销 | ~-15% tok/s | xgrammar mask 每步 ~1-3ms |
| decode 步 GPU 空转 | **69%** | profiler：169 linear/步 dispatch 开销 |
| 饱和功耗 | 416W，8.6 tok/J | ~¥0.05/百万 token 全成本估算 |

## 自研 mini-serving（教学引擎）

| 引擎 | TPOT@并发 | 总吞吐 |
|---|---|---|
| per-seq forward | 13→202ms 线性退化 | 79 tok/s 封顶 |
| batched forward | 13→26ms 几乎平 | **973.6 tok/s（12.3x）** |
| vs vLLM 差 | — | 973 vs 2762：pad KV 浪费+无 graph+逐 seq prefill |

## Kernels（Triton/CUDA，全部带 bottleneck→fix→数字）

fused softmax 942GB/s｜RMSNorm 3-7x｜matmul 160TF 超 cuBLAS｜
reduction 四版 468→1259GB/s｜vector add 955GB/s（95% 带宽）

## 复现

```bash
# vLLM SLO 扫描（离线环境必须本地 tokenizer）
HF_HUB_OFFLINE=1 vllm bench serve --model qwen7b \
  --tokenizer <model_dir> --dataset-name random \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 160 --request-rate <QPS> --ignore-eos
# mini-serving
cd mini-serving && MINI_ENGINE=batched python server.py
python benchmarks/chunked_prefill_itl.py   # ITL 干扰
python benchmarks/prefix_router.py --mode sticky  # 路由对照
python benchmarks/kv_context_sweep.py      # 长上下文曲线
python benchmarks/profile_decode_step.py   # profiler 时间线
```
