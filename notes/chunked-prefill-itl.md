# Chunked Prefill / ITL 干扰实验（M5 补）

日期：2025-11-05 · vLLM 0.11.0 · Qwen2.5-7B-Instruct · RTX 4090
脚本：`benchmarks/chunked_prefill_itl.py`（8 条背景短请求流 + 注入 7000tok 长 prompt）

## 核心问题

单个长 prefill 是一次大 GEMM（数百 ms）。若它在调度步里独占 GPU，
同 batch 的 decode 请求会被「队头阻塞」——ITL 尾部毛刺。这就是 P/D 分离存在的动机。

## 重要发现：vLLM V1 无法关闭 chunked prefill

`--no-enable-chunked-prefill` 被 CLI 接受（non-default args 显示 False），但
`arg_utils.py::_set_default_args` 对非 pooling 任务**无条件**重置为 True：
> "V1 always uses chunked prefills and prefix caching for non-pooling tasks."

所以「开/关」对照不可做，改用 **`--max-num-batched-tokens`**（每步 token 预算）
做等效对照：预算大 ≈ prefill 一把梭；预算小 = 紧切片。

## 实测（7000tok 长 prompt 注入 8 条短请求流，各 96 output tok）

| 每步预算 | ITL p50 | p95 | p99 | **max** | 长请求 TTFT |
|---|---|---|---|---|---|
| **8192**（prefill 一步跑完） | 16.6ms | 17.7ms | 220.7ms | **377.2ms** | 436ms |
| **2048**（默认） | 16.6ms | 18.8ms | 135.5ms | 188.2ms | 305ms* |
| **512**（紧切片） | 16.7ms | 52.5ms | 54.6ms | **54.8ms** | 461ms |

*2048 行长 prompt 为 6000tok，TTFT 不同量纲仅供参考趋势。

## 解读（面试可直接讲）

- **尾延迟 7x 改善**：紧切片下 max ITL 377→55ms。切片让 decode 每隔一个小步就能插进来。
- **TTFT 代价极小**：长请求 TTFT 436→461ms（+6%）。prefill 总计算量没变，只是被切开。
- **p95 反而升高**（17.7→52.5ms）：512 预算下每个 decode 步都要和 prefill chunk 共享，
  常规 token 间隔被系统性拉长 ~3x —— 这就是「中位数换尾部」的 trade-off。
- **P/D 分离动机**：chunked prefill 缓解但治不了根——p50 还是被拖长。彻底的解法是
  prefill/decode 物理分离到不同实例（TTFT 敏感业务去 prefill 池，ITL 敏感的在 decode 池）。

## 踩坑记录

- APC 会破坏实验：两次跑相同长 prompt → 命中 prefix cache → prefill 免费，
  max ITL 假降到 92ms。修复：prompt 加随机盐（`make_long_prompt`）。
- V1 强制开启证据：`logs/vllm_7b_nochunk.log`（arg False 但 engine 仍
  `chunked_prefill_enabled=True`）。
