# Prefix-Aware Routing 实验（M5 补）

日期：2025-11-05 · vLLM 0.11.0 ×2 实例 · Qwen2.5-0.5B · RTX 4090
脚本：`benchmarks/prefix_router.py`（12 前缀族 × 4 后缀 × 3 轮，每轮 48 并发）

## 设置

两个 vLLM 实例（:8000/:8001，各占 40% 显存），APC 默认开启。
- **rr**：round-robin 派发——同前缀族的请求被拆到两个实例
- **sticky**：前缀 md5 hash → 固定实例（SGLang cache-aware routing 的简化版）

两组之间重启实例保证冷启动公平。

## 实测 mean TTFT

| 轮次 | rr | sticky |
|---|---|---|
| rep0（全冷） | 181ms | 223ms |
| rep1 | **243ms** | **88ms** |
| rep2 | 85ms | 88ms |
| 整体 | 170ms | 133ms（-22%） |

## 解读

- **rep1 是核心差异**：sticky 第二轮同前缀必落同实例 → APC 全命中（88ms）；
  rr 有 ~50% 概率落到没缓存该前缀的实例 → 重付 prefill（243ms，比冷启动还慢，
  因为并发排队叠加）。
- **rep2 收敛**：rr 三轮下来已把 12 个前缀烘进两个实例，缓存容量放得下 → 
  追平 sticky。说明 sticky 的优势 = **更快达到稳态命中率 + 单副本缓存**。
- 前缀数 >> 实例数 × 缓存容量时（生产常态），rr 的 miss 永不收敛——
  本实验 12:2 的温和比例下整体仍有 22% 的 TTFT 优势。
- 这正是 SGLang RadixAttention 路由、llm-d KV-aware routing 的动机：
  把「前缀亲和」从 engine 内提升到调度层。

## 坑

- 实验间必须重启实例：rr 跑完两个后端缓存全热，直接测 sticky 会假性变好。
- 远端杀进程别用含服务名字的 grep/pgrep 直写——会匹配到 ssh 包装 shell 自杀，
  用 `[q]wen05` 括号技巧。
