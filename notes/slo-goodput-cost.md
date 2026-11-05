# SLO Goodput + Token 成本实验（生产口径压测）

日期：2025-11-05 · vLLM 0.11.0 `bench serve`（Poisson 到达）· Qwen2.5-7B · RTX 4090
负载：random 512in/128out，各 160 请求

## QPS 扫描 → goodput 曲线

| 请求率 | 实际消化 | out tok/s | 峰值并发 | TTFT p99 | TPOT p99 |
|---|---|---|---|---|---|
| 2 rps | 1.95 | 249 | 13 | 136ms | 20.0ms |
| 4 | 3.80 | 486 | 21 | 54ms | 17.6ms |
| 8 | 7.20 | 921 | 33 | 65ms | 19.6ms |
| 16 | 12.89 | 1650 | 63 | 99ms | 22.6ms |
| 32 | 20.50 | 2623 | 142 | 191ms | 29.5ms |
| 64 | **27.04**（消化不动） | 3461 | 160 | 145ms | **31.6ms** |

**SLO 解读**（口径：TPOT p99 < 30ms 且系统不积压）：
goodput ≈ **20 req/s（~2600 out tok/s）**。超过此点后到达率 > 消化率，
TTFT 开始排队劣化（99→191ms）。QPS64 下峰值并发打满 160，
吞吐 3461 tok/s 即本工作负载的饱和吞吐。

## 功耗与成本

- 压测期 nvidia-smi 采样：峰值 **416W @ ~90% GPU util**
- 按饱和吞吐 3460 tok/s、400W 估算：≈ 0.116 J/token ≈ **8.6 tok/J**
- 折算：电费 $0.10/kWh → ~$0.012/百万 token（纯能费）；
  4090 租赁 ~¥2/h → **~¥0.05/百万 token 全成本**（12.5M tok/h 容量下）
- 面试口径：单卡 4090 跑 7B，百万 token 成本在分位级——
  「量化成本账」比报峰值 TFLOPS 更打动人

## 结构化输出开销（guided_json / xgrammar）

- 自由生成 400tok：61 tok/s
- JSON schema 约束 28tok 合法对象：~52 tok/s（**约 -15%**）
- 每步 ~1-3ms 的 grammar mask 计算——agent 场景真实存在的税

## 坑（记录）

- `vllm bench serve --model` 会去 HF 拉 tokenizer：离线环境必须
  `--tokenizer <本地路径>` + `HF_HUB_OFFLINE=1`，否则空跑重试。
- guided_json + "输出数组" prompt 出现过 2-token 早停（`[]` 合法完成）——
  约束过强时模型"选择"最短合法输出，值得注意。
