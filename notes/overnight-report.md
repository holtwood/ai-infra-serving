# 通宵报告（2025-11-06 凌晨）——一句话：3 PASS + 1 FAIL(诚实负结果) + 1 微基准，全推

| Task | 状态 | 核心数字 |
|---|---|---|
| T1 equiv 重跑 | PASS | 仍 r3@17 分叉 → 排除 position bug，确认 bf16 batch-invariance |
| T2 compile vs graph | PASS | eager 13.75 / compile 8.37(间隙48%) / 手写graph 5.46ms |
| T3 AWQ 质量 | PASS | PPL: AWQ-32B 11.75 < 7B-bf16 15.56；「量化是使能」闭环 |
| T4 ngram spec | FAIL | draft 零触发/零加速，config 合法——假设待查 |
| T5 KV swap | PARTIAL | 微基准：swap 往返 36.7ms@8k vs recompute ~500ms ≈14x；引擎接入未完成 |
| T6 prefix cache | SKIP | 会话上下文耗尽，未开始 |
| T7 split-K kernel | SKIP | 同上 |

PENDING_USER：无对外发布动作。残留：无（GPU 已验证 1MiB）。
明日建议：T5 引擎接入（swap 结论已立）、T7 split-K、T4 假设验证。
