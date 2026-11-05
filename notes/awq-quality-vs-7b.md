# T3：AWQ-32B vs 7B-bf16 质量+速度三角（vLLM 离线实测）

本地 notes 语料 PPL（prompt_logprobs 法）：**7B-bf16 = 15.56，
AWQ-32B = 11.75**——4-bit 的 32B 质量仍优于 bf16 的 7B。
5 题 greedy（logs/ppl_compare.json）：两模型均正确完成推理/代码/
翻译，无可见质量塌缩。

速度（前测）：AWQ-32B conc32=631tok/s、ITL p50~50ms vs 7B ~8ms。
显存：AWQ ~19.4GB 权重装入 48G，bf16-32B(~65GB)不可能。

结论弧线：量化是**使能技术**——用 ~3x TPOT 换「同卡上跑 4x 大
模型且 PPL 更优」。面试口径：同价位选大模型+激进量化 >
小模型+bf16（质量/显存维度），代价是延迟。
