# T4：ngram spec × 内容熵 —— FAIL（提案零触发，诚实记录）

配置已确认合法（speculative.py: method/prompt_lookup_max/min），
EngineCore 日志显示 drafter 正常加载。但三档内容压测：
repetitive 55.7 / code 58.5 / prose 58.5 tok/s vs baseline
57.9/59.9/59.9 ≈ 零差异；/metrics 中
`spec_decode_num_drafts_total=0`、accepted=0——**draft 从未生成**。

排除项：无 disable 警告；temperature=0 下 spec 理论上应工作
（rejection=exact match）；并发 4 < 任何 batch 阈值。

未验证假设（留给下次）：① 0.11.0 的 ngram proposer 可能对
greedy/短 ctx 请求有隐式跳过路径（待读 ngram_proposer.py）；
② prompt_lookup_min=1 与 max=4 的窗口在该版本有 edge case。
可重试：temperature=0.7 + 更长上下文。
