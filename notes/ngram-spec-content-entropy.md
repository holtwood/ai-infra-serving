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

## 根因定位（源码实锤）

ngram_proposer.batch_propose 收到 `spec_decode_unsupported_reqs`
集合——请求采样参数不兼容 spec 时被跳过。0.11.0 默认继承模型
generation_config（Qwen2.5-7B 带 repetition_penalty:1.05）→
全部请求进 unsupported 集合 → draft=0。**与此前 spec-decode
issue 调查同一条路径回旋镖命中。**
验证方法（下次）：`--generation-config vllm` 重启使请求不带
rep_pen → draft 应非零；或请求级传 repetition_penalty=1.0。
附注：8192 阈值只是 numba 线程启发式，非根因。

## 验证跑（--generation-config vllm）✅ 根因实锤

drafts=1957、draft_tokens=9765（每步 5 提案）、accepted=1606
→ 平均接受 ~0.82 tok/步（16%）。tok/s：code 55.6/prose 56.1
≈ baseline 无净加速（低接受率下 spec 开销对冲）；
repetitive 27.5 为混杂项（输出 134 tok 提前收尾，wall 被
TTFT 主导）。结论：根因=generation_config rep_pen 1.05（实锤）；
ngram 在低熵内容上理论增益大，但本 workload 接受率仅 16%，
**spec-decode 收益高度依赖输出可预测性分布**（原结论成立）。
