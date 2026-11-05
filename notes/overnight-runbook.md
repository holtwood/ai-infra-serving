# 通宵执行 Runbook（ai-infra-serving，约 8-9h GPU）

给执行 agent 的自洽作战手册。按序执行 T1→T7，每项独立交付、
独立 commit。**任何任务跳过/失败必须如实记录并继续下一项。**

## 0. 全局约束（最高优先级，违反即事故）

**环境**：远端 `ssh -p 40003 root@222.211.217.173`，工作目录
`/data/ai-learn`，venv `.venv/bin/python`；本地仓库
`/home/shane/github/ai-learn`，推送 `holtwood/ai-infra-serving`。
GPU RTX4090-48G（49140MiB）。`/model` 只读，不许动。

**共享机器守卫协议（每个 GPU 步骤前必做）**：
1. `nvidia-smi --query-compute-apps=pid --format=csv,noheader` 列出
   GPU 进程，逐一 `ls /proc/<pid>` + `ps -p <pid> -o cmd=` 核实
2. 发现非本 runbook 清单内的活进程（/proc 存在且 cmd 非我方）→
   `sleep 600` 重查，最多 3 次 → 仍占用则**跳过该任务**，在
   overnight-report 记 `SKIP(占用)`，进入下一任务
3. 本 runbook 起的所有 PID 追加写入 `/tmp/overnight_pids`
   （每行一个）；清理**只**按此清单：对每个 PID 先取其子进程
   `pgrep -P <pid>` 一并 kill -9。vLLM 是两棵树：
   `vllm serve` API 父进程 + `VLLM::EngineCore` 子进程，都要杀
4. 每任务结束验证 `nvidia-smi memory.used < 2000MiB` 才进下一项；
   若 `compute-apps` 显示条目但 `/proc/<pid>` 不存在 = 驱动延迟，
   等 60s 复测，不追杀

**禁止**：发布 issue/PR/评论、提交外部榜单、改 git config、
改 /model、kill 清单外任何进程、sudo。需对外决策 → 在报告中
标 `PENDING_USER` 并跳过。

**交付节奏**：每任务完成 → 写对应 notes → 更新
`benchmarks/RESULTS.md` → `scp` 同步远端改动文件 → 本地
`git add -A && git commit && git push`。中断恢复：
`git log` + overnight-report 已写部分判断从哪继续。

---

## T1. 等价性测试重跑（~20min，无服务，直接跑）

目的：graph 引擎修了 `position_ids = ctx-1` 的 off-by-one，
验证 batched equiv 测试 1/4 分叉是否同源。
```bash
cd /data/ai-learn && .venv/bin/python mini-serving/test_batched_equiv.py
MINI_ENGINE=graph .venv/bin/python mini-serving/test_batched_equiv.py  # 若支持
```
产出：在 `notes/m4-summary.md` 等价性小节追加判定——
「分叉消除（position bug 实锤）/ 仍存在（确为 bf16 batch-
invariance）」。无论哪种都是有效信息。

## T2. torch.compile vs 手写 graph（~40min）

目的：框架白给的图 vs 手写图的差距。
步骤：`engine_batched.py` decode forward 包
`torch.compile(self.model, mode="reduce-overhead")`（env 开关
`MINI_COMPILE=1`），warmup 3 步后用 `benchmarks/
profile_decode_step.py` 同款计时跑 50 步取均值；对照已有
eager 13.75ms / 手写 graph 5.46ms（max_len=256）。compile 若因
StaticCache 动态 shape 反复 recompile，如实记录 guard/recompile
行为——这本身是素材。
产出：`notes/torch-compile-vs-graph.md`。

## T3. AWQ-32B 质量对照（~60min，两模型轮流，勿并起）

目的：补「量化是使能不是加速」的质量半边。
已有速度：AWQ-32B conc32=631tok/s、ITL p50 49.8ms、KV 池 85k tok。
步骤（全程离线，无外网依赖）：
1. 写 `benchmarks/ppl_compare.py`：transformers 直载模型，
   对同一段固定文本算 perplexity。语料取本地文件拼接
   （如 `notes/*.md` 英文段落 + README，~2000 token），
   滑窗 512、stride 256；先载 7B-bf16 算 PPL，del 模型+
   `torch.cuda.empty_cache()`，再载 AWQ-32B 算 PPL
2. 5 道固定题（2 推理/2 代码/1 翻译，题目写死在脚本里）两模型
   greedy 各答一遍，输出存文件供人工对评（agent 只做粗评：
   是否完成指令/明显错误数）
产出：`notes/awq-quality-vs-7b.md`（PPL 对照 + 答题样本）。

## T4. ngram spec × 内容熵（~60min）

前置验证（必做）：`grep -rn "prompt_lookup\|ngram"
/data/ai-learn/.venv/lib/python3.11/site-packages/vllm/config/`
确认 0.11.0 的 speculative-config key 名；若 key 不同按源码改。
```bash
.venv/bin/vllm serve /data/ai-learn/models/Qwen2.5-7B-Instruct \
  --served-model-name qwen7b --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":1}'
.venv/bin/python benchmarks/ngram_spec_content.py qwen7b
curl -s localhost:8000/metrics | grep -E "spec_decode"
# baseline：去掉 spec config 重启，同脚本取 tok/s 对照
```
产出：`notes/ngram-spec-content-entropy.md`——三档 tok/s +
接受 token 数/draft 数 = 平均接受长度。结论模板：「接受率∝输出
可预测性，复读 ~X tok/step、代码 ~Y、散文 ~Z」。

## T5. KV swap-to-CPU（~2h，代码任务）

目的：preemption 只测了 recompute 代价；造 swap 替代路径。
步骤：`mini-serving/engine_batched.py`
1. `ReqStatus` 加 `SWAPPED`；`swap_out(r)`：KV 张量
   `.to('cpu', non_blocking=True)` 进 pinned buffer；
   `resume(r)`：拷回 + 状态回 RUNNING
2. 挤压条件复用 preemption 实验（小 max_len + 高并发长输出），
   KV 池 >90% 时选最长 ctx 请求 swap（而非 recompute）
3. 测：swap 往返耗时 vs 同 ctx 重新 prefill 耗时，
   ctx ∈ {512, 2048, 8192} → 画盈亏交叉点
   （PCIe gen4 ~25GB/s；KV 56KiB/tok → 8k ctx 单向 ~18ms）
产出：`notes/kv-swap-vs-recompute.md` 含交叉阈值 +
「vLLM V1 只 recompute、SGLang 有 radix→CPU swap」定位。

## T6. engine 级 prefix cache（~2h，代码任务）

目的：路由层 APC 有数据，引擎内实现是自己的证据。
步骤：`mini-serving/allocator.py`
1. 块内容哈希：`(parent_block_hash, token_ids_tuple)` →
   block_id，存 `dict`；prefill 时按 prompt 切 block_size
   逐块查表，命中块直接引用（不重复 forward 该段）
2. 驱逐：LRU 或引用计数归零 + 显存压力触发，策略写注释
3. 验证：同一 500-token 前缀 × 8 条不同后缀请求，
   第 2 条起 TTFT / 第 1 条 TTFT（对标 vLLM APC 实测 4x）
产出：`notes/engine-prefix-cache.md`。

## T7. 旗舰：paged kernel v2 split-K（剩余全部时间 ~2-3h）

目的：v1 单 program 串行扫块，长 ctx 扩展差；vLLM 用 split-K
分段并行 + combine。
步骤：`kernels/triton/05_paged_attention.py` 旁建 `06_paged_splitk.py`
1. `program_id` 加 SPLIT 维：每 program 处理
   `ceil(n_blocks/SPLIT)` 块的连续区间，输出部分 (m,l,acc)
2. combine kernel：对各段 (m,l,acc) 做 log-sum-exp 归并
3. bench：ctx ∈ {1k, 4k, 16k}，SPLIT ∈ {1,4,8,16}，
   对照 v1 找拐点
进阶（时间富余）：接进 `engine_graph.py`——StaticCache 换
块池 + 每 seq block_table，graph 内调 kernel。做不到就如实记
「HF 缓存所有权挡住 kernel 注入 = vLLM 自研后端的理由」。
产出：`notes/paged-kernel-v2.md`。

## 收尾（无论做到哪必做）

写 `notes/overnight-report.md`：

```markdown
| Task | 状态 | 核心数字 | 文件 |
| T1 equiv | PASS/FAIL/SKIP | | |
...
未决事项（PENDING_USER）：...
```

全部 commit + push。报告第一行给一句话总结，供用户晨起速览。

## 时间预算（参考，非硬约束）

T1 20m · T2 40m · T3 60m · T4 60m · T5 2h · T6 2h · T7 剩余
合计 ~9h 满档；T7 是弹性项，前面超时就压缩它。
