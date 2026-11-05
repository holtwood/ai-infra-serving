# SGLang 关键机制精读：Overlap Scheduler + RadixCache 细节

日期：2025-11-05 · 源码/官方文档精读（本机装不上 sglang，用读代码替代实测）
对应面经原题："overlap scheduler 藏了什么、怎么在 token 还没生成时
就组下一个 batch" / "RadixCache 怎么驱逐"

## Overlap Scheduler（vLLM 没有的机制）

**问题**：vLLM 的 step 是串行的——forward 完，CPU 做 sample 后处理
（append token、查 stop string、组下一步 batch），这段时间 GPU 空转。
我们 profiler 实测 mini-serving decode 步 **69% 时间 GPU 在等 CPU**——
就是这个问题的量化。

**SGLang 的做法**：调度循环提前一步——GPU 跑 batch N 时，CPU 已经在
组 batch N+1（N 的输出 token 还没出来，就用**占位符 future token**
排进 N+1 的输入位置）。等 N 完成、真 token 到位，占位符被替换后
直接 launch N+1——CPU 调度时间和 GPU forward 重叠，间隙≈0。

代价（面试陷阱题）：① 占位符机制增加复杂度（FutureMap 解析）；
② 被替换的 batch 里若有请求本步该 finish，实际多跑了一步浪费一点
算力；③ 对 TP>1 跨进程同步更 tricky。vLLM V1 的答案是
async_scheduling + 独立 detokenizer——思路同源：把 CPU 工作移出
GPU 关键路径。

## RadixCache 细节（超出"前缀树+LRU"的部分）

- **存的是 KV block 引用，不是文本**：节点带 ref count，
  in-flight 请求锁住的节点不能被驱逐（lock refs）
- **驱逐是 leaf-first LRU**：只驱逐叶子——内部节点是共享前缀，
  驱了子树全废。所以"热点短前缀"天然被保护
- **前缀匹配按 token 不是按字符**（路由层的近似树才按字符，
  SGLang 官方 router 为省 tokenize 开销存 raw chars）
- **前缀感知调度的 128-request 截断**：匹配时最多回看 128 条历史
  请求建树——平衡命中率和建树开销的工程取舍
- **vs vLLM APC**：vLLM 是 block-hash → KV block 的 flat map，
  无树结构、无共享节点概念；RadixCache 的 trie 天然表达"前缀共享
  前缀"的多级复用（vLLM 用 hash 链也达到了类似效果，结构不同）

## 对照我们实测

| 机制 | SGLang | 我们对应的实测 |
|---|---|---|
| CPU 掩盖 | overlap scheduler | mini-serving 69% GPU 空转（profiler） |
| 前缀路由 | cache-aware router + 近似树 | prefix_router.py sticky vs rr -22% |
| 驱逐 | leaf-first LRU + lock refs | — |

## 面试一句话

"SGLang 两个杀手锏：overlap scheduler 用 future-token 占位符把
下一步 batch 的 CPU 组装叠进本步 GPU forward（vLLM 用 async
scheduling+独立 detokenizer 达到同效）；RadixCache 是带 ref count
的 KV trie，leaf-first LRU 驱逐、in-flight 前缀不可逐——
比 vLLM 的 block-hash map 更适合多级前缀复用。"
