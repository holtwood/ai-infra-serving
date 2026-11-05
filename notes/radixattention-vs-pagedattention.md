# RadixAttention vs PagedAttention：精读笔记

> SGLang 部署因依赖问题未跑通（apache-tvm-ffi==0.1.0b15 未发布），
> 按 README 预案做源码/论文精读。对照数据仍以 vLLM 实测为准。

## 一句话区分

- **PagedAttention（vLLM）**：KV 按固定大小 block 分页，page table
  寻址——解决"显存碎片 + 不能共享"
- **RadixAttention（SGLang）**：KV 按 radix tree（前缀树）组织，
  树节点 = token 前缀 —— 解决"前缀复用 + 缓存感知调度"

## PagedAttention（我手写过 allocator，handwrite/04）

- 物理显存切成 block（默认 16 tok/块），逻辑序列经 block table
  映射到物理块；按需 append，废块回 freelist，O(1) alloc/free
- 内部碎片 ≤ block_size-1 tokens；外部碎片 = 0（等大小块）
- 前缀共享 = 多 seq 的 block table 指到同一物理块 + refcount；
  分叉时 copy-on-write
- vLLM V1 上 APC 的 hash：block 内容 hash → block pool 字典，
  跨请求复用（实测 hit rate 95.5%，TTFT 4x）

## RadixAttention（SGLang 论文 + 源码结构）

- KV 以 **radix tree** 存：每条边是一段 token 序列的压缩前缀，
  节点持有该前缀的 KV 页引用
- 新请求到来：与树做**最长前缀匹配** → 命中的部分直接复用，
  只对尾部增量做 prefill —— 等价于 APC，但前缀匹配是树结构
  天然操作，且支持**任意位置的前缀**（不限于请求开头？——
  注意：实际也要求从 token 0 开始的公共前缀，与 APC 相同约束）
- 驱逐：**LRU**——叶子向上收缩；vLLM APC 是 hash-pool 满了
  按引用计数+LRU 驱逐，思想同源
- **cache-aware scheduling**（SGLang 独有亮点）：路由层按
  "哪个 worker 持有最长匹配前缀"分发请求 —— 集群级 hit rate 提升，
  这是树结构暴露给调度的接口，PagedAttention 没有直接对应物

## 对照表

| 维度 | PagedAttention | RadixAttention |
|---|---|---|
| 数据结构 | block table + hash pool | radix tree |
| 匹配粒度 | 16-token block（hash 命中必须整块相同）| token 级前缀（可共享部分块）|
| 驱逐 | refcount=0 可回收，满时 LRU | 树上 LRU 收缩 |
| 集群路由 | 无内建 | cache-aware 路由（核心卖点）|
| 适合场景 | 通用 serving | 高前缀共享：agent/多轮对话/模板化 prompt |

## 面试口径

- "APC 和 RadixAttention 本质都是前缀复用：vLLM 用 hash 表做
  块级匹配，SGLang 用前缀树做 token 级匹配 + 把命中率暴露给路由器"
- "哪个好？"——短上下文独立请求差别小；agentic/多轮/共享 system
  prompt 场景 RadixAttention 命中率和路由收益更高
- 实测锚点：vLLM APC on/off TTFT 差 4x、hit 95.5%（本 repo m3）

## 参考

- SGLang RadixAttention 论文：arxiv.org/abs/2312.07104
- vLLM PagedAttention：arxiv.org/abs/2309.06180
- 代码：sglang/srt/mem_cache/radix_cache.py；vllm/v1/core/kv_cache_manager.py
