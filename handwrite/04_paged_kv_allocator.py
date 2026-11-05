"""手撕系列 04：分页 KV 块分配器 —— vLLM block manager 极简版（纯 Python）

核心机制（白板高频题，要短而正确）：
- 物理块池：显存切成 block_size 个 token 的槽位，freelist 管空闲块
- block table：每条 seq 维护 逻辑块号 → 物理块 id（PagedAttention 按表寻址）
- 引用计数：ref>1 = 被多条 seq 共享（prefix caching 的收益来源）
- 内容寻址：逻辑块按内容 hash 建映射，相同前缀的 seq 命中即共享、ref+1
- copy-on-write：写共享块必须先摘出私有副本，否则改到别人的前缀
- 真 vLLM 还多：ref=0 块 LRU 缓存、抢占(swap/recompute)、per-token slot 寻址
运行：python 04_paged_kv_allocator.py
"""
from collections import deque


class PagedKVPool:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.n, self.bs = num_blocks, block_size
        self.freelist = deque(range(num_blocks))  # 空闲物理块 id
        self.ref = [0] * num_blocks               # 物理块引用计数
        self.data = [None] * num_blocks           # 教学版存 token；真系统是 KV 槽位
        self.h2p = {}                             # 内容 hash → 物理块（共享索引）
        self.p2h = [None] * num_blocks            # 物理块 → 内容 hash（free 时 O(1) 摘）
        self.tables = {}                          # seq_id → block table [物理块id,...]

    def _new(self) -> int:
        if not self.freelist:
            raise RuntimeError("KV pool OOM")     # 真系统此时：拒入/抢占队尾
        b = self.freelist.popleft(); self.ref[b] = 1
        return b

    def new_seq(self, sid: str) -> None: self.tables[sid] = []

    def map_block(self, sid: str, key) -> int:
        """追加一个逻辑块。key=块内容 tuple，本身即内容 hash（真 vLLM 用
        prefix_hash+块内 token 算 hash）。命中 → 共享物理块 ref+1
        （prefix caching）；未命中 → 拿新块。"""
        if key in self.h2p:
            b = self.h2p[key]
            self.ref[b] += 1
        else:
            b = self._new()
            self.h2p[key], self.p2h[b] = b, key
            self.data[b] = list(key)
        self.tables[sid].append(b)
        return b

    def write_block(self, sid: str, idx: int, key) -> int:
        """写第 idx 个逻辑块（key=新内容）。独占→原地写并换 hash；共享(ref>1)→ COW。"""
        b = self.tables[sid][idx]
        if self.ref[b] > 1:
            self.free(b)                          # 旧块 ref-1（仍被共享，不回收）
            b = self._new()                       # ref>1 ⇒ 减后 ≥1 ⇒ 新块必不同
            self.tables[sid][idx] = b
        else:
            old = self.p2h[b]                     # 独占块换内容：旧 hash 映射作废
            if old is not None and self.h2p.get(old) == b:
                del self.h2p[old]
        prev = self.h2p.get(key)                  # 同 key 另有主：摘其 hash 标记防误删
        if prev is not None and prev != b:
            self.p2h[prev] = None
        self.h2p[key], self.p2h[b], self.data[b] = b, key, list(key)
        return b

    def free(self, b: int) -> None:
        assert self.ref[b] > 0, "double free"
        self.ref[b] -= 1
        if self.ref[b]:
            return                                # 还有别的 seq 在用
        k = self.p2h[b]
        if k is not None and self.h2p.get(k) == b:  # 防删到同 key 的新主
            del self.h2p[k]
        self.p2h[b] = self.data[b] = None
        self.freelist.append(b)

    def free_seq(self, sid: str) -> None:
        for b in self.tables.pop(sid):
            self.free(b)

    @property
    def usage(self) -> float:
        return 1.0 - len(self.freelist) / self.n


if __name__ == "__main__":
    p = PagedKVPool(num_blocks=4, block_size=16)
    p.new_seq("A"); p.new_seq("B")

    # 1) 分配 + block table 记账（key=块内容 tuple，即内容 hash）
    b0 = p.map_block("A", (1, 2, 3))
    b1 = p.map_block("A", (4, 5))
    assert p.tables["A"] == [b0, b1] and p.ref[b0] == p.ref[b1] == 1

    # 2) 共享：B 的相同前缀块命中 → 同一物理块，ref=2，不再占新块
    assert p.map_block("B", (1, 2, 3)) == b0
    assert p.ref[b0] == 2 and len(p.freelist) == 2  # 共享不占新块

    # 3) COW：A 改写共享块 → 拿到新物理块；B 看到的还是旧内容
    b2 = p.write_block("A", 0, (1, 2, 9))
    assert b2 != b0 and p.tables["A"][0] == b2
    assert p.ref[b0] == 1 and p.data[b0] == [1, 2, 3] and p.data[b2] == [1, 2, 9]
    assert p.h2p[(1, 2, 3)] == b0                 # 共享索引仍指向 B 的块

    # 4) 缺页拒绝：池满（b0,b1,b2 占 3/4，再要 2 块第二块 OOM）
    p.map_block("B", (7, 7))                      # 4/4 满
    try:
        p.map_block("B", (8, 8))
        raise AssertionError("应该 OOM")
    except RuntimeError:
        pass

    # 5) 释放：A 归还 2 块；B 仍独占 b0 → 共享索引对 B 依然有效
    p.free_seq("A")
    assert len(p.freelist) == 2 and p.ref[b0] == 1
    assert (1, 2, 9) not in p.h2p                 # 独占块 hash 已随释放摘除

    # 6) 全释放归零；hash 表不留残
    p.free_seq("B")
    assert len(p.freelist) == 4 and not p.h2p and p.usage == 0.0
    print("alloc / share / COW / OOM-reject / free 全部 PASS")
