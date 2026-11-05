"""M4: 迷你 LLM serving 引擎 —— EngineCore（纯逻辑，无 HTTP/asyncio）

面试原题：手写一个最小 serving engine 主循环。三件套：
    Scheduler    每步决定跑谁：FCFS + token 预算 + KV 准入
    KVBlockPool  分页 KV 记账：按需分配、池满触发抢占
    EngineCore   step() = 一次迭代：组 batch → forward → 采样 → 释放

与 vLLM V1 的对应 & 刻意简化（面试要主动讲差距）：
  * vLLM 把 N 条 seq 拼成 flattened batch 走【一次】 forward，
    PagedAttention 按 block table 寻址各 seq 物理 KV 块，权重只搬一遍
    （decode memory-bound，batch 是最便宜的吞吐）。本版【逐 seq
    forward】：逻辑等价但权重每 seq 搬一遍；省 padding/掩码/块寻址。
  * KVBlockPool 只记账：真 KV 在各请求的 HF DynamicCache 里，不是
    分页物理显存；但"池满→准入控制+抢占"链路是真实的。
  * 无 prefix caching / CUDA graph / chunked prefill / swap。
"""
import math, time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

BLOCK_SIZE = 16  # KV 块大小（token/块），同 vLLM 默认 block size


class ReqStatus(Enum):
    PREFILL = auto()  # 在 waiting，待（重）做 prefill
    RUNNING = auto()  # 在 running，decode 中
    DONE = auto()     # 完成或失败，待 API 层收尾


@dataclass(eq=False)  # eq=False：past_key_values 是 Cache 对象，禁止逐字段比较
class Request:
    """一条请求的全部状态（对应 vLLM 的 Request/Sequence）。"""
    req_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    arrival_time: float = field(default_factory=time.time)  # wall clock，SSE created 用
    status: ReqStatus = ReqStatus.PREFILL
    generated: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)  # 占用的物理块（记账）
    past_key_values: object = None                      # 真 KV：HF DynamicCache
    finish_time: float = 0.0
    finish_reason: str | None = None                    # stop | length | error

    @property
    def context_len(self) -> int:
        return len(self.prompt_ids) + len(self.generated)

    def context_ids(self) -> list[int]:
        # preempt(recompute) 后要整体重算 KV：已生成 token 并入上下文
        return self.prompt_ids + self.generated


class KVBlockPool:
    """分页 KV 池——教学版只做容量记账。

    vLLM：显存切成 block_size token 的槽位；seq 的 block table 记
    逻辑块→物理块 映射，PagedAttention 按表寻址，按需分配 → 碎片
    <1 块、可 copy-on-write 共享前缀。本版块不存真数据，但
    alloc/free/池满引发的调度行为（拒入、抢占）与真系统一致。
    """
    def __init__(self, num_blocks: int, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.total = num_blocks
        self._free = deque(range(num_blocks))
        self.preemptions = 0

    def needed(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    def can_alloc(self, n: int) -> bool:
        return n <= len(self._free)

    def alloc(self, n: int) -> list[int]:  # 不足返回 []，调用方先 can_alloc
        return [self._free.popleft() for _ in range(n)] if self.can_alloc(n) else []

    def free(self, ids: list[int]) -> None:
        self._free.extend(ids)

    @property
    def usage(self) -> float:
        return 1.0 - len(self._free) / self.total


@dataclass
class SchedConfig:
    max_batched_tokens: int = 2048  # 单步 token 预算 = vLLM max_num_batched_tokens
    max_num_seqs: int = 32          # 单步最大并发 seq 数 = vLLM max_num_seqs


class Scheduler:
    """FCFS + token 预算 + KV 准入（对应 vLLM V1 Scheduler 的每步决策）。

    vLLM 同样先给 running 每条留 1 token decode 预算，剩余按到达序
    喂给 waiting 的 prefill。差别：vLLM 预算耗尽即 break，且能把超长
    prompt 切片（chunked prefill）填进预算缝；本版整条进/整条等，
    不满足条件时跳过看下一个（短请求可搭车）。"""
    def __init__(self, pool: KVBlockPool, cfg: SchedConfig):
        self.pool, self.cfg = pool, cfg

    def schedule(self, waiting: deque, running: list) -> tuple[list, list]:
        """(admitted, updated)；updated=本步状态已变的请求（此处=判死）。"""
        budget = self.cfg.max_batched_tokens - len(running)
        admitted, updated = [], []
        for req in list(waiting):
            if len(running) + len(admitted) >= self.cfg.max_num_seqs:
                break
            nblk = self.pool.needed(req.context_len)
            if nblk > self.pool.total:  # 整池都装不下 → 判死，别无限排队
                req.status, req.finish_reason = ReqStatus.DONE, "error"
                req.finish_time = time.time()
                waiting.remove(req)
                updated.append(req)
                continue
            if req.context_len > budget or not self.pool.can_alloc(nblk):
                continue  # 预算/KV 不足等下步；永远超预算的会饿死（chunked prefill 就是为此）
            budget -= req.context_len
            req.block_ids = self.pool.alloc(nblk)
            waiting.remove(req)
            admitted.append(req)
        return admitted, updated

    def preempt(self, running: list, waiting: deque) -> Request:
        """池满→踢 running 队尾回 waiting 队首（vLLM 恢复策略：
        recompute=丢 KV 重算，本版；swap=KV 搬 CPU 内存，省算力费带宽）。
        朴素抢占会 thrash——刚重进又撞墙，vLLM 用水位线+优先级缓解。"""
        victim = running.pop()
        self.pool.free(victim.block_ids)
        victim.block_ids, victim.past_key_values = [], None
        victim.status = ReqStatus.PREFILL
        waiting.appendleft(victim)
        self.pool.preemptions += 1
        return victim


def sample_next_token(logits: torch.Tensor, temperature: float = 0.0,
                      top_p: float = 1.0) -> int:
    """同 scratch-decode：T=0 greedy，否则 softmax(logits/T) + top-p 截断。"""
    if temperature <= 0:
        return int(logits.argmax().item())
    logits = logits / temperature
    if top_p < 1.0:
        s_logits, s_idx = torch.sort(logits, descending=True)
        probs = F.softmax(s_logits, dim=-1)
        s_logits[(probs.cumsum(-1) - probs) > top_p] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(0, s_idx, s_logits)
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())


class EngineCore:
    """引擎内核：waiting/running 两队列，每步一个"prefill+decode 混合批"。"""

    def __init__(self, model_path: str, device: str = "cuda",
                 num_kv_blocks: int = 256, cfg: SchedConfig | None = None):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device).eval()
        eos = self.model.generation_config.eos_token_id  # Qwen 常配多个 eos
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.pool = KVBlockPool(num_kv_blocks)
        self.scheduler = Scheduler(self.pool, cfg or SchedConfig())
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.steps = self.gen_tokens = 0
        self._t0 = time.perf_counter()

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @torch.no_grad()
    def step(self) -> list[Request]:
        """一次迭代：返回本步有新 token（或被判死）的请求，供 API 推流。"""
        self.steps += 1
        admitted, updated = self.scheduler.schedule(self.waiting, self.running)
        decode_set = list(self.running)  # 快照：本步之前已 RUNNING 的
        for req in admitted:             # 新入列：本步即 prefill
            req.status = ReqStatus.RUNNING
            self.running.append(req)
            self._forward(req, prefill=True)
            updated.append(req)
        for req in decode_set:           # 老 seq：本步 decode 1 token
            if req.status is not ReqStatus.RUNNING:
                continue                 # 本步内已被抢占/判死
            if self._ensure_kv(req):
                self._forward(req, prefill=False)
                updated.append(req)
        for req in updated:              # 统一收尾：完成的归还 KV 块
            if req.status is ReqStatus.RUNNING and self._finished(req):
                self._release(req)
        return updated

    def _forward(self, req: Request, prefill: bool) -> None:
        """单 seq 前向（简化：vLLM 把所有 seq 拼一个 batch 走一次）。"""
        ids = req.context_ids() if prefill else [req.generated[-1]]
        out = self.model(input_ids=torch.tensor([ids], device=self.device),
                         past_key_values=req.past_key_values, use_cache=True)
        req.past_key_values = out.past_key_values  # prefill 新建 / decode 追加
        req.generated.append(
            sample_next_token(out.logits[0, -1], req.temperature, req.top_p))
        self.gen_tokens += 1

    def _ensure_kv(self, req: Request) -> bool:
        """decode 前保证下个 token 有块可写；池满→踢队尾直到够/自己出局。"""
        need = self.pool.needed(req.context_len + 1)
        while need > len(req.block_ids):
            if self.pool.can_alloc(need - len(req.block_ids)):
                req.block_ids += self.pool.alloc(need - len(req.block_ids))
                return True
            if need > self.pool.total:
                req.finish_reason = "error"   # 超池容，永远跑不了
                self._release(req)
                return False
            if self.scheduler.preempt(self.running, self.waiting) is req:
                return False                  # 自己被踢 → 下轮重新 prefill
        return True

    def _finished(self, req: Request) -> bool:
        if req.generated and req.generated[-1] in self.eos_ids:
            req.finish_reason = "stop"
        elif len(req.generated) >= req.max_new_tokens:
            req.finish_reason = "length"
        return req.finish_reason is not None

    def _release(self, req: Request) -> None:
        """完成/失败收尾：归还块、丢 KV、移出 running。"""
        req.status, req.finish_time = ReqStatus.DONE, time.time()
        self.pool.free(req.block_ids)
        req.block_ids, req.past_key_values = [], None
        if any(r is req for r in self.running):
            self.running.remove(req)

    def stats(self) -> dict:
        dt = time.perf_counter() - self._t0
        return {"waiting": len(self.waiting), "running": len(self.running),
                "kv_usage": round(self.pool.usage, 3),
                "kv_free_blocks": len(self.pool._free),
                "kv_total_blocks": self.pool.total,
                "preemptions": self.pool.preemptions, "steps": self.steps,
                "gen_tokens": self.gen_tokens,
                "tok_per_s": round(self.gen_tokens / max(dt, 1e-9), 1)}
