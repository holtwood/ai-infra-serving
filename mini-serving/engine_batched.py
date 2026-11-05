"""M4+: batched forward 版引擎 —— 填平 per-seq forward 的 35x 差距。

与 engine.py 的唯一本质区别：decode 步不再逐 seq 各跑一次 forward，
而是把 running set 拼成【一个】 batch 走一次 forward——权重只搬一遍。

实现方式（教学版，对照 vLLM）：
  * KV 存放：一个共享 DynamicCache，B 条 seq 的 K/V【左 pad】到统一
    长度 L（batch 内最大 ctx），attention_mask 屏蔽 pad 位。
    vLLM 是物理分页+block table 寻址零浪费；本版 pad 位有访存浪费，
    但 decode 是 memory-bound（瓶颈在权重搬运不在 KV 读），
    多搬的 pad KV 相对权重可忽略——实测验证这个判断。
  * 新 seq 进来：单独 prefill（batch=1 出 cache）→ 左 pad 合并进
    共享 batch cache；完成/被踢 → batch_select_indices 切掉该行。
  * 每步：input_ids=[B,1]（各 seq 上一个 token）+ position_ids=
    各 seq 实际位置 + attention_mask=[B,L+1] → 一次 forward →
    每行独立 sample。
"""
import math, os, time
from collections import deque

import torch
import torch.nn.functional as F

from engine import (Request, ReqStatus, KVBlockPool, SchedConfig, Scheduler,
                    sample_next_token, BLOCK_SIZE)


class BatchedCache:
    """共享 batch KV：维护一个 DynamicCache + 每行 seq 的 pad 元信息。

    rows[i] ↔ running[i]：第 i 条 seq 的 KV 存在
    cache.layers[*].keys/values[i, :, pad_i:L, :]，左侧 pad_i 位是零填充
    （attention_mask 屏蔽）。L = 当前 batch 最大 context_len。
    """

    def __init__(self):
        self.cache = None          # transformers DynamicCache | None
        self.pad: list[int] = []   # 每行左 pad 长度
        self.L = 0                 # 统一后的序列长度

    # ---------- 合并：单 seq prefill 产出的 cache 左 pad 后并进 batch ----------
    def add(self, seq_cache, ctx_len: int) -> None:
        new_L = max(self.L, ctx_len)
        if self.cache is None:
            self.cache = seq_cache
            self.pad = [0]
            self.L = ctx_len
            if ctx_len < new_L:
                self._repad(new_L)
            return
        if new_L > self.L:
            self._repad(new_L)     # 全员向右对齐到新 L
        pad_i = new_L - ctx_len
        for i, layer in enumerate(self.cache.layers):
            k_new = F.pad(seq_cache.layers[i].keys,
                          (0, 0, pad_i, 0))                # (…, S, D) 左 pad S
            v_new = F.pad(seq_cache.layers[i].values, (0, 0, pad_i, 0))
            layer.keys = torch.cat([layer.keys, k_new], dim=0)
            layer.values = torch.cat([layer.values, v_new], dim=0)
        self.pad.append(pad_i)
        self.L = new_L

    def _repad(self, new_L: int) -> None:
        """已有行整体左 pad 到新长度（新 seq ctx 超过当前 L 时触发，罕见）。"""
        grow = new_L - self.L
        for layer in self.cache.layers:
            layer.keys = F.pad(layer.keys, (0, 0, grow, 0))
            layer.values = F.pad(layer.values, (0, 0, grow, 0))
        self.pad = [p + grow for p in self.pad]
        self.L = new_L

    def remove(self, idx: int) -> None:
        """第 idx 行 seq 离开 batch（完成/被抢占）。"""
        idxs = torch.tensor([i for i in range(len(self.pad)) if i != idx],
                            device=self.cache.layers[0].keys.device)
        if idxs.numel() == 0:
            self.cache, self.pad, self.L = None, [], 0
            return
        self.cache.batch_select_indices(idxs)
        self.pad.pop(idx)
        # L 不收缩（省 repad）；空跑时整体清空

    def masks_positions(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """decode 步的 attention_mask [B,L+1] 和 position_ids [B,1]。"""
        B = len(self.pad)
        ctx = torch.tensor([self.L - p for p in self.pad], device=device)
        mask = torch.ones(B, self.L + 1, dtype=torch.long, device=device)
        for i, p in enumerate(self.pad):
            if p:
                mask[i, :p] = 0                    # 左 pad 位屏蔽
        return mask, ctx.view(B, 1)


class EngineCoreBatched:
    """与 EngineCore 同接口；step() 里 decode 合并为一次 batched forward。"""

    def __init__(self, model_path: str, device: str = "cuda",
                 num_kv_blocks: int = 256, cfg: SchedConfig | None = None):
        from transformers import AutoModelForCausalLM
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device).eval()
        if os.environ.get("MINI_COMPILE") == "1":
            self.model = torch.compile(self.model,
                                       mode="reduce-overhead",
                                       dynamic=True)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.pool = KVBlockPool(num_kv_blocks)
        self.scheduler = Scheduler(self.pool, cfg or SchedConfig())
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.bcache = BatchedCache()
        self.steps = self.gen_tokens = 0
        self._t0 = time.perf_counter()

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @torch.no_grad()
    def step(self) -> list[Request]:
        self.steps += 1
        admitted, updated = self.scheduler.schedule(self.waiting, self.running)

        # 新入列：逐 seq prefill（教学简化：prefill 不拼批，compute-bound
        # 差异不在本实验主线上；真实系统靠 chunked prefill 拼进同一步）
        for req in admitted:
            req.status = ReqStatus.RUNNING
            self.running.append(req)
            self._prefill(req)
            updated.append(req)

        # decode：全体 RUNNING 行（含刚准入的）合成一次 forward。
        # bcache 行序 ≡ running 序；_ensure_kv 可能触发抢占同步剔行。
        for req in list(self.running):
            if req.status is ReqStatus.RUNNING and not self._ensure_kv(req):
                pass                        # 被踢/判死：状态已被改
        alive = [r for r in self.running if r.status is ReqStatus.RUNNING]
        if alive:
            self._decode_step(alive)
            for req in alive:
                if req not in updated:
                    updated.append(req)

        for req in updated:
            if req.status is ReqStatus.RUNNING and self._finished(req):
                self._release(req)
        return updated

    # ---------- 模型交互 ----------

    def _prefill(self, req: Request) -> None:
        ids = req.context_ids()
        out = self.model(
            input_ids=torch.tensor([ids], device=self.device),
            use_cache=True)
        # 独立 cache 左 pad 合并进共享 batch（req 与 running[-1] 对齐）
        self.bcache.add(out.past_key_values, len(ids))
        req.past_key_values = None               # 真 KV 统一由 bcache 持有
        req.generated.append(sample_next_token(
            out.logits[0, -1], req.temperature, req.top_p))
        self.gen_tokens += 1

    def _decode_step(self, reqs: list[Request]) -> None:
        # running 与 bcache 行序必须一致：running 里可能有刚 prefill 完
        # 的 seq（不在本次 decode_set），按 reqs 在 running 中的行序取样
        input_ids = torch.tensor([[r.generated[-1]] for r in reqs],
                                 device=self.device)
        mask, pos = self.bcache.masks_positions(self.device)
        out = self.model(input_ids=input_ids,
                         attention_mask=mask,
                         position_ids=pos,
                         past_key_values=self.bcache.cache,
                         use_cache=True)
        self.bcache.L += 1                       # 全行同时 +1 token
        for i, req in enumerate(reqs):
            req.generated.append(sample_next_token(
                out.logits[i, -1], req.temperature, req.top_p))
            self.gen_tokens += 1

    # ---------- 生命周期（与 EngineCore 相同，多一步 bcache 行剔除） ----------

    def _ensure_kv(self, req: Request) -> bool:
        need = self.pool.needed(req.context_len + 1)
        while need > len(req.block_ids):
            if self.pool.can_alloc(need - len(req.block_ids)):
                req.block_ids += self.pool.alloc(need - len(req.block_ids))
                return True
            if need > self.pool.total:
                req.finish_reason = "error"
                self._release(req)
                return False
            victim = self.scheduler.preempt(self.running, self.waiting)
            # victim 是 running 队尾 → 同步剔 bcache 末行（刚 pop 后末位）
            if self.bcache.cache is not None:
                self.bcache.remove(len(self.running))
            if victim is req:
                return False
        return True

    def _finished(self, req: Request) -> bool:
        if req.generated and req.generated[-1] in self.eos_ids:
            req.finish_reason = "stop"
        elif len(req.generated) >= req.max_new_tokens:
            req.finish_reason = "length"
        return req.finish_reason is not None

    def _release(self, req: Request) -> None:
        idx = self.running.index(req) if req in self.running else -1
        req.status, req.finish_time = ReqStatus.DONE, time.time()
        self.pool.free(req.block_ids)
        req.block_ids, req.past_key_values = [], None
        if idx >= 0:
            self.running.pop(idx)
            # running 与 bcache 行序一致 → 同步剔行
            if self.bcache.cache is not None:
                self.bcache.remove(idx)

    def stats(self) -> dict:
        dt = time.perf_counter() - self._t0
        return {"waiting": len(self.waiting), "running": len(self.running),
                "kv_usage": round(self.pool.usage, 3),
                "preemptions": self.pool.preemptions, "steps": self.steps,
                "gen_tokens": self.gen_tokens,
                "tok_per_s": round(self.gen_tokens / max(dt, 1e-9), 1)}
