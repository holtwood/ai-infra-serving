"""M4++：CUDA Graph 版 batched 引擎 —— 砍掉 decode 步 69% 的 GPU 空转。

profiler 实测：batched decode 每步 ~170 次 kernel launch，
Python/dispatch 占墙钟 69%。本版把整个 decode forward 录成图回放。

关键约束（实测 transformers 4.56 StaticCache）：
  * cache_position 是全 batch 共享的单一索引 → 所有行的 KV 必须
    【右对齐】到同一个 [0:L) 窗口（pad 在前），新 token 统一写 slot L。
    行长度不同 → 更长的请求准入时把所有行 KV 右移 delta（vLLM 里
    不存在这问题——block table 本来就允许任意逻辑位置）。
  * graph 要求全部指针稳定 → 输入用静态 buffer（ids/pos/mask/cpos），
    采样在图外，prefill 逐条不进图（KV 直接拷进 StaticCache）。
  * 分桶 BUCKETS：batch 数补 dummy 行到桶位（输出丢弃）——
    与 vLLM cudagraph_capture_sizes + padding slot 同思路。
"""
import time
from collections import deque

import torch
from transformers import StaticCache

from engine import (Request, ReqStatus, KVBlockPool, SchedConfig, Scheduler,
                    sample_next_token, BLOCK_SIZE)

BUCKETS = (1, 2, 4, 8, 16, 32)


class EngineCoreGraph:
    def __init__(self, model_path: str, device: str = "cuda",
                 num_kv_blocks: int = 256, cfg: SchedConfig | None = None,
                 max_len: int = 2048):
        from transformers import AutoModelForCausalLM
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.pool = KVBlockPool(num_kv_blocks)
        self.scheduler = Scheduler(self.pool, cfg or SchedConfig())
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []          # running[i] ↔ cache 第 i 行
        self.L = 0                                # 当前右对齐窗口长度
        self.max_len = max_len
        self.cache = StaticCache(config=self.model.config,
                                 max_cache_len=max_len)
        # 首个 update 决定 batch 维：用 dummy [Bmax,1] 撑开到 BUCKETS[-1]
        dummy = torch.zeros(BUCKETS[-1], 1, dtype=torch.long, device=device)
        self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy),
                   cache_position=torch.zeros(1, dtype=torch.long,
                                              device=device),
                   past_key_values=self.cache, use_cache=True)
        self.graphs: dict[int, tuple] = {}
        self.steps = self.gen_tokens = 0
        self._t0 = time.perf_counter()

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _shift_right(self, delta: int):
        """L 变大时全体行 KV 右移 delta（图外做，只动静态张量内容）。"""
        for layer in self.cache.layers:
            layer.keys[:, :, delta:self.L + delta] = \
                layer.keys[:, :, :self.L].clone()
            layer.values[:, :, delta:self.L + delta] = \
                layer.values[:, :, :self.L].clone()
        self.L += delta

    @torch.no_grad()
    def _prefill(self, req: Request, row: int):
        ids = torch.tensor([req.prompt_ids], device=self.device)
        out = self.model(input_ids=ids, use_cache=True)
        dk = out.past_key_values
        li = len(req.prompt_ids)
        for i, layer in enumerate(self.cache.layers):
            layer.keys[row:row+1, :, self.L-li:self.L] = \
                dk.layers[i].keys[0]
            layer.values[row:row+1, :, self.L-li:self.L] = \
                dk.layers[i].values[0]
        tok = sample_next_token(out.logits[0, -1],
                                temperature=req.temperature,
                                top_p=req.top_p)
        req.generated.append(tok)

    # ---------- decode graph ----------
    def _bucket(self, n: int) -> int:
        for b in BUCKETS:
            if n <= b:
                return b
        return BUCKETS[-1]

    def _capture(self, B: int):
        M = self.max_len + 1
        bufs = dict(
            ids=torch.zeros(B, 1, dtype=torch.long, device=self.device),
            pos=torch.zeros(B, 1, dtype=torch.long, device=self.device),
            mask=torch.zeros(B, M, dtype=torch.long, device=self.device),
            cpos=torch.zeros(1, dtype=torch.long, device=self.device),
        )
        cache = self.cache
        def fw():
            return self.model(input_ids=bufs["ids"],
                              attention_mask=bufs["mask"],
                              position_ids=bufs["pos"],
                              cache_position=bufs["cpos"],
                              past_key_values=cache,
                              use_cache=True).logits
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fw()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            bufs["logits"] = fw()
        self.graphs[B] = (g, bufs)

    @torch.no_grad()
    def _decode_graph(self) -> bool:
        n = len(self.running)
        if self.L + 1 > self.max_len:
            return False
        B = BUCKETS[-1]          # StaticCache batch 维锁定 Bmax
        if B not in self.graphs:
            self._capture(B)
        g, bufs = self.graphs[B]
        ids, pos, mask, cpos = (bufs["ids"], bufs["pos"],
                                bufs["mask"], bufs["cpos"])
        mask.zero_()
        for i, req in enumerate(self.running):
            li = req.context_len               # 含已生成
            ids[i, 0] = req.generated[-1]
            pos[i, 0] = li - 1                 # 输入 token 的绝对 rope 位置
            mask[i, self.L+1-li:self.L+1] = 1  # 右对齐窗口 + 新 slot
        for i in range(n, B):
            ids[i, 0] = next(iter(self.eos_ids))
            mask[i, 0] = 1
        cpos[0] = self.L
        g.replay()
        self.L += 1                  # 全体右对齐窗口同步前推一格
        for i, req in enumerate(self.running):
            tok = sample_next_token(bufs["logits"][i, -1],
                                    temperature=req.temperature,
                                    top_p=req.top_p)
            req.generated.append(tok)
        return True

    def _compact_rows(self, k: int):
        """第 k 行请求完成 → 后续行 KV 下移一格，保持 running[i]↔行 i。"""
        B = self._bucket(len(self.running) + 1)
        for layer in self.cache.layers:
            layer.keys[k:B-1] = layer.keys[k+1:B].clone()
            layer.values[k:B-1] = layer.values[k+1:B].clone()

    @torch.no_grad()
    def step(self) -> list[Request]:
        self.steps += 1
        admitted, updated = self.scheduler.schedule(self.waiting, self.running)
        for req in admitted:
            li = len(req.prompt_ids)
            if li > self.L:                    # 新最长 → 全体右移
                self._shift_right(li - self.L)
            req.status = ReqStatus.RUNNING
            self.running.append(req)
            self._prefill(req, len(self.running) - 1)
            updated.append(req)
        if self.running:
            if not self._decode_graph():
                # 超出 max_len：最老请求判 length 结束
                req = self.running.pop(0)
                req.status = ReqStatus.DONE
                req.finish_reason = "length"
                req.finish_time = time.time()
                updated.append(req)
                self._compact_rows(0)
            for req in list(self.running):
                if (req.generated[-1] in self.eos_ids
                        or len(req.generated) >= req.max_new_tokens):
                    req.status = ReqStatus.DONE
                    req.finish_reason = ("stop" if req.generated[-1]
                                         in self.eos_ids else "length")
                    req.finish_time = time.time()
                    k = self.running.index(req)
                    self.running.remove(req)
                    self.pool.free(req.block_ids)
                    self._compact_rows(k)
                    updated.append(req)
                else:
                    self.gen_tokens += 1
        return updated
