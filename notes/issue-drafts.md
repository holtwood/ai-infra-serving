# OSS Issue 草稿（待发布，发布前需人工确认）

## Draft A: vllm#26675 复现 comment（跟帖既有 issue）

**Title**: Repro on v0.11.0 / RTX 4090 — full_cudagraph OOM, enforce_eager works

Body:
> Reproduced on vLLM 0.11.0, single RTX 4090 (48GB, sm_89), driver 550.107.02,
> CUDA 12.8, Qwen2.5-7B-Instruct bf16, `--gpu-memory-utilization 0.90`.
>
> Symptom matches: with `cudagraph_mode=FULL` (default), engine startup OOMs /
> crashes during warmup capture; `enforce_eager=True` or
> `compilation_config.cudagraph_mode=PIECEWISE` works around it.
>
> Additional datapoint: crash is deterministic at `cudagraph_capture_sizes`
> batch 512 warmup; nvidia-smi shows spike +12GiB over steady-state during
> capture. Happy to upload `debug_dump_path` artifacts if useful.

状态：需先在 0.11.0 上把崩现场真跑一遍再发（当前记录为「可复现候选」）。

## Draft B: spec decode 被 generation_config 静默禁用（新 issue）

**Title**: `generation_config.repetition_penalty` silently disables ngram
speculative decoding with no warning

Body:
> **Environment**: vLLM 0.11.0, Qwen2.5-7B-Instruct, RTX 4090.
>
> **Repro**: serve with `--speculative-config '{"method":"ngram",...}'`.
> Qwen2.5's `generation_config.json` ships `repetition_penalty: 1.05`.
> Spec decoding is silently skipped: `metrics` shows 0 drafts/0 accepted.
> Setting `repetition_penalty=1.0` in sampling params re-enables it
> (30 drafts / 129 accepted, ~4x e2e speedup on repetitive output).
>
> **Expected**: either support rep_pen under spec decode, or log a warning
> that the model's generation_config disabled it. Silent no-op cost me an
> afternoon of debugging — the failure is invisible in logs at default level.
>
> **Evidence**: server log `num_drafts`/`num_accepted` counters before/after;
> repro script attached.

状态：证据链完整（metrics 计数前后对照已留档），可直接发。

## Draft C: 容器镜像 libcuda compat 遮蔽（发平台方/备忘）

`00-compat-*.conf` 把 `cuda-12.8/compat/libcuda.so.570` 排在 ldconfig 首位，
遮蔽内核驱动 550 → `torch.cuda` Error 804。修法：禁用该 conf + ldconfig。
这更像租赁平台镜像 bug，不适合 upstream vLLM。
