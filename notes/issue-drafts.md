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

## Draft B: spec decode 被 generation_config 静默禁用 —— ❌ 已撤回（核实后）

**发布前核实结论（2025-11-05）：不发。** 三条证据：

1. **"无警告"不成立**：我们自己的 `logs/vllm_7b_spec.log` 启动期就有
   `WARNING ... Default sampling parameters have been overridden by the
   model's Hugging Face generation config`（#15281 引入），并打印了
   `{'repetition_penalty': 1.05, ...}` 来源。只是没点名 spec decode。
2. **上游已修**：main 分支 `is_spec_decode_unsupported` 函数整体移除——
   penalties 类参数由 #19482 的 rejection-sampler logit processors 支持，
   其余不兼容参数由 #31982（2026-01 合并）改为请求级显式报错。
3. 发出去 = 重复 issue + 事实错误，负资产。

**保留的正确面试叙事**（debug 价值不受影响）：
"0.11.0 上实测 spec decode 不生效 → metrics 0 drafts → 追到
`is_spec_decode_unsupported` → 发现是 generation_config 的 rep_pen=1.05
→ rep_pen=1.0 后 4x 加速。事后核实上游已重构该路径。"

## Draft C: 容器镜像 libcuda compat 遮蔽（发平台方/备忘）

`00-compat-*.conf` 把 `cuda-12.8/compat/libcuda.so.570` 排在 ldconfig 首位，
遮蔽内核驱动 550 → `torch.cuda` Error 804。修法：禁用该 conf + ldconfig。
这更像租赁平台镜像 bug，不适合 upstream vLLM。
