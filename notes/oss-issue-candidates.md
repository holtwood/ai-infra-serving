# OSS issue/PR 候选池（M7）

目标：从本次实验踩的真 bug 里挑 1-2 个，写成规范英文 issue 提到
vllm-project/vllm（或对应 repo）。诚实原则：只报实测复现过的，
不声称已 merge。

## 候选 1：ngram spec decode 被 generation_config 默认采样参数静默禁用 ⭐首选

**标题草稿**：
`Speculative decoding silently disabled when model generation_config sets repetition_penalty/min_p`

**复现**：
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5}'
# /metrics: spec_decode_num_drafts_total 恒为 0，无任何 WARNING
```
根因：Qwen2.5 的 generation_config.json 带 `repetition_penalty:1.05`，
vLLM `generation_config="auto"` 默认套用 → `is_spec_decode_unsupported()`
判全量请求不兼容 → `spec_decode_unsupported_reqs` 收集所有请求。
**建议修复**：spec_decode_unsupported_reqs 非空时输出一次 WARNING；
或在 spec decode 启用时忽略 generation_config 里的冲突参数。
**证据**：metrics 截图 + rep_pen=1.0 后 drafts>0 的对照（本 repo 已测）。

## 候选 2：SharedStorageConnector + cudagraph 预热崩溃

**标题草稿**：
`SharedStorageConnector crashes on startup with default cudagraph: 
save_kv_layer called during warmup _dummy_run with no connector_metadata`

**复现**：
```python
LLM(model, kv_transfer_config=KVTransferConfig(
    kv_connector="SharedStorageConnector", kv_role="kv_producer",
    kv_connector_extra_config={"shared_storage_path": "/tmp/kv"}))
# enforce_eager=False（默认）→ EngineCore init 即 AssertionError
# enforce_eager=True 可绕过
```
堆栈：`compile_or_warm_up_model` → `_dummy_run` →
`maybe_save_kv_layer_to_connector` → `save_kv_layer` →
`assert self._connector_metadata is not None`
**建议修复**：dummy run 无 metadata 时 save hook 直接 return。
**价值**：直接阻断任何想用 SharedStorageConnector 做 P/D demo 的用户。

## 候选 3（昇腾，用户提供素材）：vllm-ascend 三个 issue

- W8A16 量化路径被拒绝/不支持
- cudagraph 模式下 stream 耗尽/重复 capture
- conv1d fallback 到 CPU/慢路径
→ 等拿到昇腾机器复现细节后，整理成 vllm-ascend issue；conv1d
fallback 若能量化慢多少倍，可升级为 kernel 补齐 PR。

## 候选 4（可选）：transformers 5.x 与 vllm 0.11 兼容上限未声明

`transformers==5.x` 移除 `all_special_tokens_extended` →
vllm 0.11.0 tokenizer 初始化崩 AttributeError。
setup 要求 `transformers>=4.55.2` 无上限 → 可提"pin <5 或兼容"。
（低优先级：可能已被上游知晓）

## 查重结果（2026-11-05 已核实）

- **候选 2 已有 issue**：vllm#26675 —— 完全同款
  （graph 模式 _dummy_run → save_kv_layer → connector_metadata assert）。
  → 改为去该 issue 下补充"0.11.0 + RTX4090 复现 + enforce_eager 可绕过"
  的确认 comment，仍是有效贡献。
- **候选 1 部分被修**：PR #31982（2026-01 合并）把"静默忽略"改成
  "请求级校验报错"。在 0.11.0 上静默禁用仍真实存在；且衍生新角度——
  generation_config 默认 rep_pen=1.05 会让【所有】请求触发该校验，
  新版行为可能是"全员报错"（需新版本验证后才能定稿）。
- vllm-ascend#10245 是同类 dummy-batch 污染 connector 的先例，
  可作引用。

## 发布前 checklist

- [ ] 用最小脚本复现（<30 行）
- [ ] 附环境：vllm 0.11.0 / torch 2.8.0 / CUDA 12.8 / RTX 4090 / driver 550
- [ ] 附完整 traceback（候选2）和 metrics 输出（候选1）
- [ ] 说明期望行为 vs 实际行为
- [ ] 先搜 vllm issues 确认未重复（发布前做一次）
