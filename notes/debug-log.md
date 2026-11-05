# Debug Log —— 真实排障记录

> 用途：①面试官爱问"你解决过什么困难问题"；②每个 bug 都是潜在 OSS issue/PR 素材。
> 记录格式：现象 → 定位 → 根因 → 解法 → 可复用教训。

## 2026-11-05 M0 环境搭建（租用容器，RTX 4090 48G / driver 550 / CUDA 12.8）

### Bug 1：vLLM 0.33 拉起 CUDA 13 依赖，与驱动不兼容
- **现象**：`pip install vllm` 解析出 `torch==2.14.0 + nvidia-*-cu13`
- **根因**：vllm 0.33 默认绑定最新 torch（cu13 runtime），需要 driver ≥580；本机 550.107 最高支持 CUDA 12.8
- **解法**：钉版本 `torch==2.8.0`（PyPI 默认轮为 cu128）+ `vllm==0.11.0`（官方配套组合）
- **教训**：装 GPU 栈先对 `driver CUDA 版本` 和 wheel 的 `cuXY` 后缀做兼容性检查，别装最新

### Bug 2：`hf` CLI 改名 + httpx2 解压栈崩溃
- **现象**：`huggingface-cli: No such file`；换 `hf download` 后中途 `TypeError: process() takes no keyword arguments`（httpx2 `_decoders.py` 调 brotli/zstd `process()` 带 kwargs，本机库版本不兼容）
- **根因**：系统内 brotli/zstandard 版本过旧，不支持新版调用签名；`HF_HUB_DISABLE_XET` 无效（常规路径也走同一栈）
- **解法**：绕过 Python 解压栈——`curl` 直接拉 `hf-mirror.com/<repo>/resolve/main/<file>`，文件清单来自 `/api/models/<repo>/tree/main`，断点续传 `-C -` + 低速看门狗
- **教训**：网络/解压层坏掉时降到 HTTP 原语自己拼，比修依赖快

### Bug 3：pip 大 wheel 下载反复静默 stall
- **现象**：pip 下载 cudnn(706MB)/torch(888MB) 时进度停住，进程 sleeping 无网络连接，缓存字节数零增长
- **根因**：镜像源长连接挂死（无 read timeout 触发）
- **解法**：①从 simple index 解析真实 wheel URL，`curl -C - --speed-time 30 --speed-limit 10240` 循环断点续传，本地 `pip install *.whl`；②dep 解析交给 uv（Rust 并行下载、超时健壮）
- **教训**：`pip --timeout/--retries` 救不了挂死的 socket；大文件用可续传工具、包管理用 uv

### Bug 4：自伤式 pkill
- **现象**：`ssh bash -c 'pkill -f setup_env.sh ...'` 返回 255，整个会话被杀
- **根因**：`pkill -f` 按全命令行匹配，把自己的 bash -c 也匹配了
- **解法**：`pgrep -f "[s]etup_env"`——正则里包一个字符类，字符串本身不匹配自身
- **教训**：远程一键命令里 pkill 永远用 `[x]` 技巧

### 待验证的 OSS-issue 候选
- [ ] vllm 0.33 在 driver 550 环境直接拉 cu13 依赖且无任何前置检查 → 是否值得给 vLLM 提 "installer should warn when resolved torch CUDA > driver support"？
- [ ] hf_hub 在 brotli 版本不兼容时崩溃无 fallback → huggingface_hub issue 候选（先复现最小案例）

## 2026-11-05 vllm-ascend 素材（用户另一台昇腾机，由用户提供）
- W8A16 量化路径被拒绝/不支持 → 整理英文 issue 提 vllm-ascend
- cudagraph 模式下 stream 耗尽/重复 capture → 整理成 issue，可能升级 PR
- conv1d fallback 到 CPU/慢路径 → issue + 可能的 kernel 补齐 PR

## 2026-11-05 vllm 0.11.0：ngram 投机解码被 generation_config 静默禁用 ⭐OSS issue 候选

**现象**：`--speculative-config '{"method":"ngram","num_speculative_tokens":5,...}'`
启动正常（drafter 加载成功），但 `/metrics` 中
`spec_decode_num_drafts_total` / `num_accepted_tokens_total` 恒为 0，
TPOT 与基线无差——整个 spec decode 静默空转。

**根因链**：
1. Qwen2.5-7B-Instruct 的 `generation_config.json` 带 `repetition_penalty: 1.05`
2. vLLM `generation_config="auto"` 默认把模型默认采样参数套用到所有请求
3. `vllm/v1/spec_decode/utils.py::is_spec_decode_unsupported()` 规定
   `repetition_penalty != 1.0` 的请求不支持 spec decode
4. → 全部请求进 `spec_decode_unsupported_reqs`，proposer 跳过
5. **日志/metrics 无任何"spec decode 被采样参数禁用"的提示**

**验证**：客户端显式 `repetition_penalty: 1.0` 后 drafts=30、
accepted=129（接受长度 ~4.3/5），重复性输出 e2e 0.88s→0.22s（4x）。

**Issue 素材**：建议 vLLM 在 spec_decode_unsupported_reqs 非空时打
WARNING（"spec decode disabled for request X due to sampling params"），
或在 generation_config 读取处提示默认采样参数与 spec decode 冲突。
同类受害者：所有 generation_config 带 repetition_penalty/min_p 的模型。

## 2026-11-05 容器镜像 libcuda compat 遮蔽真实驱动（Error 804）
- 现象：torch.cuda.is_available()=False，Error 804 forward compat
- 根因：/etc/ld.so.conf.d/00-compat-*.conf 让 ldconfig 优先命中
  /usr/local/cuda-12.8/compat/libcuda.so.570 而非真实驱动 550
- 修复：重命名 conf + ldconfig → libcuda.so.1 解析回 x86_64-linux-gnu 版
- 同类问题：ncu 报 ERR_NVGPUCTRPERM（宿主机 RmProfilingAdminOnly=1，
  容器内无解 → 用 events+ptxas+compute-sanitizer 替代）

## 2026-11-05 vllm 0.11.0 + transformers 5.x 崩
- `Qwen2Tokenizer has no attribute all_special_tokens_extended`
- transformers 5.22.1 移除了该属性 → 锁 transformers==4.56.2
- （vllm 0.11.0 setup 要求 transformers>=4.55.2 但未卡上限 → 可提 issue）

## 2026-11-05 vllm 0.11.0：SharedStorageConnector + cudagraph 预热崩 ⭐bug 候选#2

**现象**：`kv_connector=SharedStorageConnector` + 默认 cudagraph →
EngineCore 启动即崩 `AssertionError`。
**根因**：`compile_or_warm_up_model` 的 `_dummy_run` 触发
`attention/layer.py::maybe_save_kv_layer_to_connector` →
`save_kv_layer` 要求 `_connector_metadata` 非空，但 dummy run
没有 scheduler 下发的 metadata → `assert self._connector_metadata is not None` 炸。
**绕过**：`enforce_eager=True`（跳过 cudagraph warmup 路径）。
**修法建议**：connector save hook 应在 `is_dummy_run/forward_context`
无 metadata 时直接 return，而非 assert。
P/D demo 本体跑通：producer 把 ~1200tok KV 序列化到
/data/ai-learn/kv_share（16MB），consumer 免 prefill decode；
0.5B 小模型上 传输成本≈重算成本（热 0.32s vs 冷 0.29s），
印证"PD 只对大模型/长上下文/高带宽互联划算"的行业口径。

## 2026-11-05 SGLang 部署失败（依赖链问题，非我方 bug）
- sglang[all]==0.5.4 → 依赖 flashinfer-python==0.4.1 →
  要求 apache-tvm-ffi==0.1.0b15，该 beta 未在任何 index 发布
- sglang==0.5.3.post1 → 硬钉 transformers==4.57.0，该版本已被
  PyPI yanked（setup 出错）→ 无解
- 结论：本机无法部署，按预案完成 RadixAttention 源码精读笔记
