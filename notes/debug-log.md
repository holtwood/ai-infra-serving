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
