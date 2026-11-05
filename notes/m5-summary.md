# M5 小结：量化 / CUDA Graph / 投机解码（全部实测）

日期：2026-11-05 | vllm 0.11.0 | RTX 4090 (sm_89，无 FP8 tensor core)

## 一、ngram 投机解码：从"静默失效"到 4x 加速 ⭐本模块最大收获

**对照实验**（Qwen2.5-7B，重复性输出 prompt）：
| 请求参数 | spec decode | e2e (128 tok) | 吞吐 |
|---|---|---|---|
| 默认（repetition_penalty=1.05） | **静默禁用** | 0.88s | 125 chars/s |
| repetition_penalty=1.0 | 激活 | **0.22s** | **498 chars/s (4x)** |

- metrics 实证：30 drafts / 129 accepted tokens，**接受长度 ~4.3/5**
- 根因：模型 `generation_config.json` 带 `repetition_penalty:1.05`
  → vLLM 套用到所有请求 → `is_spec_decode_unsupported()` 判定不兼容
  → proposer 全跳过，**日志零提示**（详见 debug-log.md，OSS issue 候选）

**面试话术**："投机解码的收益 = 接受长度 × 每步多算的 draft 成本；
ngram 免 draft 模型但只在输出可预测（重复/模板化/代码）时有效；
我在 vllm 0.11 上实测并定位了一个静默失效 case——
模型 generation_config 的默认采样参数会全局禁用 spec decode。"

## 二、CUDA Graph 消融（0.5B，--enforce-eager vs 默认）

| 模式 | TPOT c=1 | TPOT c=32 | 总吞吐@32 |
|---|---|---|---|
| cudagraph（默认 FULL_AND_PIECEWISE） | 2.6ms | 4.4ms | 2762 tok/s |
| enforce-eager | 7.2ms | 8.8ms | 2149 tok/s |

- 小模型 launch 开销 ~5ms/step 与真实计算同量级 → graph 化 **TPOT -64%**
- 面试点：decode 每步几十个 kernel，每个 launch ~20-30μs CPU 开销，
  graph 把整个 decode 步录成一次提交；**收益∝模型小、并发低**

## 三、量化：AWQ INT4 实测（Qwen2.5-32B-AWQ）

- vLLM 自动路由 `awq_marlin` kernel（启动日志）
- 32B 模型 bf16 需 ~65GB 装不下 48GB 卡 → INT4 ~19GB 轻松跑
- TPOT 21.5ms（c=1）≈ 19GB/939GB/s + dequant 开销
- **权重量化收益本质**：decode memory-bound 阶段搬运量 -75%；
  dequant 增加的 compute 在 memory-bound 段几乎免费

## 四、FP8 在 sm_89 的边界（诚实负结果）

`/model/Qwen3.8-27B-FP8` 加载失败——但**不是 FP8 问题**：
checkpoint 是 `model_type: qwen3_5`，transformers 4.56.2（vllm 0.11
兼容上限）不认识该架构。FP8-on-Ada 的实验本身没做成：
sm_89 无 FP8 tensor core，vLLM 只能 dequant→bf16 跑，
收益只剩显存压缩。留给后续有更新 vllm 的环境复测。

## 五、profiling 工具链（容器约束下的替代方案）

- ncu：宿主机 `RmProfilingAdminOnly=1` → **不可用**（容器内无解）
- 替代栈实测可用：CUDA events 计时 / `nvcc --ptxas-options=-v`
  （寄存器、smem、spill）/ compute-sanitizer memcheck /
  torch.profiler / vllm `/metrics` + 服务端吞吐日志
- 面试话术："受限环境先验证 perf counter 权限，没有就用
  events+occupancy API 算带宽利用率，照样能定位 memory/compute bound"

## 六、未完成（诚实清单）

- LMCache KV offload：依赖重，本期未部署（读文档+P/D 路由模拟替代）
- MTP/EAGLE draft model：需配套 draft 权重，ngram 已覆盖机制演示
- nsys timeline：未安装，torch.profiler 覆盖同需求
