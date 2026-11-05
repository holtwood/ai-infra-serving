# 深度调研：AI Infra 面试的差异化亮点（2025-11 调研）

> 调研方法：真实 JD 关键词 × 亲历者面经（牛客/阿里/蚂蚁）× 国外社区（GPU MODE、r/LocalLLaMA、HN）× 在职 inference 工程师简历反向工程。

## 结论先行

**现有课程设计方向是对的**——手写 decode loop、kernel+profiling、mini-serving、vLLM 源码，全部命中面经。但调研发现 **5 个高性价比增量**，其中"OSS merged PR"是从"完成课程"到"拿到 offer"最关键的一步：

| 优先级 | 增量项 | 为什么 | 成本 |
|---|---|---|---|
| P0 | **vLLM/SGLang merged PR ≥1** | 转行案例的共性杀手锏："2 CUDA PRs into llama.cpp" 式简历线。面试官眼里 = 代码能力被第三方背书 | 高（嵌入 M3 后） |
| P0 | **可复现的公开 benchmark 报告** | "published with configs, raw exports and repro steps"——在职工程师简历原话。我们的三引擎对比天然就是 | 低（M3 产出物换个包装） |
| P1 | **GPU MODE popcorn 打榜** | gpumode.com leaderboard 是真实竞技场，提交 Triton kernel 有排名；社区里有人靠这个被看见 | 低（M2 kernel 顺手提交） |
| P1 | **P/D 分离 + KV-aware 路由演示** | 2025-26 JD 热词：llm-d、NVIDIA Dynamo、disaggregation、KV-aware routing。单卡可用两个 vLLM 实例演示 | 中（M4 扩展） |
| P2 | **batch invariance 演示** | Thinking Machines "Defeating Nondeterminism" 是年度热帖，vLLM 0.11 已内置开关，一行 flag 出效果 | 极低 |

## 一、面试官到底考什么（牛客面经实证）

### 阿里云 AI infra 三面（2025 暑期实习）
- 全程项目深聊，**零八股零编程题**
- 考"工程真实性"：你的算子开发工作流？怎么判断正确性？需求→实现→测试→验证的完整链路
- 考"路径自洽"：为什么从 DL 转 infra？学习路径能不能讲成一条线
- 甚至考了"你怎么用 AI 工具协作"（长上下文怎么处理）——他们想看真实工作方式

### 蚂蚁 AI infra 一面
- 算子优化判断框架：**是否热点 → 瓶颈类型（算力/访存/调度）→ 优化路径（kernel/融合/图优化/布局）**
- 量化深挖：AWQ 加了反量化步骤为什么还快？（memory-bound + 融合减少搬运）
- "你的 timing 插件为什么不用 ncu？"（接入成本、迭代速度 vs 完整 profiler 的场景区分）
- vLLM 源码浅问都接不住 → "光知道概念不够，要进源码"
- OOM/随机崩溃的真实排查链路 + 1 道 LeetCode

### 国外侧（NVIDIA/inference 岗经验贴）
- "walk through **one kernel you wrote**: bottleneck → fix → measured speed-up"（三个要素缺一不可）
- "system design where **the GPU is the scarce resource**: batching, scheduling, memory"
- 简历要能撑住**每个决策三连追问**（why this approach → alternatives → tradeoff）

## 二、2025-26 热词（JD 关键词频率，从高到低）

vLLM V1 内部（scheduler/KV manager/prefix caching/chunked prefill）＞ FP8/AWQ/GPTQ 量化 ＞ P/D 分离 + KV-aware routing（llm-d/NVIDIA Dynamo/NIXL）＞ speculative decoding（EAGLE3/n-gram）＞ CUDA Graph/torch.compile ＞ 多 LoRA serving ＞ KV offload/分层缓存（LMCache）。

面试话术锚点：**TTFT/TPOT/goodput/SLO**——开口闭口延迟吞吐指标而不是"框架很快"。

## 三、OSS 贡献怎么打（vLLM 实战路径）

来源：vLLM 官方 contributing doc + 社区新人 Q&A + 亲历者博客（ym2132）。

- **别从 good-first-issue 列表抢**（AI 生成的 PR 已经挤占，maintainer 审不过来）。正确路径：**自己用出问题** → 复现 → issue+PR 一起交。我们 M3 大量跑 vLLM 实验，天然会遇到 bug/边界 case——每个都是候选
- 次选：`good first issue` 里挑冷门的（发布时间久、评论少的），先 comment 领任务
- 文档/benchmark 类 PR 也算 merged PR，门槛最低：docs.vllm.ai 的 contributing 有 job board（good first issues / onboarding tasks / new model requests）
- 流程：comment 认领 → 最小实现 + 测试 → draft PR → 按 review 改。PR 描述写清复现步骤和动机
- **我们环境的一个真实候选**：4090 48G（Ada, sm89）上 FP8/batch-invariance/AWQ 的边界问题——sm89 在官方支持矩阵边缘，容易踩到真 bug

## 四、GPU MODE 打榜玩法

- gpumode.com 上有常驻 leaderboard 题目（kernel 优化挑战），用 `popcorn-cli` 本地跑分后提交
- 题目典型形态："优化这个 GEMM/attention/通信 kernel"，按加速比排名，对标人类和 AI 提交
- 门槛：需要进 Discord（discord.gg/gpumode）拿 popcorn-cli；题目偏难但提交即留痕
- **策略**：M2 的 Triton kernel 写完顺手提交一版，哪怕排名一般，"在 GPU MODE leaderboard 有提交"本身就是信号

## 五、给课程的补丁（已并入 README 计划）

- **M3 增加**：batch-invariance 演示（`VLLM_BATCH_INVARIANT` 或对应 flag）；实验中发现任何 bug → 记 DEBUGGING.md，作为 OSS PR 候选池
- **M4 扩展**：加"P/D 分离 + prefix-aware 路由"迷你演示——同卡起 2 个 vLLM replica（4B/7B 模型），自写路由器按 prefix 相似度分发，量出 prefix-cache 命中率差异
- **M5 增加**：LMCache（vLLM 官方 KV offload 插件）演示：重复前缀请求 TTFT 变化；投机解码开 ngram draft 测加速比
- **新增 M7（机动/冲刺）**：① vLLM OSS PR（从 DEBUGGING 候选池挑）② popcorn 提交 ③ 把 benchmark 报告发博客（知乎/dev.to）④ 更新简历话术
- **notes 新增**：`debug-log.md` 真实排障记录（面经证明面试官爱问"你解决过什么困难 bug"）

## 六、追加发现（论坛深挖第二轮）

### 面试官视角原帖（牛客，推理加速工程师面校招生实录）⭐
三轮追问链全记录：项目（问题来源→方法→结果）→ CUDA 基础三连：**shared memory bank conflict**（候选人答错）、**CUDA stream 异步执行顺序与 latency**、**寄存器溢出后果** → encoder/decoder 区别 → iteration-level scheduling → "vLLM 最大创新点是什么" → "vLLM 调度还有什么可改进" → **weight-only 量化为什么只量化 weight 不量化 activation**。面试官评价"简历一般但工作量实在，校招 top10%"——**扎实经历 > 包装**。

### 同类项目竞争水位（重要校准）
国内"从零手写推理引擎"已是常见敲门砖：KuiperInfer（2.5k star）、**MiniInfer**（Paged KV + continuous batching + CUDA Graph + FA2/FlashInfer 双后端，自称接近工业级 vLLM 性能）、个人 LLM-inference-engine（CUDA 从零跑 DeepSeek-7B）。**我们的 mini-serving 要差异化，建议做完基础版后加 1-2 个 MiniInfer 级特性（CUDA graph decode / 真 batch forward / prefix cache），并在 README 诚实写清取舍**——面试官会追问"你改了哪、精度掉多少、极限在哪"，只复现会被打穿。

### vLLM Talent Pool（官方招聘通道）
talentpool@vllm.ai —— 官方运营，已帮多位学生拿到顶级 AI lab/infra 实习；定向需求：CUDA/CUTLASS kernel、分布式、投机解码/RL、多模态、模型支持。**2-3 个 merged PR 后值得直接投**。

### 真实转行 playbook（Inesh Reddy 案例）
Tata Elxsi ML 工程师 → **vLLM PRs + Tensara/GPU MODE 榜成绩 → Meta AI Systems 推理工程师**。简历写法："#3 globally on GELU kernel"、"LMCache+vLLM 6.5× TTFT 降低"——**量化表述的成绩单**。

### popcorn-cli 细节（比预想更好）
评测跑在**官方托管的真卡上**（A100/H100/B200/MI300，Modal 提供，免费）——本地没 GPU 也能打榜；`--profile` 可远程跑 Nsight Compute 下载报告。入门题就是 PMPP 练习级别。`popcorn register discord` → `popcorn submit submission.py --leaderboard <名> --gpu H100 --mode leaderboard`。GPUMODE 官方承认前排已被职业玩家+AI 占据，**所以价值定位=学习驱动+公开信号+人脉（Discord 里帮人 review kernel 是公认 referral 路径），不拼名次**。

### JD 信号（头部团队原话）
- DeepSeek 社招 JD："**欢迎没有大模型经验，但计算机基础素质扎实、对 AGI 有热情的同学**"——基础素质 > 现成经验；加分项：开源深度贡献、IOI/ICPC/SC 超算竞赛
- NVIDIA JD 直接写 "Build production-grade software inside vLLM/SGLang/Dynamo/FlashInfer"——**OSS 贡献就是工作内容本身**
- Moonshot 校招有独立"RL Infra"方向：vLLM 二开优化 rollout KV 复用

### 新增面试题型（国内外互证）
- **手写题**：带 KV cache 的 GQA attention、paged block allocator、continuous batching scheduler、top-p sampler（inferenceinterviews.com 总结，国内外都考）
- **口算题**：KV cache bytes/token、decode 时延下界、batch 吞吐拐点
- **系统设计**：Anthropic 的 "inference API" 设计题在一亩三分地被反复讨论，已成国外大厂 SD 轮标准题
- **资源**：B站"月球大叔"（硅谷 infra 从业者求职系列）、知乎"草原小可汗"面经专栏（华为/百度/阿里/蚂蚁全程记录）——建议 M6 前通读

## 七、参考链接存档

- GPU MODE: https://www.gpumode.com/ , https://github.com/gpu-mode/kernelbot , https://github.com/gpu-mode/lectures
- vLLM 贡献指南: https://docs.vllm.ai/en/stable/contributing/ , good first issues: https://github.com/vllm-project/vllm/contribute
- vLLM 新人 Q&A（中文友好）: https://princepride.github.io/live-streaming-tools/vllm_beginner_qa/final/blog.en/
- OSS 心路: https://ym2132.github.io/vllm_make_awq_models_work_batch_invariance.html
- 牛客面经: 阿里 /discuss/875005711049052160 , 蚂蚁 /discuss/872458011837591552
- 推理工程师简历参考（背 JD 词）: richardjoy.ai , ml-infra.pro/en/cv
- Thinking Machines batch invariance: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
