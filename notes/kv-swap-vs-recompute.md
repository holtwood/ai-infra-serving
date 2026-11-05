# T5：KV swap-to-CPU vs recompute（微基准实测，未接引擎）

7B KV=56KiB/tok，pinned buffer 实测 PCIe ~22-24GiB/s：
ctx 512→往返 2.4ms｜2048→9.4ms｜8192→36.7ms。
对照 recompute=重新 prefill（7B 实测 ~16ktok/s → 8k ctx ≈500ms）：
**swap 在 8k ctx 便宜 ~14x**，且全程占优（512 时 2.4ms vs ~30ms）。
结论：本机 1TB 内存条件下，swap 显著优于 vLLM V1 的 recompute-only；
vLLM 的选择是通用性保守（不依赖 host mem、实现简单），
但在内存充裕机器上留了性能。引擎级接入留待后续（allocator
页粒度 swap 比整条 KV 更优——只需搬被逐出的块）。
