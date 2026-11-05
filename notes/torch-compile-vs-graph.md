# T2：torch.compile vs 手写 CUDA Graph（decode step）

环境：MINI_COMPILE=1 → `torch.compile(model, mode="reduce-overhead",
dynamic=True)`，30 步稳态，8 seq batched decode，0.5B。

| 版本 | step 耗时 | GPU 间隙 | vs eager |
|---|---|---|---|
| eager batched | 13.75ms | ~69% | 1x |
| **torch.compile** | **8.37ms** | ~48% | 1.6x |
| 手写 CUDA graph | 5.46ms | — | 2.5x |

结论：compile（cudagraph trees + triton 融合，profiler 里 720 次
fused kernel）**免费拿到手写图 ~60% 的收益**，但 guard 检查 +
Python 残留让间隙仍有 48%；手写图对 KV 指针/分桶控制更细，
再快 1.5x。面试口径：「框架图是 1 行代码的 60% 答案，
剩下 40% 在自定义内存布局上」。
