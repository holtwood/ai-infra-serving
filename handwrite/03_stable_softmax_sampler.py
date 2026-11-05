"""手撕系列 03：数值稳定 softmax + log_softmax/LSE + 温度/top-k/top-p 采样器

面试必问：
- 为什么减 max 而不是 clamp：softmax 对 logits 平移不变（分子分母同乘
  e^{-c} 抵消），减 max 把最大指数压到 e^0=1，数学结果不变、只防溢出；
  clamp 会改变相对大小 = 直接改分布
- 溢出阈值：fp32/bf16 指数域相同，exp(x) 在 x ≈ 88.7 处撞 FLT_MAX(3.4e38)
- LSE（log-sum-exp）是 FlashAttention online softmax 的原语：logits 不落地
运行：python 03_stable_softmax_sampler.py
"""
import torch


def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """(a) 稳定 softmax：先减 max（softmax(x-c)=softmax(x)，平移不变性）。

    不减 max：e^1000=inf → inf/inf=NaN（见 __main__ 反例）。
    减了 max 最大项恒为 e^0=1，分母 ∈ [1, V]；下溢最多该项概率≈0，无害。
    """
    e = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax_lse(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """(b) log_softmax = x - LSE(x)，LSE 用同一个减 max trick 防溢出。

    log Σ e^{x_i} = m + log Σ e^{x_i - m}   （把最大项提出 log 外）
    CE loss 数值上需要 log p 而不是 p：p=0 时 log(0)=-inf，先取 log 更稳。
    """
    m = x.max(dim=dim, keepdim=True).values
    lse = m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))
    return x - lse                                          # log p_i = x_i - LSE


def sample(logits: torch.Tensor, temperature: float = 1.0,
           top_k: int = 0, top_p: float = 1.0, gen=None) -> int:
    """(c) 从 logits 到 token id 全手写。logits: (V,) → int。

    顺序有讲究：先除温度（改变分布形状），再 top-k/top-p 截断。
    采样用逆 CDF：u~U(0,1)，返回 cumsum 首个 ≥u 的位置 —— 与
    torch.multinomial 同分布（multinomial 本质就是 inverse-CDF/Gumbel 等价物）。
    """
    if temperature <= 0:
        return int(logits.argmax())                         # T→0 时分布退化到 argmax
    z = logits.float() / temperature                        # T<1 尖锐 / T>1 平坦
    if top_k > 0:                                           # 只留前 k 个
        kth = z.topk(min(top_k, z.numel())).values[-1]
        z = z.masked_fill(z < kth, float("-inf"))
    if top_p < 1.0:                                         # nucleus：保累计≥p 最小集合
        s_z, idx = z.sort(descending=True)
        s_p = softmax_stable(s_z)
        # "去掉自己后的累计 > p"才砍 → 首个让累计超 p 的 token 仍保留（HF 同款语义）
        s_z = s_z.masked_fill(s_p.cumsum(-1) - s_p > top_p, float("-inf"))
        z = torch.full_like(z, float("-inf")).scatter_(0, idx, s_z)
    probs = softmax_stable(z)
    u = torch.rand((), generator=gen, device=z.device)
    return int(torch.searchsorted(probs.cumsum(-1), u).clamp(max=z.numel() - 1))


def smoke_test():
    torch.manual_seed(0)
    V = 8
    logits = torch.randn(V) * 2
    probs = softmax_stable(logits)

    # 分布合理性：10000 次逆 CDF 采样，经验频率应≈真实概率（界取 ~10σ 宽松）
    g = torch.Generator().manual_seed(0)
    cnt = torch.zeros(V)
    for _ in range(10000):
        cnt[sample(logits, gen=g)] += 1
    err = (cnt / 10000 - probs).abs().max().item()
    assert err < 0.05, f"freq={cnt/10000} vs p={probs}"
    print(f"inverse-CDF 采样分布   max|freq-p|={err:.3f}  PASS")

    # top-k=3：2000 次采样必须全部落在 top3 内
    keep = set(logits.topk(3).indices.tolist())
    assert all(sample(logits, top_k=3, gen=g) in keep for _ in range(2000))
    print("top-k=3 截断          PASS")

    # top-p=0.5：只能落在 nucleus 内（nucleus 定义与实现同款：cum-p ≤ p）
    s_p, s_idx = probs.sort(descending=True)
    nucleus = set(s_idx[(s_p.cumsum(0) - s_p) <= 0.5].tolist())
    assert all(sample(logits, top_p=0.5, gen=g) in nucleus for _ in range(2000))
    print("top-p=0.5 截断        PASS")

    assert sample(logits, temperature=0) == int(logits.argmax())
    print("greedy argmax         PASS")


if __name__ == "__main__":
    # 反例：不减 max，fp32 直接 NaN（bf16 指数域同 fp32，一样死）
    x = torch.tensor([1000.0, 1001.0, 999.0])
    print(f"x={x.tolist()}")
    print(f"  naive : {x.exp() / x.exp().sum()}   ← e^1000=inf，inf/inf=NaN")
    print(f"  stable: {softmax_stable(x)}   (应≈[0.245,0.665,0.090])")

    # 对拍 torch 官方实现
    r = torch.randn(4, 1024)
    assert (softmax_stable(r) - r.softmax(-1)).abs().max() < 1e-6
    assert (log_softmax_lse(r) - r.log_softmax(-1)).abs().max() < 1e-5
    print("softmax/log_softmax 对拍 F.softmax  PASS")

    smoke_test()
