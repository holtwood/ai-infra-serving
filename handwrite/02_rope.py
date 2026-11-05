"""手撕系列 02：RoPE 旋转位置编码（两版实现 + 朴素 ground truth 对拍）

必背三问：
- 为什么加在 q/k 上：attention 里只有 q·k 内积吃位置；RoPE 让
  <R_m q, R_n k> = q^T R_{n-m} k —— 内积只依赖【相对】距离 n-m，v 不需要位置
- 为什么叫"旋转"：head_dim 切成 D/2 个二维平面，第 i 对按角度 pos·θ_i 旋转
- 长上下文外推为何衰减：θ 在训练见过的 pos 范围内是插值，超出后高频维先
  混叠 → NTK-aware / YaRN 通过缩放 base/频率把外推变回插值（一句话带过）

两种"配对约定"（数学等价，差一个维度置换）：
  (a) 教科书交错式：相邻两维 (x_{2i}, x_{2i+1}) 一对 —— 原论文形态
  (b) HF half-split：首尾配对 (x_i, x_{i+D/2}) —— transformers
      apply_rotary_pos_emb 的形态；cos/sin 前后两半复制一份
运行：python 02_rope.py
"""
import math
import torch


def inv_freqs(head_dim: int, base: float, device=None):
    """θ_i = base^{-2i/D}, i = 0..D/2-1 —— 第 i 对维度的角速度，从高频到低频。"""
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    return base ** (-i / head_dim)                          # (D/2,)


def rope_textbook(x, pos_ids, base=10000.0):
    """(a) 教科书版：交错两两旋转。x:(B,H,S,D)，pos_ids:(S,) → (B,H,S,D)

    等价于构造 (D,D) 块对角旋转矩阵 R(pos) 再右乘，但 pairwise 写法省掉
    大矩阵：每对独立做二维旋转 [[cos,-sin],[sin,cos]]。
    """
    ang = pos_ids.float()[:, None] * inv_freqs(x.shape[-1], base, x.device)[None]
    cos, sin = ang.cos(), ang.sin()                         # (S,D/2)
    xe, xo = x[..., 0::2], x[..., 1::2]                     # 偶/奇位 (B,H,S,D/2)
    oe = xe * cos - xo * sin                                # cos/sin 广播到 (B,H,S,D/2)
    oo = xe * sin + xo * cos
    return torch.stack((oe, oo), dim=-1).flatten(-2)        # 交错放回 → (B,H,S,D)


def rotate_half(x):
    """(b) HF 小工具：后半维取负轮转到前半 —— (-x2, x1)，配对 (i, i+D/2)。"""
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_hf(x, pos_ids, base=10000.0):
    """(b) 工程版：x·cos + rotate_half(x)·sin，half-split 配对。

    cos/sin 前后两半各复制一份，使 dim i 与 dim i+D/2 同频 θ_i。
    """
    ang = pos_ids.float()[:, None] * inv_freqs(x.shape[-1], base, x.device)[None]
    cos = ang.cos().repeat(1, 2)[None, None]                # (1,1,S,D)：[θ0..θh|θ0..θh]
    sin = ang.sin().repeat(1, 2)[None, None]
    return x * cos + rotate_half(x) * sin                   # (B,H,S,D)


def rope_naive(x, pos_ids, base=10000.0):
    """ground truth：逐位置、逐对（交错约定）纯 Python 套旋转公式，慢但显然对。"""
    x = x.float().clone()
    S, D = x.shape[2], x.shape[3]
    for s in range(S):
        for i in range(D // 2):
            t = float(pos_ids[s]) * base ** (-2.0 * i / D)  # 角度 = pos·θ_i
            c, sn = math.cos(t), math.sin(t)
            a, b = x[..., s, 2 * i].clone(), x[..., s, 2 * i + 1].clone()
            x[..., s, 2 * i] = a * c - b * sn               # 二维旋转矩阵第一行
            x[..., s, 2 * i + 1] = a * sn + b * c           # 第二行
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, S, D = 2, 4, 16, 8
    x = torch.randn(B, H, S, D)             # fp32 自检（bf16 下同样适用，atol 放宽到 2e-2）
    pos = torch.arange(S)

    # (a) 对拍 naive ground truth
    ref = rope_naive(x, pos)
    err = (rope_textbook(x, pos) - ref).abs().max().item()
    assert err < 1e-4, err
    print(f"textbook vs naive    max_err={err:.3e}  PASS")

    # (b) 与 (a) 差一个维度置换：交错序 [e0,o0,e1,o1..] ↔ half 序 [e0,e1..|o0,o1..]
    # perm 把交错存储转成 half-split 存储；inv 是逆置换，转回来对拍。
    h = D // 2
    perm = torch.cat((torch.arange(0, D, 2), torch.arange(1, D, 2)))   # [0,2,..,1,3,..]
    inv = torch.stack((torch.arange(h), torch.arange(h) + h), 1).flatten()
    y = rope_hf(x[..., perm], pos)[..., inv]                # 换序→旋转→换回
    err = (y - ref).abs().max().item()
    assert err < 1e-4, err
    print(f"hf(perm)  vs naive   max_err={err:.3e}  PASS")

    # 旋转是正交变换 → 保范数：验证"真是旋转"的最快 sanity check
    err = (rope_textbook(x, pos).norm(dim=-1) - x.norm(dim=-1)).abs().max().item()
    assert err < 1e-5, err
    print(f"norm preserved       max_err={err:.3e}  PASS")

    # 相对性验证：<R_m q, R_n k> = <q, R_{n-m} k> —— RoPE 的全部意义
    q, k = torch.randn(1, 1, 1, D).float(), torch.randn(1, 1, 1, D).float()
    lhs = float((rope_textbook(q, torch.tensor([3])) *
                 rope_textbook(k, torch.tensor([7]))).sum())
    rhs = float((q * rope_textbook(k, torch.tensor([7 - 3]))).sum())
    assert abs(lhs - rhs) < 1e-4
    print(f"rel-pos identity     |{lhs:.4f}-{rhs:.4f}| < 1e-4  PASS")
