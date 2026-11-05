# 手撕系列 05：反向传播推导（面试"讲题"版）

> 面试考的不是默写公式，是**讲清楚**：每步说 Jacobian 是谁、链式乘完形状对不对。
> 记号：上游梯度 dY = ∂L/∂Y；L 是标量 ⇒ 任何中间量的梯度**与自身同形**（形状守恒律，推导的第一检查点）。

## 1. softmax + 交叉熵：dL/dz = p − y（必推）

**前向**：logits z ∈ R^V → p_i = e^{z_i} / Σ_j e^{z_j} → L = −log p_y（y 为真标签，one-hot 向量）

**第一步：softmax 的 Jacobian**

∂p_i/∂z_j = p_i(δ_ij − p_j)，写成矩阵 J = diag(p) − p pᵀ ∈ R^{V×V}

- 对角：∂p_i/∂z_i = p_i(1−p_i)；非对角：∂p_i/∂z_j = −p_i p_j
- 直觉：归一化约束下 p_i 变大必然挤占其他项 → 非对角恒负

**第二步：链式乘过去**

dL/dz_j = Σ_i (∂L/∂p_i)·(∂p_i/∂z_j)，其中 ∂L/∂p_i = −y_i/p_i

代入 Jacobian：**−Σ_i (y_i/p_i)·p_i(δ_ij − p_j) = −y_j + p_j·Σ_i y_i = p_j − y_j**

**形状视角**：dL/dp (V,) 右乘 J (V×V) → (V,)，与 z 同形 ✓

**工程意义**：
- 梯度天然有界 ∈ [−1,1] → CE 训练稳定；logits 差多少梯度都一样大，不会炸
- 实现上 `F.cross_entropy` 融合 log_softmax + NLL：先用 LSE 求 log p（防 log(0) 与 exp 溢出），反传直接 p−y，不用存中间 Jacobian

## 2. LayerNorm backward：讲思路，不默写

**前向**（对最后一维 N 归一化）：x → μ=mean(x), σ²=var(x), x̂=(x−μ)/σ → y = γ⊙x̂+β

- **dγ = Σ dY⊙x̂，dβ = Σ dY**（对前导维归约，形状 (N,)）——elementwise 部分是白送的
- **dx 的难点**：x 通过三条路径影响输出——x̂ 分子、μ、σ²（μ 和 σ² 都依赖全部 x_i）
- 三条链全部展开合并后（记 g = γ⊙dY，逐维）：

  **dx = (1/Nσ) · [ N·g − Σ_j g_j − x̂ ⊙ Σ_j (g_j x̂_j) ]**

- **怎么讲**：g 先被投影掉两个方向——常数方向（Σg 项=去均值）与 x̂ 方向（Σ g·x̂ 项=去该方向的协方差分量），再缩放回 x 的空间（/σ）
- 形状检查：Σg、Σ(g·x̂) 是 (…,1) 归约后广播，dx 与 x 同形 ✓
- 加分点：var 用**有偏**估计（1/N 而非 1/(N−1)）；RMSNorm 砍掉 mean 分支 → dx 少一项，这就是它便宜的原因

## 3. Attention VJP：维度流转（一层 causal MHA）

**前向**（B,H,S,D）：S₀ = QKᵀ/√D → (B,H,S,S)；P = softmax(S₀)；O = PV → (B,H,S,D)

给定 dO (B,H,S,D)，沿图倒推——每步只问"这个算子的 Jacobian 是什么"：

| 步骤 | 算子 | VJP | 结果形状 |
|---|---|---|---|
| dV | O = P·V | dV = Pᵀ·dO | (B,H,S,D) |
| dP | O = P·V | dP = dO·Vᵀ | (B,H,S,S) |
| dS₀ | P = softmax(S₀) | dS₀ = P ⊙ (dP − rowsum(dP⊙P)) | (B,H,S,S) |
| dQ | S₀ = QKᵀ/√D | dQ = dS₀·K /√D | (B,H,S,D) |
| dK | S₀ = QKᵀ/√D | dK = dS₀ᵀ·Q /√D | (B,H,S,D) |

**口诀与坑**：
- matmul VJP：C=AB ⇒ dA = dC·Bᵀ，dB = Aᵀ·dC（转置补在"另一边"，形状凑回去就对）
- softmax VJP 别构造 (S×S) Jacobian：J·v = p⊙v − p·(v·p) 是 O(S) 公式，即表中形式
- scale 在 S₀ 里：dQ、dK 各乘 1/√D（常被问反成 √D）
- causal mask：被 mask 元素梯度天然为 0，masked_fill 处梯度直通未 mask 元素，无需单独反传
- FlashAttention 反传要点：前向不存 P（S×S 太大），反传时用 saved O + LSE **重算** P 再做 VJP——recompute 换 memory

## 4. 面试讲题模板

1. 先写前向表达式 + 逐步标形状
2. "L 是标量 ⇒ 每个中间量的梯度与自身同形"——每次链式乘完先核形状
3. 沿计算图倒推，每步只处理一个算子的 Jacobian
4. 收尾主动讲数值与工程注意点（LSE 稳定化、融合 kernel、recompute 换显存）
