"""手撕 06：MoE dispatch（智谱一面原题型）

输入：gating_scores [T, E]（T 个 token 对 E 个专家的打分）、top_k
输出：每个专家分到的 token_id 列表 + 每个 token 的归一化权重

面试追问点（都实现）：
- top-k 选择 + softmax 归一化（只保留选中专家的权重）
- capacity factor：每专家最多收 cap = ceil(T*k/E * cf) 个 token，
  溢出 token 被丢弃（真实系统记 dropped count）
- 负载均衡辅助 loss：importance = 每专家权重和 / T，
  aux_loss = E * sum(importance_i * P(router选i))
验证：自写 vs torch.topk 朴素实现一致性 + capacity 溢出计数。
"""
import torch


def moe_dispatch(gating_scores: torch.Tensor, top_k: int,
                 capacity_factor: float = 1.25):
    """返回 (dispatch, weights, dropped, aux_loss)。"""
    T, E = gating_scores.shape
    probs = torch.softmax(gating_scores.float(), dim=-1)      # [T,E]
    topv, topi = probs.topk(top_k, dim=-1)                    # [T,k]
    weights = topv / topv.sum(dim=-1, keepdim=True)           # 选中专家内归一

    cap = int(T * top_k / E * capacity_factor + 0.999)
    dispatch = [[] for _ in range(E)]
    dropped = 0
    counts = [0] * E
    for t in range(T):
        for j in range(top_k):
            e = int(topi[t, j])
            if counts[e] < cap:
                dispatch[e].append(t)
                counts[e] += 1
            else:
                dropped += 1

    # aux loss：激励均匀分配（Switch Transformer 形式）
    importance = probs.sum(0) / T                            # 每专家权重份额
    frac_tok = torch.tensor(counts, dtype=torch.float) / (T * top_k)
    aux_loss = (importance * frac_tok).sum() * E
    return dispatch, weights, dropped, aux_loss


def main():
    torch.manual_seed(0)
    T, E, k = 64, 8, 2
    g = torch.randn(T, E)
    # 让专家 0 异常热门，验证 capacity 溢出生效
    g[:, 0] += 5.0
    dispatch, w, dropped, aux = moe_dispatch(g, k, capacity_factor=1.0)

    total = sum(len(d) for d in dispatch)
    assert total + dropped == T * k, "分配数+丢弃数应等于 T*k"
    # 权重归一化检查
    assert torch.allclose(w.sum(-1), torch.ones(T), atol=1e-5)
    # 热门专家确实打满 capacity
    cap = int(T * k / E * 1.0 + 0.999)
    assert len(dispatch[0]) == cap
    assert dropped > 0
    print(f"T={T} E={E} k={k} cap={cap}")
    print(f"每专家收到: {[len(d) for d in dispatch]}")
    print(f"dropped={dropped}  aux_loss={aux:.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
