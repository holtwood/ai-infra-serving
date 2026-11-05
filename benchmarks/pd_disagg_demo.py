"""P/D 分离演示：同机双角色 —— producer(prefill) 产 KV → 共享存储 →
consumer(decode) 免 prefill 直接出 token。

真实生产里 KV 走 NIXL/RDMA/LMCache；单机演示用 SharedStorageConnector
（vLLM 自带，KV 序列化到共享目录）。要点是【角色分离】本身：
producer 只算 prefill、不生成；consumer 命中 KV 后跳过 prefill。

测量：
  - producer：prefill + KV dump 耗时
  - consumer：KV load + 首 token 耗时（对比同 prompt 冷启动 TTFT）

用法：python pd_disagg_demo.py
"""
import os
import shutil
import time

SHARE = "/data/ai-learn/kv_share"
MODEL = "/data/ai-learn/models/Qwen2.5-0.5B-Instruct"


def make_llm(role: str):
    from vllm import LLM
    from vllm.config import KVTransferConfig
    ktc = KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role=role,
        kv_connector_extra_config={"shared_storage_path": SHARE},
    )
    # enforce_eager=True：绕过 vllm 0.11.0 的 bug —— cudagraph 预热
    # _dummy_run 会触发 save_kv_layer 但此时无 connector_metadata → assert 崩
    return LLM(model=MODEL, kv_transfer_config=ktc,
               max_model_len=2048, gpu_memory_utilization=0.4,
               enforce_eager=True)


def main():
    from vllm import SamplingParams
    shutil.rmtree(SHARE, ignore_errors=True)

    # 一个够长的 prompt（模拟 RAG/长上下文 prefill）
    prompt = "请总结以下内容：" + (
        "KV cache 是大模型推理中缓存 key/value 的中间结果的机制，"
        "它把每步 attention 的历史状态存起来，避免重复计算。") * 40
    sp = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)

    print("=== producer (prefill only) ===")
    t0 = time.perf_counter()
    prod = make_llm("kv_producer")
    out = prod.generate([prompt], sp)   # prefill + dump KV 到共享目录
    t1 = time.perf_counter()
    print(f"prefill+dump: {t1-t0:.2f}s, KV files: {os.listdir(SHARE)}")
    del prod
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    print("\n=== consumer (decode only，命中共享 KV) ===")
    cons = make_llm("kv_consumer")
    t2 = time.perf_counter()
    out = cons.generate([prompt], sp)   # 命中共享 KV → 跳过 prefill
    t3 = time.perf_counter()
    print(f"decode 输出: {out[0].outputs[0].text[:80]}...")
    print(f"[热] 生成 32tok 耗时: {t3-t2:.2f}s")

    print("\n=== 对照：清掉共享 KV，同 consumer 冷跑 ===")
    shutil.rmtree(SHARE, ignore_errors=True)
    t4 = time.perf_counter()
    out2 = cons.generate([prompt], sp)  # 无 KV 可命中 → 自己 prefill
    t5 = time.perf_counter()
    print(f"[冷] 生成 32tok 耗时: {t5-t4:.2f}s")
    print(f"\nP/D 收益: 免 prefill 省下 ~{t5-t4-(t3-t2):.2f}s")


if __name__ == "__main__":
    main()
