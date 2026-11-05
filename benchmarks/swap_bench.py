import torch, time
torch.manual_seed(0)
KV_PER_TOK = 56 * 1024  # 7B: 28层×2×4头×128×2B
for ctx in (512, 2048, 8192):
    kv = torch.randn(ctx * KV_PER_TOK // 2, dtype=torch.bfloat16, device="cuda")
    pin = torch.empty_like(kv, device="cpu", pin_memory=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pin.copy_(kv, non_blocking=False)
    torch.cuda.synchronize(); t_out = time.perf_counter() - t0
    t0 = time.perf_counter()
    kv.copy_(pin, non_blocking=False)
    torch.cuda.synchronize(); t_in = time.perf_counter() - t0
    mib = ctx * KV_PER_TOK / 2**20
    print(f"ctx={ctx:5d} KV={mib:7.1f}MiB out={t_out*1e3:6.1f}ms in={t_in*1e3:6.1f}ms "
          f"rt={((t_out+t_in)*1e3):6.1f}ms bw={mib/1024/t_out:5.1f}GiB/s")
