"""Split-count sweep: decode and combine kernel time (CUPTI) per explicit split count, vs the host policy's choice.

  python benchmarks/split_sweep.py [--csv out.csv]
"""
import argparse
import csv
import math
import statistics

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import nvfp4_mla
from nvfp4_mla.testing import RandomCache

SHAPES = [(128, 1, 65536), (128, 16, 4096), (128, 16, 16384), (128, 16, 65536), (128, 64, 4096), (128, 64, 16384),
          (128, 64, 65536), (16, 1, 65536), (16, 16, 65536), (16, 64, 16384), (16, 64, 65536)]
CONFIGS = [("fp8", 3, 2), ("fp8", 2, 3), ("bf16", 3, 2)]  # (path, stages, CTAs per SM on RTX PRO 6000, from ncu)
SPLITS = [1, 2, 3, 4, 6, 8, 11, 16, 24, 32, 48, 64, 96, 128]
TILE, C0 = 32, 256


def ru(x, m):
    return (x + m - 1) // m * m


def policy(L, ctas, wave):
    """Python copy of decode_impl's split choice."""
    min_chunk, best, chunk, s = ru(max(int(math.sqrt(L)), 1), TILE), None, ru(L, TILE), 1
    while True:
        c = ru((L + s - 1) // s, TILE)
        if s > 1 and c < min_chunk:
            break
        cost = (ctas * ((L + c - 1) // c) + wave - 1) // wave * (c + C0)
        if best is None or cost < best:
            best, chunk = cost, c
        if c == TILE:
            break
        s += 1
    return (L + chunk - 1) // chunk


def kernel_ms(fn, iters=15):
    """Median per-call GPU time of the decode and combine kernels, L2 flushed before each call."""
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.int8, device="cuda")
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            flush.zero_()
            with record_function("fn"):
                fn()
        torch.cuda.synchronize()
    gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    calls = []
    for a in (e for e in gpu if e.name == "fn"):
        d = {"decode": 0.0, "combine": 0.0}
        for k in gpu:
            if k.name != "fn" and a.time_range.start <= k.time_range.start and k.time_range.end <= a.time_range.end:
                d["combine" if "mla_combine" in k.name else "decode"] += k.device_time / 1e3
        calls.append(d)
    return {k: statistics.median(c[k] for c in calls) for k in ("decode", "combine")}


ap = argparse.ArgumentParser()
ap.add_argument("--csv")
a = ap.parse_args()
sms = torch.cuda.get_device_properties(0).multi_processor_count
f = nvfp4_mla._load().decode
rows = []
for H, B, L in SHAPES:
    c = RandomCache([L] * B, 64, H)
    for path, stages, per_sm in CONFIGS:
        auto = policy(L, B * H // 16, sms * per_sm)
        res = {}
        for s in sorted(set(SPLITS + [auto])):
            chunk = ru((L + s - 1) // s, TILE)
            n = (L + chunk - 1) // chunk
            if n in res or chunk < TILE:
                continue
            t = kernel_ms(lambda: f(*c.args(), L, n, stages, int(path == "fp8")))
            res[n] = t
            rows.append(dict(path=path, stages=stages, heads=H, batch=B, s_kv=L, splits=n, auto=int(n == auto),
                             decode_ms=t["decode"], combine_ms=t["combine"], total_ms=t["decode"] + t["combine"]))
        best = min(res, key=lambda n: res[n]["decode"] + res[n]["combine"])
        tot = lambda n: res[n]["decode"] + res[n]["combine"]
        print(f"H={H:<3} B={B:<2} L={L:<5} {path}/{stages}: policy {auto:>3} splits {tot(auto):7.3f} ms "
              f"(dec {res[auto]['decode']:.3f} comb {res[auto]['combine']:.3f}) | best {best:>3} {tot(best):7.3f} ms "
              f"(dec {res[best]['decode']:.3f} comb {res[best]['combine']:.3f}) | policy/best {tot(auto) / tot(best):.3f}",
              flush=True)
    del c
    torch.cuda.empty_cache()
if a.csv:
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
