"""Achievable DRAM read BW and mma.sync throughput / latency of the local GPU, and the ridge point they imply.

  python benchmarks/roofline.py      (needs ~8 GB of free GPU memory)
The read BW is what bench.py's %BW column is relative to (pass it as --bw on GPUs it doesn't know).
"""
import hashlib
import itertools
import os
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load

MAJOR, MINOR = torch.cuda.get_device_capability()
CC = f"{MAJOR}{MINOR}"
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{MAJOR}.{MINOR}a"  # the kind:: mma variants need the arch-specific target
SRC = pathlib.Path(__file__).with_suffix(".cu")
mod = load(f"roofline_sm{CC}_{hashlib.sha1(SRC.read_bytes()).hexdigest()[:8]}", [str(SRC)], extra_cuda_cflags=["-O3"])
SMS = torch.cuda.get_device_properties(0).multi_processor_count


def ms(fn, iters=20):
    fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in ev)


print(f"{torch.cuda.get_device_name()}  CC{CC}  SMs={SMS}")
flag = torch.zeros(1, dtype=torch.int32, device="cuda")
best_bw = 0.0
x = torch.empty(4 << 30, dtype=torch.uint8, device="cuda")  # 4 GB: a 1 GB buffer still hits L2 left from the last run
for unroll, bps, thr in itertools.product((1, 4, 8), (2, 4, 8), (256, 512)):
    t = ms(lambda: mod.read(x, unroll, SMS * bps, thr, flag))
    bw = x.numel() / t / 1e6
    best_bw = max(best_bw, bw)
    print(f"  read 4GB unroll={unroll} blocks/SM={bps} threads={thr}: {bw:7.0f} GB/s")
y = torch.empty_like(x)
print(f"  torch copy_ 4GB (r+w): {2 * x.numel() / ms(lambda: y.copy_(x)) / 1e6:7.0f} GB/s")
del x, y
torch.cuda.empty_cache()

# (name, flop per instruction); kinds 2 and 3 are sm_120a / sm_121a only
KINDS = [("bf16 m16n8k16", 4096), ("e4m3 m16n8k32", 8192), ("kind::f8f6f4 e4m3 m16n8k32", 8192),
         ("kind::mxf4nvf4 e2m1 m16n8k64", 16384)]
out = torch.zeros(1, device="cuda")
cyc = torch.zeros(SMS * 64, dtype=torch.int64, device="cuda")
iters, tflops = 4096, {}


def cycles_per_mma(kind, chains, threads):
    """Mean clock64 cycles per mma per scheduler, one block on one SM (warps spread over the 4 schedulers)."""
    mod.mma(kind, chains, iters, 1, threads, cyc, out)
    torch.cuda.synchronize()
    warps = threads // 32
    return cyc[:warps].double().mean().item() / (iters * chains * max(1, warps // 4))


print("  mma.sync kind                    latency   cycles/mma/scheduler    flop/cycle/SM   TFLOPS (whole GPU)")
for kind in range(4 if MAJOR == 12 else 2):
    name, flop = KINDS[kind]
    cycles_per_mma(kind, 1, 32)  # warm up
    lat = cycles_per_mma(kind, 1, 32)
    one, two = cycles_per_mma(kind, 8, 128), cycles_per_mma(kind, 8, 256)
    best = 0.0
    for bps, thr in itertools.product((1, 2, 4), (128, 256)):
        t = ms(lambda: mod.mma(kind, 8, iters, SMS * bps, thr, cyc, out), iters=5)
        best = max(best, SMS * bps * (thr // 32) * iters * 8 * flop / (t * 1e-3) / 1e12)
    tflops[name] = best
    print(f"  {name:31s} {lat:6.1f}   {one:5.1f} (1 warp) {two:5.1f} (2 warps)   {4 * flop / min(one, two):7.0f}"
          f"   {best:7.1f}")
best_tf = tflops[KINDS[0][0]]

ridge = best_tf * 1e12 / (best_bw * 1e9)
print(f"best read BW {best_bw:.0f} GB/s | mma.sync BF16 {best_tf:.1f} TFLOPS | ridge {ridge:.0f} flop/B "
      f"-> NVFP4 decode (6.2*H*q_len flop/B) is memory-bound for H*q_len < {ridge / 6.2:.0f}")
