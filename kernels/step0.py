"""Step 0: achievable read BW and BF16 mma.sync peak -> ridge point, per GPU.

  uv run modal run bench.py::sh --cmd "python kernels/step0.py"            (GPU=B200 for SM100)
"""
import hashlib
import itertools
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load

CC = "".join(map(str, torch.cuda.get_device_capability()))
SRC = pathlib.Path(__file__).with_suffix(".cu")
mod = load(f"step0_sm{CC}_{hashlib.sha1(SRC.read_bytes()).hexdigest()[:8]}", [str(SRC)], extra_cuda_cflags=["-O3"])
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
for gb in (4,):  # 1 GB runs read partly from L2 (128 MB) left over from the previous run
    x = torch.empty(gb << 30, dtype=torch.uint8, device="cuda")
    for unroll, bps, thr in itertools.product((1, 4, 8), (2, 4, 8), (256, 512)):
        t = ms(lambda: mod.read(x, unroll, SMS * bps, thr, flag))
        bw = x.numel() / t / 1e6
        best_bw = max(best_bw, bw)
        print(f"  read {gb}GB unroll={unroll} blocks/SM={bps} threads={thr}: {bw:7.0f} GB/s")
    y = torch.empty_like(x)
    print(f"  torch copy_ {gb}GB (r+w): {2 * x.numel() / ms(lambda: y.copy_(x)) / 1e6:7.0f} GB/s")
    del x, y
    torch.cuda.empty_cache()

out = torch.zeros(1, device="cuda")
best_tf, iters = 0.0, 4096
for bps, thr in itertools.product((1, 2, 4, 8), (128, 256)):
    t = ms(lambda: mod.mma(iters, SMS * bps, thr, out), iters=5)
    tf = SMS * bps * (thr // 32) * iters * 8 * 2 * 16 * 8 * 16 / (t * 1e-3) / 1e12
    best_tf = max(best_tf, tf)
    print(f"  mma.sync bf16 m16n8k16 blocks/SM={bps} threads={thr}: {tf:7.1f} TFLOPS")

ridge = best_tf * 1e12 / (best_bw * 1e9)
print(f"best read BW {best_bw:.0f} GB/s | best mma.sync BF16 {best_tf:.1f} TFLOPS | ridge {ridge:.0f} flop/B "
      f"-> NVFP4 decode (6.2*H*q_len flop/B) is memory-bound for H*q_len < {ridge / 6.2:.0f}")
