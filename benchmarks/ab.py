"""Same-run A/B of the FP8 path: a reference kernel source vs the package kernel, GPU time via CUPTI.

  git show <commit>:nvfp4_mla/csrc/mla_decode.cu > /tmp/ref.cu
  python benchmarks/ab.py --ref /tmp/ref.cu [--heads 16,128] [--batch 1,16,64] [--s-kv 4096,16384,65536]

Run-to-run variation on the same GPU model is up to 8%, and small shapes (tens of microseconds) also depend on the
GPU's clock state, so the two kernels are timed alternately (--rounds times, after a warm-up) and the median of the
per-round ratios is reported.
"""
import argparse
import hashlib
import statistics
import os
import pathlib

import torch
from torch.utils.cpp_extension import load

import baselines
import nvfp4_mla
from nvfp4_mla.testing import RandomCache, rel_l2

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="reference mla_decode.cu")
ap.add_argument("--ref-stages", type=int, default=3)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--precision", type=int, default=1, help="precision of the package kernel (1 FP8, 2 FP4 QK)")
ap.add_argument("--heads", default="16,128")
ap.add_argument("--batch", default="1,16,64")
ap.add_argument("--s-kv", default="4096,16384,65536")
a = ap.parse_args()
ints = lambda s: [int(x) for x in s.split(",")]
major, minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}a"
src = pathlib.Path(a.ref)
ref = load(f"ref_{hashlib.sha1(src.read_bytes()).hexdigest()[:10]}", [str(src)], extra_cuda_cflags=["-O3", "-lineinfo"])
new = nvfp4_mla._load()
warm = RandomCache([65536] * 64, 64, 128)  # bring the clocks up before the first timed shape
for _ in range(20):
    new.decode(*warm.args(), 65536, 0, 3, a.precision)
torch.cuda.synchronize()
del warm
print(f"{'H':>4} {'B':>3} {'s_kv':>6} | {'ref ms':>8} {'new ms':>8} {'new/ref':>8} {'min-max':>13} | {'rel diff':>9}")
for H in ints(a.heads):
    for B in ints(a.batch):
        for S in ints(a.s_kv):
            c = RandomCache([S] * B, 64, H)
            fr = lambda: ref.decode(*c.args(), S, 0, a.ref_stages, True)
            fn = lambda: new.decode(*c.args(), S, 0, 3, a.precision)
            rounds = [(baselines.gpu_ms(fr), baselines.gpu_ms(fn)) for _ in range(a.rounds)]
            ratios = [tn / tr for tr, tn in rounds]
            tr, tn = statistics.median(r for r, _ in rounds), statistics.median(n for _, n in rounds)
            print(f"{H:>4} {B:>3} {S:>6} | {tr:8.3f} {tn:8.3f} {statistics.median(ratios):8.3f} "
                  f"{min(ratios):6.3f}-{max(ratios):.3f} | {rel_l2(fn(), fr()):9.2e}", flush=True)
            del c
            torch.cuda.empty_cache()
