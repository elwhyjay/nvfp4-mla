"""Same-run A/B of the FP8 path: a reference kernel source vs the package kernel, GPU time via CUPTI.

  git show <commit>:nvfp4_mla/csrc/mla_decode.cu > /tmp/ref.cu
  python benchmarks/ab.py --ref /tmp/ref.cu [--heads 16,128] [--batch 1,16,64] [--s-kv 4096,16384,65536]

Run-to-run variation on the same GPU model is up to 8%, so compare versions within one process.
"""
import argparse
import hashlib
import os
import pathlib

import torch
from torch.utils.cpp_extension import load

import baselines
import nvfp4_mla
from nvfp4_mla.testing import RandomCache, rel_l2

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="reference mla_decode.cu")
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
print(f"{'H':>4} {'B':>3} {'s_kv':>6} | {'ref ms':>8} {'new ms':>8} {'new/ref':>8} | {'rel diff':>9}")
for H in ints(a.heads):
    for B in ints(a.batch):
        for S in ints(a.s_kv):
            c = RandomCache([S] * B, 64, H)
            fr = lambda: ref.decode(*c.args(), S, 0, 3, True)
            fn = lambda: new.decode(*c.args(), S, 0, 3, True)
            tr, tn = baselines.gpu_ms(fr), baselines.gpu_ms(fn)
            print(f"{H:>4} {B:>3} {S:>6} | {tr:8.3f} {tn:8.3f} {tn / tr:8.3f} | {rel_l2(fn(), fr()):9.2e}", flush=True)
            del c
            torch.cuda.empty_cache()
