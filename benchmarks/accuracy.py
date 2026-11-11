"""Accuracy of every nvfp4_mla path, and of FlashInfer Cake on SM100, on the same random NVFP4 caches.

  python benchmarks/accuracy.py

The measure is the relative L2 error against fp32 attention over the unquantized BF16 KV: what a user of the cache
gets, format and kernel error together. The ratio to the CUDA kernel's BF16 mode shows what a path adds on top of
the format. The error against the dequantized cache (the kernel's own arithmetic) is printed in brackets.

Two global scales: 1 (block scales around 1) and calibrated (amax / (6 x 448), block scales up to the E4M3 maximum,
as a serving engine would set it; RoPE bytes up to 448).
"""
import torch

import baselines
import bench
from nvfp4_mla.testing import RandomCache, rel_l2

sm100 = torch.cuda.get_device_capability()[0] == 10
providers = ["nvfp4", "nvfp4-fp8"] + (["nvfp4-tc", "nvfp4-tc/0/v8", "cake"] if sm100 else ["nvfp4-fp4q"])
SCALES = {"scale 1": (1.0, 1.0), "calibrated": (2.0 ** -9, 2.0 ** -6)}
print(f"{'shape':22s} {'scales':10s} " + "".join(f"{p:>22s}" for p in providers))
for H, B, S in [(16, 4, 4096), (128, 4, 4096), (16, 16, 16384)]:
    for sname, sc in SCALES.items():
        c = RandomCache([S] * B, 64, H, *sc)
        ref, ref16 = c.reference(), c.reference_bf16()
        cells, base = [], None
        for p in providers:
            try:
                o = bench.make(p, c, None, S)().float().reshape(ref.shape)
                e16, e = rel_l2(o, ref16), rel_l2(o, ref)
                base = base or e16
                cells.append(f"{e16:.4f} x{e16 / base:.2f} ({e:.4f})")
            except Exception as ex:
                cells.append(baselines.first_line(ex)[:22])
        print(f"H={H:<3} B={B:<2} S={S:<6} {sname:10s} " + "".join(f"{x:>22s}" for x in cells), flush=True)
