"""Accuracy of nvfp4_mla, and of FlashInfer Cake on SM100, on the same random NVFP4 cache.

  python benchmarks/accuracy.py

Relative L2 error against two fp32 references: attention over the dequantized NVFP4 cache (the kernel's own error)
and over the unquantized BF16 cache (kernel and format error together).
"""
import torch

import baselines
import bench
from nvfp4_mla.testing import RandomCache, rel_l2

providers = ["nvfp4"] + (["cake"] if torch.cuda.get_device_capability()[0] == 10 else [])
for H, B, S in [(16, 4, 4096), (128, 4, 4096), (16, 16, 16384)]:
    c = RandomCache([S] * B, 64, H)
    ref, ref16 = c.reference(), c.reference_bf16()
    for p in providers:
        try:
            o = bench.make(p, c, None, S)().float().reshape(ref.shape)
            print(f"H={H:<3} B={B:<2} S={S:<5} {p:5s}  vs dequantized {rel_l2(o, ref):.4f}  vs BF16 {rel_l2(o, ref16):.4f}")
        except Exception as e:
            print(f"H={H:<3} B={B:<2} S={S:<5} {p:5s}  {baselines.first_line(e)}")
