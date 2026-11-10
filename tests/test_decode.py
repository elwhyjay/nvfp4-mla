"""uv run modal run bench.py::sh --cmd "pytest -q test_decode.py"   (GPU=B200 for SM100)"""
import hashlib
import os
import pathlib

import pytest
import torch
from torch.utils.cpp_extension import load

import ref

SRC = str(pathlib.Path(__file__).parent / "kernels" / "MLA_decode.cu")
MAJOR, MINOR = torch.cuda.get_device_capability()
CC = f"{MAJOR}{MINOR}"
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{MAJOR}.{MINOR}a"  # cvt.rn.bf16x2.e2m1x2 needs sm_100a/sm_120a
# arch in the name: the torch_extensions cache on the Volume is shared by both GPUs.
# source hash in the name: Modal-mounted sources don't look newer than cached objects, so ninja would skip rebuilds.
TAG = f"sm{CC}_{hashlib.sha1(pathlib.Path(SRC).read_bytes()).hexdigest()[:8]}"
ext = {fast: load(f"mla_decode_{TAG}{'_fast' if fast else ''}", [SRC],
                  extra_cuda_cflags=["-O3", "-lineinfo"] + (["--use_fast_math"] if fast else []))
       for fast in (False, True)}

CASES = [  # (kv_lens, page_size, heads)
    ([1], 1, 16),
    ([17, 1, 64, 65], 16, 16),           # lengths off the page / tile boundary, length 1
    ([1000, 4097, 3, 64], 64, 16),
    ([513, 2048], 1, 128),
    ([4096, 1, 777, 5000], 64, 128),
    ([31, 33, 95, 97], 16, 16),          # around the v3 32-token tile and 3-stage prologue
]


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("ckv_scale,kpe_scale", [(1.0, 1.0), (0.37, 3.1)])
def test_dequant_bitwise(fast, ckv_scale, kpe_scale):
    c = ref.Cache([100, 37], page_size=16, heads=16, ckv_scale=ckv_scale, kpe_scale=kpe_scale)
    out = torch.cat(ext[fast].dequant(c.ckv, c.sf, c.kpe, ckv_scale, kpe_scale, False), -1)
    want = torch.cat(c.dequant(), -1)
    assert torch.equal(out.isnan(), want.isnan())
    ok = ~want.isnan()
    assert torch.equal(out[ok].view(torch.int32), want[ok].view(torch.int32))


def check(out, c, label):
    err = ref.rel_l2(out, ref.ref_fp4(c))
    print(f"{label}: rel-L2 {err:.2e}")
    assert not out.isnan().any()
    assert err < 4e-3  # bf16 output alone ~6e-4; v2+ also rounds P to bf16


def args(c):
    return c.q_nope, c.q_pe, c.ckv, c.sf, c.kpe, c.page_table, c.kv_lens, ref.SM_SCALE, c.ckv_scale, c.kpe_scale


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("kv_lens,page_size,heads", CASES)
@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_decode(version, fast, kv_lens, page_size, heads):
    c = ref.Cache(kv_lens, page_size, heads, ckv_scale=0.5, kpe_scale=2.0)
    out = getattr(ext[fast], f"decode_{version}")(*args(c))
    check(out, c, f"{version} fast={fast} {kv_lens} ps={page_size} H={heads}")


@pytest.mark.parametrize("num_splits", [0, 3, 7])  # 0 = heuristic; splits cross page boundaries, some are empty
@pytest.mark.parametrize("kv_lens,page_size,heads", CASES)
def test_decode_v4(num_splits, kv_lens, page_size, heads):
    c = ref.Cache(kv_lens, page_size, heads, ckv_scale=0.5, kpe_scale=2.0)
    out = ext[False].decode_v4(*args(c), max(kv_lens), num_splits)
    check(out, c, f"v4 splits={num_splits} {kv_lens} ps={page_size} H={heads}")


def test_ref_vs_fa2():
    """The fp32 reference agrees with an independent kernel on the same (dequantized) data."""
    c = ref.Cache([1000, 4097, 3, 64], page_size=64, heads=128)
    assert ref.rel_l2(ref.fa2_dequant(c)(), ref.ref_fp4(c)) < 4e-3
