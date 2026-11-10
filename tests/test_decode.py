"""pytest tests/  (needs a Blackwell GPU and flashinfer-python; the first run builds the extension three ways)"""
import pytest
import torch

pytest.importorskip("flashinfer")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
    pytest.skip("needs a Blackwell GPU (SM100 / SM120)", allow_module_level=True)

import nvfp4_mla  # noqa: E402
from nvfp4_mla.testing import RandomCache, fa2_on_dequantized, rel_l2  # noqa: E402

BUILDS = {"native": (), "fast_math": ("--use_fast_math",), "compat_cvt": ("-DNVFP4_MLA_COMPAT_CVT",)}

CASES = [  # (kv_lens, page_size, heads)
    ([1], 1, 16),
    ([17, 1, 64, 65], 16, 16),           # lengths off the page / tile boundary, length 1
    ([1000, 4097, 3, 64], 64, 16),
    ([513, 2048], 1, 128),
    ([4096, 1, 777, 5000], 64, 128),
    ([31, 33, 95, 97], 16, 16),          # around the 32-token tile and the 3-stage prologue
]


def run(c, build="native", splits=0, stages=3):
    return nvfp4_mla._load(*BUILDS[build]).decode(*c.args(), int(c.kv_lens.max()), splits, stages)


def check(out, c, label):
    err = rel_l2(out, c.reference())
    print(f"{label}: rel-L2 {err:.2e}")
    assert not out.isnan().any()
    assert err < 4e-3  # bf16 output alone gives ~6e-4; P is also rounded to bf16


@pytest.mark.parametrize("build", ["native", "fast_math"])
@pytest.mark.parametrize("ckv_scale,kpe_scale", [(1.0, 1.0), (0.37, 3.1)])
def test_dequantize_bitwise(build, ckv_scale, kpe_scale):
    c = RandomCache([100, 37], page_size=16, heads=16, ckv_scale=ckv_scale, kpe_scale=kpe_scale)
    out = torch.cat(nvfp4_mla._load(*BUILDS[build]).dequantize(
        c.ckv_cache, c.ckv_sf_cache, c.kpe_cache, ckv_scale, kpe_scale, False), -1)
    want = torch.cat(c.dequantize(), -1)
    assert torch.equal(out.isnan(), want.isnan())
    ok = ~want.isnan()
    assert torch.equal(out[ok].view(torch.int32), want[ok].view(torch.int32))


@pytest.mark.parametrize("num_splits", [0, 1, 3, 7])  # 0 = heuristic; splits cross page boundaries, some are empty
@pytest.mark.parametrize("kv_lens,page_size,heads", CASES)
def test_decode(num_splits, kv_lens, page_size, heads):
    c = RandomCache(kv_lens, page_size, heads, ckv_scale=0.5, kpe_scale=2.0)
    check(run(c, splits=num_splits), c, f"splits={num_splits} {kv_lens} ps={page_size} H={heads}")


@pytest.mark.parametrize("stages", [2, 4, 6])
@pytest.mark.parametrize("kv_lens,page_size,heads", CASES[1:3])
def test_stages(stages, kv_lens, page_size, heads):
    c = RandomCache(kv_lens, page_size, heads, ckv_scale=0.5, kpe_scale=2.0)
    check(run(c, stages=stages), c, f"stages={stages} {kv_lens} ps={page_size} H={heads}")


@pytest.mark.parametrize("kv_lens,page_size,heads", CASES)
def test_fast_math(kv_lens, page_size, heads):
    c = RandomCache(kv_lens, page_size, heads, ckv_scale=0.5, kpe_scale=2.0)
    check(run(c, "fast_math"), c, f"fast_math {kv_lens} ps={page_size} H={heads}")


@pytest.mark.parametrize("kv_lens,page_size,heads", CASES[1:3] + CASES[4:5])
def test_compat_cvt_matches_native(kv_lens, page_size, heads):
    """The pre-CUDA-13.2 conversion path (through f16) is exact, so the output is bitwise identical."""
    c = RandomCache(kv_lens, page_size, heads, ckv_scale=0.5, kpe_scale=2.0)
    assert torch.equal(run(c, "compat_cvt"), run(c))


def test_public_api():
    c = RandomCache([300, 70], page_size=16, heads=16, ckv_scale=0.5, kpe_scale=2.0)
    check(nvfp4_mla.decode(*c.args()), c, "nvfp4_mla.decode")
    got = torch.cat(nvfp4_mla.dequantize(c.ckv_cache, c.ckv_sf_cache, c.kpe_cache, 0.5, 2.0), -1)
    want = torch.cat(c.dequantize(), -1)
    ok = ~want.isnan()
    assert got.dtype == torch.bfloat16 and torch.equal(got[ok], want[ok].bfloat16())


def test_rejects_bad_inputs():
    c = RandomCache([64], page_size=16, heads=16)
    a = list(c.args())
    with pytest.raises(RuntimeError, match="bfloat16"):
        nvfp4_mla.decode(c.q_nope.half(), *a[1:])
    with pytest.raises(RuntimeError, match="multiple of 16"):
        nvfp4_mla.decode(c.q_nope[:, :8].contiguous(), c.q_pe[:, :8].contiguous(), *a[2:])
    with pytest.raises(RuntimeError, match="int32"):
        nvfp4_mla.decode(*a[:5], c.page_table.long(), *a[6:])
    with pytest.raises(RuntimeError, match="float8_e4m3fn"):
        nvfp4_mla.decode(*a[:3], c.ckv_sf_cache.view(torch.uint8), *a[4:])


def test_reference_vs_fa2():
    """The fp32 reference agrees with an independent kernel on the same (dequantized) data."""
    c = RandomCache([1000, 4097, 3, 64], page_size=64, heads=128)
    assert rel_l2(fa2_on_dequantized(c)(), c.reference()) < 4e-3
