"""The SM100 tcgen05 kernel (nvfp4_mla.sm100): pytest tests/test_sm100.py (needs SM100 and nvidia-cutlass-dsl; each page
size compiles the kernel once with CuTe DSL)."""
import pytest
import torch

pytest.importorskip("flashinfer")
pytest.importorskip("cutlass")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("needs an SM100 GPU", allow_module_level=True)

import nvfp4_mla  # noqa: E402
from nvfp4_mla import sm100  # noqa: E402
from nvfp4_mla.testing import RandomCache, rel_l2  # noqa: E402

# block scales near the E4M3 maximum, as with a global scale calibrated to amax / (6 x 448); the RoPE bytes near 448
CALIBRATED = (2.0 ** -9, 2.0 ** -6)

CASES = [  # (kv_lens, page_size, heads, num_splits)
    ([200], 64, 16, 1),
    ([128, 1000, 77], 16, 32, 1),
    ([4096] * 4, 64, 16, 0),
    ([5000, 300], 32, 128, 3),
    ([16384] * 8, 64, 128, 0),
    ([1, 129, 300], 8, 16, 1),          # page size < 16: the cp.async load
]


def check(c, out, page_size, unquantized=True):
    """The default path is v3 from page size 16 on: V and P in E4M3, so against the dequantized cache only to that
    rounding, and against the unquantized KV within 3% of the CUDA kernel's BF16 mode. v2.4 below that: exact V."""
    err = rel_l2(out, c.reference())
    assert not out.isnan().any()
    if page_size < 16:
        assert err < 3e-3
        return err
    assert err < 0.04
    if unquantized:
        ref16 = c.reference_bf16()
        assert rel_l2(out, ref16) <= 1.03 * rel_l2(nvfp4_mla.decode(*c.args()), ref16)
    return err


@pytest.mark.parametrize("kv_lens,page_size,heads,splits", CASES)
@pytest.mark.parametrize("scales", [(0.5, 2.0), CALIBRATED])
def test_decode(kv_lens, page_size, heads, splits, scales):
    c = RandomCache(kv_lens, page_size, heads, *scales)
    err = check(c, sm100.decode(*c.args(), num_splits=splits), page_size)
    print(f"{kv_lens} ps={page_size} H={heads}: rel-L2 vs dequantized {err:.2e}")


@pytest.mark.parametrize("kv_lens,page_size,heads,splits", [c for c in CASES if c[1] >= 16])
@pytest.mark.parametrize("scales", [(0.5, 2.0), CALIBRATED])
def test_v2(kv_lens, page_size, heads, splits, scales):
    """v2.4 (DecodeV2) where v3 is the default: exact V, so close to the dequantized cache."""
    c = RandomCache(kv_lens, page_size, heads, *scales)
    out = sm100.decode(*c.args(), num_splits=splits, version=3)
    assert not out.isnan().any()
    assert rel_l2(out, c.reference()) < 3e-3


@pytest.mark.parametrize("page_size", [1, 16, 64, 128])
def test_lengths(page_size):
    """Every length around the 128-token tile and the stage ring, two requests per call (the tiling, so against
    the dequantized cache only)."""
    worst = 0.0
    for L in (1, 2, 63, 64, 65, 127, 128, 129, 255, 300, 513):
        c = RandomCache([L, 3 * L + 5], page_size, 16, seed=L)
        worst = max(worst, check(c, sm100.decode(*c.args()), page_size, unquantized=False))
    print(f"ps={page_size}: worst rel-L2 vs dequantized {worst:.2e}")


@pytest.mark.parametrize("scales", [(1.0, 1.0), CALIBRATED])
@pytest.mark.parametrize("heads", [16, 128])
def test_against_unquantized(heads, scales):
    """Against attention over the unquantized BF16 KV, the error the user sees: within 3% of the CUDA kernel's BF16
    mode on the same cache (both are then limited by the format)."""
    c = RandomCache([4096] * 4, 64, heads, *scales)
    ref = c.reference_bf16()
    e_tc = rel_l2(sm100.decode(*c.args()), ref)
    e_cuda = rel_l2(nvfp4_mla.decode(*c.args()), ref)
    print(f"H={heads} scales={scales}: vs unquantized sm100 {e_tc:.4f}, CUDA bf16 mode {e_cuda:.4f}")
    assert e_tc <= 1.03 * e_cuda
