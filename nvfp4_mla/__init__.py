"""Dense MLA decode attention that reads the FlashInfer NVFP4 paged KV cache (flashinfer#4676) directly.

The CUDA extension is compiled on first use for the current GPU (Blackwell: SM100 / SM120) and cached by PyTorch
under ``~/.cache/torch_extensions``.
"""
import functools
import hashlib
import os
import pathlib

import torch

__all__ = ["decode", "dequantize"]
__version__ = "0.1.0"

_SRC = pathlib.Path(__file__).parent / "csrc" / "mla_decode.cu"


@functools.lru_cache(maxsize=None)
def _load(*extra_cuda_cflags: str):
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    if major not in (10, 12):
        raise RuntimeError(f"nvfp4_mla needs a Blackwell GPU (SM100 / SM120), found SM{major}{minor}")
    # The e2m1/e4m3 -> 16-bit cvt instructions are arch-specific (sm_100a / sm_120a) features.
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}a"
    # Source hash in the name: a checkout whose files keep old mtimes would otherwise reuse a stale build.
    tag = hashlib.sha1(_SRC.read_bytes() + repr(extra_cuda_cflags).encode()).hexdigest()[:10]
    try:
        return load(
            f"nvfp4_mla_sm{major}{minor}_{tag}", [str(_SRC)],
            extra_cuda_cflags=["-O3", "-lineinfo", *extra_cuda_cflags],
            verbose=os.environ.get("NVFP4_MLA_VERBOSE") == "1",
        )
    finally:
        if arch_list is None:
            del os.environ["TORCH_CUDA_ARCH_LIST"]
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list


def decode(q_nope, q_pe, ckv_cache, ckv_sf_cache, kpe_cache, page_table, kv_lens, sm_scale,
           ckv_scale=1.0, kpe_scale=1.0, max_kv_len=None, num_splits=0):
    """Absorbed-MLA decode (one query token per request) on an NVFP4 paged KV cache.

    Args:
        q_nope: ``[batch, heads, 512]`` bfloat16, the absorbed no-RoPE query. ``heads`` must be a multiple of 16.
        q_pe: ``[batch, heads, 64]`` bfloat16, the RoPE query.
        ckv_cache: ``[pages, page_size, 256]`` uint8, packed E2M1 (low nibble = even channel).
        ckv_sf_cache: ``[pages, page_size, 32]`` float8_e4m3fn, one scale per 16 channels.
        kpe_cache: ``[pages, page_size, 64]`` float8_e4m3fn.
            The three caches are written by ``flashinfer.nvfp4_quantize_append_paged_mla_kv_cache``.
            ``page_size`` must be a power of two.
        page_table: ``[batch, max_pages]`` int32, page ids of each request.
        kv_lens: ``[batch]`` int32, tokens per request (slots past it are never read).
        sm_scale: softmax scale applied to ``q . k``.
        ckv_scale, kpe_scale: the global decode scales used when the cache was written.
        max_kv_len: an upper bound of ``kv_lens``, used to size the KV split. Pass it to avoid a device sync and
            to keep the launch shape fixed under CUDA graphs. Default: ``kv_lens.max()``.
        num_splits: KV splits per request; 0 picks it from the GPU size, batch and ``max_kv_len``.

    Returns:
        ``[batch, heads, 512]`` bfloat16 attention output (the latent value; apply W_UV outside).
    """
    if max_kv_len is None:
        max_kv_len = int(kv_lens.max())
    return _load().decode(q_nope, q_pe, ckv_cache, ckv_sf_cache, kpe_cache, page_table, kv_lens, sm_scale,
                          ckv_scale, kpe_scale, max_kv_len, num_splits, 3, 0)


def dequantize(ckv_cache, ckv_sf_cache, kpe_cache, ckv_scale=1.0, kpe_scale=1.0, dtype=torch.bfloat16):
    """Dequantize a whole NVFP4 MLA cache to ``(ckv [pages, page_size, 512], kpe [pages, page_size, 64])``.

    ``dtype`` is bfloat16 or float32. BF16 is exact when ``ckv_scale`` and ``kpe_scale`` are powers of two.
    """
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"dtype must be bfloat16 or float32, got {dtype}")
    ckv, kpe = _load().dequantize(ckv_cache, ckv_sf_cache, kpe_cache, ckv_scale, kpe_scale, dtype == torch.bfloat16)
    return ckv, kpe
