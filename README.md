# NVFP4 Dense MLA Decode kernel

MLA decode attention (DeepSeek-style, weight-absorbed) that reads the **FlashInfer NVFP4 paged KV cache** directly.
I added this cache format and its quantize/append kernel to FlashInfer in
[flashinfer#4676](https://github.com/flashinfer-ai/flashinfer/pull/4676); this repository is the decode side.
The latent KV is stored as E2M1 with one E4M3 scale per 16 channels and the RoPE part as FP8, so one token is
352 bytes (FP8: 576, BF16: 1152). The kernel dequantizes in registers and keeps Q, P and the MMA in BF16 with FP32
accumulation.

It is written from scratch in CUDA (`mma.sync`, `cp.async`, split-KV) and uses one code path for both
**SM120 (RTX PRO 6000)** and **SM100 (B200)**.

Status: dense decode, one query token per request (see [Roadmap](#roadmap)). How the kernel got from a CUDA-core
version to 85% of DRAM bandwidth, step by step with the profiles behind each change:
[docs/optimization-log.md](docs/optimization-log.md).

## Why

FlashInfer can write an MLA KV cache in NVFP4 (#4676). Who can read it back for dense decode, as of 2026-11-10
(FlashInfer v0.7.3 and v0.7.4rc1, vLLM v0.33.0, SGLang v0.5.23, TensorRT-LLM v1.4.0rc3):

- **SM12x** (RTX PRO 6000 and RTX 50 series are SM120, DGX Spark is SM121): FlashInfer, vLLM and SGLang have no
  dense NVFP4 MLA decode. vLLM's dense MLA backends take BF16 or FP8 KV only, and its NVFP4 MLA cache
  (`nvfp4_ds_mla`) is for sparse MLA on SM100 only. SGLang's FP4 MLA cache uses its own layout and is dequantized to
  BF16 before attention. TensorRT-LLM's `fp4_mla` (FP4 query, KV and P, its own cache layout) is validated on SM100
  and SM107 only.
- **SM100:** FlashInfer v0.7.3 added `cake_mla_nvfp4_paged_decode` (experimental, needs CUDA 13.4), which reads
  this cache with `tcgen05` MMA and an NVFP4 query. It is faster than this kernel on B200 (see
  [Performance](#performance)). An open vLLM PR ([#63203](https://github.com/vllm-project/vllm/pull/63203)) uses it
  for `--kv-cache-dtype nvfp4` on dense MLA models.
- FlashInfer's generic MLA decode (`trtllm_batch_decode_with_kv_cache_mla`) still raises for this cache.

This kernel reads the cache on SM120 and SM100 and keeps the query in BF16.

## Requirements

- NVIDIA Blackwell GPU: SM120 (RTX PRO 6000) or SM100 (B200). Other Blackwell parts (SM103, SM121) are not tested.
- CUDA toolkit 12.8 or newer, with `nvcc` on `PATH`. With 13.2 or newer the kernel uses the native FP4/FP8 to BF16
  conversion instructions; older toolkits convert through FP16, which gives the same values with a few more
  instructions.
- PyTorch 2.7 or newer for the same CUDA major version, and `ninja`.
- `flashinfer-python >= 0.7.0` to write the cache (and for the tests and benchmarks).

Tested with CUDA 13.4.2, PyTorch 2.12.0, Python 3.13 and FlashInfer 0.7.4rc1 on RTX PRO 6000 Blackwell Server
Edition and B200; the benchmarks below use this setup. The tests also pass with CUDA 13.3 and FlashInfer 0.7.0 and
0.7.3.

## Install

```bash
git clone https://github.com/elwhyjay/nvfp4-mla.git
cd nvfp4-mla
pip install -e .            # in the same environment as SGLang / vLLM / FlashInfer
```

The CUDA extension is compiled on the first call (about one minute) and cached in `~/.cache/torch_extensions`.
Set `NVFP4_MLA_VERBOSE=1` to see the build output.

## Usage

```python
import flashinfer
import nvfp4_mla

# NVFP4 MLA paged cache, written by FlashInfer
ckv_cache = torch.empty(num_pages, page_size, 256, dtype=torch.uint8, device="cuda")
ckv_sf_cache = torch.empty(num_pages, page_size, 32, dtype=torch.float8_e4m3fn, device="cuda")
kpe_cache = torch.empty(num_pages, page_size, 64, dtype=torch.float8_e4m3fn, device="cuda")
flashinfer.nvfp4_quantize_append_paged_mla_kv_cache(
    ckv, kpe, batch_indices, positions, ckv_cache, ckv_sf_cache, kpe_cache,
    kv_indices, kv_indptr, kv_last_page_len, ckv_scale, kpe_scale)

out = nvfp4_mla.decode(
    q_nope, q_pe,                        # [B, H, 512], [B, H, 64] bfloat16 (absorbed query)
    ckv_cache, ckv_sf_cache, kpe_cache,
    page_table, kv_lens,                 # [B, max_pages] int32 page ids, [B] int32 lengths
    sm_scale, ckv_scale, kpe_scale,
    max_kv_len=max_context_len)          # an upper bound: no host sync, fixed launch shape for CUDA graphs
# out: [B, H, 512] bfloat16 (latent output; apply W_UV outside)
```

`page_table` is the same block table as `block_tables` in FlashInfer's `MLAPlanMetadata.dense`.
`nvfp4_mla.dequantize(ckv_cache, ckv_sf_cache, kpe_cache, ckv_scale, kpe_scale)` returns the cache in BF16 or FP32,
for example for a prefill kernel.

Current limits: one query token per request (no MTP verify yet), heads a multiple of 16, page size a power of two,
head dims 512 + 64 (DeepSeek-V2/V3/R1, Kimi-K2).

## Accuracy

- Kernel vs an FP32 PyTorch reference on the same dequantized cache: relative L2 error at most 1.9e-3, from rounding
  the output and P to BF16.
- The in-kernel dequantization is bitwise equal to the PyTorch reference, also with `--use_fast_math`.
- Quantization error of the format itself (random data, NVFP4 cache vs BF16 cache): relative L2 0.08-0.12, cosine
  about 0.995. For comparison, FP8 KV with FP8 Q (Triton FP8, XQA) has cosine 0.998 against BF16.
- FlashInfer Cake also quantizes the query to NVFP4. On the same caches its relative L2 error against BF16 is
  0.13-0.17, 1.4-1.7x the error of this kernel (`benchmarks/accuracy.py` on B200, three shapes).

## Performance

GPU kernel time per call in ms, measured with CUPTI after an L2 flush, so host launch time is not included.
Page size 64, every request has the same length, median of 20 calls. All providers run in the same process on the same
random data. Full sweeps (H 16/128, batch 1-64, 1K-64K tokens): [rtx_pro_6000.csv](benchmarks/results/rtx_pro_6000.csv),
[b200.csv](benchmarks/results/b200.csv).

Baselines: FlashInfer `BatchMLAPagedAttentionWrapper` (fa2, BF16 KV); FlashInfer
`trtllm_batch_decode_with_kv_cache_mla` (XQA on SM120, trtllm-gen on SM100); FlashInfer `cake_mla_nvfp4_paged_decode`
(SM100, same NVFP4 cache, NVFP4 query); SGLang's Triton MLA decode (the vLLM/SGLang default on SM120, vendored in
`benchmarks/triton_mla.py`); and "dequantize + fa2" (`nvfp4_mla.dequantize` to BF16, then fa2).

**RTX PRO 6000 (SM120)**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/compare-rtx-pro-6000-dark.svg">
  <img alt="Latency of nvfp4-mla and other MLA decode kernels on RTX PRO 6000 for four shapes" src="docs/img/compare-rtx-pro-6000-light.svg">
</picture>

| heads | batch x kv_len | nvfp4 | Triton FP8 | fa2 BF16 | dequantize + fa2 | XQA FP8 |
|---|---|---|---|---|---|---|
| 16 | 1 x 4K | **0.018** | 0.045 | 0.036 | 0.038 | n/a (needs 128 heads) |
| 16 | 1 x 64K | **0.043** | 0.115 | 0.127 | 0.180 | n/a |
| 16 | 16 x 64K | **0.330** | 0.574 | 1.507 | 2.603 | n/a |
| 16 | 64 x 64K | **1.119** | 1.850 | 6.780 | 11.23 | n/a |
| 128 | 1 x 64K | **0.103** | 0.225 | 0.161 | 0.213 | 1.113 |
| 128 | 16 x 64K | 1.336 | 2.198 | 3.168 | 4.272 | **1.127** |
| 128 | 64 x 64K | 4.995 | 8.387 | 10.04 | 14.39 | **2.360** |

**B200 (SM100)**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/compare-b200-dark.svg">
  <img alt="Latency of nvfp4-mla and other MLA decode kernels on B200 for four shapes" src="docs/img/compare-b200-light.svg">
</picture>

| heads | batch x kv_len | nvfp4 | Cake NVFP4 | trtllm-gen FP8 | trtllm-gen BF16 | fa2 BF16 | Triton FP8 |
|---|---|---|---|---|---|---|---|
| 16 | 1 x 4K | **0.012** | 0.013 | 0.013 | 0.017 | 0.026 | 0.051 |
| 16 | 1 x 64K | 0.031 | **0.020** | 0.030 | 0.044 | 0.083 | 0.113 |
| 16 | 16 x 64K | 0.197 | **0.088** | 0.118 | 0.237 | 0.562 | 0.532 |
| 16 | 64 x 64K | 0.741 | **0.318** | 0.386 | 0.844 | 2.473 | 1.908 |
| 128 | 1 x 64K | 0.102 | **0.033** | 0.037 | 0.046 | 0.099 | 0.298 |
| 128 | 16 x 64K | 1.454 | 0.300 | **0.148** | 0.256 | 1.522 | 3.515 |
| 128 | 64 x 64K | 5.761 | **0.548** | 0.587 | 1.061 | 4.928 | 14.56 |

How to read these numbers:

- **SM120, 16 heads** (memory bound): fastest in all 16 measured shapes (batch 1-64, 1K-64K tokens): 1.6-3.4x
  faster than Triton FP8 and 1.5-6.1x faster than fa2. At 64 x 64K it reads 1.32 TB/s, 85% of the achievable
  1.56 TB/s (`benchmarks/roofline.py`). Triton FP8 reaches the same fraction there, so the 1.65x gain is the byte
  ratio 576 / 352.
- **SM120, 128 heads**: fastest at batch 1-4, where XQA has no KV split and is slow with long contexts. From batch 16
  the problem is compute bound and XQA's FP8 MMA wins (2.1x at 64 x 64K). The BF16 `mma.sync` floor for that shape
  is 2.57 ms, already above XQA, so closing this gap needs lower-precision MMA (see [Roadmap](#roadmap)). It is
  faster than Triton FP8 and fa2 in all shapes.
- **B200, 16 heads**: FlashInfer Cake (`tcgen05` MMA) is the fastest in most shapes, up to 2.3x faster than this
  kernel; this kernel is faster only at batch 1-4 with up to 4K tokens. It is faster than trtllm-gen BF16 in all
  16 shapes (1.03-1.47x), than fa2 (1.6-3.3x) and than Triton FP8 (2.5-5.1x). This kernel uses `mma.sync` on both
  GPUs; on B200 it is limited by latency within the SM, not by memory.
- **B200, 128 heads**: compute bound, and `mma.sync` is far below `tcgen05`: Cake and trtllm-gen FP8 are up to
  10-11x faster.
- Cake's query is NVFP4, so its error against the BF16 cache is 1.4-1.7x higher (see [Accuracy](#accuracy)).

## Optimization history

Latency of each kernel version on the same shapes. v1 runs on CUDA cores; v2 moves to tensor cores with
dequantization in registers; v3 adds the `cp.async` pipeline; v4 splits the KV range across CTAs; v5 cuts
instructions per token. Each step, and the profile that motivated it, is in
[docs/optimization-log.md](docs/optimization-log.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/versions-dark.svg">
  <img alt="Latency per kernel version, v1 to v5, for three shapes on RTX PRO 6000 and B200" src="docs/img/versions-light.svg">
</picture>

The same versions on a roofline (batch 64, 64K tokens). At 16 heads RTX PRO 6000 reaches 85% of the DRAM roof; at
128 heads both GPUs sit at about a third to a half of the BF16 `mma.sync` roof, which is the next target.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/roofline-dark.svg">
  <img alt="Roofline of RTX PRO 6000 and B200 with kernel versions v1 to v5 and the fastest other library" src="docs/img/roofline-light.svg">
</picture>

## Reproduce

```bash
pip install -e ".[test,bench]"
pytest tests                       # correctness; the first run builds the extension three ways (~3 min)
python benchmarks/roofline.py      # achievable read BW and BF16 mma.sync peak of this GPU
python benchmarks/baselines.py     # sanity check: every baseline agrees with FlashInfer fa2
python benchmarks/accuracy.py      # error against the dequantized and the BF16 reference
python benchmarks/bench.py         # full sweep; --batch / --s-kv / --heads / --providers to narrow it
uv run --with matplotlib python benchmarks/plot.py   # redraw docs/img from benchmarks/results
```

The largest sweep points (64 x 64K, with BF16 copies for the baselines) need a lot of GPU memory; narrow the sweep on
smaller GPUs.

The `cake` provider needs SM100, CUDA 13.4 and `flashinfer-python>=0.7.4rc1`. The PyPI wheel does not ship Cake's
CUDA sources ([flashinfer#7977](https://github.com/flashinfer-ai/flashinfer/issues/7977)); copy
`flashinfer/experimental/` from the source distribution of the same version into the installed package.

## How it works

- One CTA handles 16 heads of one request and one KV split, with 4 warps. Each warp owns 128 of the 512 latent
  channels, and each thread owns 32 channels, which are exactly two FP4 scale blocks.
- FP4 to BF16 happens in registers (`cvt.rn.bf16x2.e2m1x2`), and the E4M3 block scale is multiplied on the MMA
  operand. E2M1 x E4M3 fits in BF16 exactly, so the MMA sees exact values.
- The QK reduction axis is permuted so that each thread's Q fragments stay in registers for the whole kernel.
- The dequantized QK B fragments are transposed with `movmatrix` and reused as PV B fragments, so every token is
  dequantized once.
- Partial scores of the 4 warps are summed through shared memory behind a split arrive/wait `mbarrier`; the
  transposes run while a warp waits.
- KV tiles of 32 tokens stream in through a 3-stage `cp.async` pipeline into XOR-swizzled shared memory
  (no bank conflicts). Online softmax rescales lazily (only when the running max grows by more than 2^8).
- Long contexts are split across CTAs. The split size is the smallest `waves x (chunk + C0)` with chunk >= sqrt(L),
  and a separate kernel merges the partial results.

## Cache layout (flashinfer#4676)

| tensor | shape | dtype | content |
|---|---|---|---|
| `ckv_cache` | `[pages, page_size, 256]` | uint8 | 512 latent channels as E2M1, two per byte, low nibble = even channel |
| `ckv_sf_cache` | `[pages, page_size, 32]` | float8_e4m3fn | one scale per 16 latent channels |
| `kpe_cache` | `[pages, page_size, 64]` | float8_e4m3fn | 64 RoPE channels |

Dequantization: `ckv = e2m1 * scale * ckv_scale`, `kpe = fp8 * kpe_scale`.

## Roadmap

- Compute-bound shapes on SM120 (128 heads, and MTP below): block-scaled FP4 `mma.sync` for QK, where the cache's
  scales match the MMA's operand scales, and more warps per scheduler to keep the tensor pipe busy.
- A `tcgen05` path for SM100 that keeps the BF16 query; the target is FlashInfer Cake at 16 heads.
- MTP / speculative decode (2-4 query tokens per request).
- An SGLang integration (attention backend plugin plus a KV pool in this layout).

## Acknowledgements

- The cache format and its quantize/append kernel are from my FlashInfer PR
  [#4676](https://github.com/flashinfer-ai/flashinfer/pull/4676), which FlashInfer's `cake_mla_nvfp4_paged_decode`
  (SM100) also reads.
- `benchmarks/triton_mla.py` is vendored from [SGLang](https://github.com/sgl-project/sglang) (commit c311bc961164,
  Apache-2.0) to benchmark against the Triton MLA decode kernel. SGLang has not changed these kernels for NVIDIA GPUs
  since then (checked against v0.5.23, 2026-11-10).

## License

Apache-2.0, see [LICENSE](LICENSE).
