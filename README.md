# NVFP4 Dense MLA Decode kernel

MLA decode attention (DeepSeek-style, weight-absorbed) that reads the **FlashInfer NVFP4 paged KV cache** directly.
I added this cache format and its quantize/append kernel to FlashInfer in
[flashinfer#4676](https://github.com/flashinfer-ai/flashinfer/pull/4676); this repository is the decode side.
The latent KV is stored as E2M1 with one E4M3 scale per 16 channels and the RoPE part as FP8, so one token is
352 bytes (FP8: 576, BF16: 1152). By default the kernel dequantizes in registers and keeps Q, P and the MMA in BF16
with FP32 accumulation; an FP8 mode (E4M3 Q, K, V, P and MMA) is up to 1.7x faster on compute-bound shapes.

There are two kernels, both written from scratch: a CUDA kernel (`mma.sync`, `cp.async`, split-KV) for
**SM120 (RTX PRO 6000)**, which also runs on SM100, and a warp-specialized `tcgen05` kernel in CuTe DSL for
**SM100 (B200)** (`nvfp4_mla.sm100`).

Status: dense decode, one query token per request (see [Roadmap](#roadmap)). How both kernels were built, step by
step with the hypothesis, the profile and the result of each change (including the ones that did not work): the CUDA
kernel from a CUDA-core version to 85% of DRAM bandwidth and an FP8 mode for compute-bound shapes, and the SM100
kernel from a serial `tcgen05` version to TMA and warp specialization:
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
  this cache with `tcgen05` MMA and an NVFP4 query. It is still faster than this repository's SM100 kernel on B200
  (1.4x at 16 heads, batch 64 and 64K tokens; more with 128 heads; see [Performance](#performance)). An open vLLM
  PR ([#63203](https://github.com/vllm-project/vllm/pull/63203)) uses it for `--kv-cache-dtype nvfp4` on dense MLA
  models.
- FlashInfer's generic MLA decode (`trtllm_batch_decode_with_kv_cache_mla`) still raises for this cache.

This kernel reads the cache on SM120 and SM100 and keeps the query in BF16 by default; its FP8 mode trades a few
percent of accuracy for speed on compute-bound shapes.

## Requirements

- NVIDIA Blackwell GPU: SM120 (RTX PRO 6000) or SM100 (B200). Other Blackwell parts (SM103, SM121) are not tested.
- CUDA toolkit 12.8 or newer, with `nvcc` on `PATH` (`precision="fp4-qk"` needs 12.9). With 13.2 or newer the
  kernel uses the native FP4/FP8 to BF16 conversion instructions; older toolkits convert through FP16, which gives the
  same values with a few more instructions.
- PyTorch 2.7 or newer for the same CUDA major version, and `ninja`.
- `flashinfer-python >= 0.7.0` to write the cache (and for the tests and benchmarks).
- For the SM100 kernel: `nvidia-cutlass-dsl[cu13] >= 4.8.1` (`pip install -e ".[sm100]"`).

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
    max_kv_len=max_context_len,          # an upper bound: no host sync, fixed launch shape for CUDA graphs
    precision="bf16")                    # or "fp8": faster with 128 heads and batch >= 4 (see Performance)
# out: [B, H, 512] bfloat16 (latent output; apply W_UV outside)
```

`precision` sets the arithmetic of the attention, not the cache format:

| `precision` | Q, P and MMA | when to use it | extra error over `"bf16"` |
|---|---|---|---|
| `"bf16"` (default) | BF16 | 16 heads, small batches, and on B200 | |
| `"fp8"` | E4M3 (Q quantized per row in the kernel; K and V rounded to E4M3) | 128 heads, batch 4 and up, SM120: 1.3-1.7x faster | 0-11% |
| `"fp4-qk"` | as `"fp8"`, QK in block-scaled FP4 (FP4 query, as FlashInfer Cake) | 128 heads, large batch, SM120: about 3% faster than `"fp8"` | 25-55% |

On SM100, `nvfp4_mla.sm100.decode` takes the same arguments (without `precision`) and returns the same output:

```python
from nvfp4_mla import sm100

out = sm100.decode(q_nope, q_pe, ckv_cache, ckv_sf_cache, kpe_cache, page_table, kv_lens,
                   sm_scale, ckv_scale, kpe_scale, max_kv_len=max_context_len)
```

On B200 it is faster than `nvfp4_mla.decode` from about 16 requests x 16K tokens (up to 1.6x with 16 heads and
1.8x with 128 heads). Below that, its fixed cost of about 40 µs per call makes it slower (see
[Performance](#performance)). The first call for each page size compiles it with CuTe DSL.

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
- `precision="fp8"`: 0.02-0.06 against the dequantized cache (E4M3 rounding of Q, K, V and P); against the BF16 cache
  0-11% more error than the default, because the cache format dominates. `precision="fp4-qk"`: 25-55% more.
  [`benchmarks/precision.py`](benchmarks/precision.py) simulates the options on the CPU.
- SM100 kernel: 1.6e-3 to 1.7e-3 against the same reference. Its QK runs on the cache's FP4 bytes with block-scaled
  FP4 MMA, but the BF16 query is not rounded to one NVFP4 value: it is written as a sum of 4 NVFP4 terms with one
  shared scale per head (rest 5.5e-5), and the RoPE part as 3 E4M3 terms (rest 1.2e-5).
- FlashInfer Cake also quantizes the query to NVFP4. On the same caches its relative L2 error against BF16 is
  0.13-0.17, 1.4-1.7x the error of this kernel's default (`benchmarks/accuracy.py` on B200, three shapes).

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

| heads | batch x kv_len | nvfp4 bf16 | nvfp4 fp8 | nvfp4 fp4-qk | Triton FP8 | fa2 BF16 | dequantize + fa2 | XQA FP8 |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 x 4K | **0.018** | 0.019 | 0.021 | 0.045 | 0.036 | 0.038 | n/a (needs 128 heads) |
| 16 | 1 x 64K | **0.046** | 0.052 | 0.052 | 0.116 | 0.128 | 0.179 | n/a |
| 16 | 16 x 64K | 0.332 | **0.324** | 0.325 | 0.574 | 1.518 | 2.613 | n/a |
| 16 | 64 x 64K | 1.121 | **1.105** | **1.105** | 1.852 | 6.790 | 11.23 | n/a |
| 128 | 1 x 64K | 0.105 | **0.080** | **0.080** | 0.226 | 0.164 | 0.213 | 1.110 |
| 128 | 16 x 64K | 1.353 | **0.778** | 0.797 | 2.195 | 3.180 | 4.264 | 1.120 |
| 128 | 64 x 64K | 4.815 | 2.836 | 2.762 | 8.156 | 9.750 | 14.16 | **2.306** |

**B200 (SM100)**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/compare-b200-dark.svg">
  <img alt="Latency of nvfp4-mla and other MLA decode kernels on B200 for four shapes" src="docs/img/compare-b200-light.svg">
</picture>

| heads | batch x kv_len | nvfp4 CUDA | nvfp4 tcgen05 | Cake NVFP4 | trtllm-gen FP8 | trtllm-gen BF16 | fa2 BF16 | Triton FP8 |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 x 4K | **0.012** | 0.037 | 0.013 | **0.012** | 0.016 | 0.025 | 0.049 |
| 16 | 1 x 64K | 0.030 | 0.049 | **0.021** | 0.031 | 0.044 | 0.079 | 0.111 |
| 16 | 16 x 64K | 0.202 | 0.130 | **0.087** | 0.116 | 0.238 | 0.560 | 0.532 |
| 16 | 64 x 64K | 0.748 | 0.456 | **0.320** | 0.385 | 0.844 | 2.487 | 1.907 |
| 128 | 1 x 64K | 0.105 | 0.082 | **0.034** | 0.037 | 0.045 | 0.097 | 0.295 |
| 128 | 16 x 64K | 1.512 | 0.851 | 0.301 | **0.149** | 0.253 | 1.515 | 3.515 |
| 128 | 64 x 64K | 5.875 | 3.383 | **0.541** | 0.566 | 1.002 | 5.058 | 14.53 |

How to read these numbers:

- **SM120, 16 heads** (memory bound): the default mode is the fastest in all 16 measured shapes (batch 1-64, 1K-64K
  tokens): 1.6-3.3x faster than Triton FP8 and 1.5-6.1x faster than fa2. At 64 x 64K it reads 1.32 TB/s, 85% of the
  achievable 1.56 TB/s (`benchmarks/roofline.py`). Triton FP8 reaches the same fraction there, so the 1.65x gain is
  the byte ratio 576 / 352. The FP8 mode changes little here (0.8-1.03x).
- **SM120, 128 heads** (not memory bound from batch 16): the FP8 mode is 1.3-1.7x faster than the default from batch 4.
  It is the fastest at batch 1-4 (XQA has no KV split and is slow with long contexts) and at batch 16 with 16K tokens
  or more, and faster than Triton FP8 (1.2-2.9x) and fa2 (1.1-4.1x) in all shapes. XQA's FP8 cache still wins at
  batch 64 (1.23x at 64 x 64K). This is not a hardware floor: ncu shows the kernel bound by latency (38.6% of issue
  slots, tensor pipe 49%). Test builds with wrong numbers bound each part: removing the FP4 -> FP8 conversion (46% of
  the instructions) saves 12%, removing both per-tile barriers 5%, and both together leave 2.44 ms, still above XQA.
  The rest is how the MMAs are issued, in bursts by 2 warps per scheduler, which the register budget does not let grow
  ([optimization log](docs/optimization-log.md#e3-and-e4-results-sm120-stops-at-v11c)).
- **B200, 16 heads**: FlashInfer Cake is the fastest in 11 of the 16 shapes. The SM100 kernel is 1.4x slower than
  Cake at 64 x 64K (it reads 3.2 TB/s, 44% of the achievable 7.3 TB/s; Cake 63%) and 1.5x slower at 16 x 64K. It is
  faster than the CUDA kernel from 16 x 16K tokens (up to 1.6x) and than trtllm-gen BF16 in the same shapes (up to
  1.85x). Below that it is the slower of the two kernels: a call costs at least about 40 µs with a cold L2 (24 µs
  warm: 18 µs for one tile through the kernel's start, pipeline and end, 7 µs for three small kernels that split the
  query and merge the KV splits). The CUDA kernel is the fastest of all at batch 1-4 with 1K tokens and at batch 1
  with 4K.
- **B200, 128 heads**: Cake is 1.5-6.3x and trtllm-gen FP8 2.1-7.3x faster than the SM100 kernel. One CTA of it
  handles 16 heads, so the cache is read 8 times per request; this is the next structural change (see
  [Roadmap](#roadmap)). It is faster than the CUDA kernel in 9 of the 16 shapes (up to 1.8x).
- Cake's query is NVFP4, so its error against the BF16 cache is 1.4-1.7x higher (see [Accuracy](#accuracy)).
- The CUDA kernel's FP8 mode does not help on B200 (1.5-1.7x slower than its default at 64 x 64K in an earlier run).

## Optimization history

Latency of each kernel version on the same shapes. v1 runs on CUDA cores; v2 moves to tensor cores with
dequantization in registers; v3 adds the `cp.async` pipeline; v4 splits the KV range across CTAs; v5 cuts
instructions per token. v6-v11c are the FP8 mode for compute-bound shapes: FP8 MMA (v6), QK split by token so the
scores need no cross-warp sum (v9), V loaded with a transposing 8-bit `ldmatrix` (v11), conflict-free shared-memory
layouts (v11c). Each step, the hypothesis and the profile behind it, and the steps that were measured and dropped
(v7, v8, a split-policy change, v12) are in [docs/optimization-log.md](docs/optimization-log.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/versions-dark.svg">
  <img alt="Latency per kernel version, v1 to v5 on both GPUs and v6 to v11c (FP8 mode) at 128 heads on RTX PRO 6000" src="docs/img/versions-light.svg">
</picture>

The same versions on a roofline (batch 64, 64K tokens). At 16 heads RTX PRO 6000 reaches 85% of the DRAM roof. At 128
heads the FP8 mode reaches 45% of the E4M3 `mma.sync` roof on RTX PRO 6000; there no unit is saturated (issue slots
39%, tensor pipe 49%): the kernel is bound by the latency of each warp's serial chain. On B200 the SM100 kernel
(v2.4) sits above the CUDA kernel's `mma.sync` stages: 44% of the DRAM roof at 16 heads.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/roofline-dark.svg">
  <img alt="Roofline of RTX PRO 6000 and B200 with kernel versions v1 to v5 and the fastest other library" src="docs/img/roofline-light.svg">
</picture>

Where the remaining gap to XQA at 128 heads comes from. The FP4 -> FP8 conversion is 46% of the FP8 mode's
instructions, so a test build replaced it with two logic ops per word (wrong results, same control flow). It removes
40% of the instructions but only 12% of the time at 64K tokens, and the build is still 1.09x slower than XQA: the
conversion is not the floor, the rest of each warp's chain is. At 1K tokens both builds are about 1.65x slower, a fixed
cost per CTA that the conversion does not explain either.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/sm120-gap-dark.svg">
  <img alt="RTX PRO 6000, 128 heads, batch 64: time of the FP8 mode and of a build without the conversion, relative to XQA, for 1K to 64K tokens" src="docs/img/sm120-gap-light.svg">
</picture>

The SM100 kernel, version by version on B200 (one process). v0 converts each FP4 tile to FP16 in shared memory and
runs one serial chain of `tcgen05` MMAs. v1a' runs QK on the raw FP4 bytes with block-scaled FP4 MMA (the BF16 query
as 4 NVFP4 terms). v1b converts V from the raw tile straight into tensor memory. v2.3 gives each step its own warps
(load, convert, MMA, softmax) and fixes the order of the load loop: stall samples per barrier showed that it
published every tile one release too late, so its own latency was in every tile's period. v2.4 loads the tiles by
TMA and converts V in 8 warps instead of 4. With 128 heads, v1a' and v1b are slower than v0 because they handle 16
heads per CTA instead of 32 (tensor memory), so they read the cache twice as often. The first warp-specialized
builds (v2-v2.2, 1.04-1.13 ms at 16 heads x 64 x 64K) are in the log only; the current code no longer builds them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/sm100-versions-dark.svg">
  <img alt="B200: latency of the SM100 tcgen05 kernel from v0 to v2.4 for three shapes, with FlashInfer Cake on the same shapes" src="docs/img/sm100-versions-light.svg">
</picture>

## Reproduce

```bash
pip install -e ".[test,bench]"
pytest tests                       # correctness; the first run builds the extension three ways (~3 min)
python benchmarks/roofline.py      # achievable read BW and BF16 mma.sync peak of this GPU
python benchmarks/baselines.py     # sanity check: every baseline agrees with FlashInfer fa2
python benchmarks/accuracy.py      # error against the dequantized and the BF16 reference
python benchmarks/bench.py         # full sweep; --batch / --s-kv / --heads / --providers to narrow it
                                   # (nvfp4-tc: the SM100 kernel, needs ".[sm100]")
uv run --with matplotlib python benchmarks/plot.py   # redraw docs/img from benchmarks/results
```

The largest sweep points (64 x 64K, with BF16 copies for the baselines) need a lot of GPU memory; narrow the sweep on
smaller GPUs.

The `cake` provider needs SM100, CUDA 13.4 and `flashinfer-python>=0.7.4rc1`. The PyPI wheel does not ship Cake's
CUDA sources ([flashinfer#7977](https://github.com/flashinfer-ai/flashinfer/issues/7977)); copy
`flashinfer/experimental/` from the source distribution of the same version into the installed package.

## How it works

Default (BF16) mode:

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

FP8 mode (`precision="fp8"`):

- E4M3 `mma.sync` m16n8k32 for QK and PV over 32-token tiles. Q is quantized once per row in the kernel and stays in
  registers.
- QK is split by token: each warp takes 8 tokens over all 576 channels, so the MMA itself sums the channels and the
  scores need no cross-warp reduction; each warp computes the softmax of its own tokens and the warps exchange one row
  maximum.
- Each thread converts its token's 128 latent channels (E2M1 to f16, times the block scale, then rounded to E4M3) once;
  they are QK's operand and go to an E4M3 V tile in shared memory. PV is split by channel and loads V with
  `ldmatrix.m16n16.trans.b8`, which returns exactly the transposed FP8 fragments.
- `fp4-qk` feeds QK with the cache bytes as stored, through block-scaled FP4 `mma.sync` (`kind::mxf4nvf4`), whose
  per-16 E4M3 scales match the cache's.

SM100 kernel (`nvfp4_mla.sm100`, CuTe DSL):

- `tcgen05` MMA with the operands swapped: tokens are M (128 per tile) and the CTA's 16 heads are N, so a 16-head
  query still fills the MMA. Scores, the output accumulator and V live in tensor memory.
- QK reads the cache's FP4 bytes as stored, with block-scaled FP4 MMA (`kind::mxf4nvf4`) and the cache's E4M3
  scales. The BF16 query is written as 4 NVFP4 terms with one shared scale per head, stacked along N, so K is read
  once for all terms. The RoPE part is E4M3 x E4M3 (`kind::f8f6f4`) on the raw RoPE bytes, with the query as 3 E4M3
  terms.
- V is converted from the raw FP4 tile straight into tensor memory (8-bit transposing `ldmatrix`, E2M1 to FP16,
  times the block scale) and is the A operand of the PV MMA; P comes from shared memory.
- 512 threads in roles: one warp issues TMA (one box per page) into a 4-stage ring of raw tiles, two warps lay out
  the block scales for the QK MMA and for the V conversion, 8 warps convert V, one warp issues the MMAs and 4 warps
  run the softmax. The roles hand tiles over through mbarriers; `setmaxnreg` moves registers from the MMA/TMA
  warpgroup to the softmax warps.
- Online softmax with lazy rescaling (as in the CUDA kernel) and two P buffers; V in two 64-token halves, so the
  conversion of the next tile overlaps PV.

## Cache layout (flashinfer#4676)

| tensor | shape | dtype | content |
|---|---|---|---|
| `ckv_cache` | `[pages, page_size, 256]` | uint8 | 512 latent channels as E2M1, two per byte, low nibble = even channel |
| `ckv_sf_cache` | `[pages, page_size, 32]` | float8_e4m3fn | one scale per 16 latent channels |
| `kpe_cache` | `[pages, page_size, 64]` | float8_e4m3fn | 64 RoPE channels |

Dequantization: `ckv = e2m1 * scale * ckv_scale`, `kpe = fp8 * kpe_scale`.

## Roadmap

- SM100 kernel: the remaining gap to FlashInfer Cake. The V conversion is now the busiest role, each call has a
  fixed cost of about 40 µs (mostly the kernel's start and one tile's latency), and with 128 heads the cache is read
  once per 16 heads.
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
