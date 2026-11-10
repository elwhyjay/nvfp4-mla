# NVFP4 Dense MLA Decode kernel

MLA decode attention (DeepSeek-style, weight-absorbed) that reads the **FlashInfer NVFP4 paged KV cache** directly.
I added this cache format and its quantize/append kernel to FlashInfer in
[flashinfer#4676](https://github.com/flashinfer-ai/flashinfer/pull/4676); this repository is the decode side.
The latent KV is stored as E2M1 with one E4M3 scale per 16 channels and the RoPE part as FP8, so one token is
352 bytes (FP8: 576, BF16: 1152). The kernel dequantizes in registers and keeps Q, P and the MMA in BF16 with FP32
accumulation.

It is written from scratch in CUDA (`mma.sync`, `cp.async`, split-KV) and uses one code path for both
**SM120 (RTX PRO 6000)** and **SM100 (B200)**.

Status: dense decode, one query token per request (see [Roadmap](#roadmap)).

## Why

FlashInfer can write an MLA KV cache in NVFP4 (#4676). Who can read it back for dense decode, as of 2026-11-10
(FlashInfer v0.7.3, vLLM v0.33.0, SGLang v0.5.23, TensorRT-LLM v1.4.0rc3):

- **SM12x** (RTX PRO 6000 and RTX 50 series are SM120, DGX Spark is SM121): FlashInfer, vLLM and SGLang have no
  dense NVFP4 MLA decode. vLLM's dense MLA backends take BF16 or FP8 KV only, and its NVFP4 MLA cache
  (`nvfp4_ds_mla`) is for sparse MLA on SM100 only. SGLang's FP4 MLA cache uses its own layout and is dequantized to
  BF16 before attention. TensorRT-LLM's `fp4_mla` (FP4 query, KV and P, its own cache layout) is validated on SM100
  and SM107 only.
- **SM100:** FlashInfer v0.7.3 added `cake_mla_nvfp4_paged_decode` (experimental), which reads this cache with
  `tcgen05` MMA and an NVFP4 query. An open vLLM PR ([#63203](https://github.com/vllm-project/vllm/pull/63203))
  uses it for `--kv-cache-dtype nvfp4` on dense MLA models.
- FlashInfer's generic MLA decode (`trtllm_batch_decode_with_kv_cache_mla`) still raises for this cache.

This kernel reads the cache on SM120 and SM100 and keeps the query in BF16.

## Requirements

- NVIDIA Blackwell GPU: SM120 (RTX PRO 6000) or SM100 (B200). Other Blackwell parts (SM103, SM121) are not tested.
- CUDA toolkit 12.8 or newer, with `nvcc` on `PATH`. With 13.2 or newer the kernel uses the native FP4/FP8 to BF16
  conversion instructions; older toolkits convert through FP16, which gives the same values with a few more
  instructions.
- PyTorch 2.7 or newer for the same CUDA major version, and `ninja`.
- `flashinfer-python >= 0.7.0` to write the cache (and for the tests and benchmarks).

Tested with CUDA 13.3, PyTorch 2.12.0 and Python 3.13 on RTX PRO 6000 Blackwell Server Edition and B200. The
benchmarks below use FlashInfer 0.7.0; the tests and the baseline checks also pass with FlashInfer 0.7.3.

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
- Quantization error of the format itself (random data, NVFP4 cache vs BF16 cache): relative L2 0.10, cosine about
  0.995. For comparison, FP8 KV with FP8 Q (Triton FP8, XQA) has cosine 0.998 against BF16.

## Performance

GPU kernel time per call in ms, measured with CUPTI after an L2 flush, so host launch time is not included.
Page size 64, every request has the same length, median of 20 calls. All providers run in the same process on the same
random data. Full sweeps (H 16/128, batch 1-64, 1K-64K tokens): [rtx_pro_6000.csv](benchmarks/results/rtx_pro_6000.csv),
[b200.csv](benchmarks/results/b200.csv).

Baselines: FlashInfer `BatchMLAPagedAttentionWrapper` (fa2, BF16 KV); FlashInfer
`trtllm_batch_decode_with_kv_cache_mla` (XQA on SM120, trtllm-gen on SM100); SGLang's Triton MLA decode
(the vLLM/SGLang default on SM120, vendored in `benchmarks/triton_mla.py`); and "dequantize + fa2"
(`nvfp4_mla.dequantize` to BF16, then fa2).

**RTX PRO 6000 (SM120)**

| heads | batch x kv_len | nvfp4 | Triton FP8 | fa2 BF16 | dequantize + fa2 | XQA FP8 |
|---|---|---|---|---|---|---|
| 16 | 1 x 4K | **0.018** | 0.046 | 0.037 | 0.038 | n/a (needs 128 heads) |
| 16 | 1 x 64K | **0.043** | 0.115 | 0.130 | 0.181 | n/a |
| 16 | 16 x 64K | **0.333** | 0.573 | 1.554 | 2.661 | n/a |
| 16 | 64 x 64K | **1.109** | 1.850 | 7.057 | 11.49 | n/a |
| 128 | 1 x 64K | **0.105** | 0.226 | 0.166 | 0.215 | 1.135 |
| 128 | 16 x 64K | 1.366 | 2.197 | 3.322 | 4.428 | **1.136** |
| 128 | 64 x 64K | 5.143 | 8.521 | 10.52 | 14.93 | **2.440** |

**B200 (SM100)**

| heads | batch x kv_len | nvfp4 | Triton FP8 | fa2 BF16 | trtllm-gen BF16 | trtllm-gen FP8 |
|---|---|---|---|---|---|---|
| 16 | 1 x 4K | 0.013 | 0.050 | 0.025 | 0.016 | **0.012** |
| 16 | 1 x 64K | 0.036 | 0.111 | 0.081 | 0.044 | **0.030** |
| 16 | 16 x 64K | 0.204 | 0.536 | 0.563 | 0.236 | **0.116** |
| 16 | 64 x 64K | 0.772 | 1.907 | 2.496 | 0.857 | **0.388** |
| 128 | 1 x 64K | 0.105 | 0.292 | 0.098 | 0.046 | **0.036** |
| 128 | 64 x 64K | 5.949 | 14.56 | 4.966 | 1.024 | **0.545** |

How to read these numbers:

- **SM120, 16 heads** (memory bound): fastest in all 16 measured shapes (batch 1-64, 1K-64K tokens): 1.6-3.6x
  faster than Triton FP8 and 1.5-6.4x faster than fa2. At 64 x 64K it reads 1.33 TB/s, 85% of the achievable
  1.56 TB/s (`benchmarks/roofline.py`). Triton FP8 reaches the same fraction there, so the 1.66x gain is the byte
  ratio 576 / 352.
- **SM120, 128 heads**: fastest at batch 1-4, where XQA is slow with long contexts. From batch 16 the problem is
  compute bound and XQA's FP8 MMA wins; the BF16 `mma.sync` peak of this GPU is the limit for this kernel. It is
  faster than Triton FP8 and fa2 in all shapes.
- **B200, 16 heads**: faster than trtllm-gen BF16 in all 16 shapes (1.1-1.5x), than fa2 (1.6-3.2x) and than
  Triton FP8 (2.4-4.6x). trtllm-gen FP8 (`tcgen05` MMA) is up to 2x faster except at batch 1-4 with 1K tokens.
  This kernel uses `mma.sync` on both GPUs; on B200 it is limited by latency within the SM, not by memory.
- **B200, 128 heads**: compute bound, and `mma.sync` is far below `tcgen05`: up to 12x slower than trtllm-gen FP8.
- FlashInfer's `cake_mla_nvfp4_paged_decode` (v0.7.3, SM100, reads the same cache with an NVFP4 query) is not in these
  tables yet; it needs CUDA 13.4.

## Reproduce

```bash
pip install -e ".[test,bench]"
pytest tests                       # correctness; the first run builds the extension three ways (~3 min)
python benchmarks/roofline.py      # achievable read BW and BF16 mma.sync peak of this GPU
python benchmarks/baselines.py     # sanity check: every baseline agrees with FlashInfer fa2
python benchmarks/bench.py         # full sweep; --batch / --s-kv / --heads / --providers to narrow it
```

The largest sweep points (64 x 64K, with BF16 copies for the baselines) need a lot of GPU memory; narrow the sweep on
smaller GPUs.

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

- MTP / speculative decode (2-4 query tokens per request).
- An SGLang integration (attention backend plugin plus a KV pool in this layout).
- A B200 comparison with FlashInfer's `cake_mla_nvfp4_paged_decode`.

## Acknowledgements

- The cache format and its quantize/append kernel are from my FlashInfer PR
  [#4676](https://github.com/flashinfer-ai/flashinfer/pull/4676), which FlashInfer's `cake_mla_nvfp4_paged_decode`
  (SM100) also reads.
- `benchmarks/triton_mla.py` is vendored from [SGLang](https://github.com/sgl-project/sglang) (commit c311bc961164,
  Apache-2.0) to benchmark against the Triton MLA decode kernel. SGLang has not changed these kernels for NVIDIA GPUs
  since then (checked against v0.5.23, 2026-11-10).

## License

Apache-2.0, see [LICENSE](LICENSE).
