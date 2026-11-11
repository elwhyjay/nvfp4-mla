# Optimization log

How the kernel went from v1 to v5. Each step lists the hypothesis, the measurement behind it, the change, and the
result. Steps that did not work are kept. Latency is GPU kernel time for MLA decode with q_len 1, page size 64, and
DeepSeek geometry (512 latent + 64 RoPE channels). Unless noted, the shape is **H=16, batch 64, 64K tokens**, the
shape that streams the most KV bytes. Per-version numbers for all 32 shapes are in
[`benchmarks/results/versions.csv`](../benchmarks/results/versions.csv).

## Setup

| | RTX PRO 6000 (SM120) | B200 (SM100) |
|---|---|---|
| SMs, max smem per block | 188, 99 KB | 148, 227 KB |
| Achievable DRAM read BW ([`roofline.py`](../benchmarks/roofline.py)) | 1559 GB/s | 7300 GB/s |
| `mma.sync` BF16 m16n8k16 peak | 455 TFLOPS | 558 TFLOPS (not tcgen05) |
| Ridge point | 293 flop/B | 76 flop/B |

A decode step does 2 · (576 + 512) = 2176 flop per token and head, and reads 352 B per token from the NVFP4 cache.
The arithmetic intensity is therefore about 6.2 · H flop/B.
- **H=16 (99 flop/B):** memory-bound on SM120, so the target is DRAM bandwidth. On B200 the same shape is already
  above the `mma.sync` ridge.
- **H=128 (791 flop/B):** compute-bound on both GPUs.

**How the measurements were taken**
- Profiles use Nsight Compute 2025.2.1 with `--clock-control none`.
- Up to v4, latency was timed with CUDA events. From v4 on it is CUPTI kernel time, with L2 flushed before each
  call (see "Measurement fix" below). For the large shapes in this log the two methods agree within 1–2%.

## v0: reference

The fp32 PyTorch reference dequantizes the cache and runs attention. "Dequantize, then FlashInfer fa2" serves as the
first baseline.
- Relative L2 error of the NVFP4 path against the unquantized BF16 cache is 0.1025 at H=16 and 0.1003 at H=128. This
  matches what was measured for the cache format in flashinfer#4676 (0.092–0.102).
- Every later version is checked against the dequantized reference. Kernel dequantization is bitwise equal to it.
  Attention output stays within 2e-3 relative L2 (BF16 rounding of P and the output).

## v1: CUDA cores (correctness first)

One thread block per (request, head group), fp32 FMA on CUDA cores. The only goal was to confirm the dequantization
rule (low nibble = even channel, one E4M3 scale per 16 channels).

| | RTX PRO 6000 | B200 |
|---|---|---|
| H=16, B64, 64K | 21.7 ms | 35.2 ms |
| H=128, B64, 64K | 134.8 ms | 212.7 ms |

## v2: tensor cores with dequantization in registers

- **Hypothesis.** v1 is far from both roofs, so the arithmetic itself is the bottleneck. Move QK and PV to BF16
  `mma.sync`, and dequantize FP4 in registers so the BF16 cache never exists in memory.
- **Design.**
  - A CTA handles 16 heads with 4 warps. Warp *w* owns latent channels [128w, 128w+128) and RoPE channels
    [16w, 16w+16).
  - The QK reduction axis is permuted so that each thread holds 32 contiguous channels, which are two scale blocks.
    Q is permuted the same way and stays in registers for the whole kernel.
  - `cvt.rn.bf16x2.e2m1x2` turns two FP4 codes into BF16, then `mul.rn.bf16x2` applies the block scale.
    E2M1 × E4M3 is exact in BF16, so this adds no rounding.
  - The QK B fragments are transposed in registers with `movmatrix.trans` and reused as PV B fragments. Each token
    is dequantized once and serves both GEMMs.
  - The 4 warps each hold a partial QK score. These are summed through 4 KB of shared memory.
- **Result.** H=16: 21.7 → **5.15 ms** (4.2×). H=128: 134.8 → **6.09 ms** (22×).
- **Measurement after the change** (ncu, H=16 B64 16K):
  - DRAM throughput is 11.9% and achieved occupancy 8.3%.
  - Long-scoreboard stalls are 48.6% of all stalls. Every tile does a dependent load chain
    (page table → page → data) that nothing hides.

## v3: cp.async pipeline

- **Hypothesis.** v2 waits on dependent global loads. A multi-stage asynchronous copy can keep the loads in flight
  while the tensor cores work on the previous tile.
- **Design.**
  - 32-token tiles in 3 `cp.async` stages, 45 KB of shared memory per CTA, which allows 2 CTAs per SM on SM120.
  - Each thread copies one token's contiguous 64 B segment. The page id for the next tile is fetched one tile ahead.
  - All three smem arrays use a 16 B-chunk XOR swizzle, so the fragment loads have no bank conflicts.
- **Result.**
  - H=16: 5.15 → **3.45 ms**. DRAM throughput went from 11.9% to 27.3%.
  - H=128 at batch 64 regressed, 6.09 → 8.16 ms. This was not profiled at the time. The likely cause is fewer
    resident CTAs per SM, since smem per CTA went up. v5 later removed the regression.
- **Measurement after the change.** Occupancy did not move (8.3%), and every batch size took the same 3.45 ms. The
  grid was the limit: H=16 at batch 64 launches only 64 CTAs on 188 SMs, so most SMs were idle.

## v4: split-KV

- **Hypothesis.** v3 is limited by the number of CTAs, not by any unit inside an SM. Splitting each request's KV
  range across CTAs, then merging the partial results, should fill the GPU.
- **Design.**
  - Fixed chunks that are multiples of 32 tokens, with at least 256 tokens per chunk in auto mode.
  - A combine kernel merges the partial outputs using their LSE values.
  - The caller passes `max_kv_len`, so there is no host synchronization and the launch shape works under CUDA Graphs.
  - The first combine kernel loaded the partials sequentially and took 95 µs at batch 1. Parallelizing it over
    (rows × 64-channel slices) brought the whole call at batch 1, 64K down to 47 µs.
- **Result** (RTX PRO 6000):

  | | v3 | v4 |
  |---|---|---|
  | H=16, B64, 64K | 3.45 ms | **1.11 ms** (85% of achievable BW, ncu DRAM 88%) |
  | H=16, B1, 64K | 3.45 ms | **0.046 ms** |
  | H=128, B64, 64K | 8.16 ms | 8.19 ms |

  The memory-bound goal on SM120 (75% of achievable BW) was met at this step. H=128 did not move, because it is
  compute-bound.

## Measurement fix: CUDA events → CUPTI kernel time

At batch 1, every FlashInfer trtllm-gen call measured at least about 85 µs with CUDA events, and the floor changed
between sessions.
- Timing the host and the kernel separately showed where it came from: the trtllm-gen wrapper spends about 135 µs of host time per call,
  while its kernel runs for 8.9 µs (1K tokens) to 24.6 µs (64K tokens). The events were timing the host.
- All benchmarks since then sum the GPU kernel time inside a profiler range (CUPTI through `torch.profiler`), with L2
  flushed before each call ([`baselines.gpu_ms`](../benchmarks/baselines.py)).
- This kernel has little host overhead, so its own numbers moved by only 1–2%. The baselines at small batch moved a
  lot.

## v5: fewer instructions per token

- **Measurement first** (ncu on B200, v4, H=16 B64 64K). On B200 this shape is above the `mma.sync` ridge, so time
  inside the SM matters.
  - Issue slots were 54% busy, about 10 warps were active per SM (16%), and the tensor pipe was 35% busy. No pipe was
    saturated.
  - A warp spent about 480 instructions per 16 tokens. That is about 2000 cycles per CTA tile, while DRAM delivers a
    tile every ~225 cycles.
  - Static SASS had 334 FMUL against 68 HMMA. The FMULs come from `o *= correction` on every tile and from precise
    `exp2f`.
  - The top stalls were the cross-warp score reduction (barrier → LDS → FADD chain) and the row max (SHFL → FMNMX).
    Integer division in the `cp.async` address math was another ~5%.
- **Hypothesis.** The SM is latency- and instruction-bound. Cutting instructions on the per-tile path, and shortening
  the reduction chain, should help both GPUs.
- **Changes.**
  - *Lazy rescale* (as in FlashAttention-4): rescale `o` only when the running max grows by more than 2^8. P stays
    ≤ 256, which is safe in fp32 and BF16.
  - Use `ex2.approx.ftz`, and fold the softmax scale and max subtraction into one FMA: `ex2(fma(s, scale, -m))`.
  - Interleave the two HMMA dependency chains in QK. This removed 16 NOPs between dependent HMMAs.
  - Score reduction:
    - Store each thread's four partials as one `float4` with stride 16, which is conflict-free and saves 4 KB of smem.
    - Read the thread's own partial from registers.
    - Keep the summation order the same in every warp.
  - Split the mbarrier into arrive and wait: write the partials, arrive, run the 32 `movmatrix` needed by PV, then wait.
    Registers went from 168 to 202 (B200: 3 → 2 CTAs per SM). It was still a net gain.
  - Mask only the last sub-tile. Require power-of-two page sizes, so address math uses shifts and masks.
  - New split policy: choose the chunk that minimizes `waves × (chunk + 256)` subject to `chunk ≥ √L`. CTA time grows
    with the chunk, combine time with L / chunk, so the optimum sits near √L. This matched the measured best chunk
    from 1K to 64K tokens, and it avoids a last wave that is half empty.
- **Tried and dropped.**
  - `__launch_bounds__(128, 3)` to force 3 CTAs per SM: it spilled 80 B per thread, and H=128 got 15–20% slower.
  - Pipeline depth 2 / 3 / 4 / 6:
    - B200 changed by ±3%, so the memory pipeline is not the B200 bottleneck.
    - On SM120, 4 or more stages drop to 1 CTA per SM, and H=128 gets 50%+ slower.
    - 3 stages stays.
  - Larger head groups for H=128:
    - 32 heads per CTA with 4 warps needs 72 Q + 128 O registers per thread, which does not fit.
    - 8 warps × 64 channels makes the cross-warp reduction 112 floats per thread, which costs more than the
      dequantization it saves.
    - The L2 hit rate is already 87%, so KV reuse across head groups is mostly there anyway.
- **Result** (GPU time):

  | | RTX PRO 6000 v4 → v5 | B200 v4 → v5 |
  |---|---|---|
  | H=16, B64, 64K | 1.110 → 1.109 ms (memory-bound, unchanged) | 0.955 → **0.772 ms** |
  | H=16, B1, 1K | 0.024 → **0.014 ms** | 0.021 → **0.010 ms** |
  | H=128, B64, 64K | 8.19 → **5.14 ms** | 8.97 → **5.95 ms** |

## Where it stands

Latency, H=16, batch 64, 64K tokens:

| GPU | v1 | v5 | v1 → v5 | % of the roof for this shape |
|---|---|---|---|---|
| RTX PRO 6000 | 21.7 ms | 1.11 ms | 20× | 85% of DRAM BW |
| B200 | 35.2 ms | 0.77 ms | 46× | 34% of `mma.sync` BF16 peak |

On RTX PRO 6000, H=16 is the fastest MLA decode measured at every batch size and context length, including SGLang
Triton FP8 (1.65× at batch 64). The compute-bound shapes are the open problem; Part 2 below.

## Part 2: compute-bound shapes (in progress)

The target is RTX PRO 6000 at H=128, batch 64, 64K tokens. There v5 takes 5.0 ms and XQA (FP8 KV, FP8 MMA) takes
2.36 ms. Three measurements came first; the choice for v6 follows from them.

### Measurement: where v5 spends time at H=128

ncu on v5, H=128 B64 64K (4.99 ms):
- The tensor pipe is busy 54% of the time. Issue slots are 35% busy (IPC 1.39 of 4), so the SM is not short of
  issue bandwidth.
- Only 2 warps per scheduler are resident (occupancy 16.7%): 206 registers and 45 KB of smem per CTA each allow
  2 CTAs per SM.
- Stalls: waiting on a fixed-latency dependency 27%, math pipe throttle 15%, short scoreboard 12%, long scoreboard 8%.
- There are 9.8 other instructions per HMMA. About 2.6 of them make the BF16 operands (`F2FP.E2M1.UNPACK`,
  `HMUL2` for the scale, `MOVM` for the V transpose), about 1 is the cross-warp score reduction (`FADD`), and the
  rest are moves, softmax, addressing and control.

With 2 warps per scheduler, the chain dequantize → HMMA → reduce is not hidden, and the tensor pipe idles about half
of the time.

### Measurement: `mma.sync` kinds on SM120

[`roofline.py`](../benchmarks/roofline.py), one warp per scheduler with 8 independent accumulators:

| kind | latency (cycles) | cycles per mma per scheduler | TFLOPS (whole GPU) |
|---|---|---|---|
| BF16 m16n8k16 | 29.8 | 15.6 | 449 |
| E4M3 m16n8k32 (also `kind::f8f6f4`) | 29.7 | 15.6 | 897 |
| `kind::mxf4nvf4` block-scaled E2M1 m16n8k64 | 29.8 | 15.6 | 1862 |

Every kind costs the same issue slot and latency. FP8 does 2x and block-scaled FP4 4x the reduction length per
instruction, so a lower-precision operand cuts tensor time proportionally.

### Measurement: what precision costs

[`precision.py`](../benchmarks/precision.py) simulates each option on the same NVFP4 cache. Error is relative L2
against attention over the unquantized BF16 cache (random data, three shapes):

| option | error | vs v5 |
|---|---|---|
| BF16 Q, BF16 P (v5) | 0.084-0.137 | |
| FP8 Q, K and V rounded to FP8, FP8 P | 0.087-0.133 | -3% to +7% |
| FP4 Q (as in FlashInfer Cake), BF16 or FP8 P | 0.131-0.179 | +28% to +55% |

Almost all of the error comes from storing the cache in NVFP4. Computing in FP8 adds almost nothing; quantizing the
query to FP4 adds a lot. (For reference, an FP8 KV cache with FP8 Q and P, as XQA uses, gives 0.046-0.054.)

### Options

Tensor work per warp per 32-token tile. The floor scales the BF16 floor (2.57 ms) by the mma count, since every kind
takes the same 15.6 cycles.

| option | mma per tile | tensor floor at H=128 B64 64K | error vs v5 |
|---|---|---|---|
| A. BF16 QK and PV (v5) | 68 | 2.57 ms | |
| B. FP8 QK and PV (RoPE part stays BF16) | 36 | 1.36 ms | -3% to +7% |
| C. FP4 QK (block-scaled, needs FP4 Q), BF16 PV | 44 | 1.66 ms | +28% to +55% |
| D. FP4 QK, FP8 PV | 28 | 1.06 ms | +28% to +55% |

- Option A cannot beat XQA: its floor is above XQA's time.
- Block-scaled FP4 is the only kind whose scale format matches the cache (E4M3 per 16 channels along the QK reduction
  axis), and it needs both operands in FP4, so FP4 QK implies an FP4 query.
- PV cannot use block scaling: the V scales run along the output channels, not along the reduction axis (tokens).
- XQA itself runs at 55% of the FP8 peak for this shape (floor 1.30 ms, measured 2.36 ms).

### Choice for v6: option B (FP8 QK and PV)

1. It is the largest cut in tensor work that keeps the accuracy (-47% mma per tile). It keeps the accuracy advantage
   over FlashInfer Cake, whose query is FP4.
2. Its main risk is the work of making FP8 operands: converting NVFP4 to E4M3 in registers, and transposing 8-bit V
   for PV, where `movmatrix` only handles 16-bit elements. Option D needs the same FP8 PV, so solving this first is a
   step toward D, not a detour.
3. D (FP4 query) comes next as a speed option, measured against v6 for both time and error.

### v6 predictions (written before the implementation)

- HMMA instructions per warp per tile: 68 -> 36 (static SASS count).
- Tensor pipe busy time: about -47%, from about 2.7 ms to about 1.4 ms at H=128 B64 64K.
- Other instructions per tile do not shrink: the dequantization now ends in E4M3, and V needs an 8-bit transpose.
  While the kernel stays latency-bound with 2 warps per scheduler, the expected time is 3.5-4.0 ms (-20% to -30%),
  still slower than XQA.
- Error against BF16 within -3% to +7% of v5 (`benchmarks/accuracy.py` on GPU).
- H=16 is memory-bound on SM120 and should not change by more than 3%.
- Decision rule: if H=128 improves by less than 10%, the non-tensor instruction stream is the limit, and the next
  step attacks it before the FP4 query. One candidate is that at H=128 eight CTAs dequantize the same KV tile.

### v6 result: FP8 QK and PV

Implementation (`tile_fp8` in the kernel, `fp8=True`):
- A whole 32-token tile per step, since PV with k32 consumes 32 tokens at once.
- Q is quantized to E4M3 at the start with one scale per row and part (latent / RoPE); the scales fold into the
  score scale.
- K/V: E2M1 to f16, times the block scale (exact), then rounded to E4M3. The RoPE part of K is used as stored.
- The PV token axis is permuted to match the QK accumulator layout, so P needs no shuffle. V is transposed with
  `movmatrix` on byte pairs, then split into two channels with a byte permute.
- P is E4M3 without scaling; the row sum stays fp32.
- The PV output lands in the same registers as in the BF16 path, so the softmax update and the stores are shared.
- One cross-warp reduction per tile, so a single buffer is enough: smem goes from 45 KB to 43 KB.

Measured (RTX PRO 6000; v5 and v6 in the same run):

| | v5 | v6 | predicted |
|---|---|---|---|
| H=128 B64 64K | 4.81 ms | **3.56 ms (-26%)** | 3.5-4.0 ms |
| H=128 B64 4K | 0.417 ms | 0.324 ms (-22%) | |
| H=128 B1 64K | 0.104 ms | 0.082 ms (-21%) | |
| H=16 B64 64K | 1.122 ms | 1.125 ms | within 3% |
| MMA per warp per tile | 68 | **36** | 36 |
| Tensor pipe busy time (ncu) | 2.69 ms | **1.48 ms (-45%)** | about -47% |
| Other instructions per warp per tile | 666 | 679 | no decrease |
| Error vs BF16 cache (9 shapes) | 0.078-0.150 | 0.085-0.161 | |

- Error grew by 0% to 11% over v5, a little more than the simulated -3% to +7%.
- Instruction mix per warp per tile: conversions (`F2FP`) 84 -> 140 for the E4M3 rounding and P, `MOVM` 64 -> 32
  plus `PRMT` 8 -> 40 for the byte-pair transpose, address math 120 -> 86 (one tile step instead of two).
- Not predicted: registers fell from 206 to 168 (FP8 Q fragments are half the size), so registers would now allow
  3 CTAs per SM. Shared memory (43 KB per CTA) is now the only limit at 2.
- Issue slots 35% -> 48%; stalls: dependency wait 23%, short scoreboard 14%. Still latency-bound with 2 warps per
  scheduler.

### v6b hypothesis: 2 pipeline stages, 3 CTAs per SM

- Two stages cut smem to about 32 KB, so 3 CTAs fit per SM (168 registers x 384 threads = 64512 of 65536).
- The memory pipeline keeps the same depth per SM: 2 stages x 3 CTAs = 3 x 2 = 6 tiles in flight.
- A third warp per scheduler should hide part of the dependency and short-scoreboard stalls (37% together).
- Predictions: occupancy 2 -> 3 CTAs per SM; H=128 B64 64K 3.56 ms -> 2.9-3.2 ms (-10% to -20%); H=16 within 3%.
- Decision rule: adopt 2 stages for the FP8 path if H=128 improves by 10% or more; otherwise keep 3.

### v6b result: occupancy went up, the time did not

| | 3 stages (v6) | 2 stages | predicted |
|---|---|---|---|
| CTAs per SM (achieved occupancy) | 2 (16.6%) | 3 (24.4%) | 3 |
| Kernel time, H=128 B64 64K (ncu) | 3.71 ms | 3.50 ms (-5.4%) | |
| KV splits chosen at that shape | 8 (11 waves) | 11 (10 waves) | |
| End-to-end, H=128 B64 64K | 3.511 ms | 3.485 ms (-0.7%) | 2.9-3.2 ms |
| H=128 B64 4K | 0.325 ms | 0.269 ms (-17%) | |
| H=128 B16 64K | 0.950 ms | 0.891 ms (-6%) | |
| H=128 B1 4K | 0.024 ms | 0.029 ms (+21%) | |

- The latency-hiding hypothesis is mostly wrong. With 50% more warps the kernel got only 5% faster; issue slots
  went from 48% to 53% and the stall mix barely changed.
- The 17% gain at 4K comes from wave quantization, not latency hiding: 512 CTAs fit in one wave at 3 per SM and
  need two at 2 per SM.
- At 64K a larger wave made the policy choose 11 splits instead of 8, which adds about 50 MB of fp32 partial results
  written and read by the combine kernel. The policy's cost model has no term for combine traffic, which grows with
  the number of rows (8x larger at H=128 than at H=16). Whether this ate the kernel gain is not established: the
  -5.4% is ncu kernel time and the -0.7% is the benchmark total, and the two tools differ by about 5% on the same
  kernel.
- Decision: keep 3 stages (the rule asked for 10% at the target shape). Two findings carry forward: the split
  policy should count combine traffic, and the non-tensor instruction stream (679 instructions per warp per tile
  against 36 MMA) is the limit, not the number of warps.

### Split policy: hypothesis

The KV split count is chosen on the host by minimizing `waves x (chunk + C0)` with `C0 = 256` tokens and
`chunk >= sqrt(L)`. It was fitted to split sweeps of the BF16 path in v5.

- Hypothesis: the policy ignores the combine kernel's cost. The partial outputs are fp32, 2 KB per row and split,
  written once and read once, so that cost grows with rows x splits and is 8x larger at H=128 than at H=16.
  The v5 fit absorbed it into `C0` for the shapes it was fitted on; the FP8 path and a different wave size move
  the optimum.
- Measurement first, with one timer: for each shape and each split count, the decode and combine kernel times
  separately (CUPTI), for the FP8 path with 2 and 3 stages and for the BF16 path. Then compare the best split
  count with the policy's choice.
- Prediction: at H=128 with batch 16-64, the best split count is lower than the policy's choice and saves at least
  3%; at H=16 the policy stays within 3% of the best.

### Split policy: result

[`split_sweep.py`](../benchmarks/split_sweep.py): 11 shapes x 3 configurations (FP8 with 3 and 2 stages, BF16 with 3 stages), every split count
from 1 to 128, decode and combine time each measured with CUPTI.

- The prediction is wrong for the target. For the FP8 path with 3 stages at H=128 the policy picks the best split
  count or one within 0.2% (H=128 B64 64K: 8 splits, the best).
- Combine is real but small: about 0.012 ms per split at H=128 batch 64 (8 splits = 2.5% of the total).
- Over all 33 cases the policy is 1.5% slower than the best split count on average and 7.1% in the worst case.
  The worst cases are memory-bound H=16 shapes, where the policy splits batch 64 into 11-17 parts and 3-8 is best:
  bandwidth is already saturated with fewer CTAs than SMs, so the policy's wave model over-splits.
- A cost model with a combine term (`K x splits x rows / wave`, refit together with `C0` and the `sqrt(L)` floor,
  84 combinations) does no better than the current one on these 33 cases.
- Decision: keep the policy. Fixing the memory-bound case needs a second model for little gain.
- Side result for the FP8 path: with the policy's own choices, 2 stages beat 3 stages on 8 of 11 shapes (geometric
  mean -3.2%; H=128 B64 4K -17%, H=128 B1 64K +4.9%). This matters when the FP8 path becomes the default.

### v7 hypothesis: share the K/V conversion between head groups

At H=128 the 8 CTAs of one request and split (16 heads each) all convert the same KV tile. From the v6 instruction
mix, per warp and tile, the work that does not depend on the head group is the E2M1 -> E4M3 conversion (132
`F2FP`), the scale multiply (64 `HMUL2`), the V transpose (32 `MOVM` + 32 `PRMT`) and the raw KV reads (12 `LDS`):
about 270 of 715 instructions.

Options:

| option | conversion per head group | cost |
|---|---|---|
| a warp computes two head groups (32 heads, 128 channels) | 1/2 | accumulators alone need 128 registers; with Q, K, V and scores it exceeds 255 |
| a warp computes 32 heads x 64 channels, 8 warps per CTA | 1/2 | the cross-warp score reduction becomes 8-way over twice the rows: about 224 `FADD` and 56 `LDS` per tile |
| 8 warps per CTA = 2 head groups; warps w and w + 4 own the same channels and each converts half of the tile's tokens, then they swap halves through smem | 1/2 | 4 `STS.128` + 4 `LDS.128` + one 64-thread barrier per tile; smem about 68 KB, so 1 CTA (8 warps) per SM, the same warps per SM as now |

Choice: the third option. Registers stay at 168, the score reduction stays 4-way, and the copy work (`cp.async`
issue and address math) is spread over 256 threads instead of 128. The V transpose stays in registers in this step
(each warp still needs its own PV fragments); moving it to smem is a later step.

Predictions (FP8 path, H=128; H=16 keeps one head group per CTA):
- Instructions per warp per tile: about 715 -> 605 (-15%): conversion -98, raw reads -6, copy about -15, exchange +10.
- H=128 B64 64K: 3.55 ms -> 3.15-3.35 ms (-6% to -12%). Less than the instruction cut, because the kernel is
  latency-bound and the pair barrier adds a wait.
- Decision rule: keep if H=128 B64 64K improves by 5% or more and no H=128 shape gets more than 3% slower. If the
  instruction count falls as predicted but the time does not, issue bandwidth is not what limits the kernel.

### v7 result: shared conversion made it slower

| per warp per tile (ncu, H=128 B64 64K) | v6 | v7 (2 head groups) | predicted |
|---|---|---|---|
| `F2FP` (conversion) | 140 | **74** | halved |
| `HMUL2` (scale) | 64 | **32** | halved |
| address and integer math | 86 | **156** | about -15 |
| all instructions | 715 | 682 (-4.7%) | about 605 (-15%) |
| registers | 168 | 178 | 168 |
| issue slots busy | 48% | 39% | |
| tensor pipe busy | 40% | 34% | |
| kernel time (ncu) | 3.71 ms | 4.22 ms (+14%) | |

End-to-end on the H=128 shapes (CUPTI): B64 64K 3.447 -> 4.043 ms (+17%), B16 64K +15%, B64 4K +12%;
only B1 16K got faster.

- The sharing itself worked: conversion and scale instructions halved.
- Two things were not predicted:
  - Address math grew by 70 instructions per tile: the slot-to-token rotation, the exchange indices and the
    8-way copy split all need runtime index arithmetic.
  - The SM now holds one 8-warp CTA instead of two independent 4-warp CTAs. Every `__syncthreads` and every pair
    barrier stops warps that used to belong to a different CTA and kept running. Issue slots fell from 48% to 39%,
    and barrier and MIO stalls appeared (4% and 10%).
- The hypothesis "fewer instructions per tile make it faster" is wrong for this kernel. What limits it is the latency
  of the per-warp chain (load, convert, QK, cross-warp sum, softmax, PV) and how much independent work the
  schedulers can switch to. Two small independent CTAs per SM give more of that than one large one.
- Decision: drop the two-group CTA; its code is not kept on main.

### v8 hypothesis: convert the next tile while the current one finishes

The measurements since v6 point to the latency of each warp's chain, not to the amount of work: tensor pipe 40% busy,
issue slots about 50%, and dependency stalls (`wait` and short scoreboard) 37% of all samples. Within a tile the
chain is strictly ordered: read and convert K/V, QK, write the partial scores, wait for the other three warps, sum,
softmax, PV. With 2 warps per scheduler, little else is ready while a warp waits.

Change (FP8 path):
- Convert one tile ahead. At iteration i the K/V of tile i are already E4M3 fragments in registers. The conversion
  of tile i+1 (its smem reads plus about 210 conversion instructions) runs between the warp's arrival at the
  cross-warp barrier and its wait, after the V transposes.
- Tile i's shared memory is fully read during iteration i-1, so its stage can be refilled at iteration i. With the
  same 3 stages the prefetch depth stays 2 tiles.
- The current tile's fragments are dead after the V transposes, so the next tile's can reuse their registers.

Predictions:
- Registers 168 -> about 190; still 2 CTAs per SM (shared memory is the limit).
- Instructions per tile about the same (within 3% of 715).
- Issue slots 48% -> 55-60%; the share of dependency and barrier stalls drops.
- H=128 B64 64K: 3.45 ms -> 3.0-3.2 ms (-7% to -13%). H=16 (memory-bound) within 3%.
- Decision rule: keep if H=128 B64 64K improves by 5% or more and no measured shape gets more than 3% slower.

### v8 result: slower, and the reason is in the structure

Same-run A/B ([`ab.py`](../benchmarks/ab.py): the v6 source and v8 built side by side; outputs bitwise identical):

| | v6 | v8 | predicted |
|---|---|---|---|
| H=128 B64 64K | 3.745 ms | 3.867 ms (+3.3%) | -7% to -13% |
| other H=128 shapes | | +2% to +6% (B1 4K -16%) | |
| H=16 shapes | | -1% to +6% | within 3% |
| registers | 168 | 242 | about 190 |
| instructions per warp per tile | 715 | 748 | about the same |
| issue slots busy | 48% | 49% | 55-60% |
| tensor pipe busy | 40% | 38% | |

The v6 reference measured 3.745 ms here and 3.45-3.55 ms in earlier runs: run-to-run variation on the rented GPUs is
up to 8%, so from here on comparisons are made within one run.

- The next tile's fragments did not reuse the current tile's registers (242 instead of about 190), and computing
  the next stage's addresses added about 50 integer instructions per tile.
- No overlap was gained. All four warps of a CTA do the same work and reach the cross-warp barrier at nearly the
  same time, so that wait was short to begin with. The softmax and PV after the barrier are a separate basic block
  (the `try_wait` loop ends one), so the scheduler cannot interleave the conversion with them.
- Decision: drop v8.

Where the v6 warps actually wait (stall samples at H=128 B64 64K, by the instruction the warp waits at):

| instruction | share | main reason |
|---|---|---|
| `QMMA` (FP8 MMA) | 20% | math pipe throttle (64%): QK issues 20 and PV 16 MMAs back to back, so the tensor pipe is saturated in bursts and idle in between |
| `FADD` | 13% | short scoreboard (66%): the cross-warp sum waits for its shared-memory loads |
| `MOVM` + `PRMT` | 13% | fixed-latency dependency: the V transpose chain |
| `F2FP` (conversions) | 11% | |
| `FMNMX` | 6% | short scoreboard: the row maximum waits for its shuffles |

No single item dominates. The cross-warp sum and the row maximum (19% together) come from the decomposition
itself: the four warps of a CTA split QK by channel, so every tile ends with a 4-way sum of 16 x 32 partial scores.

### v9 hypothesis: split QK by token instead of by channel

v6's warps split QK by channel, which costs a 4-way sum of 16 x 32 fp32 partial scores every tile, and every warp
then computes the softmax of all 32 tokens. The new decomposition for the FP8 path (CTA = 16 heads, 4 warps,
32-token tiles):

- QK: warp w takes tokens 8w..8w+7 over all 576 channels. Thread t's part of the reduction axis is latent channels
  [128t, +128) and RoPE [16t, +16), so the MMA itself sums over the four threads and the scores need no cross-warp
  sum. Q (E4M3, 72 registers) stays in registers; its per-row scale needs only shuffles, since each warp holds the
  whole row.
- Softmax: each warp computes p only for its own 8 tokens. The warps exchange one row maximum per row through
  shared memory; the row sums stay per warp and are added once at the end.
- Each thread converts its token's 128 latent channels (the same number of elements as in v6), uses them for QK, and
  stores them to a 16 KB E4M3 V tile in shared memory. P (16 x 32 E4M3) also goes through shared memory.
- PV: warp w takes latent channels [128w, +128) over all 32 tokens. V fragments come from `ldmatrix.trans` and a byte
  permute (no `movmatrix`). The V tile's channel order is chosen so that PV writes the same accumulator registers as
  v6, and every shared-memory access pattern is XOR-swizzled to be bank-conflict free.
- Shared memory is 40 KB with 2 stages (2 CTAs per SM); 3 barriers per tile instead of 2.

Predictions:
- Instructions per warp per tile: 715 -> about 420 (-40%): no cross-warp score sum (about -64), no `movmatrix`
  (-32), a quarter of the softmax and P conversion (about -90), fewer tile-level reloads.
- Registers about 220-240, no spills.
- H=128 B64 64K, same run as v6: -20% to -30% (from about 3.6 ms to 2.6-2.9 ms).
- H=16 (memory-bound): within 3%. Error against the BF16 cache within 2% of v6.
- Decision rule: keep if H=128 B64 64K is 10% or more faster and no H=16 shape is more than 3% slower.

### v9 result: faster where it is compute-bound, slower on small problems

243 registers, no spills; output within 2e-4 of v6 (summation order). `ab.py` now times the two kernels alternately
(5 rounds after a warm-up): two plain runs of the same build differed by up to 40% on 25-80 us shapes, depending on
what ran before them.

v9 time / v6 time (median of 5 rounds; the per-round spread is within 2% except where noted):

| | B1 | B4 | B16 | B64 |
|---|---|---|---|---|
| H=128, 4K tokens | 1.15 | 1.16 | 0.97 | 0.91 |
| H=128, 16K | 1.12 | 0.97 | 0.90 | 0.88 |
| H=128, 64K | 1.01 | 0.89 | 0.87 | **0.86** |
| H=16, 4K | 1.18 | 1.11 | 1.03 | 1.03 |
| H=16, 16K | 1.24 | 1.07 | 1.03 | 1.00 |
| H=16, 64K | 1.13 | 1.04 | 1.00 | 1.00 |

ncu at H=128 B64 64K:

| | v6 | v9 | predicted |
|---|---|---|---|
| instructions per warp per tile | 715 | 513 (-28%) | about 420 |
| cross-warp sum (`FADD`) | 66 | 18 | |
| `MOVM` | 32 | 0 | 0 |
| softmax (`FFMA`, `MUFU`) | 52 | 16 | a quarter |
| tensor pipe busy | 40% | 46% | |
| kernel time | 3.71 ms | 3.08 ms (-17%) | -20% to -30% |

- The mechanism works as intended, at about two thirds of the predicted size: the cross-warp score sum, the transposes
  and three quarters of the softmax are gone. The largest stall source is now the conversion chain (`F2FP`, `HMUL2`,
  `PRMT`: 31% of samples), then the MMAs (21%).
- Small problems (batch 1-4) got 4-24% slower. Against v6 with 2 stages they are still 9-12% slower at H=16 batch 1,
  so the stage count is not the whole cause. With few tiles per CTA these shapes are latency-bound, and v9's tile is
  a longer serial chain (three barriers; P goes through shared memory).
- The decision rule is not met: H=128 B64 64K improved by 14% (needed 10%), but H=16 shapes got up to 24% slower
  (allowed 3%).
- Decision, with that stated: v9 becomes the FP8 path. The rule assumed the FP8 path would also serve small and
  memory-bound shapes, but there the BF16 path (v5) is as fast or faster than both FP8 versions and more accurate
  (H=16 B1 64K: 0.043 ms vs 0.045 for v6 and 0.050 for v9). Choosing between the two paths per shape is a separate,
  measured step before the FP8 path becomes the default.

### v10 hypothesis: FP4 QK with block scales (speed mode, FP4 query)

After v9 the largest stall source is the conversion chain (31% of samples), then the MMAs (21%). FP4 QK attacks
only the second: V still has to be converted to E4M3 for PV, so the conversion stays.

Layout, checked first with a probe of `mma.sync kind::mxf4nvf4.block_scale.scale_vec::4X m16n8k64` against a PyTorch
reference (exact match, with a different scale per column):
- A: `a0` = row g, k 8t..8t+7 (8 codes, low nibble first), `a1` row g+8, `a2`/`a3` the same at k 32+8t.
  B: `b0` = column g, k 8t..8t+7; `b1` k 32+8t.
- Scales: lane (g, t) passes 4 E4M3 bytes (one per k16 block): row g + 8 (t % 2) for A, column g for B.

Design (QK only; RoPE stays FP8; softmax, P and PV unchanged from v9):
- k64 step s covers latent channels 64s..64s+63 in order, so each k16 block is one 16-channel scale block of the
  cache, and K's B fragments are the cache bytes as stored: 4 `ldmatrix.x4` per tile, no conversion for QK. The
  B scales are the token's 32 scale bytes.
- The same registers are the source of the E4M3 conversion for the V tile, which removes the separate cache reads.
  Thread t then holds exactly chunks 2t and 2t+1 of each 128-channel block of the V tile, and v9's swizzle stays
  conflict-free for these stores.
- Q: E2M1 codes with an E4M3 scale per 16 channels and an fp32 scale per row (FlashInfer Cake's scheme), 48
  registers instead of 72.

Predictions:
- MMAs per warp per tile: 34 -> 26; instructions per tile about 513 -> 495.
- H=128 B64 64K, same run as v9: -3% to -8%.
- Error against the BF16 cache: +28% to +55% over v9 (as simulated), about the level of FlashInfer Cake.
- Decision rule: keep as an opt-in mode if H=128 B64 64K is 3% or more faster; it never replaces the FP8 mode.

### v10 result: FP4 QK, a small gain at a large accuracy cost

239 registers, no spills; the FP8 path is unchanged by the refactor (bitwise equal to v9).

| | v9 (FP8) | v10 (FP4 QK) | predicted |
|---|---|---|---|
| H=128 B64 64K (same run, 5 rounds) | 3.027 ms | 2.877 ms (-4.5%) | -3% to -8% |
| other H=128, batch 16-64 | | -2% to -5% | |
| H=128 batch 1, H=16 | | 0% to +9% | |
| MMAs per warp per tile | 34 | 26 | 26 |
| instructions per warp per tile | 513 | 564 (+10%) | about 495 |
| tensor pipe busy | 46% | 37% | |
| stall share at the FP8 MMAs | 21% | 10% | |
| error vs the BF16 cache (9 shapes) | 0.085-0.161 | 0.096-0.201 (+3% to +55%) | +28% to +55% |

- The MMA count and the MMA stalls fell as predicted, and the time fell by the predicted amount.
- The instruction count rose instead of falling: integer work (`IMAD`, `LOP3`; 16% of stall samples together) for
  the `ldmatrix` row addresses, the runtime byte selectors and the scale extraction.
- Decision: keep as an opt-in mode (`precision=2`); the rule asked for 3%. It trades 25-55% more error on most shapes
  for 4-5% on large compute-bound ones.

### Last SM120 hypotheses (v11)

Where v9 and v10 spend instructions and stall samples, per warp and 32-token tile (H=128 B64 64K):

| group | v9 instructions | v9 stalls | v10 instructions | v10 stalls |
|---|---|---|---|---|
| conversion (`F2FP`, `HMUL2`) | 199 | 21% | 203 | 22% |
| integer and address math | 73 | 12% | 146 | 24% |
| `PRMT` | 40 | 10% | 56 | 11% |
| MMA | 34 | 21% | 26 | 12% |
| shared memory | 32 | 12% | 33 | 8% |
| FP32 (softmax and sums) | 49 | 10% | 50 | 11% |

The integer instructions in the tile loop are address math redone every tile: the 64-bit global address of each
`cp.async` (page to token byte offset) and the swizzled shared-memory offsets added to the stage base. The stage
base changes every tile, and at about 240 registers the compiler recomputes the per-lane offsets instead of keeping
them.

1. An 8-bit transposing `ldmatrix` (`.m16n16.trans.b8`, available on sm_120a) loads V's PV fragments directly.
   32 of v9's 40 `PRMT` only split `ldmatrix.trans` (b16) results into even and odd channels. Prediction: `PRMT`
   40 -> 8, its stall share 10% -> about 3%, time -3% to -6%.
2. Unrolling the tile loop by the stage count (2) makes each copy's stage base a compile-time constant, so the
   swizzled offsets can leave the loop and the global addresses are needed once per page. Prediction: integer
   instructions -40% to -60%, time -3% to -8% (more in the FP4 QK mode). Risk: holding the offsets may need registers
   the kernel does not have; checked first with ptxas.
3. The conversion is a floor on SM120. Every element needs `cvt` (E2M1 to f16), a multiply by its block scale, and
   `cvt` (to E4M3): 1.5 instructions per element, 39% of v9's instructions. A lookup-table conversion, moving the
   scale into P, or a BF16 V tile were each checked on paper and cost as much or more (or leave one CTA per SM),
   and block-scaled MMA cannot take V's scales (they run along the output channels). XQA, reading an FP8 cache,
   does not pay this. Prediction: after 1 and 2 the H=128 B64 64K time is about 2.6-2.7 ms, still above XQA's
   2.30 ms, so on SM120 this kernel does not overtake XQA at H=128.

These are the last steps for SM120; the work then moves to B200.

### v11 results

**1. Transposing 8-bit `ldmatrix` (v11a).** A probe of `ldmatrix.m16n16.x2.trans.b8` showed that lane (g, t) gets
channel g and g + 8 of tokens 4t..4t+3 (first 16 rows) and 16+4t..16+4t+3 (second 16 rows): exactly the PV B
fragments of two n8 blocks, in natural token order. The V tile now holds plain 16-channel chunks (no byte permute
on either side), P is stored in token order, and the FP8 path has its own output stores (channel 128w + 16k + 8h +
2t + {0, 1}).

| FP8 mode | v10 | v11a | predicted |
|---|---|---|---|
| `PRMT` per warp per tile | 40 | 8 | 8 |
| its share of stall samples | 10% | 2% | about 3% |
| H=128 B64 64K (same run) | 3.037 ms | 2.860 ms (-5.4%) | -3% to -6% |
| other shapes | | H=128 -3% to -6%, H=16 0% to -3% | |

Output bitwise equal to v10. In the FP4 QK mode the same change gained nothing (-4% to +2%).

**2. Tile loop unrolled by the stage count (v11b).**

| | FP8 mode v11a -> v11b | FP4 QK mode v10 -> v11b | predicted |
|---|---|---|---|
| integer instructions per warp per tile | 68 -> 33 | 146 -> 30 | -40% to -60% |
| all instructions | 469 -> 428 | 564 -> 440 | |
| H=128 B64 64K | 2.860 -> 2.855 ms (0%) | 2.936 -> 2.785 ms (-5.2%) | -3% to -8% |

The integer work fell as predicted in both modes, but the FP8 mode did not get faster: 9% fewer instructions and
no change in time, another sign that this kernel is limited by latency, not by issue slots. The FP4 QK mode, which
had twice the integer work, gained 5%. Registers 245 -> 253 (FP8 mode), no spills.

FP4 QK mode against FP8 mode in the same build: H=128 B64 64K -3.0%, other large H=128 shapes -2% to -3%, small
shapes up to +8%. That is exactly the 3% the rule asked for, at 25-55% more error; it stays an opt-in mode.

**3. The prediction held; the "floor" explanation did not.** After 1 and 2, H=128 B64 64K takes 2.79-2.86 ms in
the FP8 mode and 2.71 ms in the FP4 QK mode (predicted 2.6-2.7 ms), against 2.30 ms for XQA, so this kernel does not
overtake XQA there. The first version of this section called the conversion a floor; a review of the v11b ncu report
(FP8 mode, H=128 B64 64K) showed that it is not one:

| | v11b |
|---|---|
| issue slots busy | 38.6% (no instruction issued in 61% of cycles) |
| warps per scheduler: active / eligible | 1.96 / 0.48 |
| busiest pipes (instructions, % of peak) | tensor 49%, LSU 33%, ALU 19%, FMA 10% |
| stalls | wait 23%, short scoreboard 18%, math pipe throttle 11%, barrier 6% |

A throughput floor would show one pipe near its peak and stalls on that pipe. Here no pipe passes 49% and most
stalls wait for the result of the previous instruction, so the kernel is bound by latency. The pipe counts alone
would allow 2.72 ms x 49% = 1.33 ms (tensor), well below XQA. The gap has two parts:

- The format: XQA reads FP8 and converts nothing.
- This kernel's structure: 247 registers allow 2 CTAs (2 warps per scheduler), and each warp runs conversion -> QK
  -> barrier -> softmax -> P to shared memory -> PV in series, so the conversion sits in the middle of a chain
  whose latency nothing hides. v6b (more CTAs per SM) and v8 (converting one tile ahead in the same warps) tried to
  hide it and did not pay off; separate converter warps were not tried.

Short contexts add fixed costs on top (query quantization, pipeline fill, the combine kernel): at batch 64 the best
mode is 1.69x slower than XQA at 1K, 1.30x at 4K, 1.17x at 16K and 1.20x at 64K.

SM120 summary at H=128 B64 64K: v5 (BF16) 4.81 ms -> v6 (FP8 MMA) 3.51 -> v9 (QK by token) 3.01 -> v11 2.79-2.86
(FP8) / 2.71 (FP4 QK). At H=16 the BF16 path stays at 85% of DRAM bandwidth and is the fastest MLA decode measured on
SM120.

### v12 hypothesis: converter warps (warp specialization)

The review above leaves one structural option untried. v7 shared the conversion between two head groups but put
them in one CTA with CTA-wide barriers, and v8 moved the conversion within the same warps; both kept the conversion
inside the warps that run QK, softmax and PV. Here it moves to other warps.

Hypothesis: the latency of the MMA warps' chain limits v11. If separate converter warps produce the E4M3 tile and
hand it over through mbarriers (no CTA-wide barrier), the MMA warps' chain loses the conversion, its shared-memory
reads and the copy address math, and the schedulers can issue conversion work while the MMA warps wait. At H=128
two head groups can then share one conversion, which v7 showed halves the conversion instructions.

Where a v11b warp spends one tile (stall samples by instruction class, FP8 mode, H=128 B64 64K, about 2240 cycles
per warp per tile):

| class | instructions | stall samples | in v12's MMA warps |
|---|---|---|---|
| conversion (`F2FP`, `HMUL2`) | 46% | 20% | gone |
| MMA | 8% | 20% | stays (the tensor pipe alone needs about 530 cycles per tile) |
| integer, `MOV`, `PRMT` | 20% | 22% | about half stays (`LDSM` -> `MOV` -> MMA, address math) |
| softmax and row max (FP32, `SHFL`) | 14% | 14% | stays |
| shared loads and stores, `cp.async` | 7% | 16% | raw reads and V stores leave |
| `LDSM`, barriers, other | 5% | 8% | stays |

A second review turned this into a timing model. The MMA warps' chain shrinks from about 2240 to about 1500 cycles
per tile (530 MMA + 970 other). If both head groups take each tile from the same full barrier, they run in lock step
and queue on the tensor pipe: 2 x 530 + 970 = about 2030 cycles per pair of tiles, about 2.45 ms, level with XQA
(2.30 ms is about 950 cycles per head-group tile). If group B issues its MMAs while group A is in softmax (FA3's
ping-pong: two named barriers around the QK issue), a pair takes max(2 x 530, 530 + 970) = 1500 cycles, about
1.8 ms. v11's two independent CTAs per SM get this offset for free from their random phase; one shared ring loses
it, so the ping-pong belongs in v12 from the start.

**Experiment E1, before building v12.** v11's FP8 path with the conversion stubbed out (raw words go into QK as if
they were E4M3 and into the V tile unchanged: wrong numbers, same control flow). This approximates v12's MMA-warp
chain, with the random phase of 2 CTAs per SM standing in for the ping-pong.
- 2.0 ms or less: build v12 with the ping-pong.
- 2.4 ms or more: the chain after the conversion (barrier, row max, P exchange) is the limit, and v12 cannot pass
  XQA; that chain comes first.
- Its ptxas register count is the first answer to the 216-register budget below.

**Free fixes, measured in the same run (v11c, output bitwise equal):**
- The FP8 path's raw-stage swizzle `c ^ ((c >> 2) & 2) ^ (r & 1)` is conflict-free for the reads, but each
  `cp.async` warp instruction (8 rows x 4 chunks) hits only 4 of the 8 16-byte slots, so it takes 2x the wavefronts
  (8 `LDGSTS` x 8.4M excess wavefronts, 72% of all excess). `c ^ (r & 7)`, the FP4 QK path's swizzle, is
  conflict-free for both.
- P has a 32-byte row stride, so its `ldmatrix` and `STS.U16` are 2-way (2 x 8.4M and 4 x 2.1M excess); a 48-byte
  stride removes both.
- Prediction: small, since the LSU (33%) is not the limit: 0-3% at H=128, none at H=16.

Design (FP8 mode; H=16 stays on v11, which is memory bound at 85%):

- CTA = 2 head groups (32 heads) = 8 MMA warps (4 per group, the v11 split: QK by token, PV by channel) + 4
  converter warps; one CTA per SM, 3 warps per scheduler instead of 2. The two groups ping-pong.
- Converter warps: `cp.async` raw stages (with the conflict-free swizzle), then E2M1 x scale -> E4M3 into a ring of
  4 converted tiles in the existing V layout (16 KB + 2 KB RoPE each; 3 is the minimum). Full and empty mbarriers
  per slot; the empty barrier counts the 8 MMA warps. Shared memory about 95 KB.
- MMA warps: QK reads K with non-transposing `ldmatrix` from the converted tile (the V layout's swizzle is
  conflict-free for 8 rows of one chunk). The 8 V fragments of PV (32 registers) are loaded right after QK, before
  the row-max barrier, so their latency hides behind the softmax; v11 runs `LDSM` -> 3 `MOV` -> MMA in series
  there. The cross-warp row max uses one named barrier per head group (`bar.sync id, 128`).
- Registers with `setmaxnreg`. The CTA is launched with 384 x 168 = 64512 registers, and `inc` can only take what
  `dec` gave back, so the split is 2 x 128 x 216 + 128 x 72 = 64512 (or 208 / 88). Checked with
  a small probe on the RTX PRO 6000: with 216 / 72 and 208 / 88 a path that keeps 200 floats live after
  `inc` runs with correct results and no spills; 216 / 80 (65536, this section's first budget) blocks in `inc`
  forever.
- One CTA per SM halves the wave (376 -> 188 CTAs), and no other CTA hides the prologue (query quantization,
  pipeline fill), so v12 is a long-context mode, dispatched by length; 1K-4K stay on v11. The 1.69x gap at 1K B64
  is a separate problem (each split CTA quantizes the query again, the combine kernel) for a separate ncu run.

Predictions:

- MMA warps fit 216 registers with the V fragments loaded early, and converter warps fit 72, both without spills
  (ptxas -v, checked before any run).
- Issue slots 39% -> 50-60%, tensor pipe 49% -> 60-75%.
- H=128 B64 64K, FP8 mode: 2.86 -> 1.8-2.2 ms. Without the ping-pong the model gives 2.4-2.5 ms.
- Decision rule: keep if H=128 B64 64K improves by 10% or more in a same-run A/B and no H=128 shape from 16K up is
  more than 3% slower.

### v11c and E1 results: v12 is not built

**Free fixes (v11c).** The swizzle the review proposed for the FP8 path, `c ^ (r & 7)`, made the FP8 reads 2-way
and left the `cp.async` writes at 2x. Both reviews and this log had counted conflicts over 32 lanes; 16-byte
accesses are served 8 lanes at a time. With that model (a short script that matches ncu for the reads and
`ldmatrix`), the FP8 path got `c ^ ((c >> 2) & 2) ^ ((r & 1) * 5)` and the FP4 QK path `c ^ (r & 7) ^ ((r & 1) << 2)`.

| H=128 B64 64K | v11b | v11c |
|---|---|---|
| excess shared wavefronts | 92.8M | 67.4M (P stores and `ldmatrix`: 25M -> 0) |
| `cp.async` writes | 2x | still 2x |
| FP8 mode, same-run A/B | 2.819 ms | 2.819 ms (H=16 and H=128 shapes within 0.5%) |
| FP4 QK mode, same-run A/B | 2.697 ms | 2.648 ms (-1.3%) |

Output is bitwise equal and all tests pass. The `cp.async` writes stay at 2x with three different swizzles, so that
2x is not a bank conflict of the layout; the likely cause is that each warp instruction touches 8 global lines (8
tokens x 64 B) and the shared-memory writes follow the lines. The prediction (0-3%, the LSU is not the limit) held.

**E1: the conversion stubbed out.** The first stub put one raw word into both halves of each MMA operand pair; the
compiler then needed 572M extra `MOV` to build the pairs (2.69 ms, not a valid measurement). The second stub writes
two different words (two `LOP3` per word):

| H=128 B64 64K, FP8 mode | v11c | E1 |
|---|---|---|
| instructions executed | 1798M | 1077M (-40%) |
| conversion (`F2FP`, `HMUL2`) | 833M | 11M |
| registers | 245 | 254 |
| issue slots busy | 38% | 26% |
| eligible warps per scheduler | 0.48 | 0.31 |
| tensor pipe | 48.5% | 55.7% |
| stall samples at MMA (math throttle) | 20% (12%) | 31% (14%) |
| kernel time, ncu | 2.96 ms | 2.56 ms (-13%) |
| same-run A/B (CUPTI) | 2.806 ms | 2.484 ms (-11.5%) |

Other H=128 shapes: -3% (B16 4K) to -10% (B16 64K). With XQA in the same process (H=128 batch 64, median of 3
alternating rounds; `docs/img/sm120-gap`):

| tokens | v11c FP8 | v11c FP4 QK | E1 | XQA FP8 | E1 / XQA |
|---|---|---|---|---|---|
| 1K | 0.137 ms | 0.134 | 0.133 | 0.081 | 1.64 |
| 4K | 0.282 | 0.270 | 0.261 | 0.215 | 1.21 |
| 16K | 0.854 | 0.816 | 0.769 | 0.673 | 1.14 |
| 64K | 2.889 | 2.738 | 2.548 | 2.328 | 1.09 |

At 1K the conversion costs 3% and the gap stays at 1.64x, so the short-context loss is a fixed cost per CTA.

- Removing all of the conversion, 40% of the instructions, saves 11-13%. Its 20% of the stall samples was not 20%
  of the time: without it the warps wait at the MMAs (bursts of QK and PV, math throttle) and in the integer and
  softmax chain instead, and fewer warps are ready (0.31 per scheduler).
- 2.48 ms is above the 2.4 ms line set before the run. Even a free conversion would leave this kernel at XQA's time
  (2.30 ms in another run; run-to-run variation is up to 8%). v12 can at best remove a little more than E1 (the
  copy, the raw reads and the V stores leave the MMA warps too), about 2.2-2.3 ms, with the one-CTA-per-SM risks
  on top. By the decision rule, v12 is not built.
- The review's model counted the conversion's stall share (about 450 of 2244 cycles) as time that would be freed;
  E1 frees about 290. Stall shares say where warps wait, not how much time disappears when the work does.
- What remains at 2.48 ms is the rest of the chain. The tensor pipe at 56% puts its own floor at 1.43 ms; the gap to
  it is the MMA bursts, the `LDSM` -> `MOV` -> MMA dependencies, the cross-warp row max and the softmax. Changing
  that is a different kernel design, not a step from v11.

SM120 ends at v11c: the FP8 mode at 2.82 ms and the FP4 QK mode at 2.65 ms on H=128 B64 64K, against 2.30 ms for XQA.

### A new SM120 structure? Pre-tests E3 and E4

After E1, two redesigns were proposed for the H=128 gap: (a) v13, converter warps plus software pipelining inside the
MMA warps (QK of tile i+1 issued before the softmax of tile i) and a ping-pong of two head groups; (b) a finer split,
8 warps of 16 heads x 256 channels, to fit 4 warps per scheduler in 128 registers. Both bet that the remaining time is
latency the scheduler could hide. On SM120, `mma.sync` is issued in order (unlike Hopper's `wgmma`), so what (a) can
hide is mainly the latency around the two per-tile barriers (row-max exchange, P exchange); (b) adds warps but brings
back a cross-warp score sum (removed in v9 for -14%) and needs a register budget v8 suggests is out of reach.

Before building either, two more stubs of v11c measure what the barriers cost (wrong numbers, same work otherwise):

- E3: no cross-warp row-max exchange; each warp uses its own maximum (one `STS` and two `LDS.128` fewer per tile).
- E4: no barrier between the P stores and PV (PV may read a stale P).
- E34: both; E134: both and no conversion (E1).

Decision rule, written before the run (H=128 B64 64K, same-run A/B against v11c):

- E34 at least 20% faster: the barriers are the main remaining limit, and (a) is worth building.
- E34 less than 10% faster: the barriers are not the limit either; with the conversion at 12% (E1), no single part of
  the chain is worth a new kernel, and the SM120 work stops at v11c.
- In between: (a) only if E134 lands below XQA's time (2.33 ms in the same-process run).

### E3 and E4 results: SM120 stops at v11c

Same-run A/B against v11c, FP8 mode (time of the stub, and how much faster than v11c):

| H=128 | E3 (no row-max exchange) | E4 (no P barrier) | E34 (both) | E134 (both, no conversion) |
|---|---|---|---|---|
| B16 16K | 0.238 ms (-3%) | 0.243 (-1%) | 0.236 (-4%) | 0.222 (-10%) |
| B16 64K | 0.789 (-4%) | 0.811 (-1%) | 0.780 (-5%) | 0.712 (-13%) |
| B64 16K | 0.811 (-4%) | 0.833 (-1%) | 0.802 (-5%) | 0.737 (-13%) |
| B64 64K | 2.760 (-4%) | 2.807 (-1%) | 2.700 (-6%) | 2.444 (-14%) |

Registers stay at 245 (254 with the conversion stubbed).

- Both per-tile barriers together cost 4-6%, below the 10% line written before the run. With the conversion
  (E1, 12%) they add up to 14%: even a kernel with neither would take 2.44 ms at H=128 B64 64K, still above XQA's
  2.33 ms. Hiding barrier latency, the main thing software pipelining with in-order `mma.sync` could do, has little to
  win, so v13 is not built.
- What remains is the MMA work itself as this structure issues it: per scheduler, one warp from each of the two CTAs,
  each alternating a burst of 18 QK and 16 PV MMAs (math throttle) with non-MMA work, while the tensor pipe is busy 49-56%
  of the time. Filling it needs more independent warps per scheduler, which the register budget does not allow; the
  finer-split proposal (b) is the only design aimed at that, and it was not tested: its 128-register budget is the
  first thing that would have to be shown.
- SM120 summary at H=128 B64 64K: v11c 2.82-2.89 ms (FP8) / 2.65-2.74 ms (FP4 QK) against XQA 2.30-2.33 ms. The
  conversion explains 12% of v11c's time, the two per-tile barriers 5%, and the rest is the MMA-issue structure.

## Part 3: B200 (in progress)

### Measurements first

FlashInfer 0.7.4rc1, CUDA 13.4.2, GPU kernel time in ms (page 64, q_len 1):

| shape | this kernel (BF16 mode) | FP8 mode | Cake (NVFP4) | trtllm-gen FP8 | trtllm-gen BF16 | fa2 | cuTile (BF16) |
|---|---|---|---|---|---|---|---|
| H=16 B64 64K | 0.766 | 1.284 | 0.320 | 0.387 | 0.839 | 2.47 | 29.4 |
| H=128 B16 64K | 1.514 | 2.235 | 0.303 | 0.149 | 0.256 | 1.52 | 1.90 |
| H=128 B64 64K | 5.880 | 8.793 | 0.549 | 0.584 | 1.061 | 4.94 | 7.33 |

- Peaks: DRAM 7.3 TB/s; cuBLAS BF16 1.5 PFLOPS and FP8 3.1 PFLOPS (tcgen05); `mma.sync` BF16 555 and E4M3 2079 TFLOPS.
- The `mma.sync` kernel is 2.4-10.7x slower than Cake; its FP8 mode is slower than its BF16 mode here.
- Cake leaves room: 63% of DRAM bandwidth at H=16 B64 64K (the NVFP4 floor is 0.202 ms), 2x slower than
  trtllm-gen FP8 at H=128 B16, and its query is FP4.
- Cake under ncu (B200, `kernel_cake_mla_nvfp4_paged_decode_*`, itself a CuTe DSL kernel):

  | | H=16 B64 64K | H=128 B16 64K |
  |---|---|---|
  | grid x block, smem | 128 x 640 threads, 203 KB | 128 x 512 threads, 208 KB |
  | waves (148 SMs, 1 CTA/SM) | 0.86 | 0.86 |
  | DRAM throughput | 55% | 28% |
  | tensor pipe active | 9% | 41% |
  | shared memory (LSU / tensor-core wavefronts, % of peak) | 56% / 35% | 44% / 25% |
  | top stall | long scoreboard 55% (mbarrier waits) | long scoreboard 38% |

  - Its MMA op counts equal 2 x B x H x tokens x dims with H = 16 or 128 exactly, so Cake also puts heads on N (no
    padding). They split as: QK over the 512 latent channels in block-scaled FP4 x FP4 (`kind::mxf4nvf4`, no K
    conversion, hence the FP4 query); RoPE QK and PV in `kind::f8f6f4`, so V is converted to E4M3 (QMUL4 + F2FP,
    12% of instructions), a rounding step on top of the FP4 query; 1.16M `tcgen05.st` (converted operands into TMEM).
  - Both shapes launch 128 CTAs on 148 SMs, so 14% of the SMs idle for the whole kernel. At H=16 the DRAM sits at 55%
    with warps mostly waiting on barriers: the pipeline does not keep enough bytes in flight per SM.
- The B200 kernel is written in CuTe DSL (nvidia-cutlass-dsl 4.8.1), with FlashInfer's CuTe DSL MLA decode (2-CTA
  M=128 tiles, 12 warps in roles, O in TMEM, P through shared memory, BF16 or FP8 KV only) as the reference design.

### v0 scope: a plain tcgen05 kernel

First plan: CTAs of 64 heads as the MMA M (a 1-CTA M=64 tile packs a 64 x 512 fp32 O into 256 of the 512 TMEM
columns), H=16 padded to 64 rows, BF16 operands. Two findings changed it before any code was written:

- CUTLASS 4.8.1 ships a mixed-input decode example (`mixed_input_fmha_decode.py`: int4/int8 KV, a converter
  warpgroup, tcgen05). It swaps A and B: tokens are the M (128 per tile) and heads the N (8-32). For MLA this removes
  the padding (H=16 would waste 75% of an M=64 tile), and with M=128 a TMEM lane is a token, so a softmax thread owns
  one token row and only reduces over heads across lanes. This is also the structure later versions grow into.
- FP16 instead of BF16 operands. e2m1 x e4m3 has at most 6 significant bits and lies in [2^-10, 2688], so it is exact
  and normal in FP16; a BF16 query converts exactly; P keeps 3 more mantissa bits. The error can only go down
  against the BF16 mode, and the conversion is `cvt.rn.f16x2.e2m1x2` + `mul.rn.f16x2`, one instruction per element.

v0 as written (`nvfp4_mla/sm100.py`):

- CTA = 1 request x 16 or 32 heads x 1 KV split, 4 warps, `cta_group::1`, no cluster. H=128 takes 4 CTAs per request,
  each reading the whole KV (the 3 extra reads should mostly hit L2).
- Per 128-token tile, in order: every thread converts one token (page lookup, 352 B of loads) into a 128 x 576 FP16
  tile in shared memory (144 KB); one warp issues QK (S^T = K Q^T, 36 MMAs of K=16) into TMEM; all threads load S^T
  (a thread = a token), reduce the column (head) max with warp reductions and shared-memory atomics, write P^T to
  shared memory and rescale O^T in TMEM; one warp issues PV (O^T += V^T P^T, 4 channel tiles x 8 MMAs); all wait.
- The FP16 tile is the K operand of QK (K-major) and the V^T operand of PV (MN-major) at the same bytes: token groups
  of 8 rows are innermost and 64-channel blocks outermost in both views.
- TMEM: S^T N columns + O^T 4N columns. Shared memory at N=32: 144 KB tile + 36 KB Q + 8 KB P, one CTA per SM.
- Split-KV partials go to the existing combine kernel, now also exposed as `combine` in the extension.

Predictions, written before the first run:

- Error at or below the SM120 BF16 mode against the dequantized reference.
- Much slower than Cake: no step overlaps another, and one CTA per SM. Beyond that, a bound that overlap alone cannot
  remove: per token the tile is written once (1152 B) and read by both MMAs (1152 + 1024 B), about 3.3 KB of
  shared-memory traffic per 352 B of HBM traffic (9.4x), while shared memory (128 B/clk/SM, assumed to also bound
  tcgen05 operand reads) is only ~5x faster than HBM per SM (7.3 TB/s / 148 SMs / 1.93 GHz = 25 B/clk). So even a
  perfectly pipelined v0 design would sit near 2x the HBM floor at H=16. If ncu shows shared memory near its peak,
  the next step is to keep the converted operand out of shared memory (TMEM, as the CUTLASS example does).

### v0 result

Correct on the first B200 run. Error against the dequantized reference (5 cases including
masked tails, page 16/32/64, explicit and automatic splits):

| | v0 (FP16 operands) | SM120 path (BF16 mode) |
|---|---|---|
| rel. L2 vs dequantized cache | 0.0016-0.0017 | 0.0018-0.0022 |

GPU kernel time in ms (B200, page 64, % of 7.3 TB/s in parentheses):

| shape | v0 | SM120 path (`mma.sync`) | Cake | trtllm-gen FP8 |
|---|---|---|---|---|
| H=16 B64 64K | 1.348 (15%) | 0.765 | 0.319 | 0.386 |
| H=128 B16 64K | 1.581 | 1.511 | 0.302 | 0.149 |
| H=128 B64 64K | 6.140 | 5.874 | 0.545 | 0.574 |

ncu:

| | H=16 B64 64K | H=128 B16 64K |
|---|---|---|
| grid, warps per SM | 128 CTAs (0.86 waves), 4 | 128 CTAs (0.86 waves), 4 |
| registers, local memory | 248 | 255, 2.2M local loads (spills) |
| DRAM throughput, bytes read | 13.5%, 1.48 GB | 3.0%, 0.37 GB (L2 absorbs the 4 head-group reads) |
| tensor pipe active | 5.7% | 9.7% |
| issue slots busy | 14% | 15% |
| shared memory: LSU / tensor-core wavefronts | 13.8% / 22.0% | 12.4% / 20.9% |
| stalls | long scoreboard 57%: 24% in the two mbarrier wait loops (MMA done), ~32% on the first use of the tile's global loads | 55%: 23% and ~27% |

- Predictions: the error held (below the BF16 mode). "Much slower than Cake" held: 4.2x at H=16, 5-11x at H=128.
  Not predicted: v0 is also slower than the `mma.sync` path, which keeps several CTAs per SM and a cp.async pipeline.
- Shared memory is not what limits v0 now (14% + 22%). One 128-token tile takes 5.2 us (H=16; 256 tiles per CTA)
  and 6.1 us (H=128): loads, conversion, QK, softmax and PV run one after another with 4 warps per SM, and the
  stalls split between waiting for the tile's bytes and waiting for the MMAs.
- The shared-memory prediction holds as a ceiling. The traffic per token is fixed by the data path, so at the H=16
  HBM floor (0.202 ms, 6.6x shorter) the tensor-core reads alone would need 146% of their peak. The lowest time this
  data path allows at H=16 is 1.34 x 0.22 = 0.30 ms if those reads have their own port, 1.34 x 0.36 = 0.48 ms if they
  share it with the LSU, i.e. Cake's time at best, however well the steps overlap.
- The 144 KB FP16 tile also cannot be double-buffered (228 KB of shared memory per SM), so a pipeline on this data
  path could only overlap the raw loads, not the conversion of the next tile with the MMAs of this one.

### v1 options

Measurement first (`benchmarks/precision.py`, CPU, random data): the query split into residual terms, each term
quantized like Cake's query (per-row FP32 scale, per-16 E4M3 scales, E2M1), so QK can run on the raw FP4 cache with
block-scaled FP4 x FP4 MMAs and the terms accumulate into the same S. Error against attention over the dequantized
cache (the kernel's own error; against the unquantized cache all rows below except the 1-term one are within 1% of
BF16, because the cache format's own error dominates there):

| query | H=16 S=4K | H=128 S=4K | H=16 S=16K |
|---|---|---|---|
| BF16 (SM120 BF16 mode) | 0.0025 | 0.0023 | 0.0020 |
| FP8 (SM120 FP8 mode) | 0.0351 | 0.0348 | 0.0350 |
| 1 FP4 term (Cake) | 0.1291 | 0.1231 | 0.1062 |
| 2 FP4 terms, 2 E4M3 RoPE terms | 0.0118 | 0.0109 | 0.0094 |
| 3 FP4 terms, 2 E4M3 RoPE terms | 0.0014 | 0.0013 | 0.0010 |

Three FP4 terms (the query's own error 0.08%, BF16 0.17%) put QK below the BF16 mode. On tcgen05 `kind::mxf4nvf4` runs
at 4x the FP16 rate, so three terms cost 0.75x of v0's FP16 QK, and the K side is never converted.

- A. Warp-specialize v0 as it is (load warp, converter warps, MMA warp, softmax warps; one FP16 tile in shared
  memory). Smallest step, but the ceiling above (0.30-0.48 ms at H=16) means it cannot pass Cake.
- B. Change the data path: QK over the 512 latent channels as `kind::mxf4nvf4` on the raw cache bytes with the
  3-term query, RoPE QK as 2 E4M3 terms (`kind::f8f6f4`) on the raw kpe; only V is converted, to FP16 (still exact),
  and written to TMEM as the A operand of PV (lanes = channels, the CUTLASS example's and Cake's placement). Shared
  memory per token falls from ~3.3 KB to ~1 KB (raw bytes in, QK reads, V-converter reads), which puts its ceiling
  below the HBM floor. Then warp specialization on top. Costs: block-scale factors through TMEM (`tcgen05.cp`), a
  transposing V conversion, and a TMEM budget of V 256 + O^T 4N + S^T N columns (416 at N=32, so V can only be
  double-buffered in 64-token halves).
- C. B with Cake's 1-term query: fastest QK, but 50x the error. At most a later speed option, like `precision=2` on
  SM120.

### Choice for v1: option B, in two steps

- **v1a (this step): QK on the raw cache.** The latent part runs as `kind::mxf4nvf4` on the cache bytes as stored,
  with the cache's block scales as the A scale factors. A small CUDA kernel before the decode writes the query as
  NVFP4 terms that share one FP32 scale per head, so all terms accumulate into one S. The RoPE part converts kpe to
  FP16 (exact) and runs as `kind::f16` into a second accumulator, combined with the latent one in the softmax. V stays
  as in v0 (FP16 in shared memory), and so does the rest of the structure, so this step measures the data path alone.
- **v1b: V into TMEM.** V converted to FP16 straight into TMEM as the A operand of PV (lanes = channels); the
  transpose comes from a transposing 8-bit `ldmatrix` on the raw bytes plus one `PRMT` per output word.
- **v2: warp specialization** (loads, conversion, MMA, softmax in separate warps) and a multi-stage load pipeline.

How many query terms. The table above used FP32 queries, so its error includes the query's own BF16 rounding. The
kernels receive BF16 queries, and the reference uses the same BF16 query, so the question is what the terms add on
top of that (CPU, against attention over the dequantized cache, FP16 P; "outliers" multiplies each channel by
`exp(N(0, 1.5))`, as activations with outlier channels have):

| query | H=16 S=4K | H=16 S=4K outliers | H=128 S=4K | H=128 S=4K outliers | H=16 S=16K | H=16 S=16K outliers |
|---|---|---|---|---|---|---|
| FP16 (v0) | 0.0001 | 0.0000 | 0.0001 | 0.0000 | 0.0001 | 0.0001 |
| 3 FP4 terms, one shared scale | 0.0010 | 0.0038 | 0.0011 | 0.0022 | 0.0009 | 0.0032 |
| 4 FP4 terms, one shared scale | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0001 | 0.0002 |

The kernels' measured error floor is their BF16 output (about 0.0016), so 3 terms with outlier channels would more
than double the error, and 4 terms add nothing measurable. Four terms are 32 `mxf4nvf4` MMAs per tile at 4x the FP16
rate, the same tensor time as v0's 32 FP16 MMAs over the latent channels.

v1a predictions:

- Error against the dequantized cache within 10% of v0's (0.0016-0.0017 on the five check cases).
- Tensor-core shared-memory reads per token: QK 1152 -> 384 B (256 B FP4 + 128 B FP16 RoPE), with PV's 1024 B
  unchanged: 2176 -> 1408 B (-35%).
- Time within 15% of v0 (1.348 ms at H=16 B64 64K, 1.581 ms at H=128 B16 64K). v0 is bound by its serial chain, this
  step changes only the QK part of it, and the query kernel adds one launch.

### v1a result: correct, 18% slower, and the K operand read once per term

The first run was wrong (error 0.85). Dumping the first tile's scores showed the RoPE part exact and the latent part
off. Controlled inputs narrowed it down: with all scales 1 the FP4 data, the term indexing and the cache scales
(SFA) were exact, and only random query scales (SFB) broke it. A least-squares fit of the scale each (head, block)
actually received (128 tokens, 32 unknowns per head) showed head h correct and blocks 0-7 repeated: the per-term
SFB offset in TMEM was added to a pointer already recast to 8-bit elements, so it moved a quarter of a column per
column and the four terms overwrote each other. With the offset added on the 32-bit pointer:

| | v0 | v1a |
|---|---|---|
| error vs dequantized cache (5 cases) | 0.0016-0.0017 | 0.0017 |

Same process on B200, ms (v0 with 16 heads per CTA as well, since v1a fits only 16):

| shape | v0, 32 heads/CTA | v0, 16 heads/CTA | v1a, 16 heads/CTA | Cake |
|---|---|---|---|---|
| H=16 B64 64K | 1.344 | 1.344 | 1.589 (+18%) | 0.319 |
| H=128 B16 64K | 1.585 | 2.585 | 3.060 (+18%) | 0.311 |
| H=128 B64 64K | 6.170 | 10.160 | 11.988 (+18%) | 0.542 |
| H=16 B64 4K | 0.112 | 0.112 | 0.134 (+20%) | 0.034 |
| H=128 B64 4K | 0.480 | 0.683 | 0.802 (+17%) | 0.049 |

ncu at H=16 B64 64K:

| | v0 | v1a |
|---|---|---|
| tensor-core shared wavefronts | 80.2M | 81.3M |
| LSU shared wavefronts | 50.4M | 71.4M (+41%) |
| instructions | 174M | 226M (+30%): shifts, `LOP3`, `VIADD` for the extra stores |
| time in the two MMA wait loops | 0.32 ms | 0.32 ms |
| registers | 248 | 202 |

- The error held. The time missed (+18% against "within 15%"), and the tensor-core reads did not fall at all
  (predicted -35%). The prediction counted the FP4 tile once, but each of the 4 terms is its own MMA and reads the
  whole K block again: 4 x 256 B + 128 B RoPE = 1152 B per token, exactly v0's. The query terms should sit along N
  instead (N = 4 x 16 = 64, one MMA per K block, the 4 term columns of a head summed in the softmax), which reads K
  once.
- The MMAs did not get slower; the extra time is the conversion phase. Each thread now stores the raw tile, its
  scale words, the RoPE tile and the V tile separately (25% more stores, +41% LSU wavefronts), with index arithmetic
  for each. In this serial structure that lands directly on the chain.
- 16 vs 32 heads per CTA matters more than the data path at H=128: v0 with 16 heads is 1.6x slower (twice the head
  groups, each converting the same tiles).

### v1a' hypothesis: the query terms along N

- Change: the B operand is the 4 terms stacked (N = 4 x 16 = 64 rows: term t, head h at row 16t + h, each row with its
  own block scales), one `mxf4nvf4` MMA per K block (8 per tile instead of 32), a 64-column latent accumulator, and the
  4 columns of a head summed after the TMEM load.
- Predictions (H=16 B64 64K): tensor-core shared wavefronts 81.3M -> about 53M (QK 1152 -> 384 B per token, PV 1024
  B unchanged); time -2% to -5%, since the MMA waits are 20% of the time and the conversion phase does not change;
  error unchanged (the same products, summed in another order).

### v1a' result: K read once

| H=16 B64 64K | v1a | v1a' | predicted |
|---|---|---|---|
| tensor-core shared wavefronts | 81.3M | 56.1M | about 53M |
| time (ncu) | 1.585 ms | 1.422 ms (-10%) | -2% to -5% |
| error vs dequantized cache | 0.0017 | 0.0017 | unchanged |

Same process, ms: H=16 B64 64K 1.426 (v0 with 16 heads/CTA 1.346), H=128 B16 64K 2.707 (2.593), H=128 B64 64K 10.534
(10.173), H=16 B64 4K 0.119 (0.112), H=128 B64 4K 0.708 (0.684). The wavefronts fell as predicted and the time fell
more: the two MMA wait loops went from 0.32 to 0.24 ms, and warp 0 now issues 8 MMAs instead of 32 (each with its own
scale-factor setup) before the others can go on. v1a' is still 3-6% behind v0: the extra stores of the conversion phase remain. They go away with the next step, which writes V
straight into TMEM instead of a 128 KB FP16 tile, and which also frees the shared memory for 32 heads per CTA.

One more DSL limit found on the way: a 32x32b TMEM load with 64 columns (`Ld32x32bOp(Repetition(64))`) crashes the
CuTe DSL 4.8.1 compiler with a segfault and no message; a staged compile-only probe isolated it, and the 64-column
accumulator is read as four 16-column loads.

### v1b hypothesis: V straight into TMEM

- Change: the V tile no longer goes through shared memory as FP16. After the QK MMAs are issued, every warp reads the
  raw FP4 tile with a transposing 8-bit `ldmatrix` (16 tokens x 16 bytes: a thread gets 4 tokens of one byte, i.e. of
  two channels), converts with `cvt.e2m1x2`, pairs tokens with one `PRMT` per word, multiplies by the two tokens'
  block scales as one f16x2, and writes the result with `tcgen05.st.16x256b` as the A operand of PV (lanes =
  channels, 2 tokens per column, 256 columns for 512 channels). Lane order within 16 lanes is even channels then odd
  ones; the epilogue maps it back. Block scales get a second, block-major copy in shared memory so 4 tokens' scales
  are one 32-bit load.
- Predictions (H=16 B64 64K, 16 heads per CTA): tensor-core shared wavefronts 56M -> about 15M (PV no longer reads
  shared memory); LSU wavefronts down by the 64 16-byte V stores per thread per tile, partly paid back by 32
  `ldmatrix` and 64 scale loads; the V conversion overlaps the QK MMAs, which it could not before. Time -10% to
  -20% against v1a'. TMEM: 464 columns at 16 heads (one CTA per SM, as now); 32 heads per CTA would need 608 and does
  not fit without reusing columns.

### v1b result: faster than v0 for the first time

Partial tiles first came out wrong (output 0 below 112 valid tokens in the last tile, 0.15-0.85 error above). Three
store-side fixes (warp reconvergence before the `.sync.aligned` instructions, scale stores outside the valid-token
branch, no branch at all) left the error identical to four digits, so the bug was deterministic and elsewhere. A dump of
the epilogue showed the softmax sum as NaN and the column maximum as -inf: the V conversion loop named its ldmatrix row
`row`, which in a CuTe DSL kernel (plain Python scope) overwrote the thread's own token row used by the mask below it.
After the loop every thread had row 112 + lane % 16, so short tiles were masked entirely. Renamed; the branch-free token
load stayed (rows past the end read the last valid token and are masked to zero), as it removes a divergent branch.

| | v1a' | v1b | predicted |
|---|---|---|---|
| error vs dequantized cache (5 cases, and L = 1-300) | 0.0016-0.0017 | 0.0016-0.0017 | |
| tensor-core shared wavefronts (H=16 B64 64K) | 56.1M | 22.5M | about 15M |
| LSU shared wavefronts | 70.7M | 58.3M | down |
| instructions | 222M | 244M | |
| registers | 211 | 250 | |

Same process, ms (16 heads per CTA for v1a' and v1b):

| shape | v0, 32 heads/CTA | v0, 16 heads/CTA | v1a' | v1b | Cake |
|---|---|---|---|---|---|
| H=16 B64 64K | 1.345 | 1.346 | 1.444 | 1.275 (-12%) | 0.319 |
| H=128 B16 64K | 1.586 | 2.586 | 2.769 | 2.427 (-12%) | 0.309 |
| H=128 B64 64K | 6.169 | 10.175 | 10.763 | 9.473 (-12%) | 0.544 |
| H=16 B64 4K | 0.112 | 0.112 | 0.122 | 0.111 (-9%) | 0.034 |
| H=128 B64 4K | 0.480 | 0.683 | 0.726 | 0.642 (-12%) | 0.049 |

- The time fell as predicted (-9% to -12% against -10% to -20%); v1b is the first version faster than v0 at the same
  heads per CTA (-5% to -7%). The tensor-core reads fell less than predicted: the estimate left out P^T and the
  query, which PV and QK still read from shared memory every tile.
- The data path is now the one the plan aimed at: K read once as FP4 by the MMA, V converted once into TMEM, about
  1 KB of shared-memory traffic per token. What limits v1b is still the serial structure: one warpgroup, issue slots
  21%, tensor pipe 6%, and 47% of the stall samples on long scoreboards (the first use of each global load and the
  MMA waits).
- At H=128 the 16 heads per CTA cost more than the data path saves (v0 with 32 heads: 1.586 ms, v1b: 2.427 ms at
  B16 64K): every tile is loaded and converted by 8 CTAs per request instead of 4. 32 heads need 608 TMEM columns
  (512 exist), so the next version has to reuse columns or share the converted V between head groups.
