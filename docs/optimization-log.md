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

### v2 hypothesis: warp specialization

v1b does one 128-token tile in about 5 us, one step after another. At the HBM floor an SM has to finish a tile every
0.92 us (45 KB at 7.3 TB/s / 148 SMs). The steps themselves are short (estimates per tile and SM: V conversion about
0.5 us of issue, softmax about 0.3 us, the MMAs well under that), so the hypothesis is that running them in separate
warps on different tiles brings the tile period down to the slowest single role.

Design (v2.0: v1b's data path, new structure; one CTA per SM, 16 heads per CTA, 416 threads):

| role | warps | per tile |
|---|---|---|
| load | 0-3 | thread = token: global loads, raw FP4 tile, scale factors (MMA layout and a block-major copy), RoPE to FP16; 3-stage ring |
| convert | 4-7 | V from the raw tile into TMEM (as v1b) |
| softmax | 8-11 | S from TMEM, masks, row maximum (named barrier of these 4 warps only, maxima in a parity-double-buffered slot array), P to shared memory, O rescale |
| MMA | 12 | QK of tile i+1 issued before PV of tile i, so QK overlaps the softmax |

Pipelines (mbarriers): load -> MMA and load -> convert (two pipelines over the same 3 stages; the load warps wait for
both to release a stage), MMA -> softmax (S ready), softmax -> MMA (S read, so QK(i+1) may overwrite it), convert ->
MMA (V ready), MMA -> convert (PV done, V buffer free), softmax -> MMA (P ready, O rescaled), MMA -> softmax (PV done).
TMEM stays at v1b's 464 columns: one S, one V, one O buffer. S is free as soon as the softmax has loaded it, and the
V buffer is the only real serialization (convert(i+1) waits for PV(i)).

Shared memory: 3 stages x 56 KB (FP4 tile 32 KB, two scale copies 8 KB, RoPE FP16 16 KB) + query, P, maxima: about
195 KB.

Predictions (H=16 B64 64K, against v1b's 1.275 ms and Cake's 0.319 ms):

- The load warps hold one tile's loads in flight (22 x 16 B per thread, then the stores), about 45 KB per 1.0-1.3 us,
  i.e. 35-45 GB/s per SM against the 49 GB/s the HBM floor needs. They should set the period, at 1.0-1.3 us per tile.
- Time 0.30-0.45 ms (3-4x faster than v1b); DRAM throughput 45-65%; issue slots and tensor pipe both well above v1b's
  21% and 6%.
- If the time is well above that range, the stall breakdown per role decides: the convert-PV loop on the single V
  buffer (V in two 64-token halves would split it), or the load warps (cp.async into the ring, v2.1).
- Error unchanged.

### v2 result: correct at once, but 3x slower than predicted

| H=16 B64 64K | v1b | v2 | predicted |
|---|---|---|---|
| time | 1.279 ms | 1.086 ms (-15%) | 0.30-0.45 ms |
| DRAM read throughput | | 16% | 45-65% |
| issue slots / tensor pipe | 21% / 6% | 29% / 7% | well above v1b |
| registers | 250 | 128 (7.4M local loads: spills) | |

H=128 B16 64K: 2.436 -> 2.023 ms. The error is unchanged (0.0016-0.0017, all lengths 1-300 and the 5 check cases).
The structure worked the first time (no deadlock), but the tile period only went from 5.0 to 4.2 us.

Which role sets the period: each role's work removed in turn (stubs: wrong numbers, every pipeline signal kept), same
process, H=16 B64 64K:

| removed | time | | removed | time |
|---|---|---|---|---|
| nothing (v2) | 1.077 ms | | load + convert | 0.343 |
| load | 0.524 (-51%) | | load + convert + PV + O rescale | 0.247 |
| convert | 0.857 (-20%) | | PV MMAs | 1.103 (no change) |
| O rescale | 0.899 (-17%) | | QK MMAs | 1.095 (no change) |

H=128 B16 64K gives the same order (load -49%, convert -19%, O rescale -12%).

- The load warps set the period, as predicted, but at 4.2 us per tile instead of 1.0-1.3 us. Each thread loads 22
  x 16 B and stores them; ptxas capped the kernel at 128 registers (416 threads), so a thread cannot hold its 22
  loads in flight and pays several global-memory round trips per tile.
- Even with almost nothing left (load, convert, PV and rescale removed), the kernel takes 0.247 ms, about 1 us per
  tile: QK(i+1) cannot start before the softmax has read S(i) out of the single S buffer, so QK, the TMEM load and the
  barrier hops form a chain once per tile. That is already above the HBM floor (0.202 ms).

### v2.1 hypothesis: cp.async for the raw bytes

- Change: the load warps copy each token's raw bytes (ckv, scales, RoPE) with `cp.async` into a 3-stage ring, keeping
  two tiles in flight, and only then turn the scales into the two layouts and the RoPE bytes into FP16 from shared
  memory. No global load goes through registers; rows past the end are zero-filled by the copy itself.
- Shared memory per stage: 32 KB FP4 + 4 KB raw scales + 8 KB raw RoPE + 8 KB scale layouts + 16 KB RoPE FP16 = 68 KB;
  3 stages + query and P = about 230 KB, over the 227 KB limit, so this version uses 2 stages (one tile in flight
  while one is processed) and leaves the RoPE path for later.
- Prediction: the load role drops to about the cost of the "load removed" stub plus the scale and RoPE processing:
  0.55-0.70 ms at H=16 B64 64K. DRAM throughput 25-35%. The next limits are then the conversion (-20% when removed),
  the O rescale (-17%) and the QK-softmax chain (0.247 ms floor).

### v2.1 result: no gain; two stages are one tile in flight

Correct (all lengths, the 5 cases), but 1.122 ms at H=16 B64 64K (v2: 1.077, predicted 0.55-0.70). The same stubs:

| removed (H=16 B64 64K) | v2 (3 stages, register loads) | v2.1 (2 stages, cp.async) |
|---|---|---|
| nothing | 1.077 ms | 1.122 ms |
| load | 0.524 | 0.826 |
| convert | 0.857 | 0.771 |
| load + convert | 0.343 | 0.434 |
| load + convert + PV + rescale | 0.247 | 0.365 |

- The prediction assumed the copies of one tile would overlap the processing of the previous one. With 2 stages the
  load warps can only start tile i + 1 once the consumers have released tile i - 1, so at most one tile's copies are in
  flight and each iteration still waits about one memory latency. The loads moved out of registers, but not out of the
  critical path; even the skeleton (everything removed) got slower, 0.247 -> 0.365 ms, for the same reason.
- No single role explains the time: removing any one of load or convert leaves 0.77-0.83 ms, both 0.43 ms. The roles
  wait on each other through three loops that each close once per tile: the load ring (stages), V (convert(i + 1) waits
  for PV(i)), and S (QK(i + 1) waits for the softmax to read S(i)). Each loop costs one or more barrier round trips per
  tile on top of the work.

Next: more stages need less shared memory per stage. The FP16 RoPE tile (16 KB per stage, plus the raw copy it is
made from) is the part that can go: RoPE as E4M3 x E4M3 (`kind::f8f6f4`) on the raw kpe bytes, with the query's RoPE
part as 3 E4M3 terms (simulated error 0.0001-0.0003, the same as exact FP16). That gives 52 KB per stage, 3 stages in
185 KB.

### Register spills: fixed, and not the limit

v2 and v2.1 ran at 128 registers with spills (7.4M local loads at H=16 B64 64K). Registers are handed out per
warpgroup, so the 13th warp (the MMA warp) cost a whole warpgroup: 416 threads were budgeted as 512, 65536 / 512 = 128.
The fix is the one CUTLASS's FMHA uses: 512 threads (the MMA warp's warpgroup gets 3 idle warps) and `setmaxnreg` per
role. Two things had to be found first:

- ptxas silently dropped every `setmaxnreg` (no `USETMAXREG` in the SASS) because the DSL only emits `reqntid`; with
  `min_blocks_per_mp=1` on the launch the register count at entry is fixed and the instructions stay (the same rule
  the SM120 probe hit).
- The fragments, TMEM tensors and copy partitions were built once for all roles before the role branches, so they were
  live across the reassignment; each role now builds its own after its `setmaxnreg`.

Budgets tried (load / convert / softmax / MMA warpgroup), same process:

| budget | H=16 B64 64K | H=128 B16 64K |
|---|---|---|
| v2.1, 416 threads, 128 each | 1.122 ms | 2.245 ms |
| 80 / 168 / 200 / 64 | 1.160 | 2.302 |
| 72 / 152 / 176 / 112 (kept) | 1.131 | 2.235 |
| 64 / 152 / 168 / 128 | 1.152 | 2.276 |
| 48 / 152 / 208 / 104 | 1.241 | 2.455 |

With 72 / 152 / 176 / 112, ncu counts 1.17M local loads and 0.67M local stores (from 7.4M and 2.9M), and the time is
the same. The spills were outside the critical path; the stall samples are still half mbarrier waits (the per-tile
loops of the v2.1 entry), plus 19% for the three idle warps parked at the exit barrier. The 512-thread layout stays:
it costs nothing, and the idle warps have room for a later role (the O rescale, -17% when removed, is FA4's
"correction" warpgroup).

### v2.2 hypotheses: four changes, one ablation

A review of v2.1 agreed with the three per-tile loops and added two points: the single MMA warp ties the S loop and
the V loop together (it issues QK(i + 1) only after PV(i - 1), so a late V also holds back QK), and the load ring needs
Little's law worth of bytes in flight: 49 GB/s per SM x 1.5-2 us under load = 75-100 KB, i.e. at least two raw tiles
in flight, three stages minimum and four safe. The changes:

- F1, lazy O rescale and two P buffers (as SM120 v11): a head's running maximum only moves when the tile maximum is
  more than 8 (log2 units) above it, so P <= 256, fine in FP16. Every softmax thread holds all heads and reads the same
  maxima, so the decision is uniform. Only a tile that rescales waits for PV(i - 1); with two P buffers the others
  write P(i) while PV(i - 1) still reads P(i - 1).
- F2, RoPE as E4M3 x E4M3 (`kind::f8f6f4`) on the raw kpe bytes, the query's RoPE part as 3 E4M3 terms (one scale per
  head) stacked along K into the same 16 columns; no FP16 RoPE tile, so the ring holds raw bytes only (44 KB per tile)
  and goes to 4 stages; the scale layouts derived from the raw scales get two buffers.
- F3, QK issued by the first softmax warp right after the 4 softmax warps have read S(i) (the existing named
  barrier), so the S loop no longer passes through the MMA warp, which keeps only PV.
- F4, V in two 64-token halves: PV issued half by half with a commit each, so the conversion of tile i + 1's first half
  overlaps PV of tile i's second half (TMEM unchanged).

Ablation, one process on B200, alternating rounds: none, all, all but one (x4), only one (x4); every configuration is
checked for correctness first. Predictions at H=16 B64 64K against v2 (1.131 ms): F1 alone -15% to -25%, F2 alone -20%
to -35%, F3 alone -5% to -15%, F4 alone -5% to -15%; all four 0.40-0.55 ms, with the all-but-one losses larger than the
only-one gains (each loop hides the others). A change stays if removing it from "all" costs at least 3%.

### v2.2 result: one change helps a little, the predictions were 2x off

Every configuration was first checked against the reference (5 cases, lengths 1-5000, page 16/32/64, splits).
rope8 was wrong in every configuration that had it and right in none that did not, which located the bug at once:
with 4 raw stages and 2 derived scale buffers, the load warps' last iterations (no new tile to copy) also skipped the
stage acquire, and that acquire was the only proof that the tile two back had been consumed, so the last two tiles'
scale layouts could be overwritten while still in use (error 0.045 at 32 tiles, 0.17 at 8, exact at 2). The acquire
now happens in every iteration.

Same process on B200, ms:

| configuration | H=16 B64 64K | H=128 B16 64K |
|---|---|---|
| none (v2 after the spill fix) | 1.132 | 2.241 |
| only lazy | 1.118 (-1%) | 2.236 |
| only rope8 | 1.037 (-8%) | 2.060 (-8%) |
| only qksm | 1.279 (+13%) | 2.536 (+13%) |
| only vhalf | 1.125 (-1%) | 2.226 (-1%) |
| all four | 1.158 | 2.310 |
| all but lazy | 1.197 | 2.382 |
| all but rope8 | 1.216 | 2.432 |
| all but qksm | 1.045 | 2.076 |
| all but vhalf | 1.175 | 2.343 |

- By the rule written before the run (keep a change if removing it from "all" costs 3% or more): rope8 (+5%) and lazy
  (+3.4%) stay; vhalf (+1.5%) and qksm (-10%: better without it) go.
- The predictions missed by about 2x: all four were to give 0.40-0.55 ms, and the best configuration is 1.04 ms. Only
  the deeper ring (rope8) moves the time at all, and only by 8%. Removing the O rescale and its wait (lazy), splitting
  the V loop (vhalf) or the S loop (qksm) changes almost nothing. So none of the three loops is what holds the tile
  period near 4 us; they were inferred from stubs that removed whole roles, which also removed those roles' issue
  slots, shared-memory traffic and TMEM traffic, not only their waits.
- qksm makes it worse: the first softmax warp now waits for the next tile's loads before it can go on with its own
  softmax, so a quarter of the softmax runs late every tile.

Next: measure the critical path instead of inferring it. Each role will record, per tile, the clock at the start of
its wait and at the end of its work, so the gaps show which role every other one waits on.

### Before the clocks: clean stubs and the stall samples per barrier

A review of the entries above found two gaps in the evidence.

- The stall breakdown per role, which the v2 hypothesis said would decide, was never done. In a loop structure,
  removing a role also removes its waits, so the stubs show what a role costs but not which barrier edge the time
  goes into.
- The "load" stub was not clean. It removed only the ckv and kpe copies. The page-table read (a synchronous global
  load per tile), the two scale copies, the scale layouts and, without rope8, the RoPE conversion still ran. So "load
  removed" and the skeleton still made one global round trip per tile through the shallow ring. The skeleton's step
  from v2 to v2.1 (0.247 -> 0.365 ms, 0.46 us per tile, about one DRAM latency) fits that. The "QK-softmax chain floor
  of about 1 us per tile" above was measured with this stub.

Changes to the measurement:

- The load stub now makes no global access: no page table, no copies, and constant scales (E4M3 1.0). K is then stale
  shared memory, so in that stub the lazy rescale threshold is infinite; the stub never rescales, like the real run
  after the first tile on random data, instead of rescaling at random.
- [`benchmarks/ncu_waits.py`](../benchmarks/ncu_waits.py) sorts the stall samples of the ncu SASS source page by
  role (markers in the SASS such as `setmaxnreg` bound each role's code) and by barrier. Each try_wait names its mbarrier by its shared-memory offset, and its samples sit on
  the branch to its out-of-line retry loop.

The script on the old v2.1 profile (2 stages, H=16 B64 64K), as a share of each role's samples:

| role | waiting for | work |
|---|---|---|
| load (4 warps) | convert to free a stage 37%, QK to free a stage 12% | 50% (copy issue) |
| convert (4 warps) | a loaded tile 47%, PV to free V 17% | 36% |
| MMA (1 warp) | a loaded tile 81% | 17% |
| softmax (4 warps) | S 62% | 38% |

The three idle warps parked at the exit barrier are another 19% of all samples. This is the 2-stage ring: the load
waits for convert to free a stage, convert waits for the load to fill the next one, and the MMA and softmax wait behind
both.

Predictions for the current default (lazy, rope8: 4 stages, 1.04 ms at H=16 B64 64K; 256 tiles per CTA, so 1 ms is
3.9 us per tile):

- Clean skeleton (load, convert, PV and rescale removed): if the round trip was the floor, it falls below 0.15 ms
  (0.6 us per tile) and the "1 us chain floor" claim is withdrawn. If it stays above 0.22 ms, the floor stands and is
  in the QK-softmax chain itself.
- Stall samples: with 4 stages the load should mostly wait for free stages, and the role that waits least while the
  others wait on it sets the period. My guess is convert (its work share the largest, the MMA warp waiting mostly for
  V). If instead the MMA warp still waits mostly for loaded tiles, the ring is still too shallow or the copies are
  issued too slowly, and the next change is in the load role (TMA).

### Result: the load loop publishes each tile one release too late

Clean stubs, same process on B200, ms:

| removed | H=16 B64 64K, lazy+rope8 | none | H=128 B16 64K, lazy+rope8 | none |
|---|---|---|---|---|
| nothing | 1.031 | 1.118 | 2.077 | 2.238 |
| load (no global access) | 0.581 | 0.760 | 1.143 | 1.505 |
| convert | 0.620 | 0.728 | 1.234 | 1.431 |
| PV / QK / rescale | 0.966 / 0.955 / 1.037 | 1.051 / 1.030 / 1.114 | | |
| load + convert | 0.263 | 0.367 | 0.513 | 0.718 |
| load + convert + PV + rescale | 0.232 | 0.318 | 0.451 | 0.620 |
| ... + QK | 0.172 | 0.277 | 0.330 | 0.540 |

- By the rule written above, the floor stands: the clean skeleton is 0.232 ms (0.91 us per tile), above 0.22 ms.
  The global round trip was a small part of it (the 2-stage skeleton went from 0.365 to 0.318 ms, and that build
  also has the spill fix). But the floor is not the "QK-softmax chain" either: without the QK MMAs it is still
  0.172 ms. It is mostly barrier hand-offs, and the stall samples show which ones.

Stall samples, current default (lazy, rope8: 4 stages), H=16 B64 64K, as a share of each role's samples:

| role | waiting for | work |
|---|---|---|
| load | convert to free a stage 41%, QK to free a stage 14% | 44% (30% long scoreboard: page-table read, copy issue) |
| convert | a loaded tile 41%, PV to free V 19% | 40% |
| MMA | a loaded tile 79% | 19% |
| softmax | S 60% | 39% |

This is the same picture as with 2 stages, and the two top waits contradict a deep ring: the load waits for
convert to free a stage while convert waits for the load to publish the next tile. The reason is the order inside
the load loop. Iteration it first acquires the stage of tile it (so tile it - 4 must be released), issues its copies,
and only then finishes tile it - 3 (waits for its copies, writes the scale layouts, commits). So tile j is
published only after tile j - 1 is released, whatever the ring depth:

convert finishes tile j - 1 -> the load warps wake -> page-table read and 22 copy issues for tile j + 3 -> scale
layouts of tile j -> commit -> convert wakes for tile j

Each tile period therefore holds convert's work (40% of 4.0 us), the load loop's own latency (44%) and two barrier
hops, which adds up to the 4 us. The ring prefetches the bytes (they landed long ago) but not the hand-off. This also
explains three earlier results: four stages (rope8) gained only 8%, "load removed" gained 44-51% (it shortens the
load loop inside the cycle, not the bytes), and the skeleton keeps a floor of about 0.9 us per tile (the hand-off
cycle with no work in it). My guess before the run (convert sets the period) was wrong; convert is in the cycle, but
the cycle is made by the load loop.

### v2.3 hypothesis: finish before acquiring

- E (`early`): each iteration first finishes tile it - 3 (wait for its copies, scale layouts, commit), then acquires
  the stage of tile it and issues its copies. Tile j is then published once tile j - 2 is released, so the consumers
  have one tile of slack and the load loop's latency leaves their period. The derived scale buffers stay safe: the
  previous iteration's acquire proves tile it - 5 released, and nd = 2 buffers need only that.
- P (`ptpf`): the page-table entry of tile it + 1 is read right after tile it's copies are issued, so its latency
  overlaps the next acquire instead of delaying the copy issue.

Predictions at H=16 B64 64K (1.031 ms now):

- E alone: convert's "waiting for a loaded tile" share falls from 41% to under 10% and the MMA warp's from 79% to
  under 40%; time 0.45-0.65 ms. The next limit should be the V loop (convert(j + 1) waits for PV(j)), which costs
  at least convert's work, about 1.6 us per tile (0.42 ms).
- P alone: -5% to -15% (a shorter load loop inside the cycle). With E, P changes less than 3%.
- With E and P, vhalf (V in two halves, no gain in v2.2) should now help, since the V loop becomes the limit.
- Rule: E stays if it gives 10% or more; P and vhalf stay if removing them costs 3% or more.

### v2.3 result: the hand-off was the period; convert is now the limit

Every configuration is correct (the 5 cases up to 16K x 8 at H=128 and lengths 1-5000: error 0.0017, as before).
Same process on B200, ms, H=16 B64 64K / H=128 B16 64K:

| configuration | run 1 | run 2 |
|---|---|---|
| lazy + rope8 (v2.2 default) | 1.047 / 2.083 | 1.045 / 2.080 |
| + E | 0.765 / 1.496 | 0.763 / 1.494 |
| + P | 1.083 / 2.159 | |
| + E + P | 0.777 / 1.523 | |
| + E + vhalf | | 0.750 / 1.459 |
| + E + P + vhalf | 0.741 / 1.442 | 0.747 / 1.438 |
| + E + P, 2 stages (no rope8) | 0.789 / 1.564 | |

With E as the only order in the code (third run): lazy + rope8 0.764 / 1.494, none 0.837 / 1.556, + vhalf 0.749 /
1.458.

- E: -27% at both shapes. It stays, as the only load order. The time is above the predicted 0.45-0.65 ms.
- P: +3% alone and +1.6% with E. It goes; the code is removed. Reading the page table one tile early is not useful
  once the load loop is outside the consumers' period.
- vhalf: -1.7% to -2.5% on top of E in both runs, under the 3% rule, so it stays off.
- The ring depth now matters little: with E and P, 2 stages are only 1.5% slower than 4.

Stall samples with E (and vhalf), H=16 B64 64K (0.759 ms under ncu):

| role | waiting for | work |
|---|---|---|
| load | convert to free a stage 35%, QK 1% | 64% |
| convert | PV to free V 5%, a loaded tile 1% | 94% |
| MMA | a loaded tile 33%, V 18% | 47% |
| softmax | S 22% | 77% |

- The hand-off is gone as predicted: convert's wait for a loaded tile fell from 41% to 1% (predicted under 10%), the
  MMA warp's from 79% to 33% (predicted under 40%).
- The time missed because convert's work per tile was not 1.6 us, as I took from the v2.2 samples, but 94% of a
  2.96 us period, about 2.8 us. Its instruction count did not change (1106 per warp and tile). Its work samples are
  mostly stalls, and those grew once the other roles ran at the same time instead of in turn.
- The V loop is not the limit (convert waits for PV 5% of its time), which is why vhalf gives little.

What convert does per tile and warp (1106 instructions): per 8 V elements, 4 `cvt` e2m1, 2 `cvt` e4m3 (scales),
4 `prmt` and 4 `mul`; and per tile 64 LDS (scales), 32 LDSM and 64 `tcgen05.st`. No pipe is busy (ALU 40%, issue
slots 38%). One convert warp per SM sub-partition cannot hide the latency. The scale conversions wait for their LDS
(short scoreboard, 18% of convert's samples), and LDS and LDSM wait on the MIO queue (15% and 12%). The
sub-partition shares that queue with the load warp's 22 `cp.async` and 40 scale-layout stores per tile.

The per-role clocks planned above are not needed for this question: the stall samples named the barrier edge
directly, and the change made from them confirmed it. They stay in reserve for a profile that does not separate the
roles this clearly.

Options for v2.4:

- (a) Fewer shared-memory instructions in convert. Lay out the transposed scales with the 8 blocks of a tile
  contiguous, so one LDS.128 replaces 4 loads (64 -> 16 per tile). Use `ldmatrix` .x2 (32 -> 16). Optionally the
  load warps could write ready f16x2 scale pairs, which removes the 128 e4m3 conversions and their wait.
- (b) More conversion warps. TMA for the raw tile (one thread issues the copies; page size >= 32, with cp.async
  kept for smaller pages) frees most of the load warpgroup, so 8 warps convert, two per sub-partition.
- (c) Word stores instead of byte stores for the transposed scales (32 `st.shared.u8` per thread and tile).

### v2.4 hypothesis: TMA for the raw tile, two convert warps per sub-partition

Option (b). Two flags, measured together with v2.3 in one run:

- `tma`: the raw bytes of a tile (FP4 latent, scales, E4M3 RoPE) come by TMA instead of 22 `cp.async` per thread.
  The cache is viewed as [pages x page size, bytes per token] and each page is one box; the shared-memory layouts
  stay the same (the FP4 tile is the SWIZZLE_128B box layout, the RoPE tile the SWIZZLE_64B one). One thread reads
  the page table and issues the copies; boxes entirely past the end are not issued. Rows past the end are no longer
  zero-filled, so the scale layouts set the scales of those tokens to 0 (V is then exactly 0 there; S is already
  masked with a select). In this flag alone the load warpgroup still writes the scale layouts.
- `cvt8` (needs `tma`): warpgroup 0 becomes a second convert warpgroup. Each of the 8 convert warps converts 2 of
  the 4 128-channel chunks, so every sub-partition runs two convert warps. Warp 13 issues the TMA and warps 14-15
  write the scale layouts (2 tokens per thread). These are no longer the threads that acquire the stage, so before
  writing the layouts of tile j they wait for tile j - 2 to be released (the empty barriers of its stage); that
  replaces the proof the acquire gave. Registers: 128 / 128 / 176 / 80 (convert, convert, softmax, MMA + TMA +
  scales).
- Page size >= 16 only (128 / page size boxes per tile); smaller pages keep the cp.async load.

Predictions at H=16 B64 64K (v2.3: 0.764 ms):

- `tma` alone: -5% to -15%. The load warp leaves the MIO queue it shares with each convert warp (the LDS and LDSM
  of convert wait on that queue for 27% of their samples).
- `tma` + `cvt8`: 0.55-0.65 ms. With two warps per sub-partition, convert's 2.8 us per tile should about halve, and
  the softmax (77% busy, about 2.3 us per tile) becomes the limit.
- Rule as before: a flag stays if removing it costs 3% or more; correctness first for every configuration.

### v2.4 result: TMA and 8 convert warps, -39%

Two problems on the way:

- The DSL wraps each TMA copy in its own `elect_one`. Issuing the copies from one thread (`if lane == 0`)
  deadlocked. A whole warp now calls the copies, and only the expect-tx arrive is inside `elect_one`.
- The first `tma`-only build issued the TMA from warp 0 inside the load warpgroup's lagged loop. It was wrong in 6
  of 36 runs (error up to 0.21, only with 3 or more tiles per CTA) and never in the `cvt8` build, which issues from
  warp 13. Extra waits for the derived buffers changed nothing. A named barrier over the 4 load warps after the TMA
  issue removed the errors; one before the issue, or before the acquire, did not. I did not find the missing
  ordering. `tma` now uses the `cvt8` structure (warp 13 issues the copies, warps 0-3 only write the scale layouts).
  That is also the cleaner ablation: the two flags then differ only in which warps convert. After that, 0 of 36 runs
  were wrong for every configuration.
- One early run measured 0.742 ms for `tma` + `cvt8`. The code this configuration runs did not change after that,
  and the three later runs give 0.466-0.468 ms. I do not know the cause; the table uses the later builds.

Same process on B200, ms, H=16 B64 64K / H=128 B16 64K (two runs, the second one shown where they differ by more
than 0.002):

| configuration | time |
|---|---|
| v2.3 (lazy, rope8) | 0.767 / 1.497 |
| + tma | 0.639 / 1.230 (-17% / -18%) |
| + tma + cvt8 | 0.468 / 0.881 (-39% / -41%) |
| + tma + cvt8 + vhalf | 0.451 / 0.852 |

- `tma` alone: -17%, a little more than predicted (-5% to -15%). `tma` + `cvt8`: 0.468 ms, better than the
  predicted 0.55-0.65 ms.
- vhalf now passes the rule (removing it costs 3.4-3.8%), because the V loop has become visible (below).
- Default for page size >= 16: lazy, rope8, tma, cvt8, vhalf; smaller pages keep lazy, rope8 (cp.async).

Stall samples, `tma` + `cvt8`, H=16 B64 64K (0.48 ms under ncu; DRAM throughput 38%, from 24%):

| role | waiting for | work |
|---|---|---|
| convert (8 warps) | PV to free V 17%, a loaded tile 3% | 80% |
| TMA warp | convert to free a stage 83%, QK 2% | 16% |
| scale warps (2) | tile j - 2 released 42%, the TMA 2% | 56% |
| MMA | a published tile 40%, V 19% | 38% |
| softmax | S 22% | 77% |

The idle warps at the exit barrier fell from 19% to 2% of all samples. The loads are now ahead: the TMA warp waits
for free stages. Convert is still the busiest role, not the softmax as I predicted. The tile period is 1.83 us, and
convert works 80% of it and waits for PV(j - 1) to free V another 17%.

Against Cake (0.319 ms): 0.451 ms, 1.41x.

The default checked on pages of 1-128 tokens and lengths 1-300 (error 0.0017 everywhere, the cp.async path below 16
tokens per page included). B200, page size 64, ms:

| H | B | 4K | 16K | 64K | v2.3 at 64K |
|---|---|---|---|---|---|
| 16 | 16 | 0.043 | 0.060 | 0.133 | 0.196 |
| 16 | 64 | 0.069 | 0.148 | 0.454 | 0.764 |
| 128 | 16 | 0.079 | 0.233 | 0.852 | 1.494 |
| 128 | 64 | 0.284 | 0.898 | 3.381 | 5.868 |

H=128 still reads the cache once per 16 heads (8 CTAs per sequence), which is the next structural item after the
convert role.

After v2.4, on the full grid (H 16/128, batch 1-64, 1K-64K tokens, one process with the other libraries;
`benchmarks/results/b200.csv`, and every buildable version in `versions.csv`): the SM100 kernel is faster than the
CUDA kernel from about 16 x 16K tokens, but below that it is slower, and slower than v0. A call costs at least about
40 us with a cold L2. With warm caches at H=16, batch 1, 1K tokens: 17.6 us for the kernel (one tile through its
start, pipeline and end), 2.3 us query terms, 1.7 us RoPE terms, 2.7 us split merge. v0 has no query split kernels
and a shorter start (no TMA warp, no role setup), 23 us cold. The fixed cost is a separate item from the tile
period and not addressed yet.

### Accuracy is now judged against the unquantized KV

Until now the kernels were checked against attention over the dequantized cache, which is the right test for bugs
(the error there is the kernel's own arithmetic, 1.6e-3 to 1.9e-3) but the wrong measure for a choice of precision:
it counts every extra rounding as if it were the whole error, while the user sees the error against the unquantized
(BF16) KV, where the FP4 format already costs 0.08-0.12. From here on, a precision change is judged by its error
against the unquantized KV, relative to the CUDA kernel's BF16 mode on the same cache (`benchmarks/accuracy.py`);
the dequantized reference stays as the bug check in the tests.

The new measurement also uses a global scale calibrated the way a serving engine sets it (amax / (6 x 448), so the
block scales reach the E4M3 maximum), besides the scale of 1 used so far. It found a bug in the CUDA kernel's FP8 and
FP4-QK modes: they round e2m1 x block scale to E4M3, but that product reaches 6 x 448 = 2688 and E4M3 stops at 448,
so with a calibrated scale the large values saturated. The error against the unquantized KV was 0.62-0.69 (5.6-8x
the BF16 mode) instead of 0.08-0.13; the tests used small scales only and missed it. The fix rounds (e2m1 x scale) / 8
(at most 336) and puts the 8 back in the score scale and the output; dividing the f16 scale by 8 is exact. The tests
now run both scales.

On B200, both scales: the SM100 kernel's error against the unquantized KV equals the CUDA kernel's BF16 mode (ratio
1.00); FlashInfer Cake's is 1.38-1.71x.

### v2.5 hypothesis: V and P in E4M3, 32 heads per CTA

With 128 heads the SM100 kernel reads the cache 8 times per request, because a CTA holds 16 heads. 32 heads do not
fit in tensor memory with an FP16 V (S 128 + RoPE S 32 + O 128 + scales 64 + V 256 = 608 of 512 columns). With V in
E4M3 it is 480. PV then becomes `kind::f8f6f4`, which needs P in E4M3 as well (P <= 256 with the lazy rescale). V is
rounded as (e2m1 x scale) / 8, as in the CUDA fix above, and the 8 goes into the output scale.

Two more limits at 32 heads: shared memory (the query terms grow by 16 KB to about 244 KB of 227 KB, so the raw ring
goes from 4 to 3 stages, which cost 1.5% in v2.3) and the softmax registers (8 arrays of one value per head; the
softmax handles the heads in two halves of 16).

Changes, measured as an ablation in one process:

- `v8`: V and P in E4M3, 16 heads. The conversion needs about 20 instructions per 8 values instead of 14 (the
  rounding to E4M3 and the scale shift), but half the TMEM stores and half the PV MMAs.
- `st3`: 3 raw stages instead of 4.
- 32 heads per CTA (needs `v8` and `st3`), for head counts that are a multiple of 32.

Predictions (H=128 B16 64K, now 0.852 ms; H=16 B64 64K, now 0.451 ms):

- Accuracy: `v8` adds at most 3% to the error against the unquantized KV (only V and P are rounded; the CUDA FP8
  mode, which also rounds Q and K, adds 0-11%). If it adds more, V stays FP16 and 32 heads wait.
- `v8` at 16 heads: 0-10% slower (convert is the busiest role and does more ALU work per value).
- 32 heads: 0.47-0.65 ms at H=128 B16 64K (-24% to -45%). Half the tiles per head; per tile the convert work stays
  the same and the softmax work doubles, so the softmax (now about 1.4 us per tile) may set a period of about 2.8 us.
- Rule: 32 heads become the default for head counts divisible by 32 if they gain 10% or more at H=128; `v8` and
  `st3` stay only where they are needed for that.

### v2.5 result: 32 heads per CTA work, but gain only 1-2%

Every configuration is correct: against the dequantized cache 0.0017 (FP16 V) and 0.029 (`v8`, the E4M3 rounding of
V and P, about 2% per value); against the unquantized KV, `v8` is at most 3.3% above the FP16-V kernel, at
lengths 1-300 where the rounding does not average out, and 1-2% on the `benchmarks/accuracy.py` shapes (4K and 16K
tokens). By the accuracy rule (judged on those shapes) `v8` passes; the 3.3% at short lengths is noted here.

Same process on B200, ms:

| configuration | H=16 B64 64K | H=128 B16 64K | H=128 B64 64K |
|---|---|---|---|
| v2.4 (FP16 V, 4 stages, 16 heads) | 0.454 | 0.852 | 3.382 |
| + v8 | 0.495 (+9%) | 0.928 (+9%) | 3.695 |
| + v8 + st3 | 0.526 (+16%) | 0.997 (+17%) | 3.954 |
| + v8 + st3 + h32 | | 0.844 (-1%) | 3.309 (-2%) |

- 32 heads alone are worth -15% (0.997 -> 0.844), but E4M3 V (+9%) and 3 stages (+7%) cost about as much, so the
  result is -1% to -2% instead of the predicted -24% to -45%. By the rule (10% at H=128), 32 heads are not the
  default; `v8`, `st3` and `h32` stay as options.
- `v8` costs 9% at 16 heads as predicted (0-10%): the conversion does more ALU work per value.

Stall samples with 32 heads, H=128 B16 64K (0.86 ms under ncu):

| role | waiting for | work |
|---|---|---|
| convert (8 warps) | PV to free V 40% | 59% |
| softmax | S 1% | 99% |
| MMA | P 74% | 23% |
| TMA warp | convert to free a stage 85%, QK 4% | 11% |

The softmax is the limit: with 32 heads it does twice the work per tile (4 latent and 1 RoPE score loads of 32
columns, the sums, 32 exponentials per thread), and at 232 registers it still spills (77 local loads in its code).
The MMA warp waits for P, so PV frees V late and convert waits for V. Two ways to cut the softmax's work, for a
later step: QK with the query terms along K instead of N (the MMA sums the 4 terms, so the softmax loads one score
column block instead of four), and a softmax split over more warps.

### v2.6 hypothesis: the query terms along K

With 32 heads the softmax sets the period (v2.5): per tile it loads 4 latent score blocks and 1 RoPE block of 32
columns and adds them. The 4 blocks exist only because the query terms are stacked along N, so that one MMA per
64-channel step serves all terms.

- Change `tk`: for each 64-channel step, 4 MMAs, one per query term, with N = the CTA's heads, accumulating into
  one score block. The tensor work is the same; the MMA warp issues 32 QK instructions per tile instead of 8 (it is
  idle 60% of the time). Each term has its own block scales, so the query's scale factors get 4 TMEM regions and 4
  shared-memory stages.
- TMEM: the latent scores shrink from 4n to n columns, the SFB regions grow by 3; 16 heads with FP16 V: 512 of 512
  columns, 32 heads with E4M3 V: 480. Shared memory grows by 12 KB, which does not fit 4 raw stages with 16 heads,
  so `tk` goes with `st3`.
- The softmax then loads 2 score blocks instead of 5 and drops 3n adds per thread per tile.

Predictions:

- 32 heads (`v8`, `st3`, `h32`, `tk`) at H=128 B16 64K: 0.65-0.75 ms (-12% to -24% against v2.4, 0.852), B64 64K
  2.6-3.0 ms (3.38). The softmax still has twice the exponentials of 16 heads, so it stays near the limit.
- 16 heads (`st3`, `tk`): within 3% of v2.4; convert sets the period there.
- Rule: `tk` stays if removing it costs 3% or more in its configuration. 32 heads become the default for head
  counts divisible by 32 if they are 10% or more faster than v2.4 at H=128 with both batch 16 and batch 64.

### v2.6 result: correct, but the cost moves to the MMA warp

Correct in every configuration (0.0017 against the dequantized cache with FP16 V; with `v8`, as v2.5). Same process
on B200, ms:

| configuration | H=16 B64 64K | H=128 B16 64K | H=128 B64 64K |
|---|---|---|---|
| v2.4 | 0.453 | 0.854 | 3.377 |
| + st3 | 0.462 | 0.871 | 3.448 |
| + st3 + tk | 0.532 (+15%) | 1.005 | 3.984 |
| + v8 + st3 + h32 (v2.5) | | 0.862 | 3.312 |
| + v8 + st3 + h32 + tk | | 0.886 (+3%) | 3.772 (+14%) |

The prediction (-12% to -24% with 32 heads) was wrong in sign. By the rule `tk` goes, and 32 heads stay an option.

Stall samples with `st3` + `tk`, H=16 B64 64K (0.55 ms under ncu): the MMA warp works 79% of its time (38% in v2.4)
and convert waits for PV to free V 32% of its time (17%). The tensor pipe is busy only 15%, so it is not the
smaller MMAs: it is the issue path. One warp now issues 32 QK MMAs per tile instead of 8, each with a new scale-factor
field, and the MMA warpgroup spills at its 72-80 registers (6.0M local loads). Since that warp issues QK(i + 1)
before PV(i), PV starts later, V is freed later and convert waits. Terms along K would need a cheaper issue path
first: a second warp for QK, or descriptors prepared once instead of per instruction.

### Where the SM100 kernel stops for now

B200, ms, every version in one process (`versions.csv`; v2-v2.2, v2.5 and v2.6 from their own runs above) and Cake from the
same sweep (`b200.csv`):

| version | H=16 B64 64K | H=128 B16 64K | H=128 B64 64K |
|---|---|---|---|
| v0 (FP16 tile in shared memory, 32 heads per CTA) | 1.349 | 1.578 | 6.152 |
| v1a' (QK on the raw FP4 bytes) | 1.447 | 2.753 | 10.68 |
| v1b (V straight into TMEM) | 1.276 | 2.410 | 9.428 |
| v2-v2.2 (warp roles; log only) | 1.04-1.13 | 2.02-2.24 | |
| v2.3 (load loop finishes before it acquires) | 0.769 | 1.498 | 5.888 |
| v2.4 (TMA, 8 convert warps; the default) | 0.455 | 0.854 | 3.378 |
| v2.5 (E4M3 V and P, 3 stages, 32 heads per CTA; options, ablation run) | 0.526 (16 heads: no `h32`) | 0.844 | 3.309 |
| v2.6 (query terms along K; option, ablation run) | 0.532 (with `st3`) | 0.886 (with v2.5) | 3.772 (with v2.5) |
| FlashInfer Cake (NVFP4 query) | 0.320 | 0.301 | 0.541 |

v2.5 and v2.6 are not the default: by the rules written before their ablations, 32 heads gain 1-2% (the softmax
sets the period), which does not pay for the E4M3 rounding of V and P (1-3% more error against the unquantized KV),
and the query terms along K are slower (the single MMA warp becomes the limit). They stay as the options `v8`,
`st3`, `h32` and `tk`.

With the query kept at BF16-level accuracy (error against the unquantized KV equal to the CUDA kernel's BF16 mode,
where Cake's is 1.4-1.7x), the kernel is 1.4x behind Cake at 16 heads and 2.8-6.2x behind at 128 heads.

What the B200 work showed about the method:

- Stubs that remove a role's work also remove its waits, so in a loop of roles they say what a role costs, not where
  the period comes from. Stall samples per barrier named the edge directly (the load loop's order in v2.3, -27%).
- Predictions from per-role stall shares missed in both directions (v2.2 2x too optimistic, v2.4 better than
  predicted, v2.5 far too optimistic): a share of samples is not time on the critical path, which the SM120 work had
  already shown.
- Measurements need a calibrated global scale as well as a scale of 1: the FP8 saturation bug only shows with block
  scales near the E4M3 maximum.

What limits it now, and the next steps in order of expected gain:

1. 128 heads: a CTA holds 16 heads, so the cache is read 8 times. 32 heads per CTA work (v2.5) but the softmax then
   sets the period (99% busy, spilling at 232 registers). The query terms along K would cut the softmax's score
   loads from 5 to 2, but with one MMA warp the 4x QK instructions delay PV (v2.6). Next: QK issued by its own warp
   (all 16 warps have a role now, so this needs a new split, e.g. one scale-layout warp instead of two), or the
   descriptors prepared once instead of per instruction; then the terms along K and 32 heads.
2. 16 heads: convert is the busiest role (80% work, 17% waiting for PV to free V). Next: fewer shared-memory
   instructions in the conversion (scales as LDS.128, `ldmatrix` .x2) and word stores of the transposed scales.
3. Small shapes: a call costs at least about 40 us with a cold L2 (24 us warm: 18 us for one tile through the
   kernel, 7 us for the query split and merge kernels). Next: the query split inside the decode kernel's prologue,
   and a profile of the kernel's start.
4. Not started, outside the kernel: attention throughput at a fixed KV memory budget (NVFP4 holds 1.64x the tokens
   of FP8), and the accuracy comparison with Cake on real model KV (DeepSeek-V2-Lite: layer-wise attention output
   against BF16), which random data cannot replace.

## Part 4: DecodeV3 groundwork

A review of v2 against FlashInfer Cake's source (the `d2829e4e` kernel, 16 heads) and the v2.4 profile proposes a
new structure instead of more options on DecodeV2. Checked against the source and the profiles:

- Cake converts V with `mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite` (PTX ISA 9.4, CUDA 13.4): 4 values per instruction,
  scale included; one thread per token; per token a power of two 2^-k (k = 0-3 from its largest scale) keeps the
  product inside E4M3 and is undone on P. The same transform warps write the token's QK scale factors straight into
  TMEM (`tcgen05.st`, lane = token). V and P go through shared memory to the PV MMA, issued by its own warp; QK has
  another warp, RoPE is accumulated into the latent score, and two softmax warpgroups take alternate tiles with an O
  accumulator each.
- Cake's shared memory is 91% busy at 16 heads (LSU 56%, tensor reads 35%); v2.4's is 68% (LSU 48%, 1811
  wavefronts per tile, not the 34% the review assumed; tensor 20%). Of the LSU wavefronts attributed to
  instructions, 43% are the two scale-layout warps', and 63% of those are bank-conflict excess (their 32-byte-stride
  token reads).
- The query with 2 terms is within 1% of BF16 against the unquantized KV (the v1 measurement above).

Two measurements decide the order of the work.

**Probe 1: QMUL4.** (a) Does CuTe DSL's inline assembly accept the PTX 9.4 instruction? If not, the toolchain comes
first. (b) Is it one SASS instruction when the four scale lanes differ, or an emulation? Cake broadcasts one scale to
all four lanes ("the native QMUL4 reads lane 0"). (c) Its throughput against the f16 path (`cvt` e2m1, `mul`, `cvt`
e4m3), which is also the exact reference: e2m1 x e4m3 fits f16 exactly, so only the final rounding differs. If
distinct lane scales are native, V can be converted per channel and token group straight into TMEM at about a
third of the instructions, which would make "cheap conversion without the shared-memory round trip" the main path.

**Probe 2: v2.4 with the conversion removed** (the `cvt` stub, H=16 B64 64K, now 1.78 us per tile). Prediction (from
the review): 1.30-1.45 us per tile, because the softmax (about 1.4 us of work per tile) then sets the period. Rule:
1.30 us or more, and the softmax at 90% or more of its samples busy in the stub's profile, means the softmax work
(fewer score loads, two softmax warpgroups) comes before a cheaper conversion; 1.0 us or less means the conversion
goes first.

### Probe results

**QMUL4** (a CUDA C++ probe and a CuTe DSL probe, B200):

- CuTe DSL's inline assembly accepts it, on the CUDA 13.4 and the 13.3 image alike (the DSL brings its own
  compiler), so no toolchain work is needed.
- One SASS instruction, also with four different scales: `QMUL4.E4M3x4.E2M1x4.E4M3x4.SATFINITE R9, R2.N3210,
  R5.B3210`, the scale operand selecting a byte per lane. Every lane matches the exact f16 path (4.2M lanes with a
  broadcast scale, 4.2M with distinct scales; 0 differ).
- Throughput in a loop of independent operations: 80.7 operations (of 4 values) per ns and SM against 26.4 for the
  f16 path, 3.1x; the loop's own ALU work alone runs at 110, so 80.7 is a lower bound.

So distinct scales per lane are native: a thread that holds 4 tokens' nibbles of one channel and those tokens'
scales makes one 32-bit E4M3 word of V^T, 4 tokens per TMEM column, in one instruction. What remains around it is
gathering the 4 tokens' nibbles of one channel (`ldmatrix` .trans gives bytes, i.e. 2 channels per token: about 7
logic operations per 8 values), against 14 instructions per 8 values now.

**v2.4 with the conversion removed** (`cvt` stub, same process, ms):

| removed | H=16 B64 64K | per tile | H=128 B16 64K |
|---|---|---|---|
| nothing | 0.437 | 1.71 us | 0.846 |
| conversion | 0.306 | 1.20 us | 0.579 |
| load + conversion + PV + rescale + QK | 0.165 | 0.64 us | 0.319 |

1.20 us per tile is below the predicted 1.30-1.45 us, but the stub's profile shows the softmax 96% busy (and the
scale-layout warps 90%, mostly waiting for their own shared-memory loads), so by the rule's second condition the
softmax sets the period once the conversion is free. Neither branch of the rule holds cleanly: a free conversion
alone would give 0.306 ms (Cake: 0.320), and going below needs the softmax's work cut as well.

### v3.0 hypothesis: V converted by QMUL4 straight into TMEM

A new class, `DecodeV3`: v2.4's structure fixed (TMA, 8 convert warps, lazy rescale, E4M3 RoPE, V in halves) without
DecodeV2's options, and one change in the data path:

- V goes to TMEM as E4M3 (as `v8`), each 32-bit word (one channel, 4 tokens) made by one QMUL4 with the 4 tokens'
  scales. `ldmatrix` .trans still gives bytes (one token, two channels), so the nibbles are regrouped first: a
  delta swap of nibbles 1 and 2 in each half-word and one `prmt` put the even channel's 4 tokens in the low half and
  the odd channel's in the high half. 7 instructions per 8 values instead of 14.
- The range: per token a power of two 2^-k (k = 0-3 from its largest block scale, as Cake), applied to the block
  scales the conversion reads (the QK scale factors stay unshifted) and undone on P. P x 2^k must stay below 448, so
  the lazy rescale threshold drops from 2^8 to 2^5. The scale-layout warps compute k and the shifted scales; the
  softmax reads 2^k per token from a two-slot buffer, which the scale warps overwrite only after P of the tile two
  back is published.

Predictions at H=16 B64 64K (v2.4: 0.437-0.455 ms, convert 80% busy) and H=128 B16 64K (0.846):

- 0.31-0.34 ms and 0.58-0.66 ms: convert's work drops to about 45-55% of the period and the softmax (96% busy once
  the conversion is free, 1.20 us per tile) sets it.
- Error against the unquantized KV: at most 1% above the BF16 mode on the `accuracy.py` shapes (P is rounded to E4M3
  as in `v8`, which cost 1-2%; the per-token shift keeps small values that `v8`'s fixed /8 lost).
- Rule: v3 continues on this path if H=16 B64 64K is at or below 0.35 ms and the error rule (3%) holds.

### v3.0 result: correct at once, but only 6% faster

Correctness (6 shapes x 3 global scales; page sizes 16-64, 16-128 heads, 1-3 splits, lengths 1-5000; now `test_v3`
in `tests/test_sm100.py`): against the dequantized cache 0.022-0.028 (V and P rounded to E4M3), against the
unquantized KV 1.007-1.024x the BF16 mode (v2.4 on the same caches: 0.998-1.000x). That is within the 3% rule but
above the predicted 1%. The ratio is the same for the three global scales (0.5, 1, calibrated), so the per-token
shift keeps the values inside E4M3's range; what is left is E4M3's rounding of V (e2m1 x scale needs up to 6
significant bits) and of P.

Time, same process, ms:

| shape | v2.4 | v3.0 | Cake |
|---|---|---|---|
| H=16 B64 64K | 0.461 | 0.435 | 0.327 |
| H=128 B16 64K | 0.865 | 0.817 | 0.314 |
| H=128 B64 64K | 3.376 | 3.180 | 0.549 |

-6% at all three shapes against the predicted 0.31-0.34 ms and 0.58-0.66 ms, so the rule (0.35 ms or less at H=16
B64 64K) fails. Per tile at H=16 (256 tiles per CTA): 1.70 us against v2.4's 1.80 us, while v2.4 without any
conversion ran at 1.20 us. Halving the conversion's arithmetic took only a tenth of the gap.

### Where v3.0's period goes: prediction

Stubs of v3.0 in one process, next to v2.4's, and the stall samples per role of v3.0 and of v3.0 without the
conversion (H=16 B64 64K, `benchmarks/ncu_waits.py`). Three candidates, each
with what would show it:

- The conversion, still: the v3 `cvt` stub falls to v2.4's (about 0.31 ms) and the convert warps are 85% or more
  busy. The rest of its cost would then be `ldmatrix` and `tcgen05.st` (32 stores per thread and tile, as `v8`).
- The scale-layout warps: 85% or more busy. They do more per token now (k, 8 shifted words), and the new wait for
  P(j - 2) puts them on the softmax's path: their work on tile j starts when the softmax publishes P(j - 2) and must
  end, with QK(j), before the softmax finishes tile j - 1. The softmax would then wait on "S ready" for 15% or more
  of its samples.
- The softmax (P x 2^k, one more barrier wait, the rescale at 2^5 instead of 2^8): 95% or more busy, with the
  convert warps waiting for PV to release V.

Expected: the second. The v2.4 profile without conversion had the scale warps 90% busy at 1.20 us per tile (mostly
on their own shared-memory loads), and v3.0 gives them more work and a shorter slack.

### Where v3.0's period goes: result

Stubs, ms (same process; v2.4 in brackets): full 0.413 (0.438), without the load 0.279 (0.353), without the
conversion 0.318 (0.306), without load and conversion 0.208 (0.234). Share of each role's stall samples (H=16 B64
64K):

| role | v3.0 | v3.0 without the conversion |
|---|---|---|
| convert (8 warps) | 48% busy, 39% waiting for PV to release V, 13% for a tile | (stub) |
| softmax | 87% busy, 11% waiting for S | 92% busy, 6% waiting for S |
| scale warps | 78% busy, 18% waiting for P(j - 2) | 61% busy, 32% waiting for P(j - 2) |
| MMA | 40% busy, 54% waiting for the scale warps' tile | 54% busy, 38% waiting for the scale warps' tile |

None of the three conditions holds as stated. The conversion is no longer the limit (48% busy, from about 80% in
v2.4). The closest is the second: the scale warps are 78% busy and 18% waiting for P(j - 2), so 96% of their time is
in a loop with the softmax (P(j - 2) -> scale layouts of j -> QK(j) -> S(j)), and the softmax waits for S 11% of its
samples. The `load` stub, which takes the scale warps' shared-memory loads off that loop, gains 32% in v3.0 against
19% in v2.4. The softmax is the other half of the period: 87% busy, 15% of its samples ready but not selected (the
convert warps share its sub-partition).

### v3.0a hypothesis: one 2^k slot per stage

The wait for P(j - 2) only protects the 2^k slot that tile j reuses. With one slot per stage (4) instead of 2, tile j
reuses tile j - 4's slot, and the scale warps already wait for the release of tile j - 2's stage, which comes after
QK(j - 2), which comes after the softmax released S(j - 3), so after P(j - 4). The wait can go and the scale warps
leave the softmax's path.

Prediction (H=16 B64 64K): 0.36-0.39 ms (0.413 now; the softmax's own work, 87% of 1.61 us per tile, gives 0.36 as
the floor); the softmax 93% or more busy and waiting for S 5% or less. H=128 B16 64K: 0.72-0.77 ms.

### v3.0a result: -12%, and the softmax sets the period

Correctness as v3.0 (the change only removes a wait). Time, same process, ms:

| shape | v2.4 | v3.0a | Cake |
|---|---|---|---|
| H=16 B64 64K | 0.455 | 0.399 | 0.319 |
| H=128 B16 64K | 0.855 | 0.752 | 0.301 |
| H=128 B64 64K | 3.386 | 2.970 | 0.546 |

-12% at all three shapes. H=16 is just above the predicted 0.36-0.39 ms, H=128 B16 inside 0.72-0.77. The softmax is
95% busy (predicted 93% or more) and waits for S 3.5% of its samples (5% or less). Behind it: the scale warps 89%
busy, the convert warps 48% busy and 45% waiting for PV to release V.

The v3.0 rule (0.35 ms or less) still fails. It asked whether a cheap conversion alone lets the softmax set a period
of 1.20 us, the period of v2.4 without conversion. The softmax does set the period now, but its work is 1.48 us per
tile (95% of 1.56), not 1.15: the convert and scale warps share its sub-partitions (14% of its samples ready but not
selected), and it multiplies P by 2^k. The next step is therefore the softmax's own work (V3.1: 2 query terms and
RoPE in the latent accumulator, 32 TMEM columns loaded per tile instead of 80), with the scale warps right behind it.

### V3.1 plan: fewer score columns for the softmax

The softmax loads 80 TMEM columns per tile and thread: 4 query terms x 16 heads of latent score and 16 of RoPE score,
combined as g (S0 + S1 + S2 + S3) + c S_pe. Two changes cut that to 32:

- 2 query terms instead of 4 (32 columns of latent score, and half the latent QK MMA work).
- RoPE accumulated into the first term's columns. The two parts then need one unit: the RoPE query is written as 3
  E4M3 terms in units of g / kpe_ratio times a power of two 2^e per head, and the RoPE MMA becomes
  `kind::mxf8f6f4` (CuTe DSL `MmaMXF8F6F4Op`) with 2^e as its UE8M0 scale factor (1 on the cache side). Without
  2^e, q_pe x kpe_ratio / g reaches 2688 x kpe_ratio (2^14 with the calibrated scales), far past E4M3's 448. The
  RoPE precision stays as now (3 E4M3 terms; the scale is rounded up to a power of two).

**X1, the query terms first.** v1 measured 2 terms at 0.0118 against the dequantized cache (3 terms 0.0014, outlier
channels 2-4x worse), and v3 already adds E4M3 rounding of V and P (x1.024 at worst). Against the unquantized KV the
format's own error dominates, so the question is whether 2 terms still fit the 3% rule there. Measured on the real
kernel: v3.0a with 2, 3 and 4 terms (query_terms gains 2), error against the unquantized KV as a ratio to the CUDA
BF16 mode, on the test shapes and the `accuracy.py` shapes, both global scales, and again with outlier channels in the
query (each channel x exp(N(0, 1.5)), as in v1). Also the time of each.

Predictions: without outliers, 2 terms at most x1.03 at worst (about +0.5% over 4 terms); with outliers, 2 terms
past x1.03 and 3 terms within it. Time at H=16 B64 64K: 2 terms 0.37-0.39 ms (0.399 with 4), from the 48 columns
loaded instead of 80 and half the latent MMA. Rule: V3.1 uses the fewest terms whose worst ratio, outliers
included, is at most x1.03.

### X1 result: 2 terms are accurate enough, but run slower

Worst ratio to the BF16 mode against the unquantized KV over 8 shapes x 2 global scales:

| query | 2 terms | 3 terms | 4 terms |
|---|---|---|---|
| random query | x1.020 | x1.015 | x1.017 |
| outlier channels | x1.012 | x1.003 | x1.007 |
| time H=16 B64 64K, ms | 0.433 | 0.395 | 0.397 |
| time H=128 B16 64K, ms | 0.823 | 0.747 | 0.748 |

2 terms add about +0.3% at worst, so by the rule V3.1 may use 2 terms. The prediction for outlier channels was
wrong: with them the format's own error grows (the BF16 mode's error is 0.24-0.46 there, 0.08-0.17 without), and the
query terms disappear under it. The time is the surprise: 2 terms are 9-10% slower than 4, 3 terms the same as 4,
against a predicted 3-7% gain. V3.1 keeps 32 score columns only once this is understood: next, the stall samples of 2
against 4 terms.

### V3.1 predictions (written before the first run)

V3.1 as built: RoPE accumulated into the first term's columns (`kind::mxf8f6f4`, query 2^e per head), with the term
count a parameter, timed in one process against v3.0a with 4 and 2 terms (a copy of the previous code), v2.4 and
Cake.

- Accuracy: the same as v3.0a with the same terms, within 0.002 of X1's ratios, since the RoPE query keeps 3 E4M3
  terms.
- Time at H=16 B64 64K: 4 terms (64 columns loaded instead of 80) 0.38-0.39 ms against v3.0a's 0.397; 3 terms (48)
  0.37-0.38. 2 terms: no prediction until the profile explains X1's slowdown.

### V3.1 result: correct, but slower than v3.0a

Accuracy as predicted: worst x1.022 / x1.024 / x1.024 for 2 / 3 / 4 terms, each within about 0.002 of v3.0a with
the same terms. Time, one process, ms:

| shape | v2.4 | v3.0a 4 terms | v3.0a 2 | V3.1 2 | V3.1 3 | V3.1 4 | Cake |
|---|---|---|---|---|---|---|---|
| H=16 B64 64K | 0.456 | 0.400 | 0.434 | 0.421 | 0.407 | 0.433 | 0.320 |
| H=128 B16 64K | 0.853 | 0.750 | 0.824 | 0.790 | 0.763 | 0.811 | 0.299 |

Every V3.1 build is slower than v3.0a with 4 terms, and V3.1 with 4 terms, which loads 64 columns instead of 80, is
8% slower than v3.0a with 4 terms. The predictions (0.38-0.39 and 0.37-0.38 ms) fail. The softmax's work does not
explain the order of these times.

The 2- against 4-term profile of v3.0a points elsewhere. With 2 terms the softmax executes fewer instructions (81.4M
against 85.4M) but takes 1.66 us per tile instead of 1.50; its hottest instructions are the loads of the tile maxima
from shared memory and the compares that wait on them. The hardware's count of shared-load bank conflicts is 3.8x
(12.75M against 3.31M) and the shared-load wavefronts +37%, while the per-instruction counts (ideal and excess
wavefronts of each load) are identical in both builds. So the extra conflicts are not in any instruction's own
address pattern; they come from loads colliding with other traffic. The only other change with 2 terms is that the
query buffer is 8 KB smaller, so every buffer after it moves 8 KB down. V3.1 moves the buffers again (two RoPE
scale-factor tiles in, the RoPE scale out).

### Is it the shared-memory placement?

Test: V3.1 with padding of 0-8 KB after the query buffer (1 KB steps; only the placement changes), for 2, 3 and 4
terms as far as shared memory allows, H=16 B64 64K, plus the conflict count of the fastest and the slowest.

Prediction: if the placement explains X1 and V3.1, the time at a fixed term count varies by 5% or more across the
paddings, and the best placement for 2 and 3 terms is at least as fast as v3.0a with 4 terms (0.400 ms).

Result: no. Padding of 0-8 KB moves V3.1 by at most 1% (2 terms 0.414-0.419 ms, 3 terms 0.404-0.405, 4 terms
0.430 at 0 and 1 KB). The placement is not the cause; the prediction fails.

What differs between the 2- and 4-term builds is in the softmax's stall reasons: MIO throttle 370 samples against
14, all of it on the loads of the tile maxima (`LDS.128`, 686 samples against 307 for the same count) and the
compares waiting on them. These are broadcast loads, 16 per thread and tile, issued right after the named barrier,
and they queue behind the convert warps' shared-memory instructions in the same sub-partitions. How long they wait
depends on how the two roles' instruction streams happen to line up, which the term count and the fold change. The
exchange of tile maxima (16 warp reductions, a shared-memory round trip, 16 loads, 64 compares per thread) is the
softmax's most expensive and least predictable step.

### V3.1b hypothesis: the tile maximum only when a score grows

With the lazy rescale, the exchange decides nothing on most tiles: a head's maximum moves only when a score passes it
by 2^5. Each thread can test that on its own scores against the running maximum (the same in every thread), and one
barrier with an OR reduction (`bar.red.or.pred` on the softmax's named barrier) tells the warpgroup whether any did.
Only then the exchange runs, as now. On the other tiles there are no reductions, no shared-memory round trip and no
second exp2 (the correction factor is 1).

Predictions (H=16 B64 64K, one process against V3.1 and v3.0a): the softmax's work per tile falls from 1.50-1.66 us
to 1.1 us or less, and the period is then set by the scale warps (about 1.39 us of work per tile now): 0.36-0.38 ms
for 2, 3 and 4 terms, within 3% of each other. Accuracy unchanged (the same P, maxima and rescales).

### V3.1b result: Cake's time at H=16

Accuracy unchanged (the same errors as V3.1 to four digits). Time, one process, ms:

| shape | v2.4 | v3.0a 4 terms | V3.1 3 terms | V3.1b 2 terms | V3.1b 3 terms | V3.1b 4 terms | Cake |
|---|---|---|---|---|---|---|---|
| H=16 B64 64K | 0.456 | 0.400 | 0.407 | 0.354 | **0.319** | 0.330 | 0.319 |
| H=128 B16 64K | 0.851 | 0.748 | 0.763 | 0.649 | **0.574** | 0.596 | 0.304 |

V3.1b with 3 terms runs as fast as Cake at H=16 B64 64K, -30% against v2.4 and -20% against v3.0a. The prediction
(0.36-0.38 ms) was too cautious, and the term counts are 11% apart, not within 3%. The profile matches the rest of
it: the softmax is now 43-59% busy and waits for S most of the remaining time, and the scale warps set the period
(86-92% busy), with the MMA warp waiting for their tiles 38-49% of its samples. 2 terms are the slowest again (the
scale warps 92% busy, more of it in dependency stalls), so v3 uses 3 terms.

At H=128 the gap to Cake stays 1.9x: 8 CTAs per request each read the whole cache for 16 heads. That is V3.2.

### V3.2 hypothesis: 32 heads per CTA

At H=128 each request has 8 CTAs of 16 heads, and each of them loads, converts and scales the whole cache, so the
cache is read 8 times per request. With 32 heads per CTA it is read 4 times, and the conversion and scale-layout work
per head halves. v2.5 tried this (`h32`) and gained 1-2% because its softmax was then 99% busy. V3.1b's softmax is
43-59% busy at 16 heads; at 32 heads its work about doubles per tile while every other role's stays the same.

Changes, as v2.5's `h32`: 3 raw stages instead of 4 (shared memory), the per-role registers 104 / 104 / 232 / 72, the
softmax reads g from shared memory. The 2^k buffer keeps 4 slots (tile j % 4): with 3 stages, a slot per stage would
be rewritten before the softmax has read it. H=16 keeps 16 heads per CTA.

Predictions: H=128 B16 64K 0.33-0.40 ms (0.574 with 16 heads; Cake 0.304), the period set by the softmax (85% or
more busy) at about 1.3-1.5 us per tile; H=128 B64 64K about 4x that of B16 less the better wave fill; H=16 unchanged.
Accuracy unchanged.

### V3.2 result: -27% at H=128, still 1.4x behind Cake there

Accuracy unchanged (worst x1.024; with 32 heads the same or slightly lower ratios). Time, one process, ms:

| shape | v2.4 | v3 16 heads | v3 32 heads | Cake |
|---|---|---|---|---|
| H=128 B16 64K | 0.858 | 0.580 | **0.423** | 0.302 |
| H=128 B64 64K | 3.387 | 2.284 | **1.481** | 0.555 |
| H=128 B64 8K | 0.497 | 0.360 | **0.305** | 0.083 |
| H=16 B64 64K | 0.456 | 0.323 | 0.324 (16 heads) | 0.321 |

-27% at H=128 B16 and -35% at B64, short of the predicted 0.33-0.40 ms. At H=128 B16 a tile takes 1.65 us and no
role is saturated: the softmax is 78% busy (18% waiting for S), the convert warps 64% (31% waiting for PV to release
V), the MMA warp 52%. The period is the latency of the loop P(j) -> PV(j) -> V buffer free -> convert(j + 1) ->
stage free -> scale layouts -> QK -> S, not one role's work. At 32 heads the softmax's work per tile (about 1.3 us)
is most of that loop.

The remaining gap at H=128 is structural. Cake's kernel for more than 48 heads is a different program: a cluster of
two CTAs with `cta_group::2` MMAs takes all 128 heads of a request, so the cache is read and its V converted once per
request; this kernel does that 4 times (one CTA per 32 heads). At B64 8K that is 3.7x.

### The epilogue: most of the fixed cost

V3.2 with 32 heads was 2x slower than with 16 at one request and short contexts (H=128 B1 1K: 0.075 against 0.036
ms). With one tile per CTA the kernel is mostly its prologue and epilogue, and the epilogue was long: per output
element it indexed the full output tensor (its dynamic strides reloaded, about 50 instructions), branched on the split
count and on l > 0, and loaded and summed the four warps' row sums again; 3,600 SASS lines with 16 heads, 7,000 with
32. Now the output scale per head is computed once, the CTA's rows are one static (heads, 512) view, and the split
branch is outside the loops. One tile, warm L2: 18.2 -> 10.4 us with 16 heads, 33.1 -> 14.9 us with 32. A whole call
at H=16 B1 1K: 0.040 -> 0.017 ms, so most of the "40 us fixed cost per call" of earlier sections was this epilogue.

With it, 32 heads per CTA lose only where the batch holds little work (H=128: B1 up to 16K tokens, B16 1K), where
16 heads give twice the CTAs and no KV split: v3 now uses 32 heads from 32K tokens in the batch.

One process against the build before, ms: H=128 B16 64K 0.418 -> 0.379, H=128 B64 16K 0.483 -> 0.381, H=16 B16 64K
0.105 -> 0.099, but H=16 B64 64K 0.320 -> 0.339. The epilogue runs once per CTA, so that is not its cost: in the
profile the scale warps, which set the period at H=16, went from 85% to 94% busy, their dependency stalls from 19% to
31% of their samples. Their code did not change; the compiler scheduled it differently. A role at the edge of the
period shows every such change.

### Hypothesis: the 2^-k shift as integer arithmetic

The scale warps shift every block scale by 2^-k through f16 (two conversions to f16, a multiply, two conversions back
per word: 4 of the 7 instructions on the narrow conversion pipe, "math" 10-12% of their stalls). On the E4M3 bytes
it is the exponent field minus k: with bit 7 set first the subtraction cannot borrow across bytes, and bit 7 of
(byte - 8k - 8) tells which bytes stay normal; the others (blocks more than 2^11 below the token's largest, which the
f16 path rounded to 1-3 bit subnormals) become 0. About 5 integer instructions per word, all on the full-rate pipe.

Predictions: H=16 B64 64K at or below 0.320 ms (the old build) and the scale warps under 85% busy; the error within
0.001 of now; H=128 unchanged within 2%.

Result: no. With the integer shift H=16 B64 64K takes 0.347 ms (0.338 with the f16 shift, 0.320 before the epilogue
change), H=16 B16 64K 0.100 (0.099), H=128 within 1%; the error is unchanged. Fewer instructions, more time: the
prediction fails and the f16 shift stays.

What does help is registers. The scale warps share a warpgroup with the MMA and TMA warps at 80 registers, and the
softmax, at 16 heads only 44-50% busy now, holds 176. Moving 24 per thread from the softmax to that warpgroup (one
process, ms; 176/80 is the build above):

| softmax / MMA-TMA-scale registers | 176 / 80 | 152 / 104 | 144 / 112 | 136 / 120 |
|---|---|---|---|---|
| H=16 B64 64K | 0.339 | **0.326** | 0.325 | 0.328 |
| H=16 B16 64K | 0.099 | **0.097** | 0.098 | 0.099 |
| H=16 B64 16K | 0.110 | **0.107** | 0.108 | 0.108 |
| H=16 B4 64K | 0.044 | **0.043** | 0.044 | 0.045 |

v3 with 16 heads now runs 152 / 104 (the softmax can only grow from the launch's 128, and that warpgroup only shrink).
H=16 B64 64K ends 2% above the build before the epilogue change; every other shape is faster.

### Where v3 stands (full B200 sweep, one process)

v3 is now `sm100.decode`'s default for page sizes of 16 and more (v2.4 below that, or with `version=3`). The version
history on three shapes, ms:

| shape | v2.4 | v3.1 (16 heads) | v3.2 | Cake |
|---|---|---|---|---|
| H=16 B64 64K | 0.455 | 0.327 | 0.327 | 0.319 |
| H=16 B1 64K | 0.049 | 0.031 | 0.032 | 0.022 |
| H=128 B16 64K | 0.857 | 0.608 | 0.378 | 0.305 |
| H=128 B64 64K | 3.387 | 2.387 | 1.383 | 0.549 |

Against Cake over the 32-shape grid: at 16 heads 1.03x at 64 x 64K, 1.16-1.31x at 64K tokens with batch 1-16,
1.23-1.74x at 16K (the smaller the batch, the larger), 1.5-1.9x at 1K-4K (the three small kernels around the decode,
about 8 us warm, are inside this kernel's timed call; Cake's query quantization is not); at 128 heads 0.89-1.02x at
batch 1 with 1K-4K tokens, 4 x 1K and 16 x 1K, 1.1-1.5x in the other shapes up to batch 16, 2.5-2.9x at batch 64. trtllm-gen FP8 is 1.2-3.1x faster than v3 at 128 heads. Error against
the unquantized KV: 1.01-1.02x the BF16 mode (Cake 1.38-1.71x).

### This kernel and Cake: where the structures differ

FlashInfer Cake (0.7.4rc1, `flashinfer/experimental/cake_mla_nvfp4_paged_decode`) reads the same cache. Its source
has two attention programs; side by side with v3:

| | v3 (this kernel) | Cake, up to 48 heads (`d2829e4e`) | Cake, more heads (`32a47bbf`) |
|---|---|---|---|
| query | BF16 query as 3 NVFP4 terms, one scale per head; RoPE as 3 E4M3 terms times 2^e per head | 1 NVFP4 term; RoPE as 1 E4M3 term in the latent unit | same as left |
| QK | `mxf4nvf4`, the terms along N; RoPE `mxf8f6f4` (UE8M0 2^e) into the first term's columns | `mxf4nvf4` and RoPE `f8f6f4` into one score block | `cta_group::2`, the query as the TMEM A operand |
| heads per CTA | 16, or 32 from 32K tokens in the batch | 16 / 32 / 48 rows | 128 rows per cluster of 2 CTAs |
| cache reads per request at 128 heads | 4 (8 at 16 heads per CTA) | | 1 |
| V conversion | 8 warps, thread = channel: transposing `ldmatrix`, QMUL4, `tcgen05.st` into TMEM; V^T is PV's A operand in TMEM | 8 warps, thread = token: `ld.shared.v4`, QMUL4, `st.shared.v4` token-major; V is PV's A operand in shared memory | shared by the 2 CTAs |
| V range | per-token 2^-k in the block scales, undone on P | the same | the same |
| QK scale factors of the cache | 2 warps lay them out in shared memory, `tcgen05.cp` | the conversion warps write them into TMEM (`tcgen05.st`, lane = token) | |
| MMA issue | one warp for QK and PV | a QK warp and a PV warp | |
| softmax | one warpgroup; lazy rescale at 2^5, maxima exchanged only when a score grows | two warpgroups on alternate tiles, each with its own O | |
| raw stages | 4 (3 with 32 heads), TMA | 2, TMA | |
| warps | 16 | 20 | 16 |
| error vs unquantized KV, x BF16 mode | 1.01-1.02 | 1.38-1.71 | 1.38-1.71 |

What the differences are worth, on B200:

- **Query terms** (3 against 1): the accuracy column above. At 16 heads the time is small (3 terms ran as fast as 4,
  X1); at 128 heads it is the larger part of the gap (see "The tensor floor at 128 heads" below).
- **V in tensor memory** (this kernel) against V in shared memory (Cake): it spares the shared-memory round trip (Cake's
  shared memory was 91% busy at 16 heads in its profile, this kernel's 68% in v2.4) at the price of a transposing load
  and a nibble shuffle. The conversion is no longer the limit (48-64% busy).
- **One MMA warp** against two: 40-54% busy in v3, so not a limit now (v2.6 showed when it becomes one).
- **One softmax warpgroup** against two: with the vote it is 44-59% busy at 16 heads. At 32 heads (78% busy) the
  period is the latency of the loop softmax -> PV -> conversion -> scale layouts -> QK; two warpgroups on alternate
  tiles, as Cake, would overlap that loop. Not tried yet.
- **The cache's QK scale factors**: here two warps lay them out in shared memory for `tcgen05.cp`, and they set the
  period at 16 heads (85-94% busy; the register split above bought 4%). Cake's conversion warps write them into tensor
  memory directly (`tcgen05.st`, lane = token), which would remove that role's shared-memory round trip.
- **128 heads**: the cluster of two CTAs reads the cache once per request, this kernel once per 32 heads, 4x the
  conversion and scale work per request. With the query terms (below) this is the 2.5-2.9x at batch 64.

### V3.3 hypothesis: the scale layouts in the convert warps, the QK scale factors straight into tensor memory

At 16 heads the two scale-layout warps set the period. In the V3.1b profile they are 92.5% busy, of which 23% of their
samples are "not selected" (the convert and softmax warps of the same sub-partitions issue instead) and 30% dependency
stalls; their stores of the QK scale factors (`sfa_w[tk % 32, tk // 32, a, dx]`) step 4 words per lane, a 4-way bank
conflict (2.9-way on average, 64% of their store wavefronts excess). Two changes, both as Cake does them:

- The cache's QK scale factors (SFA) go to tensor memory directly with `tcgen05.st` from a thread that holds the
  token: lane = token, 4 scales per column, word w of token m in column 4w + m / 32 (Cake's layout). Two SFA buffers
  of 32 columns, so QK(j) reads one while tile j + 1's is written. The 8 conflicting stores per token, the SFA buffers
  in shared memory and the MMA warp's `tcgen05.cp` per tile go.
- A warp can only write the TMEM lanes of its own quadrant, and the scale warps sit in quadrants 2 and 3, so the work
  moves to the 8 convert warps (two per quadrant; a thread takes one token, half of its 32 scales). They spend 42-45%
  of their time waiting for PV to release V. Per tile: TMA(j + 1) landed -> scale layouts of j + 1 (after QK(j - 1)
  released its SFA buffer and convert(j - 1) its derived buffer) -> QK(j + 1) may start -> convert(j). Warps 14-15
  are left idle.

Predictions (one process): H=16 B64 64K from 0.327 to 0.28-0.30 ms (4.9-5.3 TB/s, 67-72% of the achievable 7.3), the
convert warps 85% busy or less, H=128 the same or faster, the error unchanged (the same values reach the MMAs).

### The tensor floor at 128 heads

The paragraphs above put the 128-head gap on the cache being read 4 times per request. That is incomplete: the V3.2
profile at H=128 B16 already shows the tensor pipe as the most used one (41% of its issue-weighted capacity; the MMAs'
FLOP are 11% of the FP4 and FP8 peaks), and the 3 query terms are 3x Cake's QK work, whatever the structure.

`benchmarks/tcgen05_peaks.py` measures the MMA rates on B200: one thread per SM issues the shapes back to back,
operands resident, no loads.

| MMA (M = 128) | TFLOPS | of the peak |
|---|---|---|
| FP4 `mxf4nvf4`, N = 256 | 8926 | peak |
| QK, 3 terms x 32 heads (N = 96) | 7641 | 86% |
| QK, 2 terms (N = 64) | 5941 | 67% |
| QK, 1 term (N = 32) | 3564 | 40% |
| RoPE, block-scaled FP8 `mxf8f6f4`, N = 32 | 1781 | 40% of FP8 |
| FP8 `f8f6f4`, N = 256 | 4466 | peak |
| PV, N = 32, A (V) in tensor memory | 4368 | 98% |

The FP8 peak is 4.5 PFLOPS here, not the 3.1 cuBLAS reached in a GEMM; FP4 is 2x it. Small N costs: an MMA with 32
columns runs at 40%, except with A in tensor memory.

Tensor work per request and tile (128 tokens x 128 heads) and the floor at H=128 B64 64K (32768 such tiles) at the
peaks, whatever the structure:

| query | FP4 (QK) | FP8 (RoPE, PV) | floor | measured |
|---|---|---|---|---|
| 3 terms (v3) | 50.3 MFLOP | 23.1 MFLOP | 0.354 ms | 1.384 ms (26%) |
| 2 terms | 33.6 | 23.1 | 0.292 ms | |
| 1 term, 1 RoPE term (Cake) | 16.8 | 18.9 | 0.200 ms | 0.549 ms (36%) |

At the shapes this kernel issues (32 heads per CTA) the 3-term floor is 0.457 ms (the RoPE MMAs at N = 32 alone 0.116
ms). The measured gap, 2.52x, is 1.77x of tensor work (the query terms: QK 3x, RoPE 3x, PV the same) times 1.42x of
utilization (26% of the floor against Cake's 36%; reading and converting the cache once per 32 heads is in this part).
In logarithms the query terms are about 60% of the gap. So no structure brings the 3-term query to Cake's time at
batch 64 and 128 heads without reaching a higher share of the tensor peak than Cake does; it is the price of an error
of 1.01x the BF16 mode against Cake's 1.4-1.7x. With 2 terms (X1: x1.02) the work is 1.46x Cake's.

A kernel with the heads as M (as Cake's `32a47bbf` and FlashAttention 4: the query as the TMEM A operand, thread =
head in the softmax, no reduction across threads, the two CTAs of a pair converting half the tokens each) would read
the cache once per request. At Cake's utilization (36% of the floor) it would take 0.81 ms with 2 terms and 0.98 with
3: 1.5-1.8x Cake.
Such a kernel needs 2 terms as the 128-head default, and first an answer to why 2 terms ran slower than 3 here (X1,
V3.1b; the scale warps' scheduling, not yet understood).

### V3.3 result: slower; the convert warps become the limit

Correct (the same error as V3.2), but slower everywhere. One process, ms:

| shape | V3.2 | V3.3 | Cake |
|---|---|---|---|
| H=16 B64 64K | 0.328 | 0.363 | 0.319 |
| H=16 B16 64K | 0.098 | 0.103 | 0.087 |
| H=128 B16 64K | 0.381 | 0.427 | 0.303 |
| H=128 B64 64K | 1.381 | 1.580 | 0.543 |

The prediction (0.28-0.30 ms, convert 85% busy or less) fails. The convert warps are now 85% busy and wait for PV 4% of
their samples (from 42%); the MMA warp waits for V 29%. Their 42% of waiting was not free time: the scale layouts of
tile j + 1 now run before the conversion of tile j, so every tile's V comes later and the loop that sets the period
gets longer. Writing the QK scale factors straight into tensor memory itself works (the tests pass with Cake's layout).

### V3.3b hypothesis: only the QK scale factors in the convert warps

Keep the split, move less. The convert warps write the cache's QK scale factors into tensor memory (a thread loads 4
raw words of its token, 4 `tcgen05.st`; about 15 instructions per tile) and publish the stage for QK; the scale warps
keep k, the shifted block-major bytes and 2^k, without the 8 conflicting SFA stores per token. QK then waits for the
TMA and the convert warps' short step, not for the scale warps.

Predictions (one process against V3.2): H=16 B64 64K 0.30-0.32 ms; the scale warps under 85% busy, the convert warps
under 60%; H=128 the same as V3.2 or faster.

### V3.3b result: no gain; both variants stay out of the code

Correct (the same error). One process, ms:

| shape | V3.2 | V3.3b | Cake |
|---|---|---|---|
| H=16 B64 64K | 0.328 | 0.333 | 0.319 |
| H=16 B16 64K | 0.097 | 0.098 | 0.087 |
| H=128 B16 64K | 0.380 | 0.387 | 0.299 |
| H=128 B64 64K | 1.383 | 1.418 | 0.547 |

1-3% slower at the long shapes, the same at the short ones; the prediction (0.30-0.32 ms) fails. In the profile the
convert warps are 82% busy and the MMA warp waits for V 49% of its samples: the short step in front of the
conversion still waits for the next tile's TMA and the release of tile j - 2, and every such wait delays V. The scale
warps lost their conflicting stores but stay 92% busy, with more of it "not selected" (26%), since the convert warps
now issue more in the same sub-partitions.

What the two variants show: the convert warps have no slack. Their 42% of waiting in V3.2 is the single V buffer's
hand-off with PV, the loop that sets the period, so anything placed on their path lengthens it. The cache's QK scale
factors can go into tensor memory directly (the layout above works), but not from those warps. The code stays at V3.2;
the next attempt at 16 heads should take work off the scale warps without putting it on the conversion's path (for
example their 32 single-byte stores per token, which a transpose of 4 x 4 bytes across lanes would turn into 8 word
stores).
