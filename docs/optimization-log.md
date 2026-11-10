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

### v11 results and the end of the SM120 work

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

**3. Conversion floor: confirmed.** After 1 and 2, H=128 B64 64K takes 2.79-2.86 ms in the FP8 mode and 2.71 ms
in the FP4 QK mode (predicted 2.6-2.7 ms), against 2.30 ms for XQA. The conversion is now 46% of the FP8 mode's
instructions. On SM120 this kernel does not overtake XQA's FP8 cache at H=128.

SM120 summary at H=128 B64 64K: v5 (BF16) 4.81 ms -> v6 (FP8 MMA) 3.51 -> v9 (QK by token) 3.01 -> v11 2.79-2.86
(FP8) / 2.71 (FP4 QK). At H=16 the BF16 path stays at 85% of DRAM bandwidth and is the fastest MLA decode measured on
SM120.

### Later: B200

FlashInfer Cake (`tcgen05`, NVFP4 query, FlashInfer 0.7.4rc1) on B200:
- Cake runs H=16 B64 64K in 0.32 ms against v5's 0.74 ms, but at only 63% of DRAM bandwidth.
- Its FP4 query raises the error against the BF16 reference from 0.08–0.12 (this kernel) to 0.13–0.17.
- Hypothesis: a tcgen05 kernel that keeps BF16 Q stays memory-bound at H=16 and can be both faster and more accurate.
  At H=128, BF16 Q puts its compute floor (spec BF16 tensor peak) near Cake's time, so that shape would need a low-precision query too.
