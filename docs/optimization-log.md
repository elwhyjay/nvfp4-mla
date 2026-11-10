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
- At 64K the kernel gain was eaten by the split policy. A larger wave made the policy choose 11 splits instead of 8,
  which adds about 50 MB of fp32 partial results written and read by the combine kernel. The policy's cost model
  has no term for combine traffic, which grows with the number of rows (8x larger at H=128 than at H=16).
- Decision: keep 3 stages (the rule asked for 10% at the target shape). Two findings carry forward: the split
  policy should count combine traffic, and the non-tensor instruction stream (679 instructions per warp per tile
  against 36 MMA) is the limit, not the number of warps.

### Later: B200

FlashInfer Cake (`tcgen05`, NVFP4 query, FlashInfer 0.7.4rc1) on B200:
- Cake runs H=16 B64 64K in 0.32 ms against v5's 0.74 ms, but at only 63% of DRAM bandwidth.
- Its FP4 query raises the error against the BF16 reference from 0.08–0.12 (this kernel) to 0.13–0.17.
- Hypothesis: a tcgen05 kernel that keeps BF16 Q stays memory-bound at H=16 and can be both faster and more accurate.
  At H=128, BF16 Q puts its compute floor (spec BF16 tensor peak) near Cake's time, so that shape would need a low-precision query too.
