# NVFP4 MLA Decode Kernel 계획서

작성: 2026-09-23 · 개정: 2026-09-24 (SM100/SM120 동시 타깃, 선행 구현 조사 반영) · 상태: **v0–v4 완료 (2026-09-28)**, 진행 기록은 §17

## 1. 한 줄 요약

flashinfer#4676에서 추가한 NVFP4 MLA paged KV cache 레이아웃(ckv FP4 + kpe FP8, 352 B/token)을 **직접 읽는 dense decode attention 커널을 CUDA로 scratch부터 구현**한다. 타깃은 **SM120(RTX PRO 6000)과 SM100(B200) 둘 다**이며, 코드 경로는 mma.sync 기반 하나로 두고 tile/stage 예산만 arch 상수로 바꾼다. 개인 레포에서 완성하고, 이후 FlashInfer upstream PR로 가져간다. 개발과 측정은 Modal에서 한다.

## 2. 왜 이 주제인가

- 이력서의 약점은 "scratch부터 작성하고 튜닝한 커널이 없다"는 점이다. upstream PR은 기존 코드 구조를 따라야 해서 이 증거를 만들기 어렵다.
- #4676 본문에 이미 *"Dense MLA decode does not yet consume this 352-byte layout … Decode support … are follow-ups"*라고 적었다. 캐시 레이아웃 설계에서 그 레이아웃을 소비하는 커널로 이어지는 서사가 생긴다.
- 기존 템플릿에 끼울 수 없는 레이아웃이라 FP4 dequant, tensor core mma, 비동기 파이프라인, split-KV를 직접 설계해야 한다. 커널 면접에서 묻는 요소가 모두 들어 있다.
- 2026-09-24 조사(§4.3) 기준 dense NVFP4-KV MLA decode는 TRT-LLM `fp4_mla`(Triton/CuTe DSL, SM100/103/107 대상)뿐이고 **SM120에는 dense FP4 커널이 없다.** FlashInfer dense wrapper는 #4676 캐시를 받으면 "no backend has an NVFP4 MLA decode kernel"로 raise한다. 따라서 주장은 "첫 NVFP4 MLA decode"가 아니라 **"SM120 dense NVFP4-KV MLA decode 첫 구현, FlashInfer paged 레이아웃 직접 소비, BF16 Q 유지, 단일 코드 경로로 SM100까지"**로 좁힌다.

## 3. 성공 기준

| 구분 | 기준 |
|---|---|
| 정확도 | BF16 reference 대비 rel-L2가 #4676 측정치(0.092–0.102) 수준. dequant 구현 자체의 오차는 reference dequant 대비 bitwise 또는 1e-3 이하 |
| 성능 (memory-bound 구간) | SM120: Step 0 달성 가능 BW 대비 **75% 이상**. SM100: fa2 BF16 대비 BW% 우위 + trtllm-gen FP8 대비 latency (mma.sync 경로 상한 때문에 §10 참고) |
| 비교 | §10의 베이스라인 대비 latency. SM120: XQA-MLA FP8(H=128), fa2 BF16, Triton MLA fp8(H=16). SM100: trtllm-gen FP8/BF16, cutlass BF16. 이상치(1.64× vs FP8)와 실측 차이를 설명할 수 있어야 함. 베이스라인 실측은 §10에 완료 |
| 산출물 | 레포 + NCU 근거가 있는 기술 블로그 1편 + upstream PR (또는 PR용 설계 이슈) |

MVP(3주): v0–v3까지, batch=1 고정 길이, SM120에서 정확도와 BW%까지. SM100은 같은 코드가 돌고 수치가 기록되는 수준. 이것만 있어도 면접 소재로 충분하다.

## 4. 배경

### 4.1 캐시 레이아웃 (#4676)

```
ckv_cache    : uint8          [num_pages, page_size, 256]  # 512 ch, E2M1 2개/byte
ckv_sf_cache : float8_e4m3fn  [num_pages, page_size, 32]   # 16 ch당 scale 1개
kpe_cache    : float8_e4m3fn  [num_pages, page_size, 64]   # RoPE 채널, FP8
```

- token당 256 + 32 + 64 = 352 B (FP8 layout 576 B, BF16 1152 B)
- global scale은 0.7.0에 이미 있다: `ckv_scale`, `kpe_scale` 인자. dequant = `e2m1 * sf * ckv_scale`, `fp8 * kpe_scale`.
- 확인 완료(§17): block scale `sf = e4m3(amax / (6·ckv_scale))`, E2M1은 RNE, 하위 nibble = 짝수 채널, ckv는 `precise_rounding=true` 경로.

### 4.2 MLA absorbed decode 연산

weight absorption 이후 decode는 모든 head가 latent KV 하나를 공유하는 MQA 형태가 된다.

```
q      : [H, 576]  = [q_nope_absorbed (512) | q_pe (64)]
score  : s[h,t] = q_nope[h] · ckv[t] + q_pe[h] · kpe[t]      (scale 적용)
p      = softmax_t(s)
out    : [H, 512] = p @ ckv                                  # V = ckv 자체
```

- K와 V가 같은 ckv를 쓴다. **한 번 읽은 KV tile을 QK와 PV 양쪽에 재사용**해야 한다.
- NVFP4 scale은 16채널 단위다. QK에서는 reduction 축(채널) 안에, PV에서는 token마다 다르다. 두 경우 모두 scale을 mma 밖으로 꺼낼 수 없다. **따라서 mma 전에 register 안에서 dequant해야 하고, dequant throughput이 핵심 설계 과제가 된다.**

### 4.3 선행 구현 조사 (2026-09-24, 소스 검증)

vLLM·SGLang은 자체 MLA 커널이 없고 아래 라이브러리를 감싼다. 엔진이 아니라 커널 API를 베이스라인으로 삼는다.

| 구현 | arch | KV | 형태 | 비고 |
|---|---|---|---|---|
| TRT-LLM `fp4_mla` (Triton + CuTe DSL "mufu16") | Blackwell. 문서는 sm100/103, CI는 SM107(Rubin). **SM120 없음** | NVFP4 KV, **Q와 P도 FP4** | dense | block-scaled tensor core, KVCM V2 전용, page 128, chunked prefill 불가. 정확도 비용이 우리와 다른 지점 |
| FlashInfer experimental Cake (#5443, 09-24 main 머지, 0.7.0 미포함) | SM100/103 | NVFP4 Q + KV | dense, **DSv4 geometry**(512, RoPE 없음) | 레이아웃이 달라 직접 비교 불가 |
| vLLM FlashMLA fork `nvfp4_ds_mla` (`csrc/kernels/sm100/decode/sparse/nvfp4_head64/`) | SM100 | **#4676과 동일 352 B 레이아웃** | sparse(top-k) | upstream FlashMLA에 없음. FP4→mma fragment 경로의 prior art로 반드시 읽는다 |
| FlashMLA upstream V4.1 e2m1 (288 B) | SM100 | FP4 (`extra_k_cache`만) | sparse | dense decode는 SM90 전용 |
| FlashInfer SM120 sparse (`_sparse_mla_nvfp4_sm120`) | SM120/121 | NVFP4 (DSv4 384 B) | sparse | |
| TRT-LLM sparse NVFP4 / SGLang `--kv-cache-dtype nvfp4` | SM100 / SM100+SM120 | NVFP4 **저장만** | 커널 전 dequant → FP8/BF16 커널 | "dequant-then-kernel" 베이스라인. 우리 v0와 동치 |
| FlashInfer dense wrapper + #4676 캐시 | — | — | — | `NotImplementedError`. **빈 자리** |

SM120 dense MLA decode 현황: vLLM·SGLang 기본값은 Triton MLA. vendor 커널은 TRT-LLM XQA-MLA JIT(`mla_sm120.cu`, FP8 q/kv)뿐이며 FlashInfer `trtllm_batch_decode_with_kv_cache_mla(backend="xqa")`로 노출된다. trtllm-gen cubin 7273개 중 sm120은 0개.

### 4.4 타깃 하드웨어 (Modal 실측, 2026-09-24)

| 항목 | SM120 RTX PRO 6000 | SM100 B200 |
|---|---|---|
| SM 수 | 188 | 148 |
| smem/SM (블록 opt-in 최대) | 100 KB (99 KB) | 228 KB (227 KB) |
| threads/SM | 1536 (48 warp) | 2048 (64 warp) |
| L2 | 128 MB | 126 MB |
| DRAM | GDDR7 12.48 GHz, 스펙 ~1.6 TB/s | HBM3e 4.0 GHz, 스펙 8 TB/s |
| SM clock 최대 | 2.43 GHz | 1.97 GHz |
| vec_add 스모크 (ncu) | DRAM 93%, 1.48 TB/s | DRAM 40%, 3.28 TB/s (포화 안 됨, Step 0에서 큰 버퍼로 재측) |
| Modal 가격 | $3.03/h | $6.25/h |

ncu 하드웨어 카운터는 두 GPU 모두 정상(ncu 2025.2.1, `--clock-control none`). 클럭 고정이 불가하므로 절대 latency보다 **같은 조건에서 잰 달성 BW 대비 비율**을 주 지표로 쓴다.

## 5. Roofline 먼저

token당 FLOPs = H × q_len × (2·576 + 2·512) = 2176 · H · q_len
arithmetic intensity (NVFP4) = 2176 · H · q_len / 352 ≈ **6.2 · H · q_len flop/B**

| 설정 | intensity | 예상 영역 |
|---|---|---|
| H=16, q_len=1 (V2-Lite, TP 등) | ~99 | memory-bound 근처 |
| H=16, q_len=4 (MTP/spec) | ~396 | compute-bound |
| H=128, q_len=1 (V3 full heads) | ~791 | compute-bound |

- ridge point = BF16 dense tensor peak / 달성 BW. **GPU별로** Step 0에서 실측한다(스펙 수치를 그대로 믿지 않는다). SM100은 BW가 5배라 ridge가 훨씬 오른쪽에 있다. 같은 H·q_len에서 SM120은 compute-bound, SM100은 memory-bound일 수 있다.
- 결론: **FP4의 이득은 memory-bound 구간(작은 H, q_len=1)에서만 나온다.** compute-bound 구간에서는 dequant ALU 비용이 추가돼 FP8보다 느릴 수도 있다. 이 trade-off를 두 GPU에서 측정으로 보여주는 것 자체가 블로그와 면접의 핵심 내용이다.
- memory-bound 이상치: FP8 대비 576/352 = 1.64×, BF16 대비 3.27×.

## 6. 범위

**포함**
- SM120과 SM100 둘 다 1급 타깃. 커널은 mma.sync(BF16) 공통 경로 하나, tile 크기와 stage 수만 arch 상수(§7.2). SM80/SM89는 컴파일만 유지
- paged KV, page table, 가변 길이 batch
- q_len 1–4 (MTP verify까지)
- split-KV + combine kernel

**제외**
- prefill / KV append (#4676이 이미 담당)
- tcgen05 / TMEM (SM100 전용). SM100에서도 mma.sync로 간다. vendor 커널(tcgen05 기반)보다 느릴 수 있음을 받아들이고 그 격차를 측정으로 보고한다
- multi-GPU, sparse MLA

## 7. 설계 초안

### 7.1 Work partition

- CTA 1개 = (request, head group, KV split). KV tile 하나를 head group 전체가 공유한다.
- H=16: head group = 16 → mma의 M=16에 딱 맞는다. KV는 한 번만 읽힌다.
- H=128: head group을 나누면 KV를 group 수만큼 다시 읽는다(L2 hit 기대). 반대로 한 CTA에 64 head 이상을 담으면 Q와 accumulator가 register/smem 한계에 걸린다. **v5에서 group 크기를 sweep**해서 결정한다.

### 7.2 Tile과 smem 예산

| 항목 | SM120 (100 KB/SM) | SM100 (228 KB/SM) |
|---|---|---|
| Q (bf16) 16 × 576 × 2 | 18 KB, 1회 load | 18 KB |
| KV tile packed, T=64 | 22.5 KB/stage | 22.5 KB/stage |
| 기본 구성 | 3-stage → 86 KB, **1 CTA/SM** (한계 근접) | 같은 구성 2개 → 172 KB, **2 CTA/SM** |
| 대안 | T=32 4-stage → 63 KB | T=128 3-stage → 153 KB, 1 CTA/SM. 또는 6-stage T=64 |
| warp 상한 | 48 warp/SM | 64 warp/SM |

- 첫 구현은 SM120 구성(T=64, 3-stage)을 두 GPU에 공통으로 쓴다. SM100은 자동으로 2 CTA/SM이 된다. **v5에서 SM100 전용 tile/stage sweep**을 돌려 어느 쪽이 BW%가 높은지 정한다.
- dequant된 BF16 tile(64 × 576 × 2 = 73 KB)은 smem에 올리지 않는다. **packed 상태로 smem에 두고 register fragment 단위로 dequant**한다.
- ldmatrix는 16-bit 레이아웃을 가정한다. packed FP4에서 mma fragment 레이아웃으로 가는 load 패턴은 직접 설계해야 하며, 이 부분이 가장 어렵다. vLLM FlashMLA fork의 `nvfp4_head64` 커널(같은 352 B 레이아웃)이 이 문제를 어떻게 풀었는지 먼저 읽는다.

### 7.3 Dequant

- 1차: LUT(16 entry)나 bit 연산으로 E2M1 → bf16x2, E4M3 scale을 곱한다.
- 2차: 하드웨어 변환 명령(`cvt` e2m1x2 → f16x2 계열)의 SM120/SM100 지원 여부를 PTX ISA에서 확인하고 사용한다.
- NCU에서 dequant 명령이 issue slot의 몇 %를 쓰는지 두 GPU에서 측정한다.

### 7.4 Softmax / accumulate

- online softmax(running max, sum). score는 fp32.
- out accumulator: 16 × 512 fp32 = 8192개. 4 warp 기준 thread당 64 register. 8 warp로 N을 나누는 방안과 비교한다.

### 7.5 Split-KV

- 긴 context에서 batch가 작으면 SM이 놀기 때문에 KV 축을 분할한다(flash-decoding 방식).
- partial (out, lse)를 fp32로 쓰고, 별도 combine kernel로 합친다.
- split 수 heuristic: `num_splits ≈ ceil(num_SMs × occupancy / (batch × head_groups))`, 최소 tile 수를 보장한다. num_SMs와 occupancy는 arch별로 다르다(188 vs 148, 1 vs 2 CTA/SM).

### 7.6 Stretch: block-scaled FP4 mma

TRT-LLM `fp4_mla`가 이 방식이다(Q와 P까지 FP4). SM120에는 tcgen05가 없지만 `mma.sync … kind::mxf4nvf4 .block_scale`이 sm_120a에 있다(CUTLASS SM120 blockscaled GEMM 예제 참고, PTX ISA로 확인 필요). 시도한다면 Q를 FP4로 만드는 정확도 비용을 BF16 Q 경로와 나란히 측정한다. **필수 범위는 아니다.**

## 8. 마일스톤

각 단계마다 **정확도 테스트 통과 + latency + 달성 BW + NCU 주요 metric**을 두 GPU에서 기록한다. 기록 자체가 블로그 재료다. 개발 반복은 RTX PRO 6000에서 하고, B200은 단계 마감 측정에만 쓴다.

| 단계 | 내용 | 확인할 것 |
|---|---|---|
| v0 | PyTorch reference: #4676 캐시에서 dequant → BF16 attention (FlashInfer wrapper) | 이후 모든 단계의 정답지. **"dequant-then-kernel" 베이스라인 겸용** |
| Step 0 | copy kernel로 달성 가능 BW, BF16 mma microbench로 peak. **두 GPU** | GPU별 ridge point 확정 |
| Step 0.5 | **완료(09-24)**: FlashInfer 베이스라인 sweep, 두 GPU (`baselines.py` → `out/results/*.csv`) | §10 실측 요약 |
| v1 | naive CUDA (CUDA core, fp32). 정확도만 | dequant 규칙이 #4676과 일치하는지 |
| v2 | mma.sync(BF16) + register dequant, 단일 stage | tensor pipe 사용률, dequant 비중 |
| v3 | cp.async 3-stage pipeline | DRAM throughput %, stall reason 변화 |
| MVP | **v0–v3 완료 (3주 목표)**, SM120 수치 + SM100 동작 확인 | 블로그 초안 작성 |
| v4 | split-KV + combine | batch 1–8, 긴 context 구간 |
| v5 | page table, 가변 길이 batch, head group sweep, **SM100 tile/stage sweep** | H=16/128, 1 vs 2 CTA/SM |
| v6 | q_len 1–4 | compute-bound 전환 지점 실측, GPU별 |
| v7 | CUDA Graph 호환(고정 split), FlashInfer API 형태로 정리 | upstream 준비 |

## 9. 검증

- reference: v0 (fp32 연산으로 계산한 BF16 attention)
- 경계 케이스
  - seq_len이 tile 배수가 아닌 경우 (#3717에서 잡았던 경계 패딩 → softmax 분모 오염 버그를 여기서도 테스트로 막는다)
  - page_size 1 / 16 / 64
  - batch 안에서 길이가 섞인 경우, 길이 1인 경우
  - split 경계가 page 경계와 어긋나는 경우
- `-use_fast_math` 여부에 따라 결과가 달라지지 않는지 확인 (#4676의 `precise_rounding` 이슈와 동일)
- 같은 테스트를 두 GPU에서 돌린다. arch 상수 차이로 tile 경계 버그가 한쪽에서만 나올 수 있다.

## 10. 벤치마크 방법

- L2 flush(256 MB 버퍼 write) 후 CUDA event로 측정, warmup 후 중앙값 사용. 클럭 고정 불가이므로 같은 세션에서 베이스라인과 번갈아 측정한다.
- sweep: batch {1, 4, 16, 64} × seq_len {1K, 4K, 16K, 64K} × H {16, 128} × q_len {1, 2, 4} × GPU {SM120, SM100}
- 보고 항목: latency, 달성 BW(GB/s, Step 0 대비 %), TFLOPS
- NCU: `dram__throughput`, `sm__pipe_tensor_op_hmma_cycles_active`, warp stall reasons, smem bank conflict

### 베이스라인 (커널 API 직접 호출, 엔진 없이)

| GPU | KV | 구현 | 호출 |
|---|---|---|---|
| SM120 | BF16 | FlashInfer FA2 (vendor 기본) | `BatchMLAPagedAttentionWrapper(backend="fa2")` |
| SM120 | BF16 | FlashInfer cutile (SM120 네이티브) | `backend="cutile"` |
| SM120 | BF16 | Triton MLA (vLLM·SGLang 실제 기본값) | SGLang `decode_attention_fwd(..., has_mla=True)` 파일 하나 vendoring |
| SM120 | FP8 | **XQA-MLA** (유일한 vendor FP8) | `trtllm_batch_decode_with_kv_cache_mla(backend="xqa")`, H=128·q_len=1 제약 |
| SM120 | FP8 | Triton MLA fp8 (in-kernel dequant) | 위와 같은 함수, fp8 버퍼 + scale |
| SM120 | FP4 저장 | dequant-then-kernel | v0 경로 (SGLang nvfp4 방식과 동치) |
| SM100 | BF16 | **trtllm-gen** (vLLM·SGLang 기본값) | `trtllm_batch_decode_with_kv_cache_mla(backend="trtllm-gen")` |
| SM100 | BF16 | FlashInfer cutlass | `BatchMLAPagedAttentionWrapper(backend="cutlass")`, H=128 |
| SM100 | FP8 | trtllm-gen fp8 | 같은 함수, fp8 |
| SM100 | FP4 dense | TRT-LLM `fp4_mla` Triton 백엔드 | `Fp4MlaState`/KVCM-V2 메타데이터 필요. **stretch**, 되면 가장 중요한 비교 |
| SM100 | FP4 sparse | vLLM FlashMLA fork `nvfp4_ds_mla` | 접근 패턴이 달라 latency 비교 대상은 아님. 참고 수치와 코드 리딩 |

- XQA-MLA의 H=128 제약 때문에 SM120 FP8 비교는 H=128에서만 가능하다. H=16 구간은 Triton MLA fp8과 비교한다.
- FlashInfer 0.7.0으로 위 항목이 모두 호출된다. Cake(#5443)만 main이 필요하고 레이아웃이 달라 제외한다. cutile은 `cuda.tile` 패키지가 이미지에 없어 미측정.

### 베이스라인 실측 요약 (2026-09-24, page 64, q_len 1, L2 flush, 30회 중앙값, `baselines.py`)

GB/s는 각 포맷의 KV 바이트 기준(BF16 1152, FP8 576 B/token). 원자료는 `out/results/baselines_*.csv`.

**SM120 RTX PRO 6000** (달성 BW 기준 1.48 TB/s)

| 구현 | H=16 | H=128 | 비고 |
|---|---|---|---|
| fa2 BF16 | 최고 986 GB/s (B16·S16K, 67%), 대배치 ~700 GB/s | 370–490 GB/s | H=16에서 유일하게 도는 vendor 커널 |
| XQA FP8 | 불가 (H=128 전용) | 대배치 1052 GB/s (스펙 66%). **소배치 split-KV 없음**: B1·S64K 1.11 ms vs fa2 0.168 ms | SM120 유일 vendor FP8 |
| XQA BF16 | 불가 | 4–139 GB/s, fallback 수준 | 비교표에서 제외 |
| cutlass | 불가 (SM100 전용) | 불가 | |

- H=16 FP8 비교는 Triton MLA fp8만 가능하다 → **v3(MVP) 전에 vendoring**.
- SM120 승부처: (a) H=16 memory-bound에서 fa2 BF16 대비 바이트 3.27× 이득, (b) H=128 소배치 long-context에서 XQA의 split-KV 부재.

**SM100 B200** (스펙 8 TB/s, 달성 BW는 Step 0에서 재측)

| 구현 | H=16 | H=128 | 비고 |
|---|---|---|---|
| trtllm-gen BF16 | 최고 5.7 TB/s (B64·S64K, 스펙 71%) | 4.9 TB/s | ~0.08 ms 바닥은 **host 오버헤드**였다(§17, 2026-09-28). GPU 시간은 B1·S1K에서 ~11 µs |
| trtllm-gen FP8 | 최고 6.3 TB/s (79%) | 4.4 TB/s | 같은 바닥 |
| fa2 BF16 | 상한 ~2.0 TB/s (25%) | ~0.95 TB/s | mma.sync 계열 wrapper 경로. tcgen05 커널 대비 2.5–3× 느림 |
| cutlass BF16 | 불가 (H=128 전용) | ~2.3 TB/s | |

- SM100 memory-bound 비교 상대는 trtllm-gen FP8. FP4 이상치: B64·S64K에서 1.48 GB 읽기 → 6 TB/s 기준 0.25 ms vs trtllm-gen FP8 0.384 ms (1.5×).
- ~~소배치·long-context(B1·S64K)에서는 vendor 전부가 ~85 µs 바닥에 있다.~~ 이 바닥은 CUDA event 측정에 들어간 FlashInfer Python wrapper의 host 시간(~135 µs/call)이었다. GPU 시간으로는 trtllm-gen FP8 B1·S64K가 30 µs다(§17).
- fa2가 B200에서 2 TB/s에 막히는 것은 우리 mma.sync 경로의 현실적 상한을 미리 보여 준다. SM100에서 75% 목표는 mma.sync로는 도달 못 할 가능성이 크므로 **SM100 성능 기준은 "fa2 BF16 대비 BW%와 trtllm-gen FP8 대비 latency"로 두고, 75% 기준은 SM120에 적용**한다.
- 정확도: fp8 경로는 fa2 대비 cosine 0.998, bf16 경로 1.000 (두 GPU 동일).

## 11. 일정 (주당 약 12–15시간 기준)

| 주 | 작업 |
|---|---|
| 1 | v0, Step 0(두 GPU), v1, 레이아웃과 반올림 규칙 확인, vLLM FlashMLA fork nvfp4 커널 리딩 |
| 2 | v2 (fragment load 패턴 설계가 가장 오래 걸림) |
| 3 | v3, Triton MLA fp8 vendoring(SM120 H=16 비교용), SM120 측정 정리 + SM100 동작 확인, 블로그 초안 → **MVP** |
| 4–5 | v4, v5 (SM100 tile/stage sweep 포함) |
| 6 | v6, v7, FlashInfer에 설계 이슈 또는 PR 제출 |

환경: Modal. 이미지·ncu·캐시 Volume은 `bench.py`(Modal 런처)에 있다. 개발은 RTX PRO 6000($3.03/h), B200($6.25/h)은 측정 시에만.

upstream 진행 중인 작업(#4335, #4771, MoE 시리즈)의 리뷰 대응은 우선순위를 더 높게 둔다.

## 12. 리스크

| 리스크 | 대응 |
|---|---|
| fragment load 설계가 막혀 v2가 지연 | smem에서 16-bit로 dequant 후 ldmatrix하는 단순 경로로 먼저 동작시키고, 이후 교체. vLLM fork nvfp4 커널의 경로를 참고 |
| dequant가 병목이라 FP8보다 느림 | 그 자체를 결과로 보고하고, 어느 구간에서 역전되는지 GPU별로 수치화 |
| mma.sync 경로가 SM100 vendor 커널(tcgen05)보다 크게 느림 | 예상된 결과. 격차를 수치화하고, H=16 memory-bound 구간에서는 BW%로 비교 |
| TRT-LLM `fp4_mla`가 SM120 지원을 추가 | "from-scratch CUDA, FlashInfer paged 레이아웃, BF16 Q" 차별점은 유지. 나오면 직접 비교 대상으로 흡수 |
| FlashInfer #5402/#5443 계열이 dense #4676 경로로 확장 | **착수 시 #4676 follow-up 이슈를 먼저 열어 작업 의사를 남긴다** |
| SM120 장비 접근 문제 | 해소됨. Modal RTX-PRO-6000 + ncu 검증 완료 |

## 13. 산출물

```
modal_benchmark/            # 현재 디렉토리
  bench.py                  # Modal 런처 (ncu profile, 원격 run) — 이름 고정
  kernels/MLA_decode.cu     # v1–v4 + combine + dequant (단일 파일)
  ref.py                    # v0
  baselines.py              # FlashInfer/Triton MLA 호출 래퍼 (§10 표)
  test_decode.py
  bench_local.py            # CUDA event 벤치, 원격에서 실행
  kernels/                  # 스모크·microbench (Step 0)
  out/{logs,ncu,results}/   # 모든 실행의 raw output. out/README.md 참고
  README.md                 # 결과 표, roofline 그림 (GPU별)
```

- 블로그: "NVFP4 MLA decode 커널을 scratch부터: v1 → v7 병목 이동 기록, SM120과 SM100에서"
- upstream: FlashInfer PR (#4676 follow-up)

## 14. CV 문구 템플릿 (결과가 나온 뒤 수치 채우기)

> **NVFP4 MLA decode kernel (CUDA, from scratch)**: first dense decode kernel for the 352 B/token NVFP4 paged layout from flashinfer#4676 on SM120 (RTX PRO 6000), running unchanged on SM100 (B200) through a single mma.sync path. Register-level FP4 dequant feeding BF16 mma.sync with Q kept in BF16, cp.async pipeline, split-KV. **X% / X'% of achievable DRAM BW** on RTX PRO 6000 / B200; **Y× vs XQA-MLA FP8** (SM120) and **Y'× vs trtllm-gen FP8** (SM100) in the memory-bound regime; measured crossover at H·q_len ≈ Z where dequant cost dominates.

## 15. 면접에서 나올 질문 (구현하면서 답을 준비)

1. 왜 scale을 mma 밖으로 꺼낼 수 없는가?
2. H=128에서 FP4가 FP8보다 느려지는 이유는? 그 구간은 어떻게 판단했나? SM120과 SM100에서 그 지점이 왜 다른가?
3. packed FP4에서 mma fragment로 가는 load 패턴을 설명하라. bank conflict는?
4. split 수는 어떻게 정했나? combine 비용은?
5. 달성 BW가 peak의 100%가 아닌 이유를 NCU로 설명하라.
6. TRT-LLM `fp4_mla`(Q/P FP4, block-scaled mma)와 비교해 BF16 Q + register dequant를 택한 이유와 정확도/속도 trade-off는?
7. 같은 커널이 SM100에서 tcgen05 vendor 커널 대비 얼마나 느린가, 그 이유는?

## 16. 조사 근거 (2026-09-24)

- FlashInfer v0.7.0: `flashinfer/mla/_core.py` (3707-3712 NVFP4 dense raise, 3731-3736 auto 백엔드, 3767-3781 xqa), `_batch_mla/_backends/` (fa2/fa3/cutlass/cutile), `benchmarks/bench_mla_decode_backend_comparison.py`. PR #4676, #5443, 이슈 #5402, #5403, #5282
- TRT-LLM main: `tensorrt_llm/_torch/attention/backends/fp4_mla/`, `cpp/kernels/xqa/mla_sm120.cu`, `cpp/tensorrt_llm/common/attentionOp.h:602` (trtllm-gen SM120 제외), `docs/source/features/quantization.md`
- vLLM v0.30.0: `vllm/platforms/cuda.py::_get_backend_priorities` (major 12 → TRITON_MLA), `vllm/v1/attention/backends/mla/`, FlashMLA fork `csrc/kernels/sm100/decode/sparse/nvfp4_head64/`
- SGLang main: `arg_groups/model_override_base.py:247`, `arg_groups/kv_cache_hook.py:111-318` (nvfp4 규칙), `sglang/kernels/ops/attention/decode_attention.py:1156`
- FlashMLA upstream: `setup.py`, `csrc/api/common.h`, `csrc/api/sparse_decode.cpp`

## 17. 진행 기록

### 2026-09-27 ~ 28: v0 → v4, Step 0

**정확도** (`test_decode.py`, 59개, 두 GPU 모두 통과)
- v0: NVFP4 경로 vs BF16 원본 rel-L2 **0.1025 (H=16) / 0.1003 (H=128)**. #4676 측정치(0.092–0.102)와 일치
- 커널 dequant는 PyTorch reference와 **bitwise 일치**, `--use_fast_math`에서도 동일
- v1–v4 vs fp32 dequant reference: rel-L2 ≤ 1.9e-3 (bf16 출력과 bf16 P 반올림 수준)
- 경계 케이스: page_size 1/16/64, 길이 1, tile·page 경계에 걸친 길이, 섞인 길이, split 경계가 page 경계와 어긋나는 경우, 빈 split. 사용하지 않는 slot은 NaN scale을 포함한 쓰레기 바이트로 채워 두고, 커널이 읽으면 테스트가 실패하도록 함

**Step 0** (`kernels/step0.py`, 4 GB read, 1 GB는 L2 잔존 효과로 제외)

| | SM120 RTX PRO 6000 | SM100 B200 |
|---|---|---|
| 달성 read BW | **1559 GB/s** | **7300 GB/s** (스펙 91%) |
| mma.sync BF16 m16n8k16 | 455 TFLOPS | 558 TFLOPS (tcgen05 아님) |
| ridge | 293 flop/B → H·q_len < 47이면 memory-bound | **76 flop/B → H·q_len < 12** |

- B200에서는 mma.sync 경로 기준으로 H=16·q_len=1(99 flop/B)도 compute-bound다. 이론 상한은 BW의 ~77%. §10의 "SM100 75% 기준 제외" 판단을 수치로 확인

**설계 (계획 §7 대비 달라진 점)**
- CTA = 16 head × 4 warp. warp w는 ckv 채널 [128w, +128)과 kpe [16w, +16)을 맡는다. QK reduction 축을 thread별 연속 32채널(scale block 2개)이 되도록 permute하고, Q도 같은 permutation으로 **register에 상주**시킨다(Q smem 없음)
- FP4 → BF16: `cvt.rn.bf16x2.e2m1x2`(PTX 9.2, sm_120a/100a) + `mul.rn.bf16x2`로 scale을 operand 쪽에 적용한다. e2m1×e4m3는 BF16에서 exact. §7.3의 LUT 단계는 건너뜀
- QK B fragment register를 그대로 **`movmatrix.trans`로 PV B fragment로 전치**한다. dequant는 token당 1회. PV 출력이 thread의 Q·K 채널 slice와 같은 위치에 떨어진다
- warp 간 score partial은 smem 4 KB(double buffer, stride 24로 bank conflict 없음)로 합산
- v3: T=32 × 3 stage cp.async(45 KB/CTA → SM120 2 CTA/SM). thread당 tile마다 token 1개의 64 B 연속 구간을 담당해 page id 1개를 한 tile 앞서 prefetch. smem 세 배열 모두 16 B chunk XOR swizzle
- v4: split-KV + combine. 고정 chunk(32 배수, auto일 때 ≥256 token) → CUDA Graph에도 맞는 형태. `max_kv_len`은 호출자가 넘긴다(host sync 없음)
- 선행 구현(vLLM FlashMLA `nvfp4_head64`)은 tcgen05 + 190 KB smem이라 SM120에 옮길 수 없다. 레이아웃 차이: 그쪽은 token당 `[nope|rope|sf]`, sf는 permute됨

**SM120 latency (ms), q_len=1, page 64** (`out/results/bench_*_RTX*.csv`)

| 설정 | v1 | v2 | v3 | v4 | fa2 BF16 | XQA FP8 |
|---|---|---|---|---|---|---|
| H16 B64 S64K | 21.7 | 5.15 | 3.45 | **1.12 (85% BW)** | 6.78 | — |
| H16 B16 S64K | 18.3 | 5.25 | 3.45 | **0.336 (70%)** | 1.54 | — |
| H16 B1 S64K | 18.2 | 5.54 | 3.45 | **0.047** | 0.132 | — |
| H128 B1 S64K | 18.2 | 5.33 | 3.43 | **0.118** | 0.170 | 1.115 |
| H128 B64 S64K | 134.8 | 6.09 | 8.16 | 8.16 | 9.99 | **2.29** |

- 병목 이동: v2는 dependent global load(page table → data)에 걸린다(NCU: DRAM 11.9%, L1TEX long scoreboard가 stall의 48.6%, occupancy 8%). v3 pipeline으로 DRAM 27%까지 오른다. 이후 병목은 SM당 CTA 수이며, v4 split으로 해소. 소배치에서는 combine이 순차 load로 95 µs였고, 병렬화 후 전체 47 µs
- **H=16 SM120 목표(달성 BW 75%) 달성: B64·S64K에서 85%.** fa2 BF16 대비 6.1×
- H=128 대배치는 compute-bound다. BF16 mma 하한은 2.57 ms로 XQA FP8(2.29 ms, FP8 mma)보다 느리다. BF16 Q 경로의 구조적 한계로, §5 예상과 같다. 현재는 mma peak의 31%. head group 확대(v5)로 개선할 여지가 있다
- H=128 소배치 long-context는 XQA에 split-KV가 없어 **9.4×** 앞선다

**SM120 H=16 vs SGLang Triton MLA FP8** (`triton_mla.py`, sglang@c311bc961164 vendoring. SGLang 기본 split 정책, fa2 대비 cosine: bf16 1.00000 / fp8 0.99843)

| 설정 | v4 NVFP4 | Triton FP8 | 배율 |
|---|---|---|---|
| B64 S64K | 1.119 | 1.852 (84% BW) | **1.66×** (바이트 이상치 1.64×) |
| B64 S16K | 0.346 | 0.548 | 1.58× |
| B16 S64K | 0.336 | 0.575 | 1.71× |
| B4 S64K | 0.113 | 0.224 | 1.98× |
| B1 S64K | 0.047 | 0.118 | 2.51× |

- memory-bound 대배치에서는 바이트 비율(576/352)만큼 이긴다. 두 커널 모두 BW 84–85%. 소배치에서는 split 오버헤드 차이로 배율이 더 크다
- H=128: B1 S64K 0.117 vs 0.224 (1.9×). B64 S64K 8.16 vs 8.15로 비슷하다. 대배치 최강은 XQA FP8(2.30)
- 정확도 조건이 다르다. Triton FP8은 Q도 FP8로 cast한다(cosine 0.998, rel-L2 ≈ 0.056). NVFP4는 BF16 Q이고 rel-L2 0.10(cosine ≈ 0.995)

**SM100 B200 (v4)**: H16 B64 S64K 0.958 ms(BW 21%). fa2 2.49 ms보다 2.6× 빠르고, trtllm-gen FP8 0.391 ms / BF16 0.861 ms보다는 느리다. B200은 SM×clock이 SM120의 0.65배라 SM 처리량에 묶인 지금 커널은 두 GPU에서 속도가 비슷하다. 16 token당 SM이 쓸 수 있는 시간은 메모리 기준 ~225 cycle인데, 지금 CTA는 ~2000 cycle이 걸린다. SM100에서는 CTA/SM을 늘리고 token당 명령 수를 줄이는 것이 과제(v5 sweep)

**측정 방식 변경 (2026-09-28)**: `bench_local.py`는 이제 호출당 **GPU 커널 시간의 합**을 잰다. 방법은 CUPTI(torch.profiler)이고, 매 호출 전에 L2를 flush하며, host launch 시간과 커널 사이 간격은 제외한다. 이전의 CUDA event 방식은 소배치에서 host 시간을 쟀다. `kernels/floor_check.py` 결과: trtllm-gen wrapper는 host/call ~135 µs인데 커널은 8.9 µs(B1·S1K), 24.6 µs(B1·S64K)였다. 세션마다 달라지던 "85 µs 바닥"도 이것으로 설명된다. 우리 커널은 host 오버헤드가 작아서 위 SM120 표의 v4 수치는 1–2% 안에서 같다

**GPU 시간 기준 비교 (ms)** (`bench_*_20260927-1558.csv`)

| 설정 | SM120 v4 | Triton FP8 | fa2 | XQA FP8 | B200 v4 | Triton FP8 | fa2 | trtllm BF16 | trtllm FP8 |
|---|---|---|---|---|---|---|---|---|---|
| H16 B64 S64K | **1.110 (85%)** | 1.847 | 6.934 | — | 0.955 | 1.909 | 2.489 | 0.859 | **0.385** |
| H16 B16 S64K | **0.336** | 0.574 | 1.540 | — | 0.229 | 0.532 | 0.563 | 0.237 | **0.115** |
| H16 B1 S64K | **0.046** | 0.115 | 0.129 | — | 0.039 | 0.112 | 0.080 | 0.044 | **0.030** |
| H16 B1 S1K | 0.024 | 0.024 | **0.021** | — | 0.021 | 0.026 | 0.020 | 0.014 | **0.011** |
| H128 B1 S64K | **0.115** | 0.222 | 0.166 | 1.110 | 0.118 | 0.299 | 0.099 | 0.046 | **0.035** |
| H128 B64 S64K | 8.185 | 8.281 | 10.01 | **2.428** | 8.966 | 14.55 | 4.963 | 1.007 | **0.535** |

- SM120: H=16에서는 모든 구간에서 1위다(B1 S1K만 fa2와 비슷). H=128 소배치에서는 XQA보다 9.7× 빠른데, XQA 커널 자체에 split-KV가 없어서다(GPU 시간으로 확인). H=128 대배치는 XQA FP8이 1위
- B200: 모든 구간에서 trtllm-gen FP8(tcgen05)이 1위다. **소배치 우위도 없다.** v4의 소배치 바닥 ~21 µs는 CTA 하나가 최소 256 token을 직렬로 처리하는 시간이다(16 token당 ~0.8 µs). 개선 후보: 짧은 문맥에서 min chunk 축소, per-CTA 명령 수 감소(v5)
- Triton FP8과 fa2보다는 B200에서도 H=16 전 구간에서 빠르다(대배치 2.0× / 2.6×)

**남은 일**
- v5: head group sweep(H=128), SM100 tile/stage/CTA-per-SM sweep, sub-tile 32 token화(softmax·barrier 비용 절반)
- v6 q_len 1–4, v7 CUDA Graph/FlashInfer API
- #4676 follow-up 이슈 열기(§12)
