// Dense MLA decode (q_len = 1) over the NVFP4 paged KV cache of flashinfer#4676:
//   ckv_cache    [pages, page_size, 256] uint8   E2M1 x2, low nibble = even channel
//   ckv_sf_cache [pages, page_size, 32]  e4m3    one scale per 16 channels
//   kpe_cache    [pages, page_size, 64]  e4m3
//   ckv = e2m1 * sf * ckv_scale, kpe = fp8 * kpe_scale
//
// CTA = (16-head group, request, KV split), 4 warps, BF16 mma.sync m16n8k16. Warp w owns ckv channels [128w, +128)
// and kpe channels [16w, +16); thread (g = lane/4, t = lane%4) owns ckv [128w + 32t, +32) (two scale blocks) and kpe
// [16w + 4t, +4). The QK reduction axis is permuted so that k-step j of thread t covers its channels 4j..4j+3, and Q
// is loaded in the same permutation once: the A fragments of heads g and g+8 stay in registers.
// Per 16-token sub-tile a thread dequantizes tokens g and 8+g into QK B fragments (cvt to bf16x2, then the block
// scale on the operand, exact in bf16); after movmatrix.trans the same registers are the PV B fragments, so each
// token is dequantized once. Partial scores (over a warp's channels) are summed across warps through smem.
// KV tiles of 32 tokens stream through a cp.async pipeline; long contexts are split across CTAs and merged by
// mla_combine.
// FP8 variant (tile_fp8): e4m3 m16n8k32 MMA. QK is split by token across warps (no cross-warp score sum), PV by
// channel; each thread converts its token's K slice to e4m3 for QK and stores it to a V tile that PV reads with
// ldmatrix.trans; P goes through shared memory. Q is e4m3 with per-row scales, P is e4m3, l stays fp32.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <climits>
#include <cmath>

constexpr int CKV = 512, KPE = 64;

// cvt.rn.bf16x2.{e2m1x2,e4m3x2} need PTX 9.2 (CUDA 13.2). Older toolkits (12.8+) convert through f16, which is exact
// for these values but costs 3 more instructions per pair. NVFP4_MLA_COMPAT_CVT forces that path for testing.
#if (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && \
    !defined(NVFP4_MLA_COMPAT_CVT)
#define NVFP4_MLA_BF16X2_CVT
#endif

template <typename T>
const T* ptr(const torch::Tensor& t) {
  return reinterpret_cast<const T*>(t.data_ptr());
}

// ---- scalar dequant (used by dequantize(); same multiply order as nvfp4_mla.testing so results are bitwise equal)

__device__ __forceinline__ float e2m1_to_float(uint32_t c) {
  // e>0: (1 + m/2) * 2^(e-1) built directly as fp32 bits; e==0: m/2. No denormals, so -ftz safe.
  float mag = (c & 6) ? __int_as_float(((((c >> 1) & 3) + 126) << 23) | ((c & 1) << 22)) : 0.5f * (c & 1);
  return (c & 8) ? -mag : mag;
}

__device__ __forceinline__ float e4m3_to_float(uint8_t x) {
  __nv_fp8_e4m3 f;
  f.__x = x;
  return float(f);
}

__device__ __forceinline__ void dequant16(uint2 packed, uint8_t sf, float ckv_scale, float v[16]) {
  const float s = e4m3_to_float(sf);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    uint32_t word = i < 8 ? packed.x : packed.y;
    v[i] = e2m1_to_float((word >> (4 * (i % 8))) & 15) * s * ckv_scale;
  }
}

// ---- PTX helpers

#ifndef NVFP4_MLA_BF16X2_CVT
__device__ __forceinline__ uint32_t f16x2_to_bf16x2(uint32_t h) {
  uint32_t d;
  asm("{\n.reg .b16 lo, hi;\n.reg .f32 a, b;\nmov.b32 {lo, hi}, %1;\ncvt.f32.f16 a, lo;\ncvt.f32.f16 b, hi;\n"
      "cvt.rn.bf16x2.f32 %0, b, a;\n}"
      : "=r"(d)
      : "r"(h));
  return d;
}
#endif

__device__ __forceinline__ void e2m1x8_to_bf16x8(uint32_t w, uint32_t (&o)[4]) {
#ifdef NVFP4_MLA_BF16X2_CVT
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.bf16x2.e2m1x2 %0, b0;\ncvt.rn.bf16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.bf16x2.e2m1x2 %2, b2;\ncvt.rn.bf16x2.e2m1x2 %3, b3;\n}"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
      : "r"(w));
#else
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
      : "r"(w));
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = f16x2_to_bf16x2(o[i]);
#endif
}

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint16_t x) {
  uint32_t d;
#ifdef NVFP4_MLA_BF16X2_CVT
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(x));
#else
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(x));
  d = f16x2_to_bf16x2(d);
#endif
  return d;
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

// volatile: keeps the transposes between mbar_arrive and mbar_wait, where they hide the barrier latency
__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t x) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(x));
  return d;
}

// Split arrive/wait barrier (mbarrier: arrive releases, try_wait acquires) over the CTA's 128 threads.
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ uint64_t mbar_arrive(uint64_t* bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(smem_u32(bar)) : "memory");
  return state;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint64_t state) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT:\nmbarrier.try_wait.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT;\n}" ::"r"(smem_u32(bar)),
      "l"(state)
      : "memory");
}

__device__ __forceinline__ float ex2(float x) {  // exp2f adds denormal handling around MUFU.EX2; scores never need it
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Accumulator registers are passed one by one: the FP8 PV writes into the BF16 path's Acc layout.
__device__ __forceinline__ void mma_fp8(float& c0, float& c1, float& c2, float& c3, const uint32_t (&a)[4], uint32_t b0,
                                        uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 E2M1 codes (low nibble first) times a f16x2 scale, rounded to 8 E4M3 bytes: .x = codes 0-3, .y = codes 4-7.
// The f16 product is exact (e2m1 * e4m3 needs 6 significant bits); only the final cvt rounds.
__device__ __forceinline__ uint2 e2m1x8_to_e4m3x8(uint32_t w, uint32_t scale) {
  uint2 d;
  asm("{\n.reg .b8 b0, b1, b2, b3;\n.reg .b32 h0, h1, h2, h3;\n.reg .b16 e0, e1, e2, e3;\n"
      "mov.b32 {b0, b1, b2, b3}, %2;\n"
      "cvt.rn.f16x2.e2m1x2 h0, b0;\ncvt.rn.f16x2.e2m1x2 h1, b1;\ncvt.rn.f16x2.e2m1x2 h2, b2;\ncvt.rn.f16x2.e2m1x2 h3, b3;\n"
      "mul.rn.f16x2 h0, h0, %3;\nmul.rn.f16x2 h1, h1, %3;\nmul.rn.f16x2 h2, h2, %3;\nmul.rn.f16x2 h3, h3, %3;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\ncvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\ncvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
      "mov.b32 %0, {e0, e1};\nmov.b32 %1, {e2, e3};\n}"
      : "=r"(d.x), "=r"(d.y)
      : "r"(w), "r"(scale));
  return d;
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t x) {
  uint32_t d;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(x));
  return d;
}

// four floats to four e4m3 bytes, a in the lowest byte
__device__ __forceinline__ uint32_t pack_e4m3x4(float a, float b, float c, float d) {
  uint32_t r;
  asm("{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\ncvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n}"
      : "=r"(r)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}

__device__ __forceinline__ uint16_t pack_e4m3x2(float a, float b) {  // a in the low byte
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&d)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p)))
               : "memory");
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&d)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p)))
               : "memory");
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem)), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ---- per-thread state and the 16-token step

struct Acc {
  float o[8][2][4];  // [k-step j][half][c]: channels cbase + 4j + 2*half + {0,1}, heads g (c0,c1) / g+8 (c2,c3)
  float m[2], l[2];  // rows g, g+8
};

struct Frag {  // per-thread constants
  int w, g, t4, lane, cbase, pbase;
  __device__ Frag() : w(threadIdx.x / 32), g(threadIdx.x % 32 / 4), t4(threadIdx.x % 4), lane(threadIdx.x % 32) {
    cbase = 128 * w + 32 * t4, pbase = 16 * w + 4 * t4;
  }
};

__device__ __forceinline__ void load_q(const Frag& f, const __nv_bfloat16* q_nope, const __nv_bfloat16* q_pe,
                                       size_t row0, size_t row1, uint32_t (&qa)[8][4], uint32_t (&qpa)[4]) {
  uint4 r0[4], r1[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    r0[i] = reinterpret_cast<const uint4*>(q_nope + row0 * CKV + f.cbase)[i];
    r1[i] = reinterpret_cast<const uint4*>(q_nope + row1 * CKV + f.cbase)[i];
  }
  const uint32_t* w0 = &r0[0].x;
  const uint32_t* w1 = &r1[0].x;
#pragma unroll
  for (int j = 0; j < 8; ++j) qa[j][0] = w0[2 * j], qa[j][1] = w1[2 * j], qa[j][2] = w0[2 * j + 1], qa[j][3] = w1[2 * j + 1];
  const uint2 p0 = *reinterpret_cast<const uint2*>(q_pe + row0 * KPE + f.pbase);
  const uint2 p1 = *reinterpret_cast<const uint2*>(q_pe + row1 * KPE + f.pbase);
  qpa[0] = p0.x, qpa[1] = p1.x, qpa[2] = p0.y, qpa[3] = p1.y;
}

__device__ __forceinline__ void rescale_o(Acc& a, const float (&corr)[2]) {
  if (__any_sync(~0u, corr[0] != 1.f || corr[1] != 1.f))
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) a.o[j][h][e] *= corr[e / 2];
}

// Masks tokens past L, then the online-softmax step. s[nb][e] holds the unscaled score of token t0 + 8nb + 2*t4 +
// (e & 1) for row g (e < 2) or g + 8, and p on return. scale[r]: score scale of row g / g + 8 (log2 units).
template <int NB>
__device__ __forceinline__ void softmax_update(Acc& a, const Frag& f, float (&s)[NB][4], int t0, int L,
                                               const float (&scale)[2]) {
  if (t0 + 8 * NB > L)  // last (sub-)tile only (uniform across the CTA)
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        if (t0 + 8 * nb + 2 * f.t4 + (e & 1) >= L) s[nb][e] = -INFINITY;

  // Lazy rescale (as in FlashAttention-4): keep a stale running max until a row's max grows by more than 8, so
  // p = 2^(s - m) stays <= 256 (fine for fp32 l/o and bf16 or e4m3 P) and the o *= corr pass almost never runs.
  float corr[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float mx = fmaxf(s[0][2 * r], s[0][2 * r + 1]);
#pragma unroll
    for (int nb = 1; nb < NB; ++nb) mx = fmaxf(mx, fmaxf(s[nb][2 * r], s[nb][2 * r + 1]));
    mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 2)) * scale[r];  // scores are unscaled until here
    const bool grow = mx > a.m[r] + 8.f;  // always true on the first tile (m = -inf; token t0 is valid)
    corr[r] = grow ? ex2(a.m[r] - mx) : 1.f;
    a.m[r] = grow ? mx : a.m[r];
    a.l[r] *= corr[r];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int e = 2 * r; e < 2 * r + 2; ++e) a.l[r] += (s[nb][e] = ex2(fmaf(s[nb][e], scale[r], -a.m[r])));
  }
  rescale_o(a, corr);
}

// One 16-token sub-tile starting at t0. raw/s2/pe[nb] are this thread's bytes of token t0 + 8nb + g (zero if past
// L). rb: this sub-tile's cross-warp reduction buffer; bar: the CTA barrier guarding it.
__device__ __forceinline__ void subtile(Acc& a, const Frag& f, const uint32_t (&qa)[8][4], const uint32_t (&qpa)[4],
                                        const uint4 (&raw)[2], const uint16_t (&s2)[2], const uint32_t (&pe)[2],
                                        int t0, int L, float (*rb)[16][16], uint64_t* bar, float score_scale,
                                        float kpe_ratio) {
  const int w = f.w, g = f.g, t4 = f.t4;
  uint32_t kb[2][8][2], kpb[2][2];
#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {
    const uint32_t s01 = e4m3x2_to_bf16x2(s2[nb]);
    const uint32_t sc[2] = {__byte_perm(s01, 0, 0x1010), __byte_perm(s01, 0, 0x3232)};
    const uint32_t* rw = &raw[nb].x;
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // word i = channels cbase + 8i .. +8 = k-steps 2i, 2i+1
      uint32_t v[4];
      e2m1x8_to_bf16x8(rw[i], v);
      kb[nb][2 * i][0] = mul_bf16x2(v[0], sc[i / 2]);  // exact: e2m1 * e4m3 fits in bf16
      kb[nb][2 * i][1] = mul_bf16x2(v[1], sc[i / 2]);
      kb[nb][2 * i + 1][0] = mul_bf16x2(v[2], sc[i / 2]);
      kb[nb][2 * i + 1][1] = mul_bf16x2(v[3], sc[i / 2]);
    }
    kpb[nb][0] = e4m3x2_to_bf16x2(pe[nb] & 0xffff);
    kpb[nb][1] = e4m3x2_to_bf16x2(pe[nb] >> 16);
  }

  // QK: the two n8 blocks' accumulator chains interleave, so dependent HMMAs don't stall back to back
  float c[2][4] = {}, cp[2][4] = {}, s[2][4];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) mma_bf16(c[nb], qa[j], kb[nb][j][0], kb[nb][j][1]);
#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {
    mma_bf16(cp[nb], qpa, kpb[nb][0], kpb[nb][1]);
#pragma unroll
    for (int e = 0; e < 4; ++e) s[nb][e] = c[nb][e] + kpe_ratio * cp[nb][e];
  }

  // Cross-warp sum of the partial scores. Row = head; a thread's 4 values of a row (n-block 0 then 1) are one float4
  // at 4 * t4, so a quarter warp covers two adjacent 64 B rows: conflict free.
#pragma unroll
  for (int r = 0; r < 2; ++r)
    *reinterpret_cast<float4*>(&rb[w][g + 8 * r][4 * t4]) =
        make_float4(s[0][2 * r], s[0][2 * r + 1], s[1][2 * r], s[1][2 * r + 1]);
  const uint64_t token = mbar_arrive(bar);
  uint32_t vb[2][8][2];  // PV B fragments; independent of P, so they fill the wait for the other warps
#pragma unroll
  for (int nb = 0; nb < 2; ++nb)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) vb[nb][j][h] = movmatrix_trans(kb[nb][j][h]);
  mbar_wait(bar, token);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int ww = 0; ww < 4; ++ww) {  // own partial from registers; same summation order in every warp
      const float4 y = ww == w ? make_float4(s[0][2 * r], s[0][2 * r + 1], s[1][2 * r], s[1][2 * r + 1])
                               : *reinterpret_cast<const float4*>(&rb[ww][g + 8 * r][4 * t4]);
      if (ww == 0)
        x = y;
      else
        x.x += y.x, x.y += y.y, x.z += y.z, x.w += y.w;
    }
    s[0][2 * r] = x.x, s[0][2 * r + 1] = x.y, s[1][2 * r] = x.z, s[1][2 * r + 1] = x.w;
  }
  softmax_update(a, f, s, t0, L, {score_scale, score_scale});

  // QK accumulator layout of two n8 blocks == A fragment layout of one k16 step
  const uint32_t pa[4] = {pack_bf16x2(s[0][0], s[0][1]), pack_bf16x2(s[0][2], s[0][3]),
                          pack_bf16x2(s[1][0], s[1][1]), pack_bf16x2(s[1][2], s[1][3])};
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) mma_bf16(a.o[j][h], pa, vb[0][j][h], vb[1][j][h]);
}

// ---- FP8 path: QK split by token, PV split by channel (CTA = 16 heads, 4 warps, 32-token tiles)
// QK: warp w takes tokens 8w..8w+7 over all 576 channels. Thread t's share of the reduction axis is latent
// [128t, +128) and RoPE [16t, +16): k32-step j covers its latent channels 8j..8j+7 (a0/b0 +0..3, a2/b1 +4..7), RoPE
// step r its RoPE channels 8r..8r+7. The MMA sums over the four threads, so scores need no cross-warp sum.

struct QF8 {  // Q in e4m3, one scale per row and part, in registers
  uint32_t qa[16][4], qpa[2][4];
  float scale[2], kpe_ratio[2];  // per row g / g + 8
};

__device__ __forceinline__ void load_q_fp8(const Frag& f, const __nv_bfloat16* q_nope, const __nv_bfloat16* q_pe,
                                           size_t row0, size_t row1, float sm_scale_log2, float ckv_scale,
                                           float kpe_scale, QF8& q) {
  auto lo = [](uint32_t w) { return __uint_as_float(w << 16); };
  auto hi = [](uint32_t w) { return __uint_as_float(w & 0xffff0000u); };
  auto absmax8 = [&](uint4 v) {
    return fmaxf(fmaxf(fmaxf(fabsf(lo(v.x)), fabsf(hi(v.x))), fmaxf(fabsf(lo(v.y)), fabsf(hi(v.y)))),
                 fmaxf(fmaxf(fabsf(lo(v.z)), fabsf(hi(v.z))), fmaxf(fabsf(lo(v.w)), fabsf(hi(v.w)))));
  };
  auto pack8 = [&](uint4 v, float inv, uint32_t& a, uint32_t& b) {  // channels 0-3 -> a, 4-7 -> b
    a = pack_e4m3x4(lo(v.x) * inv, hi(v.x) * inv, lo(v.y) * inv, hi(v.y) * inv);
    b = pack_e4m3x4(lo(v.z) * inv, hi(v.z) * inv, lo(v.w) * inv, hi(v.w) * inv);
  };
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const size_t row = r ? row1 : row0;
    const uint4* qn = reinterpret_cast<const uint4*>(q_nope + row * CKV + 128 * f.t4);
    const uint4* qp = reinterpret_cast<const uint4*>(q_pe + row * KPE + 16 * f.t4);
    float an = 0.f, ap = fmaxf(absmax8(qp[0]), absmax8(qp[1]));
#pragma unroll
    for (int j = 0; j < 16; ++j) an = fmaxf(an, absmax8(qn[j]));
#pragma unroll
    for (int o = 1; o <= 2; o *= 2)  // the 4 threads of a row hold all of it
      an = fmaxf(an, __shfl_xor_sync(~0u, an, o)), ap = fmaxf(ap, __shfl_xor_sync(~0u, ap, o));
    an = fmaxf(an, 1e-30f), ap = fmaxf(ap, 1e-30f);  // all-zero rows quantize to zero
#pragma unroll
    for (int j = 0; j < 16; ++j) pack8(qn[j], 448.f / an, q.qa[j][r], q.qa[j][2 + r]);
#pragma unroll
    for (int k = 0; k < 2; ++k) pack8(qp[k], 448.f / ap, q.qpa[k][r], q.qpa[k][2 + r]);
    q.scale[r] = sm_scale_log2 * ckv_scale * (an / 448.f);
    q.kpe_ratio[r] = (ap * kpe_scale) / (an * ckv_scale);
  }
}

__device__ __forceinline__ void store_out(Acc& a, const Frag& f, __nv_bfloat16* out, size_t row0, size_t row1,
                                          float ckv_scale) {
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    a.l[r] += __shfl_xor_sync(~0u, a.l[r], 1);
    a.l[r] += __shfl_xor_sync(~0u, a.l[r], 2);
    const float s = a.l[r] > 0.f ? ckv_scale / a.l[r] : 0.f;
    uint4 v[4];
    uint32_t* vw = &v[0].x;
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) vw[2 * j + h] = pack_bf16x2(a.o[j][h][2 * r] * s, a.o[j][h][2 * r + 1] * s);
    uint4* dst = reinterpret_cast<uint4*>(out + (r ? row1 : row0) * CKV + f.cbase);
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = v[i];
  }
}

// Split-KV partial: o normalized by its own l (fp32), lse in log2 units (the kernel's score scale).
__device__ __forceinline__ void store_partial(Acc& a, const Frag& f, float* o_part, float* lse_part, size_t row0,
                                              size_t row1, float ckv_scale) {
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    a.l[r] += __shfl_xor_sync(~0u, a.l[r], 1);
    a.l[r] += __shfl_xor_sync(~0u, a.l[r], 2);
    const size_t row = r ? row1 : row0;
    const float s = a.l[r] > 0.f ? ckv_scale / a.l[r] : 0.f;
    float4* dst = reinterpret_cast<float4*>(o_part + row * CKV + f.cbase);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      dst[j] = make_float4(a.o[j][0][2 * r] * s, a.o[j][0][2 * r + 1] * s, a.o[j][1][2 * r] * s, a.o[j][1][2 * r + 1] * s);
    if (f.w == 0 && f.t4 == 0) lse_part[row] = a.l[r] > 0.f ? a.m[r] + log2f(a.l[r]) : -INFINITY;
  }
}

// ---- pipelined kernel
// Copy: 4 threads per token (TILE = 32 tokens x 4 = 128 threads); thread p copies ckv chunks {p, 4+p, 8+p, 12+p}
// (each instruction covers 64 contiguous bytes per token), sf chunk p (p < 2), kpe chunk p. So each thread needs one
// page id per tile, fetched one tile ahead. Tokens past the split end are zero-filled (src-size 0), never read: slots
// past kv_len may hold NaN scales. smem rows are XOR-swizzled at 16 B granularity so the per-thread fragment reads
// are bank-conflict free.

constexpr int TILE = 32;

struct Stage {  // 16 B chunk c of row r at ckv_chunk / sf_chunk / kpe_chunk (c, r)
  uint8_t ckv[TILE][CKV / 2];
  uint8_t sf[TILE][CKV / 16];
  uint8_t kpe[TILE][KPE];
};

// Stage swizzles (16 B chunk c of tile row r). BF16 path: conflict-free for its reads (thread (w, t4) reads ckv chunk
// 4w + t4 of rows g, sf/kpe chunks by warp). FP8 path: thread t of warp w reads ckv chunks 4t..4t+3 of row 8w + g.
template <bool FP8> __device__ __forceinline__ int ckv_chunk(int c, int r) {
  return FP8 ? c ^ ((c >> 2) & 2) ^ (r & 1) : c ^ ((r & 1) << 2);
}
template <bool FP8> __device__ __forceinline__ int sf_chunk(int c, int r) { return FP8 ? c : c ^ ((r >> 2) & 1); }
template <bool FP8> __device__ __forceinline__ int kpe_chunk(int c, int r) { return FP8 ? c : c ^ ((r >> 1) & 3); }

// smem after the stages: BF16 path, the cross-warp score buffers (one per 16-token sub-tile parity); FP8 path,
// Fp8Smem. Then the mbarrier (BF16 path).
struct Fp8Smem {
  // V tile in e4m3. Chunk 8b + q holds channels 128b + 32i + 4q + {0..3}, i = 0..3 (so the PV output of warp b lands
  // in the Acc layout), stored at chunk c ^ (r & 7) ^ ((c >> 2) & 6) of token row r: conflict-free for the
  // converting threads' stores and for ldmatrix.trans over 8 tokens.
  uint4 v[TILE][CKV / 16];
  uint8_t p[16][TILE];  // P in e4m3; column = PV k position (token order of the QK accumulators)
  float mx[16][4];      // per row: max over each warp's tokens; after the loop, each warp's row sum
};
template <bool FP8>
constexpr size_t SCRATCH = FP8 ? sizeof(Fp8Smem) : 2 * 4 * 16 * 16 * sizeof(float);

__device__ __forceinline__ void read_token(const Stage& s, int r, const Frag& f, uint4& raw, uint16_t& s2, uint32_t& pe) {
  const int c = f.cbase / 32;  // ckv chunk 4w + t4
  raw = *reinterpret_cast<const uint4*>(&s.ckv[r][16 * (c ^ ((r & 1) << 2))]);
  s2 = *reinterpret_cast<const uint16_t*>(&s.sf[r][16 * ((f.w / 2) ^ ((r >> 2) & 1)) + f.cbase / 16 % 16]);
  pe = *reinterpret_cast<const uint32_t*>(&s.kpe[r][16 * (f.w ^ ((r >> 1) & 3)) + 4 * f.t4]);
}

// One 32-token tile, FP8 path. st: the stage holding the tile's cache bytes.
__device__ __forceinline__ void tile_fp8(Acc& a, const Frag& f, const QF8& q, const Stage& st, Fp8Smem& sm, int t0,
                                         int L) {
  const int w = f.w, g = f.g, t = f.t4, r = 8 * w + g;  // r: this thread's token in the tile for QK
  uint4 raw[4];  // latent channels [128t, +128): chunks 4t..4t+3
#pragma unroll
  for (int i = 0; i < 4; ++i) raw[i] = *reinterpret_cast<const uint4*>(&st.ckv[r][16 * ckv_chunk<true>(4 * t + i, r)]);
  const uint2 sf8 = *reinterpret_cast<const uint2*>(&st.sf[r][8 * t]);  // scale blocks 8t..8t+7
  const uint4 pe = *reinterpret_cast<const uint4*>(&st.kpe[r][16 * t]);  // RoPE [16t, +16), already e4m3

  uint32_t kq[16][2];  // kq[j][h]: e4m3 of latent channels 128t + 8j + 4h .. +4
  const uint32_t* rw = &raw[0].x;
#pragma unroll
  for (int b = 0; b < 4; ++b) {  // scale blocks 2b, 2b + 1 = words 4b..4b+3
    const uint32_t s01 = e4m3x2_to_f16x2(uint16_t((b < 2 ? sf8.x : sf8.y) >> (16 * (b % 2))));
    const uint32_t sc[2] = {__byte_perm(s01, 0, 0x1010), __byte_perm(s01, 0, 0x3232)};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint2 e = e2m1x8_to_e4m3x8(rw[4 * b + k], sc[k / 2]);
      kq[4 * b + k][0] = e.x, kq[4 * b + k][1] = e.y;
    }
  }

  // QK over this warp's 8 tokens: four independent chains over the latent k-steps, one for RoPE
  float c[4][4] = {}, cp[4] = {}, sv[4];
#pragma unroll
  for (int j = 0; j < 16; ++j) mma_fp8(c[j % 4][0], c[j % 4][1], c[j % 4][2], c[j % 4][3], q.qa[j], kq[j][0], kq[j][1]);
  const uint32_t* pw = &pe.x;
#pragma unroll
  for (int k = 0; k < 2; ++k) mma_fp8(cp[0], cp[1], cp[2], cp[3], q.qpa[k], pw[2 * k], pw[2 * k + 1]);
#pragma unroll
  for (int e = 0; e < 4; ++e) sv[e] = ((c[0][e] + c[1][e]) + (c[2][e] + c[3][e])) + q.kpe_ratio[e / 2] * cp[e];

  // this token's latent K is also V: store it to the V tile (see Fp8Smem for the chunk order)
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int ch = 8 * t + k;
    sm.v[r][ch ^ (r & 7) ^ ((ch >> 2) & 6)] =
        make_uint4(kq[k / 2][k % 2], kq[4 + k / 2][k % 2], kq[8 + k / 2][k % 2], kq[12 + k / 2][k % 2]);
  }

  // sv[e]: token 8w + 2t + (e & 1), row g (e < 2) or g + 8
  if (t0 + TILE > L)
#pragma unroll
    for (int e = 0; e < 4; ++e)
      if (t0 + r - g + 2 * t + (e & 1) >= L) sv[e] = -INFINITY;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    float m = fmaxf(sv[2 * rr], sv[2 * rr + 1]);
    m = fmaxf(m, __shfl_xor_sync(~0u, m, 1));
    m = fmaxf(m, __shfl_xor_sync(~0u, m, 2));
    if (t == 0) sm.mx[g + 8 * rr][w] = m;
  }
  __syncthreads();  // V tile and row maxima complete

  // Same tile max, hence the same lazy-rescale decision, in every warp; l stays per warp until the end.
  float corr[2];
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const float4 m4 = *reinterpret_cast<const float4*>(sm.mx[g + 8 * rr]);
    const float mx = fmaxf(fmaxf(m4.x, m4.y), fmaxf(m4.z, m4.w)) * q.scale[rr];
    const bool grow = mx > a.m[rr] + 8.f;
    corr[rr] = grow ? ex2(a.m[rr] - mx) : 1.f;
    a.m[rr] = grow ? mx : a.m[rr];
    a.l[rr] *= corr[rr];
#pragma unroll
    for (int e = 2 * rr; e < 2 * rr + 2; ++e) a.l[rr] += (sv[e] = ex2(fmaf(sv[e], q.scale[rr], -a.m[rr])));
  }
  rescale_o(a, corr);

  // P: PV k position 4t' + i is token 2t' + i (i < 2) or 8 + 2t' + i - 2, +16 for the upper half (as in v6)
  const int kp = 16 * (w >> 1) + 4 * t + 2 * (w & 1);
  *reinterpret_cast<uint16_t*>(&sm.p[g][kp]) = pack_e4m3x2(sv[0], sv[1]);
  *reinterpret_cast<uint16_t*>(&sm.p[g + 8][kp]) = pack_e4m3x2(sv[2], sv[3]);
  __syncthreads();  // P complete

  uint32_t pa[4];
  ldsm_x4(pa, &sm.p[(f.lane / 8 & 1) * 8 + f.lane % 8][f.lane / 16 * 16]);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    // ldmatrix.trans over the 4 token blocks: thread (g, t) receives tokens 8m + 2t, 8m + 2t + 1 of chunk bytes 2g,
    // 2g + 1, i.e. channels 32(g / 2) + 4k + 2(g % 2) + {0, 1} of warp w's block; split into even / odd channels.
    const int ch = 8 * w + k, tok = f.lane;
    uint32_t x[4];
    ldsm_x4_trans(x, &sm.v[tok][ch ^ (tok & 7) ^ ((ch >> 2) & 6)]);
    float(&o)[2][4] = a.o[k];
    mma_fp8(o[0][0], o[1][0], o[0][2], o[1][2], pa, __byte_perm(x[0], x[1], 0x6420), __byte_perm(x[2], x[3], 0x6420));
    mma_fp8(o[0][1], o[1][1], o[0][3], o[1][3], pa, __byte_perm(x[0], x[1], 0x7531), __byte_perm(x[2], x[3], 0x7531));
  }
}

template <int STAGES, bool FP8>
__global__ void __launch_bounds__(128) mla_decode_kernel(
    const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe, const uint8_t* __restrict__ ckv,
    const uint8_t* __restrict__ sf, const uint8_t* __restrict__ kpe, const int* __restrict__ page_table,
    int table_stride, const int* __restrict__ kv_lens, int page_shift, int H, float sm_scale_log2, float ckv_scale,
    float kpe_scale, __nv_bfloat16* __restrict__ out, int chunk, float* __restrict__ o_part,
    float* __restrict__ lse_part) {
  const Frag f;
  const int b = blockIdx.y;
  // CTA z covers tokens [t_beg, t_end); chunk is a multiple of TILE. Empty splits write lse = -inf.
  const int t_beg = blockIdx.z * chunk, t_end = min(kv_lens[b], t_beg + chunk);
  const int* pt = page_table + size_t(b) * table_stride;
  const size_t row0 = size_t(b) * H + 16 * blockIdx.x + f.g, row1 = row0 + 8;
  extern __shared__ __align__(16) uint8_t smem[];
  Stage* st = reinterpret_cast<Stage*>(smem);
  uint8_t* scratch = smem + STAGES * sizeof(Stage);
  auto red = reinterpret_cast<float(*)[4][16][16]>(scratch);
  Fp8Smem& sm8 = *reinterpret_cast<Fp8Smem*>(scratch);
  auto bar = reinterpret_cast<uint64_t*>(scratch + SCRATCH<FP8>);
  if (threadIdx.x == 0) mbar_init(bar, 128);
  __syncthreads();

  const int ntiles = t_end > t_beg ? (t_end - t_beg + TILE - 1) / TILE : 0;
  const int cr = threadIdx.x / 4, cp = threadIdx.x % 4;  // copy role: token row, part
  auto page_of = [&](int tile) { const int t = t_beg + tile * TILE + cr; return t < t_end ? pt[t >> page_shift] : 0; };
  auto issue = [&](int tile, int page) {
    Stage& s = st[tile % STAGES];
    const int t = t_beg + tile * TILE + cr;
    const bool ok = t < t_end;
    const size_t tok = (size_t(page) << page_shift) | (t & ((1 << page_shift) - 1));
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int c = 4 * k + cp;
      cp_async16(&s.ckv[cr][16 * ckv_chunk<FP8>(c, cr)], ckv + tok * (CKV / 2) + 16 * c, ok);
    }
    if (cp < 2) cp_async16(&s.sf[cr][16 * sf_chunk<FP8>(cp, cr)], sf + tok * (CKV / 16) + 16 * cp, ok);
    cp_async16(&s.kpe[cr][16 * kpe_chunk<FP8>(cp, cr)], kpe + tok * KPE + 16 * cp, ok);
  };

  uint32_t qa[8][4], qpa[4];
  QF8 q8;
  if constexpr (FP8)
    load_q_fp8(f, q_nope, q_pe, row0, row1, sm_scale_log2, ckv_scale, kpe_scale, q8);
  else
    load_q(f, q_nope, q_pe, row0, row1, qa, qpa);
  Acc a = {{}, {-INFINITY, -INFINITY}, {0.f, 0.f}};

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < ntiles) issue(i, page_of(i));
    cp_async_commit();
  }
  int next_page = page_of(STAGES - 1);
  for (int i = 0; i < ntiles; ++i) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();  // tile i visible to all; everyone is done with the stage refilled below
    const int nxt = i + STAGES - 1;
    if (nxt < ntiles) issue(nxt, next_page);
    cp_async_commit();
    next_page = page_of(nxt + 1);

    const Stage& s = st[i % STAGES];
    if constexpr (FP8) {
      tile_fp8(a, f, q8, s, sm8, t_beg + i * TILE, t_end);
    } else {
#pragma unroll
      for (int sub = 0; sub < TILE / 16; ++sub) {
        uint4 raw[2];
        uint16_t s2[2];
        uint32_t pe[2];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) read_token(s, 16 * sub + 8 * nb + f.g, f, raw[nb], s2[nb], pe[nb]);
        const int t0 = t_beg + i * TILE + 16 * sub;
        if (t0 < t_end)  // uniform across the CTA
          subtile(a, f, qa, qpa, raw, s2, pe, t0, t_end, red[(2 * i + sub) & 1], bar, sm_scale_log2 * ckv_scale,
                  kpe_scale / ckv_scale);
      }
    }
  }
  if constexpr (FP8) {  // row sums: thread -> warp -> CTA, through mx (no warp reads it after the last tile's P barrier)
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      float l = a.l[rr];
      l += __shfl_xor_sync(~0u, l, 1);
      l += __shfl_xor_sync(~0u, l, 2);
      if (f.t4 == 0) sm8.mx[f.g + 8 * rr][f.w] = l;
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const float4 v = *reinterpret_cast<const float4*>(sm8.mx[f.g + 8 * rr]);
      a.l[rr] = f.t4 == 0 ? (v.x + v.y) + (v.z + v.w) : 0.f;  // the stores sum l over the 4 threads of a row
    }
  }
  if (o_part) {
    const size_t prow0 = (size_t(blockIdx.z) * gridDim.y + b) * H + 16 * blockIdx.x + f.g;
    store_partial(a, f, o_part, lse_part, prow0, prow0 + 8, ckv_scale);
  } else {
    store_out(a, f, out, row0, row1, ckv_scale);
  }
}

// out[row] = sum_z 2^(lse_z - M) o_z / sum_z 2^(lse_z - M). CTA = (row, 64-channel slice); its 8 groups of 16
// threads stride over splits (4 channels per thread) so many independent loads are in flight.
__global__ void __launch_bounds__(128) mla_combine(const float* __restrict__ o_part, const float* __restrict__ lse_part,
                                                   int splits, int rows, __nv_bfloat16* __restrict__ out) {
  const int row = blockIdx.x, zg = threadIdx.x / 16, c = 64 * blockIdx.y + 4 * (threadIdx.x % 16);
  __shared__ float red_m[4];
  __shared__ float4 red_acc[8][16];
  __shared__ float red_den[8][16];
  float M = -INFINITY;
  for (int z = threadIdx.x; z < splits; z += 128) M = fmaxf(M, lse_part[size_t(z) * rows + row]);
  for (int o = 16; o; o /= 2) M = fmaxf(M, __shfl_xor_sync(~0u, M, o));
  if (threadIdx.x % 32 == 0) red_m[threadIdx.x / 32] = M;
  __syncthreads();
  M = fmaxf(fmaxf(red_m[0], red_m[1]), fmaxf(red_m[2], red_m[3]));

  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float den = 0.f;
#pragma unroll 4
  for (int z = zg; z < splits; z += 8) {
    const float lse = lse_part[size_t(z) * rows + row];
    const float4 v = *reinterpret_cast<const float4*>(o_part + (size_t(z) * rows + row) * CKV + c);
    const float w = lse == -INFINITY ? 0.f : exp2f(lse - M);  // empty split
    den += w, acc.x += w * v.x, acc.y += w * v.y, acc.z += w * v.z, acc.w += w * v.w;
  }
  red_acc[zg][threadIdx.x % 16] = acc;
  red_den[zg][threadIdx.x % 16] = den;
  __syncthreads();
  if (zg) return;
  for (int k = 1; k < 8; ++k) {
    const float4 v = red_acc[k][threadIdx.x];
    den += red_den[k][threadIdx.x], acc.x += v.x, acc.y += v.y, acc.z += v.z, acc.w += v.w;
  }
  const float s = den > 0.f ? 1.f / den : 0.f;
  *reinterpret_cast<uint2*>(out + size_t(row) * CKV + c) =
      make_uint2(pack_bf16x2(acc.x * s, acc.y * s), pack_bf16x2(acc.z * s, acc.w * s));
}

// ---- host

void check_cache(const torch::Tensor& ckv, const torch::Tensor& sf, const torch::Tensor& kpe) {
  TORCH_CHECK(ckv.is_cuda() && sf.device() == ckv.device() && kpe.device() == ckv.device(),
              "caches must be on the same CUDA device");
  TORCH_CHECK(ckv.scalar_type() == at::kByte, "ckv_cache must be uint8 (packed E2M1)");
  TORCH_CHECK(sf.scalar_type() == at::kFloat8_e4m3fn && kpe.scalar_type() == at::kFloat8_e4m3fn,
              "ckv_sf_cache and kpe_cache must be float8_e4m3fn");
  TORCH_CHECK(ckv.dim() == 3 && ckv.size(2) == CKV / 2, "ckv_cache must be [pages, page_size, 256]");
  TORCH_CHECK(sf.dim() == 3 && sf.size(2) == CKV / 16, "ckv_sf_cache must be [pages, page_size, 32]");
  TORCH_CHECK(kpe.dim() == 3 && kpe.size(2) == KPE, "kpe_cache must be [pages, page_size, 64]");
  TORCH_CHECK(sf.sizes().slice(0, 2) == ckv.sizes().slice(0, 2) && kpe.sizes().slice(0, 2) == ckv.sizes().slice(0, 2),
              "ckv_cache, ckv_sf_cache and kpe_cache must have the same [pages, page_size]");
  TORCH_CHECK(ckv.is_contiguous() && sf.is_contiguous() && kpe.is_contiguous(), "caches must be contiguous");
}

template <int STAGES, bool FP8>
torch::Tensor decode_impl(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor sf,
                          torch::Tensor kpe, torch::Tensor page_table, torch::Tensor kv_lens, double sm_scale,
                          double ckv_scale, double kpe_scale, int64_t max_kv_len, int64_t num_splits) {
  check_cache(ckv, sf, kpe);
  TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(2) == CKV, "q_nope must be [batch, heads, 512]");
  const int B = q_nope.size(0), H = q_nope.size(1);
  TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(0) == B && q_pe.size(1) == H && q_pe.size(2) == KPE,
              "q_pe must be [batch, heads, 64]");
  TORCH_CHECK(q_nope.scalar_type() == at::kBFloat16 && q_pe.scalar_type() == at::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(H % 16 == 0, "heads must be a multiple of 16");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == B && page_table.scalar_type() == at::kInt,
              "page_table must be int32 [batch, max_pages]");
  TORCH_CHECK(kv_lens.dim() == 1 && kv_lens.size(0) == B && kv_lens.scalar_type() == at::kInt,
              "kv_lens must be int32 [batch]");
  for (const auto* t : {&q_nope, &q_pe, &page_table, &kv_lens})
    TORCH_CHECK(t->device() == ckv.device() && t->is_contiguous(), "inputs must be contiguous and on the cache device");
  TORCH_CHECK(ckv_scale > 0 && kpe_scale > 0, "ckv_scale and kpe_scale must be positive");
  const int ps = ckv.size(1), page_shift = __builtin_ctz(ps);
  TORCH_CHECK((ps & (ps - 1)) == 0, "page_size must be a power of two");
  const at::cuda::OptionalCUDAGuard guard(ckv.device());

  constexpr size_t smem = STAGES * sizeof(Stage) + SCRATCH<FP8> + sizeof(uint64_t);
  auto kernel = mla_decode_kernel<STAGES, FP8>;
  // once per process, outside any CUDA-graph capture; ponytail: assumes one GPU model per process
  static const int wave = [&] {
    int dev, sms, per_sm;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 128, smem));
    return sms * per_sm;
  }();
  const int max_len = std::max<int64_t>(max_kv_len, 1);
  auto round_up = [](int x, int m) { return (x + m - 1) / m * m; };
  int chunk;
  if (num_splits > 0) {
    chunk = round_up((max_len + num_splits - 1) / num_splits, TILE);
  } else {
    // Minimize waves x (chunk + C0), C0 = per-CTA fixed cost (Q load, pipeline fill) in tokens: a split whose CTAs
    // leave the last wave half empty is slower than more, shorter splits. Chunks stay >= sqrt(L): a CTA's time grows
    // with chunk and combine's with L / chunk, so their sum is smallest near sqrt(L). Both fitted to split sweeps
    // on RTX PRO 6000 and B200 (H 16/128, B 1-64, 1K-64K tokens).
    constexpr int C0 = 256;
    const long ctas = B * H / 16;
    const int min_chunk = round_up(std::max<int>(std::sqrt(max_len), 1), TILE);
    long best = LONG_MAX;
    chunk = round_up(max_len, TILE);
    for (int s = 1;; ++s) {
      const int c = round_up((max_len + s - 1) / s, TILE);
      if (s > 1 && c < min_chunk) break;
      const long waves = (ctas * ((max_len + c - 1) / c) + wave - 1) / wave, cost = waves * (c + C0);
      if (cost < best) best = cost, chunk = c;
      if (c == TILE) break;
    }
  }
  const int splits = (max_len + chunk - 1) / chunk;

  auto out = torch::empty({B, H, CKV}, q_nope.options());
  torch::Tensor o_part, lse_part;
  if (splits > 1) {
    o_part = torch::empty({splits, B, H, CKV}, q_nope.options().dtype(torch::kFloat32));
    lse_part = torch::empty({splits, B, H}, q_nope.options().dtype(torch::kFloat32));
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  kernel<<<dim3(H / 16, B, splits), 128, smem, stream>>>(
      ptr<__nv_bfloat16>(q_nope), ptr<__nv_bfloat16>(q_pe), ptr<uint8_t>(ckv), ptr<uint8_t>(sf), ptr<uint8_t>(kpe),
      page_table.data_ptr<int>(), page_table.size(1), kv_lens.data_ptr<int>(), page_shift, H,
      sm_scale * 1.4426950408889634, ckv_scale, kpe_scale, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), chunk,
      splits > 1 ? o_part.data_ptr<float>() : nullptr, splits > 1 ? lse_part.data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (splits > 1) {
    mla_combine<<<dim3(B * H, CKV / 64), 128, 0, stream>>>(o_part.data_ptr<float>(), lse_part.data_ptr<float>(), splits,
                                                           B * H, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

torch::Tensor decode(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor sf, torch::Tensor kpe,
                     torch::Tensor page_table, torch::Tensor kv_lens, double sm_scale, double ckv_scale,
                     double kpe_scale, int64_t max_kv_len, int64_t num_splits, int64_t stages, bool fp8) {
  if (fp8)  // 2 stages: with the 17 KB scratch, 3 would leave one CTA per SM on SM120; `stages` is ignored
    return decode_impl<2, true>(q_nope, q_pe, ckv, sf, kpe, page_table, kv_lens, sm_scale, ckv_scale, kpe_scale,
                                max_kv_len, num_splits);
  switch (stages) {
#define CASE(S) \
  case S: return decode_impl<S, false>(q_nope, q_pe, ckv, sf, kpe, page_table, kv_lens, sm_scale, ckv_scale, \
                                       kpe_scale, max_kv_len, num_splits);
    CASE(2) CASE(3) CASE(4) CASE(6)
#undef CASE
  }
  TORCH_CHECK(false, "stages must be 2, 3, 4 or 6");
}

// ---- whole-cache dequant (reference check and the "dequant, then a BF16 kernel" path)

__device__ __forceinline__ void store16(float* p, const float (&v)[16]) {
  for (int j = 0; j < 16; ++j) p[j] = v[j];
}
__device__ __forceinline__ void store16(__nv_bfloat16* p, const float (&v)[16]) {  // 2 x 16 B, coalesced per warp
  uint4 u[2];
  uint32_t* w = &u[0].x;
  for (int j = 0; j < 8; ++j) w[j] = pack_bf16x2(v[2 * j], v[2 * j + 1]);
  reinterpret_cast<uint4*>(p)[0] = u[0], reinterpret_cast<uint4*>(p)[1] = u[1];
}
__device__ __forceinline__ void store2(float* p, float a, float b) { p[0] = a, p[1] = b; }
__device__ __forceinline__ void store2(__nv_bfloat16* p, float a, float b) { *reinterpret_cast<uint32_t*>(p) = pack_bf16x2(a, b); }

template <typename T>
__global__ void dequant_kernel(const uint8_t* ckv, const uint8_t* sf, const uint8_t* kpe, float ckv_scale,
                               float kpe_scale, T* ckv_out, T* kpe_out, size_t n_tok) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, tok = i / 32, blk = i % 32;
  if (tok >= n_tok) return;
  float v[16];
  dequant16(reinterpret_cast<const uint2*>(ckv + tok * (CKV / 2))[blk], sf[tok * (CKV / 16) + blk], ckv_scale, v);
  store16(ckv_out + tok * CKV + 16 * blk, v);
  store2(kpe_out + tok * KPE + 2 * blk, e4m3_to_float(kpe[tok * KPE + 2 * blk]) * kpe_scale,
         e4m3_to_float(kpe[tok * KPE + 2 * blk + 1]) * kpe_scale);
}

std::vector<torch::Tensor> dequantize(torch::Tensor ckv, torch::Tensor sf, torch::Tensor kpe, double ckv_scale,
                                      double kpe_scale, bool bf16) {
  check_cache(ckv, sf, kpe);
  const at::cuda::OptionalCUDAGuard guard(ckv.device());
  const size_t n_tok = ckv.size(0) * ckv.size(1);
  auto opt = ckv.options().dtype(bf16 ? torch::kBFloat16 : torch::kFloat32);
  auto ckv_out = torch::empty({ckv.size(0), ckv.size(1), CKV}, opt), kpe_out = torch::empty({ckv.size(0), ckv.size(1), KPE}, opt);
  if (n_tok == 0) return {ckv_out, kpe_out};
  const dim3 grid((n_tok * 32 + 255) / 256);
  auto s = at::cuda::getCurrentCUDAStream();
  if (bf16)
    dequant_kernel<<<grid, 256, 0, s>>>(ptr<uint8_t>(ckv), ptr<uint8_t>(sf), ptr<uint8_t>(kpe), ckv_scale, kpe_scale,
                                        reinterpret_cast<__nv_bfloat16*>(ckv_out.data_ptr()),
                                        reinterpret_cast<__nv_bfloat16*>(kpe_out.data_ptr()), n_tok);
  else
    dequant_kernel<<<grid, 256, 0, s>>>(ptr<uint8_t>(ckv), ptr<uint8_t>(sf), ptr<uint8_t>(kpe), ckv_scale, kpe_scale,
                                        ckv_out.data_ptr<float>(), kpe_out.data_ptr<float>(), n_tok);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {ckv_out, kpe_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &decode);
  m.def("dequantize", &dequantize);
}
