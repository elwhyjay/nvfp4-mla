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
// FP8 variant (tile_fp8): e4m3 m16n8k32 for QK and PV over a whole 32-token tile. Q is quantized to e4m3 once with a
// per-row scale; K/V are e2m1 * scale rounded to e4m3; P is e4m3 (l stays fp32). The PV token axis is permuted to
// match the QK accumulator layout, and the PV output channels land where the BF16 path puts them (shared Acc).
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
  int w, g, t4, cbase, pbase;
  __device__ Frag() : w(threadIdx.x / 32), g(threadIdx.x % 32 / 4), t4(threadIdx.x % 4) {
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
  if (__any_sync(~0u, corr[0] != 1.f || corr[1] != 1.f))
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) a.o[j][h][e] *= corr[e / 2];
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

// ---- FP8 path

// Q in e4m3 with one scale per row and part (latent / RoPE), the amax taken over all 4 warps through xchg (smem,
// 4 x 16 x 2 floats). A fragment of k32-step j: thread t's channels cbase + 8j + {0..3} (a0, a1) and + {4..7} (a2,
// a3); RoPE: channels pbase + {0..3} in a0/a1, the upper k half zero. The row scales fold into the score scale.
struct QF8 {
  uint32_t qa[4][4], qpa[4];
  float scale[2], kpe_ratio[2];  // per row g / g + 8
};

__device__ __forceinline__ void load_q_fp8(const Frag& f, const __nv_bfloat16* q_nope, const __nv_bfloat16* q_pe,
                                           size_t row0, size_t row1, float* xchg, float sm_scale_log2,
                                           float ckv_scale, float kpe_scale, QF8& q) {
  auto lo = [](uint32_t w) { return __uint_as_float(w << 16); };
  auto hi = [](uint32_t w) { return __uint_as_float(w & 0xffff0000u); };
  float qn[2][32], qp[2][4], an[2], ap[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const size_t row = r ? row1 : row0;
    uint4 v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = reinterpret_cast<const uint4*>(q_nope + row * CKV + f.cbase)[i];
    const uint32_t* w = &v[0].x;
    an[r] = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      qn[r][2 * i] = lo(w[i]), qn[r][2 * i + 1] = hi(w[i]);
      an[r] = fmaxf(an[r], fmaxf(fabsf(qn[r][2 * i]), fabsf(qn[r][2 * i + 1])));
    }
    const uint2 p = *reinterpret_cast<const uint2*>(q_pe + row * KPE + f.pbase);
    qp[r][0] = lo(p.x), qp[r][1] = hi(p.x), qp[r][2] = lo(p.y), qp[r][3] = hi(p.y);
    ap[r] = fmaxf(fmaxf(fabsf(qp[r][0]), fabsf(qp[r][1])), fmaxf(fabsf(qp[r][2]), fabsf(qp[r][3])));
#pragma unroll
    for (int o = 1; o <= 2; o *= 2) {
      an[r] = fmaxf(an[r], __shfl_xor_sync(~0u, an[r], o));
      ap[r] = fmaxf(ap[r], __shfl_xor_sync(~0u, ap[r], o));
    }
    if (f.t4 == 0) xchg[2 * (16 * f.w + f.g + 8 * r)] = an[r], xchg[2 * (16 * f.w + f.g + 8 * r) + 1] = ap[r];
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int ww = 0; ww < 4; ++ww)
      an[r] = fmaxf(an[r], xchg[2 * (16 * ww + f.g + 8 * r)]), ap[r] = fmaxf(ap[r], xchg[2 * (16 * ww + f.g + 8 * r) + 1]);
    an[r] = fmaxf(an[r], 1e-30f), ap[r] = fmaxf(ap[r], 1e-30f);  // all-zero rows quantize to zero
    const float in = 448.f / an[r], ip = 448.f / ap[r];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float* x = &qn[r][8 * j];
      q.qa[j][r] = pack_e4m3x4(x[0] * in, x[1] * in, x[2] * in, x[3] * in);
      q.qa[j][2 + r] = pack_e4m3x4(x[4] * in, x[5] * in, x[6] * in, x[7] * in);
    }
    q.qpa[r] = pack_e4m3x4(qp[r][0] * ip, qp[r][1] * ip, qp[r][2] * ip, qp[r][3] * ip);
    q.qpa[2 + r] = 0;
    q.scale[r] = sm_scale_log2 * ckv_scale * (an[r] / 448.f);
    q.kpe_ratio[r] = (ap[r] * kpe_scale) / (an[r] * ckv_scale);
  }
}

// One 32-token tile starting at t0 (FP8 path). raw/s2/pe[nb] are this thread's bytes of token t0 + 8nb + g (zero if
// past L). rb: the cross-warp reduction buffer (one per tile suffices: the tile loop's __syncthreads orders reuse).
constexpr int RB_STRIDE = 36;  // floats per row: 32 tokens + 4 pad, so 8 threads' float4 stores cover all banks

__device__ __forceinline__ void tile_fp8(Acc& a, const Frag& f, const QF8& q, const uint4 (&raw)[4],
                                         const uint16_t (&s2)[4], const uint32_t (&pe)[4], int t0, int L,
                                         float (*rb)[16][RB_STRIDE], uint64_t* bar) {
  const int w = f.w, g = f.g, t4 = f.t4;
  uint32_t kq[4][4][2];  // [token block nb][k-step j][a0/a2 half]: e4m3 of token 8nb + g, channels cbase + 8j + 4h..
#pragma unroll
  for (int nb = 0; nb < 4; ++nb) {
    const uint32_t s01 = e4m3x2_to_f16x2(s2[nb]);
    const uint32_t sc[2] = {__byte_perm(s01, 0, 0x1010), __byte_perm(s01, 0, 0x3232)};
    const uint32_t* rw = &raw[nb].x;
#pragma unroll
    for (int j = 0; j < 4; ++j) {  // word j = channels cbase + 8j .. +8, scale block j / 2
      const uint2 e = e2m1x8_to_e4m3x8(rw[j], sc[j / 2]);
      kq[nb][j][0] = e.x, kq[nb][j][1] = e.y;
    }
  }

  // QK: four independent accumulator chains per k-step. RoPE: the cache bytes are already e4m3.
  float c[4][4] = {}, cp[4][4] = {}, s[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) mma_fp8(c[nb][0], c[nb][1], c[nb][2], c[nb][3], q.qa[j], kq[nb][j][0], kq[nb][j][1]);
#pragma unroll
  for (int nb = 0; nb < 4; ++nb) {
    mma_fp8(cp[nb][0], cp[nb][1], cp[nb][2], cp[nb][3], q.qpa, pe[nb], 0u);
#pragma unroll
    for (int e = 0; e < 4; ++e) s[nb][e] = c[nb][e] + q.kpe_ratio[e / 2] * cp[nb][e];
  }

  // Cross-warp sum: a thread's 8 values of a row are two float4 at 8 * t4.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float4* p = reinterpret_cast<float4*>(&rb[w][g + 8 * r][8 * t4]);
    p[0] = make_float4(s[0][2 * r], s[0][2 * r + 1], s[1][2 * r], s[1][2 * r + 1]);
    p[1] = make_float4(s[2][2 * r], s[2][2 * r + 1], s[3][2 * r], s[3][2 * r + 1]);
  }
  const uint64_t token = mbar_arrive(bar);

  // PV B fragments, independent of P. PV k position 4t + i covers token 2t + i (i < 2) or 8 + 2t + i - 2 (the QK
  // accumulator layout), +16 for the second register. movmatrix moves byte pairs (b16), so thread (g, t) receives
  // tokens 8nb + 2t, 8nb + 2t + 1 of channel pair c0, c0 + 1 (c0 = 128w + 32(g/2) + 8j + 4h + 2(g%2)); a byte
  // permute splits them into channel c0 (even block) and c0 + 1 (odd block).
  uint32_t vb[4][2][2][2];  // [j][h][even/odd][b0/b1]
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t x[4];
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) x[nb] = movmatrix_trans(kq[nb][j][h]);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        vb[j][h][0][k] = __byte_perm(x[2 * k], x[2 * k + 1], 0x6420);
        vb[j][h][1][k] = __byte_perm(x[2 * k], x[2 * k + 1], 0x7531);
      }
    }
  mbar_wait(bar, token);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float4 x0 = make_float4(0.f, 0.f, 0.f, 0.f), x1 = x0;
#pragma unroll
    for (int ww = 0; ww < 4; ++ww) {  // own partial from registers; same summation order in every warp
      const float4* p = reinterpret_cast<const float4*>(&rb[ww][g + 8 * r][8 * t4]);
      const float4 y0 = ww == w ? make_float4(s[0][2 * r], s[0][2 * r + 1], s[1][2 * r], s[1][2 * r + 1]) : p[0];
      const float4 y1 = ww == w ? make_float4(s[2][2 * r], s[2][2 * r + 1], s[3][2 * r], s[3][2 * r + 1]) : p[1];
      if (ww == 0)
        x0 = y0, x1 = y1;
      else
        x0.x += y0.x, x0.y += y0.y, x0.z += y0.z, x0.w += y0.w, x1.x += y1.x, x1.y += y1.y, x1.z += y1.z, x1.w += y1.w;
    }
    s[0][2 * r] = x0.x, s[0][2 * r + 1] = x0.y, s[1][2 * r] = x0.z, s[1][2 * r + 1] = x0.w;
    s[2][2 * r] = x1.x, s[2][2 * r + 1] = x1.y, s[3][2 * r] = x1.z, s[3][2 * r + 1] = x1.w;
  }
  softmax_update(a, f, s, t0, L, q.scale);

  // P as the A fragment in the permuted token order; e4m3 without scaling (p <= 256, l stays fp32).
  const uint32_t pa[4] = {pack_e4m3x4(s[0][0], s[0][1], s[1][0], s[1][1]), pack_e4m3x4(s[0][2], s[0][3], s[1][2], s[1][3]),
                          pack_e4m3x4(s[2][0], s[2][1], s[3][0], s[3][1]), pack_e4m3x4(s[2][2], s[2][3], s[3][2], s[3][3])};
  // Output column n = 2t + e of block (j, h, eo) is channel cbase + 8j + 4h + 2e + eo: Acc slot [2j + h][e][eo (+2)].
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int eo = 0; eo < 2; ++eo) {
        float(&o)[2][4] = a.o[2 * j + h];
        mma_fp8(o[0][eo], o[1][eo], o[0][eo + 2], o[1][eo + 2], pa, vb[j][h][eo][0], vb[j][h][eo][1]);
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

struct Stage {
  uint8_t ckv[TILE][CKV / 2];  // chunk c of row r at c ^ ((r & 1) << 2)
  uint8_t sf[TILE][CKV / 16];  // chunk c at c ^ ((r >> 2) & 1)
  uint8_t kpe[TILE][KPE];      // chunk c at c ^ ((r >> 1) & 3)
};

// smem after the stages: the cross-warp reduction buffers (BF16: one per 16-token sub-tile parity; FP8: one per tile),
// then the mbarrier.
template <bool FP8>
constexpr int RED_FLOATS = FP8 ? 4 * 16 * RB_STRIDE : 2 * 4 * 16 * 16;

__device__ __forceinline__ void read_token(const Stage& s, int r, const Frag& f, uint4& raw, uint16_t& s2, uint32_t& pe) {
  const int c = f.cbase / 32;  // ckv chunk 4w + t4
  raw = *reinterpret_cast<const uint4*>(&s.ckv[r][16 * (c ^ ((r & 1) << 2))]);
  s2 = *reinterpret_cast<const uint16_t*>(&s.sf[r][16 * ((f.w / 2) ^ ((r >> 2) & 1)) + f.cbase / 16 % 16]);
  pe = *reinterpret_cast<const uint32_t*>(&s.kpe[r][16 * (f.w ^ ((r >> 1) & 3)) + 4 * f.t4]);
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
  float* red_base = reinterpret_cast<float*>(smem + STAGES * sizeof(Stage));
  auto red = reinterpret_cast<float(*)[4][16][16]>(red_base);
  auto red8 = reinterpret_cast<float(*)[16][RB_STRIDE]>(red_base);
  auto bar = reinterpret_cast<uint64_t*>(red_base + RED_FLOATS<FP8>);
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
      cp_async16(&s.ckv[cr][16 * (c ^ ((cr & 1) << 2))], ckv + tok * (CKV / 2) + 16 * c, ok);
    }
    if (cp < 2) cp_async16(&s.sf[cr][16 * (cp ^ ((cr >> 2) & 1))], sf + tok * (CKV / 16) + 16 * cp, ok);
    cp_async16(&s.kpe[cr][16 * (cp ^ ((cr >> 1) & 3))], kpe + tok * KPE + 16 * cp, ok);
  };

  uint32_t qa[8][4], qpa[4];
  QF8 q8;
  if constexpr (FP8)
    load_q_fp8(f, q_nope, q_pe, row0, row1, red_base, sm_scale_log2, ckv_scale, kpe_scale, q8);
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
      uint4 raw[4];
      uint16_t s2[4];
      uint32_t pe[4];
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) read_token(s, 8 * nb + f.g, f, raw[nb], s2[nb], pe[nb]);
      tile_fp8(a, f, q8, raw, s2, pe, t_beg + i * TILE, t_end, red8, bar);
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

  constexpr size_t smem = STAGES * sizeof(Stage) + RED_FLOATS<FP8> * sizeof(float) + sizeof(uint64_t);
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
  if (fp8) {
    TORCH_CHECK(stages == 2 || stages == 3, "the FP8 path is built with 2 or 3 stages");
    return stages == 2 ? decode_impl<2, true>(q_nope, q_pe, ckv, sf, kpe, page_table, kv_lens, sm_scale, ckv_scale,
                                              kpe_scale, max_kv_len, num_splits)
                       : decode_impl<3, true>(q_nope, q_pe, ckv, sf, kpe, page_table, kv_lens, sm_scale, ckv_scale,
                                              kpe_scale, max_kv_len, num_splits);
  }
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
