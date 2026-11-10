// NVFP4 MLA dense decode over the flashinfer#4676 paged layout:
//   ckv [pages, ps, 256] u8  E2M1 x2, low nibble = even channel
//   sf  [pages, ps, 32]  e4m3 one scale per 16 channels
//   kpe [pages, ps, 64]  e4m3
//   ckv = e2m1 * sf * ckv_scale, kpe = fp8 * kpe_scale
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

constexpr int CKV = 512, KPE = 64;

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

// 16 channels = one scale block. Same multiply order as ref.py so results are bitwise equal.
__device__ __forceinline__ void dequant16(uint2 packed, uint8_t sf, float ckv_scale, float v[16]) {
  const float s = e4m3_to_float(sf);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    uint32_t word = i < 8 ? packed.x : packed.y;
    v[i] = e2m1_to_float((word >> (4 * (i % 8))) & 15) * s * ckv_scale;
  }
}

// v1: CUDA-core fp32, accuracy only. CTA = (head, request); warps stride over tokens with their own online
// softmax state; lane owns ckv channels [16*lane, +16) (exactly one scale block) and kpe channels [2*lane, +2).
template <int WARPS>
__global__ void mla_decode_v1(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
                              const uint8_t* __restrict__ ckv, const uint8_t* __restrict__ sf,
                              const uint8_t* __restrict__ kpe, const int* __restrict__ page_table, int table_stride,
                              const int* __restrict__ kv_lens, int page_size, float sm_scale, float ckv_scale,
                              float kpe_scale, __nv_bfloat16* __restrict__ out) {
  const int h = blockIdx.x, b = blockIdx.y, H = gridDim.x;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int L = kv_lens[b];
  const size_t qo = (size_t(b) * H + h);

  float qn[16], qp[2], acc[16] = {};
#pragma unroll
  for (int i = 0; i < 16; ++i) qn[i] = __bfloat162float(q_nope[qo * CKV + 16 * lane + i]);
  qp[0] = __bfloat162float(q_pe[qo * KPE + 2 * lane]);
  qp[1] = __bfloat162float(q_pe[qo * KPE + 2 * lane + 1]);

  float m = -INFINITY, l = 0.f;
  for (int t = warp; t < L; t += WARPS) {
    const size_t tok = size_t(page_table[b * table_stride + t / page_size]) * page_size + t % page_size;
    float v[16];
    dequant16(reinterpret_cast<const uint2*>(ckv + tok * (CKV / 2))[lane], sf[tok * (CKV / 16) + lane], ckv_scale, v);
    float d = qp[0] * e4m3_to_float(kpe[tok * KPE + 2 * lane]) * kpe_scale +
              qp[1] * e4m3_to_float(kpe[tok * KPE + 2 * lane + 1]) * kpe_scale;
#pragma unroll
    for (int i = 0; i < 16; ++i) d += qn[i] * v[i];
#pragma unroll
    for (int o = 16; o; o /= 2) d += __shfl_xor_sync(~0u, d, o);
    const float s = d * sm_scale, m_new = fmaxf(m, s);
    const float corr = expf(m - m_new), p = expf(s - m_new);
    l = l * corr + p;
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = acc[i] * corr + p * v[i];
    m = m_new;
  }

  __shared__ float sm_m[WARPS], sm_l[WARPS], sm_acc[WARPS][CKV];
  if (lane == 0) sm_m[warp] = m, sm_l[warp] = l;
#pragma unroll
  for (int i = 0; i < 16; ++i) sm_acc[warp][16 * lane + i] = acc[i];
  __syncthreads();
  float M = -INFINITY;
  for (int w = 0; w < WARPS; ++w) M = fmaxf(M, sm_m[w]);
  float denom = 0.f, wt[WARPS];
  for (int w = 0; w < WARPS; ++w) denom += sm_l[w] * (wt[w] = M == -INFINITY ? 0.f : expf(sm_m[w] - M));
  for (int c = threadIdx.x; c < CKV; c += blockDim.x) {
    float o = 0.f;
    for (int w = 0; w < WARPS; ++w) o += sm_acc[w][c] * wt[w];
    out[qo * CKV + c] = __float2bfloat16(denom > 0.f ? o / denom : 0.f);
  }
}

template <typename T>
const T* ptr(const torch::Tensor& t) {
  return reinterpret_cast<const T*>(t.data_ptr());
}

torch::Tensor decode_v1(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor sf,
                        torch::Tensor kpe, torch::Tensor page_table, torch::Tensor kv_lens, double sm_scale,
                        double ckv_scale, double kpe_scale) {
  const int B = q_nope.size(0), H = q_nope.size(1);
  TORCH_CHECK(q_nope.is_contiguous() && q_pe.is_contiguous() && page_table.is_contiguous());
  TORCH_CHECK(ckv.size(2) == CKV / 2 && sf.size(2) == CKV / 16 && kpe.size(2) == KPE);
  auto out = torch::empty({B, H, CKV}, q_nope.options());
  constexpr int WARPS = 4;
  mla_decode_v1<WARPS><<<dim3(H, B), 32 * WARPS, 0, at::cuda::getCurrentCUDAStream()>>>(
      ptr<__nv_bfloat16>(q_nope), ptr<__nv_bfloat16>(q_pe), ptr<uint8_t>(ckv), ptr<uint8_t>(sf), ptr<uint8_t>(kpe),
      page_table.data_ptr<int>(), page_table.size(1), kv_lens.data_ptr<int>(), ckv.size(1), sm_scale, ckv_scale,
      kpe_scale, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// v2: BF16 mma.sync m16n8k16 + register dequant, single stage (loads go straight to registers).
//
// CTA = (16-head group, request), 4 warps. Warp w owns ckv channels [128w, +128) and kpe channels [16w, +16);
// thread (g = lane/4, t = lane%4) owns ckv [128w + 32t, +32) (= 2 scale blocks) and kpe [16w + 4t, +4).
// QK: the reduction axis is permuted so that k-step j of thread t covers its channels 4j..4j+3; Q is loaded in
// the same permutation once, so the A fragments (heads g, g+8) never leave registers.
// Per 16-token sub-tile, thread loads tokens g and 8+g (16 B ckv + 2 B scales + 4 B kpe each), dequantizes into
// QK B fragments, and the same registers become PV B fragments after movmatrix.trans (8 tokens x 8 channels).
// Partial scores (over the warp's channels) are summed across warps through smem.
// PV output lands on the thread's own 32 channels for heads g and g+8.

__device__ __forceinline__ void e2m1x8_to_bf16x8(uint32_t w, uint32_t (&o)[4]) {
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.bf16x2.e2m1x2 %0, b0;\ncvt.rn.bf16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.bf16x2.e2m1x2 %2, b2;\ncvt.rn.bf16x2.e2m1x2 %3, b3;\n}"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
      : "r"(w));
}

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint16_t x) {
  uint32_t d;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(x));
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

__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t x) {
  uint32_t d;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(x));
  return d;
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

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

// One 16-token sub-tile starting at t0. raw/s2/pe[nb] are this thread's bytes of token t0 + 8nb + g (zero if
// past L). rb: this sub-tile's cross-warp reduction buffer. Contains one __syncthreads.
__device__ __forceinline__ void subtile(Acc& a, const Frag& f, const uint32_t (&qa)[8][4], const uint32_t (&qpa)[4],
                                        const uint4 (&raw)[2], const uint16_t (&s2)[2], const uint32_t (&pe)[2],
                                        int t0, int L, float (*rb)[16][24], float sm_scale_log2, float ckv_scale,
                                        float kpe_scale) {
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

  float s[2][4];
#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {
    float c[4] = {}, cp[4] = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) mma_bf16(c, qa[j], kb[nb][j][0], kb[nb][j][1]);
    mma_bf16(cp, qpa, kpb[nb][0], kpb[nb][1]);
#pragma unroll
    for (int e = 0; e < 4; ++e) s[nb][e] = ckv_scale * c[e] + kpe_scale * cp[e];
  }

#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {
    *reinterpret_cast<float2*>(&rb[w][g][8 * nb + 2 * t4]) = make_float2(s[nb][0], s[nb][1]);
    *reinterpret_cast<float2*>(&rb[w][g + 8][8 * nb + 2 * t4]) = make_float2(s[nb][2], s[nb][3]);
  }
  __syncthreads();
#pragma unroll
  for (int nb = 0; nb < 2; ++nb)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float2 x = make_float2(0.f, 0.f);
#pragma unroll
      for (int ww = 0; ww < 4; ++ww) {
        const float2 y = *reinterpret_cast<const float2*>(&rb[ww][g + 8 * r][8 * nb + 2 * t4]);
        x.x += y.x, x.y += y.y;
      }
      const int t = t0 + 8 * nb + 2 * t4;
      s[nb][2 * r] = t < L ? x.x * sm_scale_log2 : -INFINITY;
      s[nb][2 * r + 1] = t + 1 < L ? x.y * sm_scale_log2 : -INFINITY;
    }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float mx = fmaxf(fmaxf(s[0][2 * r], s[0][2 * r + 1]), fmaxf(s[1][2 * r], s[1][2 * r + 1]));
    mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 2));
    const float mn = fmaxf(a.m[r], mx);  // finite: token t0 is always valid
    const float corr = exp2f(a.m[r] - mn);
    a.m[r] = mn;
    a.l[r] *= corr;
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) a.o[j][h][2 * r] *= corr, a.o[j][h][2 * r + 1] *= corr;
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
      for (int e = 2 * r; e < 2 * r + 2; ++e) a.l[r] += (s[nb][e] = exp2f(s[nb][e] - mn));
  }

  // QK accumulator layout of two n8 blocks == A fragment layout of one k16 step
  const uint32_t pa[4] = {pack_bf16x2(s[0][0], s[0][1]), pack_bf16x2(s[0][2], s[0][3]),
                          pack_bf16x2(s[1][0], s[1][1]), pack_bf16x2(s[1][2], s[1][3])};
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) mma_bf16(a.o[j][h], pa, movmatrix_trans(kb[0][j][h]), movmatrix_trans(kb[1][j][h]));
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

__global__ void __launch_bounds__(128) mla_decode_v2(
    const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe, const uint8_t* __restrict__ ckv,
    const uint8_t* __restrict__ sf, const uint8_t* __restrict__ kpe, const int* __restrict__ page_table,
    int table_stride, const int* __restrict__ kv_lens, int page_size, int H, float sm_scale_log2, float ckv_scale,
    float kpe_scale, __nv_bfloat16* __restrict__ out) {
  const Frag f;
  const int b = blockIdx.y, L = kv_lens[b];
  const size_t row0 = size_t(b) * H + 16 * blockIdx.x + f.g, row1 = row0 + 8;
  uint32_t qa[8][4], qpa[4];
  load_q(f, q_nope, q_pe, row0, row1, qa, qpa);
  Acc a = {{}, {-INFINITY, -INFINITY}, {0.f, 0.f}};
  __shared__ float red[2][4][16][24];  // [parity][warp][head][token]; row stride 24 avoids bank conflicts

  for (int t0 = 0, it = 0; t0 < L; t0 += 16, ++it) {
    uint4 raw[2] = {};
    uint16_t s2[2] = {};
    uint32_t pe[2] = {};
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      const int t = t0 + 8 * nb + f.g;
      if (t < L) {  // never touch slots past kv_len: they may hold NaN scales
        const size_t tok = size_t(page_table[b * table_stride + t / page_size]) * page_size + t % page_size;
        raw[nb] = *reinterpret_cast<const uint4*>(ckv + tok * (CKV / 2) + f.cbase / 2);
        s2[nb] = *reinterpret_cast<const uint16_t*>(sf + tok * (CKV / 16) + f.cbase / 16);
        pe[nb] = *reinterpret_cast<const uint32_t*>(kpe + tok * KPE + f.pbase);
      }
    }
    subtile(a, f, qa, qpa, raw, s2, pe, t0, L, red[it & 1], sm_scale_log2, ckv_scale, kpe_scale);  // 2 buffers: 1 barrier/sub-tile
  }
  store_out(a, f, out, row0, row1, ckv_scale);
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

// ---------------------------------------------------------------------------------------------------------------
// v3: v2's math + cp.async multi-stage pipeline of T-token tiles in smem.
// Copy: 4 threads per token (T = 32 tokens x 4 = 128 threads); thread p copies ckv chunks {p, 4+p, 8+p, 12+p}
// (each instruction covers 64 contiguous bytes per token), sf chunk p (p < 2), kpe chunk p. So each thread needs one
// page id per tile, fetched one tile ahead. Tokens past L are zero-filled (src-size 0), never read.
// smem rows are XOR-swizzled at 16 B granularity so the per-thread fragment reads are bank-conflict free.

constexpr int V3_T = 32;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

struct Stage {
  uint8_t ckv[V3_T][CKV / 2];  // chunk c of row r at c ^ ((r & 1) << 2)
  uint8_t sf[V3_T][CKV / 16];  // chunk c at c ^ ((r >> 2) & 1)
  uint8_t kpe[V3_T][KPE];      // chunk c at c ^ ((r >> 1) & 3)
};

template <int STAGES>
__global__ void __launch_bounds__(128) mla_decode_v3(
    const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe, const uint8_t* __restrict__ ckv,
    const uint8_t* __restrict__ sf, const uint8_t* __restrict__ kpe, const int* __restrict__ page_table,
    int table_stride, const int* __restrict__ kv_lens, int page_size, int H, float sm_scale_log2, float ckv_scale,
    float kpe_scale, __nv_bfloat16* __restrict__ out, int chunk, float* __restrict__ o_part,
    float* __restrict__ lse_part) {
  const Frag f;
  const int b = blockIdx.y;
  // v4 split-KV: CTA z covers tokens [t_beg, t_end); chunk is a multiple of V3_T. Empty splits write lse = -inf.
  const int t_beg = blockIdx.z * chunk, t_end = min(kv_lens[b], t_beg + chunk);
  const int* pt = page_table + size_t(b) * table_stride;
  const size_t row0 = size_t(b) * H + 16 * blockIdx.x + f.g, row1 = row0 + 8;
  extern __shared__ __align__(16) uint8_t smem[];
  Stage* st = reinterpret_cast<Stage*>(smem);
  auto red = reinterpret_cast<float(*)[4][16][24]>(smem + STAGES * sizeof(Stage));

  const int ntiles = t_end > t_beg ? (t_end - t_beg + V3_T - 1) / V3_T : 0;
  const int cr = threadIdx.x / 4, cp = threadIdx.x % 4;  // copy role: token row, part
  auto page_of = [&](int tile) { const int t = t_beg + tile * V3_T + cr; return t < t_end ? pt[t / page_size] : 0; };
  auto issue = [&](int tile, int page) {
    Stage& s = st[tile % STAGES];
    const int t = t_beg + tile * V3_T + cr;
    const bool ok = t < t_end;
    const size_t tok = size_t(page) * page_size + t % page_size;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int c = 4 * k + cp;
      cp_async16(&s.ckv[cr][16 * (c ^ ((cr & 1) << 2))], ckv + tok * (CKV / 2) + 16 * c, ok);
    }
    if (cp < 2) cp_async16(&s.sf[cr][16 * (cp ^ ((cr >> 2) & 1))], sf + tok * (CKV / 16) + 16 * cp, ok);
    cp_async16(&s.kpe[cr][16 * (cp ^ ((cr >> 1) & 3))], kpe + tok * KPE + 16 * cp, ok);
  };

  uint32_t qa[8][4], qpa[4];
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
#pragma unroll
    for (int sub = 0; sub < V3_T / 16; ++sub) {
      uint4 raw[2];
      uint16_t s2[2];
      uint32_t pe[2];
#pragma unroll
      for (int nb = 0; nb < 2; ++nb) {
        const int r = 16 * sub + 8 * nb + f.g, c = f.cbase / 32;  // ckv chunk 4w + t4
        raw[nb] = *reinterpret_cast<const uint4*>(&s.ckv[r][16 * (c ^ ((r & 1) << 2))]);
        s2[nb] = *reinterpret_cast<const uint16_t*>(&s.sf[r][16 * ((f.w / 2) ^ ((r >> 2) & 1)) + f.cbase / 16 % 16]);
        pe[nb] = *reinterpret_cast<const uint32_t*>(&s.kpe[r][16 * (f.w ^ ((r >> 1) & 3)) + 4 * f.t4]);
      }
      const int t0 = t_beg + i * V3_T + 16 * sub;
      if (t0 < t_end)  // uniform across the CTA
        subtile(a, f, qa, qpa, raw, s2, pe, t0, t_end, red[(2 * i + sub) & 1], sm_scale_log2, ckv_scale, kpe_scale);
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

// num_splits = 0: enough CTAs for one full wave, at least 256 tokens per split. max_kv_len comes from the caller
// (as in FlashInfer's plan) to avoid a device sync.
torch::Tensor decode_v4(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor sf,
                        torch::Tensor kpe, torch::Tensor page_table, torch::Tensor kv_lens, double sm_scale,
                        double ckv_scale, double kpe_scale, int64_t max_kv_len, int64_t num_splits) {
  const int B = q_nope.size(0), H = q_nope.size(1);
  TORCH_CHECK(H % 16 == 0, "heads must be a multiple of 16");
  TORCH_CHECK(q_nope.is_contiguous() && q_pe.is_contiguous() && page_table.is_contiguous());
  TORCH_CHECK(ckv.size(2) == CKV / 2 && sf.size(2) == CKV / 16 && kpe.size(2) == KPE);
  constexpr int STAGES = 3;
  constexpr size_t smem = STAGES * sizeof(Stage) + 2 * 4 * 16 * 24 * sizeof(float);
  auto kernel = mla_decode_v3<STAGES>;
  // once per process, outside any CUDA-graph capture; ponytail: assumes a single device per process
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
    chunk = round_up((max_len + num_splits - 1) / num_splits, V3_T);
  } else {
    const int ctas = B * H / 16, splits = std::max(1, wave / ctas);
    chunk = std::max(256, round_up((max_len + splits - 1) / splits, V3_T));
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
      page_table.data_ptr<int>(), page_table.size(1), kv_lens.data_ptr<int>(), ckv.size(1), H,
      sm_scale * 1.4426950408889634, ckv_scale, kpe_scale, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), chunk,
      splits > 1 ? o_part.data_ptr<float>() : nullptr, splits > 1 ? lse_part.data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (splits > 1) {
    mla_combine<<<dim3(B * H, CKV / 64), 128, 0, stream>>>(o_part.data_ptr<float>(), lse_part.data_ptr<float>(), splits, B * H,
                                           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

torch::Tensor decode_v3(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor sf,
                        torch::Tensor kpe, torch::Tensor page_table, torch::Tensor kv_lens, double sm_scale,
                        double ckv_scale, double kpe_scale) {
  const int max_pages = page_table.size(1), ps = ckv.size(1);  // one split covering any request
  return decode_v4(q_nope, q_pe, ckv, sf, kpe, page_table, kv_lens, sm_scale, ckv_scale, kpe_scale,
                   int64_t(max_pages) * ps, 1);
}

torch::Tensor decode_v2(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor sf,
                        torch::Tensor kpe, torch::Tensor page_table, torch::Tensor kv_lens, double sm_scale,
                        double ckv_scale, double kpe_scale) {
  const int B = q_nope.size(0), H = q_nope.size(1);
  TORCH_CHECK(H % 16 == 0, "heads must be a multiple of 16");
  TORCH_CHECK(q_nope.is_contiguous() && q_pe.is_contiguous() && page_table.is_contiguous());
  TORCH_CHECK(ckv.size(2) == CKV / 2 && sf.size(2) == CKV / 16 && kpe.size(2) == KPE);
  auto out = torch::empty({B, H, CKV}, q_nope.options());
  mla_decode_v2<<<dim3(H / 16, B), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      ptr<__nv_bfloat16>(q_nope), ptr<__nv_bfloat16>(q_pe), ptr<uint8_t>(ckv), ptr<uint8_t>(sf), ptr<uint8_t>(kpe),
      page_table.data_ptr<int>(), page_table.size(1), kv_lens.data_ptr<int>(), ckv.size(1), H,
      sm_scale * 1.4426950408889634, ckv_scale, kpe_scale, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Dequant the whole cache with the kernel's device functions. fp32 output: bitwise check vs ref.py.
// bf16 output: the "dequant-then-kernel" baseline (exact for power-of-two ckv_scale).
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

std::vector<torch::Tensor> dequant(torch::Tensor ckv, torch::Tensor sf, torch::Tensor kpe, double ckv_scale,
                                   double kpe_scale, bool bf16) {
  const size_t n_tok = ckv.size(0) * ckv.size(1);
  auto opt = ckv.options().dtype(bf16 ? torch::kBFloat16 : torch::kFloat32);
  auto ckv_out = torch::empty({ckv.size(0), ckv.size(1), CKV}, opt), kpe_out = torch::empty({ckv.size(0), ckv.size(1), KPE}, opt);
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
  m.def("decode_v1", &decode_v1);
  m.def("decode_v2", &decode_v2);
  m.def("decode_v3", &decode_v3);
  m.def("decode_v4", &decode_v4);
  m.def("dequant", &dequant);
}
