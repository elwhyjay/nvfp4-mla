// Achievable DRAM read BW (decode is read-only) and mma.sync throughput and latency per kind; see roofline.py.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

template <int UNROLL>
__global__ void read_bw(const int4* __restrict__ x, size_t n, int* out) {
  int acc = 0;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
    int4 v[UNROLL];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) v[k] = __ldcs(x + i + k * stride);  // streaming: don't keep in L2
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
  }
  for (; i < n; i += stride) acc ^= x[i].x ^ x[i].y ^ x[i].z ^ x[i].w;
  if (acc == 0x7eadbeef) out[0] = acc;  // keeps the loads alive
}

void read_launch(torch::Tensor x, int64_t unroll, int64_t blocks, int64_t threads, torch::Tensor out) {
  auto s = at::cuda::getCurrentCUDAStream();
  auto p = reinterpret_cast<const int4*>(x.data_ptr());
  size_t n = x.numel() / 16;
  int* o = out.data_ptr<int>();
  switch (unroll) {
    case 1: read_bw<1><<<blocks, threads, 0, s>>>(p, n, o); break;
    case 4: read_bw<4><<<blocks, threads, 0, s>>>(p, n, o); break;
    case 8: read_bw<8><<<blocks, threads, 0, s>>>(p, n, o); break;
    default: TORCH_CHECK(false, "unroll");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// mma.sync kinds: 0 bf16 m16n8k16, 1 e4m3 m16n8k32 (sm_89+), 2 kind::f8f6f4 e4m3 m16n8k32,
// 3 kind::mxf4nvf4 block-scaled e2m1 m16n8k64; 2 and 3 exist on sm_120a/sm_121a only.
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define HAS_SM120_KINDS 1
#endif

template <int KIND>
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sf) {
  if constexpr (KIND == 0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (KIND == 1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
#ifdef HAS_SM120_KINDS
    if constexpr (KIND == 2) {
      asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, "
                   "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
      asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32."
                   "ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%10}, {0,0};\n"
                   : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf));
    }
#else
    __trap();
#endif
  }
}

// CHAINS independent accumulators per warp cover the mma latency; operands never leave registers. Each warp also
// reports its clock64 cycles, so a launch with one warp per scheduler gives cycles per mma per scheduler.
template <int KIND, int CHAINS>
__global__ void mma_bench(int iters, long long* cycles, float* out) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i) a[i] = 0x3c383c38u ^ (threadIdx.x << 3);
  for (int i = 0; i < 2; ++i) b[i] = 0x3c383c38u ^ (threadIdx.x << 5);
  const uint32_t sf = 0x38383838u;  // four ue4m3 scales of 1.0
  float c[CHAINS][4] = {};
  const long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) mma<KIND>(c[j], a, b, sf);
  }
  const long long t1 = clock64();
  if (threadIdx.x % 32 == 0) cycles[(blockIdx.x * blockDim.x + threadIdx.x) / 32] = t1 - t0;
  float s = 0.f;
  for (int j = 0; j < CHAINS; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 1234.5f) out[0] = s;
}

template <int KIND>
void launch_kind(int64_t chains, int iters, int blocks, int threads, long long* cyc, float* out, cudaStream_t s) {
  switch (chains) {
    case 1: mma_bench<KIND, 1><<<blocks, threads, 0, s>>>(iters, cyc, out); break;
    case 2: mma_bench<KIND, 2><<<blocks, threads, 0, s>>>(iters, cyc, out); break;
    case 4: mma_bench<KIND, 4><<<blocks, threads, 0, s>>>(iters, cyc, out); break;
    case 8: mma_bench<KIND, 8><<<blocks, threads, 0, s>>>(iters, cyc, out); break;
    default: TORCH_CHECK(false, "chains must be 1, 2, 4 or 8");
  }
}

void mma_launch(int64_t kind, int64_t chains, int64_t iters, int64_t blocks, int64_t threads, torch::Tensor cycles,
                torch::Tensor out) {
  TORCH_CHECK(cycles.numel() >= blocks * threads / 32, "cycles needs one slot per warp");
  auto s = at::cuda::getCurrentCUDAStream();
  auto cyc = reinterpret_cast<long long*>(cycles.data_ptr<int64_t>());
  float* o = out.data_ptr<float>();
  switch (kind) {
    case 0: launch_kind<0>(chains, iters, blocks, threads, cyc, o, s); break;
    case 1: launch_kind<1>(chains, iters, blocks, threads, cyc, o, s); break;
    case 2: launch_kind<2>(chains, iters, blocks, threads, cyc, o, s); break;
    case 3: launch_kind<3>(chains, iters, blocks, threads, cyc, o, s); break;
    default: TORCH_CHECK(false, "kind");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("read", &read_launch);
  m.def("mma", &mma_launch);
}
