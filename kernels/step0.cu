// Step 0: achievable DRAM read BW (decode is read-only) and BF16 mma.sync peak.
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

// CHAINS independent accumulators per warp to cover mma latency; operands never leave registers.
template <int CHAINS>
__global__ void mma_peak(int iters, float* out) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i) a[i] = 0x3f803f80u ^ (threadIdx.x << 3);
  for (int i = 0; i < 2; ++i) b[i] = 0x3f803f80u ^ (threadIdx.x << 5);
  float c[CHAINS][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
          "{%0,%1,%2,%3};\n"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  float s = 0.f;
  for (int j = 0; j < CHAINS; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 1234.5f) out[0] = s;
}

void mma_launch(int64_t iters, int64_t blocks, int64_t threads, torch::Tensor out) {
  mma_peak<8><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(iters, out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("read", &read_launch);
  m.def("mma", &mma_launch);
}
