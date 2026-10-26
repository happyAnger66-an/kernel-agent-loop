#include <cuda_fp16.h>

// 128-bit loads plus warp-shuffle reduction; scalar fallback when hidden is not a multiple of 8.
extern "C" __global__ void rmsnorm(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ y,
    int rows,
    int hidden) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  const __half* row_x = x + static_cast<long long>(row) * hidden;
  __half* row_y = y + static_cast<long long>(row) * hidden;
  const bool vec = (hidden % 8) == 0;

  float sum = 0.f;
  if (vec) {
    const uint4* xv = reinterpret_cast<const uint4*>(row_x);
    for (int i = threadIdx.x; i < hidden / 8; i += blockDim.x) {
      uint4 packed = xv[i];
      const __half2* h = reinterpret_cast<const __half2*>(&packed);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 f = __half22float2(h[k]);
        sum += f.x * f.x + f.y * f.y;
      }
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float value = __half2float(row_x[i]);
      sum += value * value;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  __shared__ float warp_sums[32];
  __shared__ float inv_shared;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float s = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
    for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_xor_sync(0xffffffffu, s, offset);
    if (lane == 0) inv_shared = rsqrtf(s / static_cast<float>(hidden) + 1e-6f);
  }
  __syncthreads();
  const float inv = inv_shared;

  if (vec) {
    const uint4* xv = reinterpret_cast<const uint4*>(row_x);
    const uint4* wv = reinterpret_cast<const uint4*>(weight);
    uint4* yv = reinterpret_cast<uint4*>(row_y);
    for (int i = threadIdx.x; i < hidden / 8; i += blockDim.x) {
      uint4 px = xv[i];
      uint4 pw = wv[i];
      uint4 out;
      const __half2* hx = reinterpret_cast<const __half2*>(&px);
      const __half2* hw = reinterpret_cast<const __half2*>(&pw);
      __half2* ho = reinterpret_cast<__half2*>(&out);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 fx = __half22float2(hx[k]);
        const float2 fw = __half22float2(hw[k]);
        ho[k] = __floats2half2_rn(fx.x * inv * fw.x, fx.y * inv * fw.y);
      }
      yv[i] = out;
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      row_y[i] = __float2half(__half2float(row_x[i]) * inv * __half2float(weight[i]));
    }
  }
}
