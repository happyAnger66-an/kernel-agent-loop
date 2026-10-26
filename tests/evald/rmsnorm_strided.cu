#include <cuda_fp16.h>

// Correct but uncoalesced: each thread owns a contiguous chunk, so a warp's loads land hidden/blockDim
// elements apart. Needs hidden % blockDim == 0. Labeled sample for the uncoalesced-access rule.
extern "C" __global__ void rmsnorm(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ y,
    int rows,
    int hidden) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int chunk = hidden / blockDim.x;
  const __half* row_x = x + static_cast<long long>(row) * hidden + threadIdx.x * chunk;
  __half* row_y = y + static_cast<long long>(row) * hidden + threadIdx.x * chunk;
  const __half* w = weight + threadIdx.x * chunk;

  float sum = 0.f;
  for (int i = 0; i < chunk; ++i) {
    const float value = __half2float(row_x[i]);
    sum += value * value;
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
  for (int i = 0; i < chunk; ++i) {
    row_y[i] = __float2half(__half2float(row_x[i]) * inv * __half2float(w[i]));
  }
}
