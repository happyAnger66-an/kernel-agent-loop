#include <cuda_fp16.h>

// Naive one-block-per-row RMSNorm. The evaluator times this against later candidates.
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

  float sum = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float value = __half2float(row_x[i]);
    sum += value * value;
  }

  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }

  const float inv = rsqrtf(partial[0] / static_cast<float>(hidden) + 1e-6f);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float value = __half2float(row_x[i]) * inv * __half2float(weight[i]);
    row_y[i] = __float2half(value);
  }
}
