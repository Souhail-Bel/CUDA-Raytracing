#include "cuda_utils.cuh"
#include "renderer.cuh"

__global__ void render_kernel(float *fb, int width, int height) {
  int px = threadIdx.x + blockIdx.x * blockDim.x;
  int py = threadIdx.y + blockIdx.y * blockDim.y;

  if ((px >= width) || (py >= height))
    return;

  int p_idx = py * width * 3 + px * 3;

  fb[p_idx + 0] = float(px) / width;
  fb[p_idx + 1] = float(py) / height;
  fb[p_idx + 2] = 0.2;
}

__host__ void launch_render(float *d_fb, const RenderParams &params) {
  // 16x16 tiles
  // 16x16/32 = 8 warps per block
  const dim3 block_size(16, 16);

  const dim3 grid_size((params.width + block_size.x - 1) / block_size.x,
                       (params.height + block_size.y - 1) / block_size.y);

  render_kernel<<<grid_size, block_size>>>(d_fb, params.width, params.height);

  // Check kernel execution
  CUDA_CHECK(cudaGetLastError());

  // wait for framebuffer copy to CPU
  CUDA_CHECK(cudaDeviceSynchronize());
}
