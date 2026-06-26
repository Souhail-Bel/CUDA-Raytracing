#include "cuda_utils.cuh"
#include "renderer.cuh"
#include "vec3.cuh"
#include <cstdint>

// ABGR8888, SDL_PIXELFORMAT_RGBA32 on LE Linux/x86
__device__ __forceinline__ uint32_t color_to_rgba(const color &c) {
  uint8_t r = uint8_t(fminf(c.x, 1.f) * 255.99f);
  uint8_t g = uint8_t(fminf(c.y, 1.f) * 255.99f);
  uint8_t b = uint8_t(fminf(c.z, 1.f) * 255.99f);
  return (255u << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | r;
}

__global__ void render_kernel(uint32_t *fb, int width, int height) {
  int px = threadIdx.x + blockIdx.x * blockDim.x;
  int py = threadIdx.y + blockIdx.y * blockDim.y;

  if ((px >= width) || (py >= height))
    return;

  int p_idx = py * width + px;

  // parametric
  float u = float(px) / float(width - 1);
  float v = float(py) / float(height - 1);

  color pixel_color(u, v, 0.2f);

  fb[p_idx] = color_to_rgba(pixel_color);
}

__host__ void launch_render(uint32_t *d_fb, const RenderParams &params) {
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
