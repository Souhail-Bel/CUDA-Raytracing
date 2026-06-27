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

__global__ void render_kernel(uint32_t *fb, int width, int height, float t) {
  int px = threadIdx.x + blockIdx.x * blockDim.x;
  int py = threadIdx.y + blockIdx.y * blockDim.y;

  if ((px >= width) || (py >= height))
    return;

  int p_idx = py * width + px;

  // parametric
  float u = float(px) / float(width - 1);
  float v = float(py) / float(height - 1);

  // float r = 0.5f + 0.5f * sinf(u * 10.f + t + sinf(v * 8.f - t * 0.5f)) -
  //           cosf(v * 10.f + t * 2.f);
  // float g = 0.5f + 0.5f * sinf(v * 15.f - t * 5.f + sinf(u * 16.f + t));
  // float b = 0.5f + 0.5f * cosf((u + v) * 5.f + t * 2.f + sinf(t * 5.f));
  // color pixel_color(r, g, b);

  //
  // TUNNEL EFFECT
  // https://lodev.org/cgtutor/tunnel.html
  // 

  float center_offset_x = 0.5f + 0.15f * sinf(t * 1.0f);
  float center_offset_y = 0.5f + 0.12f * cosf(t * 1.5f);

  float cx = u - center_offset_x;
  float cy = v - center_offset_y;

  float r_polar = sqrtf(cx * cx + cy * cy);
  r_polar = fmaxf(r_polar, 0.005f); // center approaches 0
  float theta = atan2f(cy, cx) / 3.14159265f;

  float tunnel_u = 0.7f / r_polar;
  float tunnel_v = 2.f * theta;

  float u_anim = tunnel_u + t * 1.0f; // move
  float v_anim = tunnel_v + t * 0.4f; // rotate

  float tiles_along_depth = 4.0f;
  float tiles_around_circle = 8.0f;

  float check_u = sinf(u_anim * tiles_along_depth * 3.14159265f);
  float check_v = sinf(v_anim * tiles_around_circle * 3.14159265f);
  float mask = (check_u * check_v > 0.0f) ? 1.0f : 0.0f;
  float depth_shading = fminf(r_polar * 2.f, 1.0f);

  // palette
  color c1(.0f, .0f, .0f);
  color c2(1.0f, 1.0f, 1.0f);

  color pixel_color = ((mask*c1) + ((1.0f - mask) * c2)) * depth_shading;
  fb[p_idx] = color_to_rgba(pixel_color);
}

__host__ void launch_render(uint32_t *d_fb, const RenderParams &params) {
  // 16x16 tiles
  // 16x16/32 = 8 warps per block
  const dim3 block_size(16, 16);

  const dim3 grid_size((params.width + block_size.x - 1) / block_size.x,
                       (params.height + block_size.y - 1) / block_size.y);

  render_kernel<<<grid_size, block_size>>>(d_fb, params.width, params.height,
                                           params.time);

  // Check kernel execution
  CUDA_CHECK(cudaGetLastError());

  // wait for framebuffer copy to CPU
  CUDA_CHECK(cudaDeviceSynchronize());
}
