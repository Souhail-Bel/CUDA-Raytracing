#include "camera.cuh"
#include "cuda_utils.cuh"
#include "hittable.cuh"
#include "ray.cuh"
#include "renderer.cuh"
#include "sphere.cuh"
#include "vec3.cuh"
#include <cstdint>

static constexpr int MAX_SPHERES = 64;

__constant__ Camera d_cam;
__constant__ Sphere d_spheres[MAX_SPHERES];
__constant__ int d_num_spheres;

// ABGR8888, SDL_PIXELFORMAT_RGBA32 on LE Linux/x86
__device__ __forceinline__ uint32_t color_to_rgba(const color &c) {
  uint8_t r = uint8_t(fminf(c.x, 1.f) * 255.99f);
  uint8_t g = uint8_t(fminf(c.y, 1.f) * 255.99f);
  uint8_t b = uint8_t(fminf(c.z, 1.f) * 255.99f);
  return (255u << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | r;
}

// Traverse scene
__device__ color ray_color(const Ray &r) {
  HitRecord rec;
  bool hit_anything = false;
  float curr_closest = 1e30f;

  for (int i = 0; i < d_num_spheres; ++i) {
    HitRecord curr_record;
    if (d_spheres[i].hit(r, 0.001f, curr_closest, curr_record)) {
      hit_anything = true;
      curr_closest = curr_record.t;
      rec = curr_record;
    }
  }

  if (hit_anything) {
    return 0.5f *
           color(rec.normal.x + 1.f, rec.normal.y + 1.f, rec.normal.z + 1.f);
  }

  const vec3 unit_dir = r.dir.normalized();
  const float t = 0.5f * (unit_dir.y + 1.f);
  return lerp(color(1.f, 1.f, 1.f), color(0.7f, 0.4f, 1.f), t);
}

// kernel
__global__ void render_kernel(uint32_t *fb, int width, int height, float t) {
  int px = threadIdx.x + blockIdx.x * blockDim.x;
  int py = threadIdx.y + blockIdx.y * blockDim.y;

  if ((px >= width) || (py >= height))
    return;

  int p_idx = py * width + px;

  float u = float(px) / float(width - 1);
  float v = float(py) / float(height - 1);

  const Ray r = d_cam.get_ray(u, v);

  fb[p_idx] = color_to_rgba(ray_color(r));
}

__host__ void launch_render(uint32_t *d_fb, const RenderParams &params) {
  // SETUP
  const float angle = params.time * 1.f;
  const float distance = 3.f;
  const Camera cam(point3(sinf(angle) * distance, 1.f, cosf(angle) * distance),
                   point3(0.f, 0.f, 0.f), vec3(0.f, 1.f, 0.f), 90.f,
                   params.aspect_ratio);
  CUDA_CHECK(cudaMemcpyToSymbol(d_cam, &cam, sizeof(Camera)));

  const int num_spheres = 4;
  const Sphere spheres[num_spheres]{Sphere(point3(0.f, 0.f, 0.f), 0.5f),
                                    Sphere(point3(0.f, -100.5f, 0.f), 100.f),
                                    Sphere(point3(10.f, 0.f, 0.f), 4.f),
                                    Sphere(point3(0.f, 2.f, 0.f), 1.f)};

  CUDA_CHECK(
      cudaMemcpyToSymbol(d_spheres, &spheres, num_spheres * sizeof(Sphere)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_num_spheres, &num_spheres, sizeof(int)));

  // LAUNCH
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
