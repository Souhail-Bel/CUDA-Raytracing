#include "camera.cuh"
#include "cuda_utils.cuh"
#include "hittable.cuh"
#include "material.cuh"
#include "ray.cuh"
#include "renderer.cuh"
#include "sphere.cuh"
#include "vec3.cuh"
#include <cstdint>

static constexpr int MAX_OBJECTS = 64;

__constant__ Camera d_cam;
__constant__ Sphere d_spheres[MAX_OBJECTS];
__constant__ Material d_materials[MAX_OBJECTS];
__constant__ int d_num_spheres;

static constexpr int MAX_DEPTH = 8;
static constexpr int NUM_SAMPLING = 16;

// ABGR8888, SDL_PIXELFORMAT_RGBA32 on LE Linux/x86
__device__ __forceinline__ uint32_t color_to_rgba(const color &c) {
  uint8_t r = uint8_t(fminf(c.x, 1.f) * 255.99f);
  uint8_t g = uint8_t(fminf(c.y, 1.f) * 255.99f);
  uint8_t b = uint8_t(fminf(c.z, 1.f) * 255.99f);
  return (255u << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | r;
}

// Sky
__device__ __forceinline__ color sky(const Ray &r) {
  const vec3 unit_dir = r.dir.normalized();
  const float t = 0.5f * (unit_dir.y + 1.f);
  return lerp(color(1.f, 1.f, 1.f), color(0.7f, 0.4f, 1.f), t);
}

// Traverse scene
__device__ color ray_color(Ray r, curandState *s) {
  color throughput(1.f, 1.f, 1.f); // accumulate attentuations

  for (int bounce = 0; bounce < MAX_DEPTH; ++bounce) {

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

    if (!hit_anything) // miss
      return throughput * sky(r);

    color attentuation;
    Ray scattered;
    const Material &mat = d_materials[rec.mat_idx];

    if (!scatter(mat, rec, r, attentuation, scattered, s))
      return color(0.f, 0.f, 0.f); // absorb

    throughput = throughput * attentuation;
    r = scattered;
  }

  return color(0.f, 0.f, 0.f);
}

// kernel
__global__ void render_kernel(uint32_t *fb, int width, int height, int samples,
                              curandState *rand_states) {
  const int px = threadIdx.x + blockIdx.x * blockDim.x;
  const int py = threadIdx.y + blockIdx.y * blockDim.y;

  if ((px >= width) || (py >= height))
    return;

  const int p_idx = py * width + px;

  // reg for speed
  curandState state = rand_states[p_idx];

  color pixel(0.f, 0.f, 0.f);

  for (int i = 0; i < samples; ++i) {
    const float u = (float(px) + rand_f(&state)) / float(width - 1);
    const float v = (float(py) + rand_f(&state)) / float(height - 1);
    pixel += ray_color(d_cam.get_ray(u, v), &state);
  }

  rand_states[p_idx] = state; // for next frame

  fb[p_idx] = color_to_rgba(pixel / float(samples));
}

// called once at startup, each thread gets unique sseed
static __global__ void init_rand_kernel(curandState *states, int width,
                                        int height, unsigned long long seed) {
  const int px = threadIdx.x + blockIdx.x * blockDim.x;
  const int py = threadIdx.y + blockIdx.y * blockDim.y;

  if ((px >= width) || (py >= height))
    return;

  const int p_idx = py * width + px;

  // independent random stream seed per pixel
  curand_init(seed + static_cast<unsigned long long>(p_idx), 0, 0,
              &states[p_idx]);
}

void *alloc_rand_states(int width, int height) {
  curandState *states = nullptr;
  size_t pool_size = static_cast<size_t>(width) * height * sizeof(curandState);
  CUDA_CHECK(cudaMalloc(&states, pool_size));
  const dim3 block(16, 16);
  const dim3 grid((width + 15) / 16, (height + 15) / 16);

  init_rand_kernel<<<grid, block>>>(states, width, height, 69ULL);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("[CUDA] curandState pool intilaized %dx%d (%zu bytes)", //
         width, height, pool_size);

  return static_cast<void *>(states);
}

__host__ void init_scene() {
  const int num_spheres = 6;
  const int num_materials = 5;

  const Material mats[num_materials] = {
      {color(0.8f, 0.8f, 0.f), 0.f, 0.f, MatType::Lambertian},
      {color(0.3f, 0.3f, 0.3f), 0.f, 0.f, MatType::Metal},
      {color(1.f, 1.f, 1.f), 0.f, 1.5f, MatType::Dielectric},
      {color(0.2f, 0.6f, 0.2f), 0.f, 0.f, MatType::Lambertian},
      {color(0.1f, 0.1f, 0.4f), 0.5f, 0.f, MatType::Metal},
  };

  const Sphere spheres[num_spheres]{
      Sphere(point3(0.f, 0.f, 0.f), 0.5f, 0),
      Sphere(point3(0.f, -100.5f, 0.f), 100.f, 1),
      Sphere(point3(-10.f, 0.f, 0.f), 4.f, 2),
      Sphere(point3(0.f, 2.f, 0.f), 1.f, 3),
      Sphere(point3(1.f, 1.f, 1.5f), 0.5f, 2),
      Sphere(point3(-1.f, 1.f, 1.f), 0.75f, 4),
  };

  CUDA_CHECK(
      cudaMemcpyToSymbol(d_materials, &mats, num_materials * sizeof(Material)));
  CUDA_CHECK(
      cudaMemcpyToSymbol(d_spheres, &spheres, num_spheres * sizeof(Sphere)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_num_spheres, &num_spheres, sizeof(int)));
}

// KERNEL LAUNCH
__host__ void launch_render(uint32_t *d_fb, void *d_rand_states,
                            const RenderParams &params) {
  auto *states = static_cast<curandState *>(d_rand_states);

  // SETUP
  const float angle = params.time * 0.1f;
  const float distance = 3.f;
  const Camera cam(point3(sinf(angle) * distance, 1.f, cosf(angle) * distance),
                   point3(0.f, 0.f, 0.f), vec3(0.f, 1.f, 0.f), 90.f,
                   params.aspect_ratio);
  CUDA_CHECK(cudaMemcpyToSymbol(d_cam, &cam, sizeof(Camera)));

  // LAUNCH
  const dim3 block_size(16, 16);

  const dim3 grid_size((params.width + block_size.x - 1) / block_size.x,
                       (params.height + block_size.y - 1) / block_size.y);

  render_kernel<<<grid_size, block_size>>>(d_fb, params.width, params.height,
                                           NUM_SAMPLING, states);

  // Check kernel execution
  CUDA_CHECK(cudaGetLastError());

  // wait for framebuffer copy to CPU
  CUDA_CHECK(cudaDeviceSynchronize());
}
