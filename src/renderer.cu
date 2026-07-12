#include "math/ray.cuh"
#include "math/vec3.cuh"
#include "render/material.cuh"
#include "render/renderer.cuh"
#include "scene/camera.cuh"
#include "scene/hittable.cuh"
#include "scene/plane.cuh"
#include "scene/rect.cuh"
#include "scene/sphere.cuh"
#include "utils/cuda_utils.cuh"
#include <cstdint>

// LIMITS
static constexpr int MAX_DEPTH = 8;
static constexpr int NUM_SAMPLING = 16;
static constexpr int MAX_SPHERES = 32;
static constexpr int MAX_PLANES = 8;
static constexpr int MAX_RECTS = 32;
static constexpr int MAX_MATERIALS = 32;

// CONSTANT MEMORY
__constant__ Camera d_cam;
__constant__ Sphere d_spheres[MAX_SPHERES];
__constant__ Plane d_planes[MAX_PLANES];
__constant__ Rect d_rects[MAX_RECTS];
__constant__ Material d_materials[MAX_MATERIALS];
__constant__ int d_num_spheres;
__constant__ int d_num_planes;
__constant__ int d_num_rects;

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

__device__ bool hit_scene(const Ray &r, float t_min, float t_max,
                          HitRecord &rec) {
  bool hit_anything = false;
  float curr_closest = t_max;

  for (int i = 0; i < d_num_spheres; ++i)
    if (d_spheres[i].hit(r, t_min, curr_closest, rec)) {
      hit_anything = true;
      curr_closest = rec.t;
    }

  for (int i = 0; i < d_num_planes; ++i)
    if (d_planes[i].hit(r, t_min, curr_closest, rec)) {
      hit_anything = true;
      curr_closest = rec.t;
    }

  for (int i = 0; i < d_num_rects; ++i)
    if (d_rects[i].hit(r, t_min, curr_closest, rec)) {
      hit_anything = true;
      curr_closest = rec.t;
    }

  return hit_anything;
}

// Traverse scene
__device__ color ray_color(Ray r, curandState *s) {
  color accumulated(0.f, 0.f, 0.f);
  color throughput(1.f, 1.f, 1.f); // accumulate attenuations

  for (int bounce = 0; bounce < MAX_DEPTH; ++bounce) {

    HitRecord rec;

    if (!hit_scene(r, 0.001f, 1e30f, rec)) { // miss
      // accumulated += throughput * sky(r);
      break;
    }

    const Material &mat = d_materials[rec.mat_idx];
    // collect emission
    accumulated += throughput * emit(mat, rec);
    // then, scatter
    color attentuation;
    Ray scattered;

    if (!scatter(mat, rec, r, attentuation, scattered, s))
      break; // absorb

    throughput = throughput * attentuation;
    r = scattered;

    // RUSSIAN ROULETTE
    if (bounce > 3) {
      float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
      p = fminf(p, 0.95f); // avoid p==1 edge case
      if (rand_f(s) > p)
        break;
      throughput = throughput / p; // compensate
    }
  }

  return accumulated;
}

// kernel
__global__ void render_kernel(uint32_t *fb, vec3 *accum_fb, int width,
                              int height, int samples, curandState *rand_states,
                              int frame_index) {
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

  color curr_frame_color = pixel / float(samples);
  color accumulated_color = curr_frame_color;
  if (frame_index > 1) {
    color old_color = accum_fb[p_idx];
    // Stable rolling average formula:
    accumulated_color =
        old_color + (curr_frame_color - old_color) / float(frame_index);
  }

  // Save the high-precision float color for the next frame
  accum_fb[p_idx] = accumulated_color;

  // Output to the display texture
  fb[p_idx] = color_to_rgba(accumulated_color);
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

__host__ void init_scene(const RenderParams &params) {

  // SETUP
  const float distance = -8.f;
  const Camera cam(point3(distance, 3.f, 0), point3(0.f, 1.f, 0.f),
                   vec3(0.f, 1.f, 0.f), 60.f, params.aspect_ratio);
  CUDA_CHECK(cudaMemcpyToSymbol(d_cam, &cam, sizeof(Camera)));

  const int num_materials = 8;
  const int num_spheres = 3;
  const int num_planes = 2;
  const int num_rects = 10;

  const Material mats[num_materials] = {
      make_lambertian(color(1.f, 1.f, 1.f)),
      make_lambertian(color(1.f, 0.f, 0.f)),
      make_lambertian(color(0.f, 0.f, 1.f)),
      make_emitter(color(1.f, 1.f, 1.f), 20.f),
      make_metal(color(0.3f, 0.3f, 0.3f), 0.3f),
      make_dielectric(),
      make_lambertian(color(0.f, 1.f, 0.f)),
      make_lambertian(color(1.f, 1.f, 0.f)),
  };

  const point3 &pedestal_ball_pos = point3(0.f, 2.5f, -2.f);
  const Sphere &pedestal_ball = Sphere(pedestal_ball_pos, 0.75f, 7);
  const float pedestal_thickness = 0.7f;
  const point3 &pedestal_offset = pedestal_ball_pos -
                                  pedestal_thickness * point3(1.f, 0.f, 1.f) -
                                  point3(0.f, pedestal_ball.radius, 0.f);

  const Sphere spheres[num_spheres] = {
      Sphere(point3(-3.f, 1.f, -2.f), 1.f, 4),
      Sphere(point3(-2.f, 1.5f, 2.f), 1.5f, 5),
      pedestal_ball,
  };

  const Plane planes[num_planes] = {
      Plane(point3(0.f, 0.f, 0.f), vec3(0.f, 1.f, 0.f), 0),  // ground
      Plane(point3(0.f, 5.f, 0.f), vec3(0.f, -1.f, 0.f), 0), // roof
  };

  const Rect rects[num_rects] = {
      Rect(0.f, 1.f, -1.f, 1.f, 5.f, RectAxis::XZ, 3, true), // lamp
      Rect(0.f, 5.f, -50.f, 50.f, 4.f, RectAxis::YZ, 6),     // far wall
      Rect(-10.f, 10.f, 0.f, 5.f, -4.f, RectAxis::XY, 1),    // right wall
      Rect(-10.f, 10.f, 0.f, 5.f, 4.f, RectAxis::XY, 2),     // left wall
      Rect(0.f, 5.f, -50.f, 50.f, -15.f, RectAxis::YZ, 0),   // back wall
      Rect(pedestal_offset.x, pedestal_offset.x + 2 * pedestal_thickness,
           pedestal_offset.z, pedestal_offset.z + 2 * pedestal_thickness,
           pedestal_offset.y, RectAxis::XZ,
           0), // top pedestal
      Rect(pedestal_offset.x, pedestal_offset.x + 2 * pedestal_thickness, 0.f,
           pedestal_offset.y, pedestal_offset.z + 2 * pedestal_thickness,
           RectAxis::XY, 0), // right pedestal
      Rect(pedestal_offset.x, pedestal_offset.x + 2 * pedestal_thickness, 0.f,
           pedestal_offset.y, pedestal_offset.z, RectAxis::XY,
           0), // left pedestal
      Rect(0.f, pedestal_offset.y, pedestal_offset.z,
           pedestal_offset.z + 2 * pedestal_thickness,
           pedestal_offset.x + 2 * pedestal_thickness, RectAxis::YZ,
           0), // far pedestal
      Rect(0.f, pedestal_offset.y, pedestal_offset.z,
           pedestal_offset.z + 2 * pedestal_thickness, pedestal_offset.x,
           RectAxis::YZ,
           0), // back pedestal
  };

  CUDA_CHECK(
      cudaMemcpyToSymbol(d_materials, &mats, num_materials * sizeof(Material)));
  CUDA_CHECK(
      cudaMemcpyToSymbol(d_spheres, &spheres, num_spheres * sizeof(Sphere)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_planes, &planes, num_planes * sizeof(Plane)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_rects, &rects, num_rects * sizeof(Rect)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_num_spheres, &num_spheres, sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_num_planes, &num_planes, sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_num_rects, &num_rects, sizeof(int)));
}

// KERNEL LAUNCH
__host__ void launch_render(uint32_t *d_fb, vec3 *d_accum_fb,
                            void *d_rand_states, const RenderParams &params) {
  auto *states = static_cast<curandState *>(d_rand_states);

  // LAUNCH
  const dim3 block_size(16, 16);

  const dim3 grid_size((params.width + block_size.x - 1) / block_size.x,
                       (params.height + block_size.y - 1) / block_size.y);

  render_kernel<<<grid_size, block_size>>>(d_fb, d_accum_fb, params.width,
                                           params.height, NUM_SAMPLING, states,
                                           params.frame_index);

  // Check kernel execution
  CUDA_CHECK(cudaGetLastError());
}
