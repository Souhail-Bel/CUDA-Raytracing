#pragma once
#include "../math/vec3.cuh"
#include <cstdint>
#include <cuda_runtime.h>

struct RenderParams {
  int width;
  int height;
  float aspect_ratio;
  float time;
  int frame_index;
};

// allocate and init curandState/pixel
void *alloc_rand_states(int width, int height);

// initializes scene data
__host__ void init_scene(const RenderParams &rp);

// Launch render kernel
__host__ void launch_render(uint32_t *d_fb, vec3 *d_accum_fb,
                            void *d_rand_states, const RenderParams &params);
