#pragma once
#include <cstdint>

struct RenderParams {
  int width;
  int height;
  float aspect_ratio;
  float time;
};

// allocate and init curandState/pixel
void *alloc_rand_states(int width, int height);

// initializes scene data
void init_scene();

// Launch render kernel
__host__ void launch_render(uint32_t *d_fb, void *d_rand_states,
                            const RenderParams &params);
