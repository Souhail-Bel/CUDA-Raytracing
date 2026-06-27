#pragma once
#include <cstdint>

struct RenderParams {
  int width;
  int height;
  float aspect_ratio;
  float time;
};

// Launch render kernel
__host__ void launch_render(uint32_t *d_fb, const RenderParams &params);
