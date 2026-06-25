#pragma once
#include <cstdint>

struct RenderParams {
  int width;
  int height;
  float aspect_ratio;
};

// Launch render kernel
__host__ void launch_render(float *d_fb, const RenderParams &params);
