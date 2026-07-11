#pragma once
#include "hittable.cuh"
#include <cstdint>

enum class RectAxis : uint8_t { XZ = 0, XY = 1, YZ = 2 };

struct Rect {
  float a0, a1;
  float b0, b1;
  float k;
  RectAxis axis;
  int mat_idx;

  __host__ __device__ constexpr Rect()
      : a0(0), a1(0), b0(0), b1(0), k(0), axis(RectAxis::XZ), mat_idx(0) {}

      
};