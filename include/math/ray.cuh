#pragma once
#include "vec3.cuh"

struct Ray {
  point3 origin;
  vec3 dir;

  __host__ __device__ Ray() {}
  __host__ __device__ Ray(const point3 &o, const vec3 &d) : origin(o), dir(d) {}

  __host__ __device__ point3 at(float t) const { return origin + (t * dir); }
};