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

  __host__ __device__ Rect(float a0, float a1, float b0, float b1, float k,
                           const RectAxis &axis, int m)
      : a0(a0), a1(a1), b0(b0), b1(b1), k(k), axis(axis), mat_idx(m) {}

  __device__ bool hit(const Ray &r, float t_min, float t_max,
                      HitRecord &rec) const {
    float origin_k, dir_k;
    float origin_a, origin_b;
    float dir_a, dir_b;
    vec3 outward_normal;

    switch (axis) {
    case RectAxis::XZ:
      origin_k = r.origin.y;
      dir_k = r.dir.y;
      origin_a = r.origin.x;
      origin_b = r.origin.z;
      dir_a = r.dir.x;
      dir_b = r.dir.z;
      outward_normal = vec3(0.f, 1.f, 0.f);
      break;
    case RectAxis::XY:
      origin_k = r.origin.z;
      dir_k = r.dir.z;
      origin_a = r.origin.x;
      origin_b = r.origin.y;
      dir_a = r.dir.x;
      dir_b = r.dir.y;
      outward_normal = vec3(0.f, 0.f, 1.f);
      break;
    case RectAxis::YZ:
      origin_k = r.origin.x;
      dir_k = r.dir.x;
      origin_a = r.origin.y;
      origin_b = r.origin.z;
      dir_a = r.dir.y;
      dir_b = r.dir.z;
      outward_normal = vec3(1.f, 0.f, 0.f);
      break;
    }

    if (fabsf(dir_k) < 1e-6f) // parallel
      return false;

    const float t = (k - origin_k) / dir_k;
    if (t < t_min || t > t_max)
      return false;

    // hit position on the varying axes
    const float a_hit = origin_a + t * dir_a;
    const float b_hit = origin_b + t * dir_b;

    if (a_hit < a0 || a_hit > a1 || b_hit < b0 || b_hit > b1)
      return false;

    rec.t = t;
    rec.point = r.at(t);
    rec.mat_idx = mat_idx;
    rec.set_face_normal(r, outward_normal);
    return true;
  }
};
