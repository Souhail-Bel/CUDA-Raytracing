#pragma once
#include "hittable.cuh"

struct Plane {
  point3 point;
  vec3 normal;
  int mat_idx;

  __host__ __device__ constexpr Plane() : mat_idx(0) {}
  __host__ __device__ Plane(const point3 &p, const vec3 &n, int m)
      : point(p), normal(n.normalized()), mat_idx(m) {}

  __device__ bool hit(const Ray &r, float t_min, float t_max,
                      HitRecord &rec) const {

    const float denom = dot(r.dir, normal);

    if (fabsf(denom) < 1e-6f) // parallel
      return false;

    const float t = dot(point - r.origin, normal) / denom;
    if (t < t_min || t > t_max)
      return false;

    rec.t = t;
    rec.point = r.at(t);
    rec.mat_idx = mat_idx;
    rec.set_face_normal(r, normal);

    return true;
  }
};