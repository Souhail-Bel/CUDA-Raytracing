#pragma once
#include "ray.cuh"
#include "vec3.cuh"

struct HitRecord {
  point3 point;
  vec3 normal;
  float t;
  bool is_front_face;

  __device__ void set_face_normal(const Ray &r, const vec3 &outward_normal) {
    is_front_face = dot(r.dir, outward_normal) < 0.f;
    normal = is_front_face ? outward_normal : -outward_normal;
  }
};