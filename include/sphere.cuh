#pragma once
#include "hittable.cuh"

struct Sphere {
  point3 center;
  float radius;

  __host__ __device__ constexpr Sphere() : center{}, radius(0.f) {}
  __host__ __device__ Sphere(const point3 &c, float r) : center(c), radius(r) {}

  __device__ bool hit(const Ray &r, float t_min, float t_max,
                      HitRecord &rec) const {
    const vec3 oc = r.origin - center;
    const float a = r.dir.length_squared();
    const float half_b = dot(oc, r.dir);
    const float c = oc.length_squared() - radius * radius;
    const float del_prime = half_b * half_b - a * c;

    if (del_prime < 0.f)
      return false;

    const float sqrtd = sqrtf(del_prime);

    // try nearest, then farthest
    float root = (-half_b - sqrtd) / a;
    if (root <= t_min || root >= t_max) {
      root = (-half_b + sqrtd) / a;
      if (root <= t_min || root >= t_max)
        return false;
    }

    rec.t = root;
    rec.point = r.at(rec.t);

    const vec3 outward_normal = (rec.point - center) / radius;
    rec.set_face_normal(r, outward_normal);

    return true;
  }
};