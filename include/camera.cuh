#pragma once
#include "ray.cuh"
#include <cmath>

struct Camera {
  point3 eye_point;
  point3 pixel_00;
  vec3 horizontal;
  vec3 vertical;

  __host__ __device__ constexpr Camera() {}

  __host__ Camera(const point3 &look_from, const point3 &look_at,
                  const vec3 &vup,
                  float vfov, // degrees
                  float aspect_ratio) {
    const float theta = vfov * 3.14159265f / 180.f;
    const float vp_height = 2.f * tanf(theta * 0.5f);
    const float vp_width = aspect_ratio * vp_height;

    const vec3 w = (look_from - look_at).normalized();
    const vec3 u = cross(vup, w).normalized(); // right
    const vec3 v = cross(u, w);                // up

    eye_point = look_from;
    horizontal = vp_width * u;
    vertical = vp_height * v;

    pixel_00 = eye_point - horizontal * 0.5f - vertical * 0.5f - w;
  }

  __host__ __device__ Ray get_ray(float s, float t) const {
    const vec3 dir = pixel_00 + s * horizontal + t * vertical - eye_point;
    return Ray(eye_point, dir);
  }
};