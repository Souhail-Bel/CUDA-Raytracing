#pragma once
#include <cmath>
#include <cuda_runtime.h>

struct vec3 {
  float x, y, z;

  // def
  __host__ __device__ constexpr vec3() : x(0.f), y(0.f), z(0.f) {}
  __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

  // arith
  __host__ __device__ vec3 operator-() const { return {-x, -y, -z}; }
  __host__ __device__ vec3 operator+(const vec3 &v) const {
    return {x + v.x, y + v.y, z + v.z};
  }
  __host__ __device__ vec3 operator-(const vec3 &v) const {
    return {x - v.x, y - v.y, z - v.z};
  }
  __host__ __device__ vec3 operator*(const vec3 &v) const {
    return {x * v.x, y * v.y, z * v.z};
  }
  __host__ __device__ vec3 operator*(float t) const {
    return {x * t, y * t, z * t};
  }
  __host__ __device__ vec3 operator/(float t) const {
    return (*this) * (1.f / t);
  }

  __host__ __device__ vec3 &operator+=(const vec3 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  __host__ __device__ vec3 &operator-=(const vec3 &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  __host__ __device__ vec3 &operator*=(float t) {
    x *= t;
    y *= t;
    z *= t;
    return *this;
  }
  __host__ __device__ vec3 &operator/=(float t) { return (*this) *= (1. / t); }

  // ops
  __host__ __device__ float dot(const vec3 &v) const {
    return x * v.x + y * v.y + z * v.z;
  }
  __host__ __device__ vec3 cross(const vec3 &v) const {
    return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
  }
  __host__ __device__ float length_squared() const {
    return x * x + y * y + z * z;
  }
  __host__ __device__ float length() const { return sqrtf(length_squared()); }
  __host__ __device__ vec3 normalized() const { return (*this) / length(); }

  __host__ __device__ bool near_zero() const {
    const float eps = 1e-8f;
    return fabsf(x) < eps && fabsf(y) < eps && fabsf(z) < eps;
  }
};

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
  return v * t;
}
__host__ __device__ inline float dot(const vec3 &a, const vec3 &b) {
  return a.dot(b);
}
__host__ __device__ inline vec3 cross(const vec3 &a, const vec3 &b) {
  return a.cross(b);
}

__host__ __device__ inline vec3 lerp(const vec3 &a, const vec3 &b, float t) {
  return a * (1.f - t) + b * t;
}

__host__ __device__ inline vec3 clamp(const vec3 &v, float lo, float hi) {
  return {fmaxf(lo, fminf(hi, v.x)), fmaxf(lo, fminf(hi, v.y)),
          fmaxf(lo, fminf(hi, v.z))};
}

// Reflection
//  u : incident ray
//  n : surface normal
__host__ __device__ inline vec3 reflect(const vec3 &u, const vec3 &n) {
  return u - 2.f * dot(u, n) * n;
}

// Snell's Law
//  uv : unit incident dir
//  n  : surface normal
//  etai_etat : n1/n2
__host__ __device__ inline vec3 refract(const vec3 &uv, const vec3 &n,
                                        float etai_etat) {
  float cos_theta = fminf(dot(-uv, n), 1.f);
  vec3 r_perp = etai_etat * (uv + cos_theta * n);
  vec3 r_parallel = -sqrtf(fabsf(1.f - r_perp.length_squared())) * n;
  return r_perp + r_parallel;
}

// aliases
using color = vec3;
using point3 = vec3;