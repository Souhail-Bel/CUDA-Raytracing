#pragma once
#include "../scene/hittable.cuh"
#include "../math/ray.cuh"
#include "../math/vec3.cuh"
#include <cstdint>
#include <curand_kernel.h>

// MATERIAL REG
enum class MatType : uint8_t { //
  Lambertian = 0,
  Metal = 1,
  Dielectric = 2
};

struct Material {
  color albedo;
  float fuzz; // 0 mirror -> 1 very rough
  float ior;  // dielectric, 1.33 water
  MatType type;

  __host__ __device__ constexpr Material()
      : albedo{}, fuzz(0.f), ior(1.f), type{} {}

  __host__ __device__ Material( //
      color albedo, float fuzz, float ior, MatType type)
      : albedo(albedo), fuzz(fuzz), ior(ior), type(type) {}
};

// HELPERS
__device__ __forceinline__ float rand_f(curandState *s) {
  return curand_uniform(s);
}

__device__ __forceinline__ vec3 rand_unit_vector(curandState *s) {
  float z = 1.f - 2.f * rand_f(s);
  float r = sqrtf(fmaxf(0.f, 1.f - z * z));
  float phi = 2.f * 3.14159265f * rand_f(s);
  return vec3(r * cosf(phi), r * sinf(phi), z);
}

__device__ vec3 rand_in_unit_sphere(curandState *s) {
  return rand_unit_vector(s) * powf(rand_f(s), 1.f / 3.f);
}

// SCHLICK APPROX
__device__ __forceinline__ float schlick(float cos_theta, float ref_idx) {
  float r0 = (1.f - ref_idx) / (1.f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.f - r0) * powf(1.f - cos_theta, 5.f);
}

// SCATTER FUNCTIONS
__device__ bool scatter_lambertian( //
    const Material &mat, const HitRecord &rec, color &attenuation,
    Ray &scattered, curandState *s) {
  vec3 dir = rec.normal + rand_unit_vector(s);
  if (dir.near_zero()) // if rand cancels normal
    dir = rec.normal;

  scattered = Ray(rec.point, dir);
  attenuation = mat.albedo;
  return true;
}

__device__ bool scatter_metal( //
    const Material &mat, const HitRecord &rec, const Ray &r_in,
    color &attenuation, Ray &scattered, curandState *s) {
  // fuzz: 0 - mirror, 1 - matte
  const vec3 reflected = reflect(r_in.dir.normalized(), rec.normal);
  scattered = Ray(rec.point, reflected + mat.fuzz * rand_in_unit_sphere(s));
  attenuation = mat.albedo;

  // absorb ray below surface
  return dot(scattered.dir, rec.normal) > 0.f;
}

__device__ bool scatter_dielectric( //
    const Material &mat, const HitRecord &rec, const Ray &r_in,
    color &attenuation, Ray &scattered, curandState *s) {
  attenuation = color(1.f, 1.f, 1.f);

  // air-to-glass?
  const float rri = rec.is_front_face ? (1.f / mat.ior) : mat.ior;
  const vec3 unit_dir = r_in.dir.normalized();

  const float cos_theta = fminf(dot(-unit_dir, rec.normal), 1.f);
  const float sin_theta = sqrtf(1.f - cos_theta * cos_theta);

  const bool beyond_critical = rri * sin_theta > 1.f;

  vec3 direction;
  if (beyond_critical || schlick(cos_theta, rri) > rand_f(s))
    direction = reflect(unit_dir, rec.normal);
  else
    direction = refract(unit_dir, rec.normal, rri);

  scattered = Ray(rec.point, direction);
  return true;
}

//
__device__ bool scatter( //
    const Material &mat, const HitRecord &rec, const Ray &r_in,
    color &attenuation, Ray &scattered, curandState *s) {
  switch (mat.type) {
  case MatType::Lambertian:
    return scatter_lambertian(mat, rec, attenuation, scattered, s);
  case MatType::Metal:
    return scatter_metal(mat, rec, r_in, attenuation, scattered, s);
  case MatType::Dielectric:
    return scatter_dielectric(mat, rec, r_in, attenuation, scattered, s);
  default:
    return false;
  }
}