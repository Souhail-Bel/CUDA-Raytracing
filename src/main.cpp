#include "cuda_utils.cuh"
#include "renderer.cuh"
#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

int main() {
  RenderParams rp = {1280, 720, 16. / 9};

  int nx = rp.width;
  int ny = rp.height;
  int num_pixels = ny * nx;
  size_t fb_size = 3 * num_pixels * sizeof(float);

  float *fb;
  CUDA_CHECK(cudaMallocManaged((void **)&fb, fb_size));

  launch_render(fb, rp);

  std::ofstream out("output_1.ppm");
  if (!out) {
    std::cerr << "Error opening PPM for write.\n";
    return 1;
  }

  out << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * 3 * nx + i * 3;
      float r = fb[pixel_index + 0];
      float g = fb[pixel_index + 1];
      float b = fb[pixel_index + 2];
      int ir = int(255.99 * r);
      int ig = int(255.99 * g);
      int ib = int(255.99 * b);
      out << ir << " " << ig << " " << ib << "\n";
    }
  }

  out.close();

  CUDA_CHECK(cudaFree(fb));
}
