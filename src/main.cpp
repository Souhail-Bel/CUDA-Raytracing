#include <cuda_runtime.h>
#include <iostream>

int main() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  cudaDeviceProp props;

  for (int i = 0; i < deviceCount; ++i) {
    cudaGetDeviceProperties(&props, i);
    std::cout << "GPU " << i << ": " << props.name << "\n";
    std::cout << "  Compute capability: " << props.major << "." << props.minor
              << "\n";
    std::cout << "  Global memory: "
              << props.totalGlobalMem / (1024. * 1024 * 1024) << " GB\n";
    std::cout << "  SM count: " << props.multiProcessorCount << "\n";
    std::cout << "  Max threads/block: " << props.maxThreadsPerBlock << "\n";
  }
  return 0;
}
