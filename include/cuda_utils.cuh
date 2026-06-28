#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifdef __clang__
// for clangd's parser for <<< >>>
extern "C" unsigned cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream = 0);
#endif

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)\

//