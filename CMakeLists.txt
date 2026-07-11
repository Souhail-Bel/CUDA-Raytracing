cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Use nvcc for .cu
project(cuda_raytracing LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# NVCC is broken with GCC 16
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- COMPUTER CAPABILITY ----
# --- CHECK WITH NVIDIA-SMI ---
# nvidia-smi --query-gpu=name,compute_cap
# Ampere architecture is used here (8.6 -> 86)
set(CMAKE_CUDA_ARCHITECTURES 86)


# i just use SDL2
find_package(SDL2 REQUIRED)
find_package(CUDAToolkit REQUIRED) # used for CUDA::curand

add_executable(raytracer
    src/main.cpp
    src/renderer.cu
)

target_include_directories(raytracer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SDL2_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(raytracer PRIVATE
    ${SDL2_LIBRARIES}
    CUDA::curand
)

# CUDA compiler flags
target_compile_options(raytracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -use_fast_math          # trig/sqrt
    >
)

# Compile device functions code separately
set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
