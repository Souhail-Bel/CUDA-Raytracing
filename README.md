# CUDA Raytracer
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/Nvidia_CUDA_Logo.jpg" alt="CUDA Logo" width="300">
</p>

Having made a raytracer on the CPU, and wanting to see what I can do with CUDA, I decided to give it a spin on the Nvidia metal.

## Build & Run
This project runs exclusively on Nvidia hardware (CUDA platform). \
Make sure you have the toolkit installed, then run:
```bash
nvidia-smi --query-gpu=name,compute_cap
```
To check the compute capability. \
Name alone doesn't suffice (RTX 2050 mobile for instance is actually of Ampere architecture, cc 8.6). \
Make the appropriate changes in the `CMakeLists.txt` file and then simply build and run with:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./raytracer
```
