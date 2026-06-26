#include "cuda_utils.cuh"
#include "renderer.cuh"
#include <SDL2/SDL.h>
#include <SDL_events.h>
#include <SDL_keycode.h>
#include <SDL_render.h>
#include <SDL_video.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

static void SDL_DIE(const char *msg) {
  fprintf(stderr, "[SDL FAIL] %s: %s\n", msg, SDL_GetError());
  exit(EXIT_FAILURE);
}

// CONFIGURATION
static constexpr int WIDTH = 1280;
static constexpr int HEIGHT = 720;

int main() {

  // --- SDL SETUP ---
  if (SDL_Init(SDL_INIT_VIDEO) != 0)
    SDL_DIE("SDL_Init");

  SDL_Window *win =
      SDL_CreateWindow("CUDA Raytracer", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
  if (!win)
    SDL_DIE("SDL_CreateWindow");

  SDL_Renderer *rend = SDL_CreateRenderer(
      win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (!rend)
    SDL_DIE("SDL_CreateRenderer");

  SDL_Texture *tex = SDL_CreateTexture(
      rend, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
  if (!tex)
    SDL_DIE("SDL_CreateTexture");

  // --- FRAMEBUFFER ---
  const size_t fb_bytes =
      static_cast<size_t>(WIDTH) * HEIGHT * sizeof(uint32_t);

  // * * * * device VRAM
  uint32_t *d_framebuffer = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_framebuffer), fb_bytes));

  // * * * * host RAM
  std::vector<uint32_t> h_framebuffer(WIDTH * HEIGHT);

  // --- RENDER ---
  RenderParams rp = {WIDTH, HEIGHT, float(WIDTH) / float(HEIGHT)};

  printf("[CUDA] Rendering at %dx%d (%zu threads)...\n", WIDTH, HEIGHT,
         size_t(WIDTH) * HEIGHT);

  launch_render(d_framebuffer, rp);

  printf("[CUDA] Render done.\n");

  // --- TEXTURE COPY ---
  // * * * * GPU -> CPU
  CUDA_CHECK(cudaMemcpy(h_framebuffer.data(), d_framebuffer, fb_bytes,
                        cudaMemcpyDeviceToHost));

  // * * * * CPU -> SDL Texture
  SDL_UpdateTexture(tex, nullptr, h_framebuffer.data(),
                    WIDTH * static_cast<int>(sizeof(uint32_t)));

  printf("[SDL] Texture loaded.");

  // --- SDL LOOP ---
  bool is_running = true;
  SDL_Event event;

  while (is_running) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT)
        is_running = false;
      if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
        is_running = false;
    }

    SDL_RenderClear(rend);
    SDL_RenderCopy(rend, tex, nullptr, nullptr);
    SDL_RenderPresent(rend);
  }

  // --- END ---
  CUDA_CHECK(cudaFree(d_framebuffer));
  SDL_DestroyTexture(tex);
  SDL_DestroyRenderer(rend);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return 0;

}
