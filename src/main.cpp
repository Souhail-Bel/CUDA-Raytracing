#include "cuda_utils.cuh"
#include "renderer.cuh"
#include <SDL2/SDL.h>
#include <SDL_events.h>
#include <SDL_keycode.h>
#include <SDL_render.h>
#include <SDL_stdinc.h>
#include <SDL_timer.h>
#include <SDL_video.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

static void SDL_DIE(const char *msg) {
  fprintf(stderr, "[SDL FAIL] %s: %s\n", msg, SDL_GetError());
  exit(EXIT_FAILURE);
}

// CONFIGURATION
static constexpr int WIDTH = 1280;
static constexpr int HEIGHT = 720;
static constexpr int TARGET_FPS = 60;
static constexpr double TARGET_FRAME_MS = 1000. / TARGET_FPS; // 16.66ms

// TIMERS
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

static double ms_since(const TimePoint &t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static float s_since(const TimePoint &t) {
  return std::chrono::duration<float>(Clock::now() - t).count();
}

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
      win, -1, SDL_RENDERER_ACCELERATED); // No VSync prevention, we manage FPS
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

  // Rand states allocation
  void *d_rand_states = alloc_rand_states(WIDTH, HEIGHT);

  // --- RENDER SETUP ---
  RenderParams rp = {WIDTH, HEIGHT, float(WIDTH) / float(HEIGHT), 0.f};
  init_scene();

  int fps_frame_count = 0;
  float fps_display = 0.f;
  TimePoint t_fps = Clock::now();

  // --- SDL LOOP ---
  const TimePoint t_0 = Clock::now();
  bool is_running = true;
  SDL_Event event;

  while (is_running) {
    const TimePoint t_frame = Clock::now();
    rp.time = s_since(t_0);

    // * * RENDER AND TEXTURE COPY
    launch_render(d_framebuffer, d_rand_states, rp);

    // * * * * GPU -> CPU
    CUDA_CHECK(cudaMemcpy(h_framebuffer.data(), d_framebuffer, fb_bytes,
                          cudaMemcpyDeviceToHost));

    // * * * * CPU -> SDL Texture
    SDL_UpdateTexture(tex, nullptr, h_framebuffer.data(),
                      WIDTH * static_cast<int>(sizeof(uint32_t)));

    // * * * * PRESENT
    SDL_RenderClear(rend);
    SDL_RenderCopy(rend, tex, nullptr, nullptr);
    SDL_RenderPresent(rend);

    // * * EVENTS
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT)
        is_running = false;
      if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
        is_running = false;
    }

    // * * FPS CAP
    double frame_ms = ms_since(t_frame);
    if (frame_ms < TARGET_FRAME_MS)
      SDL_Delay(static_cast<Uint32>(TARGET_FRAME_MS - frame_ms));

    // * * FPS DISPLAY
    fps_frame_count++;
    float fps_elapsed = s_since(t_fps);
    if (fps_elapsed >= 0.5f) {
      fps_display = fps_frame_count / fps_elapsed;
      fps_frame_count = 0;
      t_fps = Clock::now();

      // won't work, console just hangs
      // printf("FPS %.1f | Took %.2f ms per frame", fps_display,
      //        1000.f / fps_display);

      char title[96];
      snprintf(title, sizeof(title),
               "CUDA Raytracer  |  %.1f FPS  |  %.2f ms/frame", fps_display,
               1000.f / fps_display);
      SDL_SetWindowTitle(win, title);
    }
  }

  // --- END ---
  CUDA_CHECK(cudaFree(d_rand_states));
  CUDA_CHECK(cudaFree(d_framebuffer));
  SDL_DestroyTexture(tex);
  SDL_DestroyRenderer(rend);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return 0;
}
