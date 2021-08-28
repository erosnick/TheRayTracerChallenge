#pragma once

#include <cstdint>

void addKernel();

void juliaKernel(int32_t width, int32_t height);

void waveKernel(int32_t width, int32_t height, int32_t ticks);

void dotKernel(int32_t width, int32_t height);

void sharedMemoryKernel(int32_t width, int32_t height);

void rayTracingKernel(int32_t width, int32_t height);