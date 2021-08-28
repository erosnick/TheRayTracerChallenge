#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA.h"
#include "GPUTimer.h"

#include <string>
#include <algorithm>
#include <random>

#define gpuErrorCheck(ans) { Utils::gpuAssert((ans), __FILE__, __LINE__); }

namespace Utils {
    inline double randomDouble(double start = 0.0, double end = 1.0) {
        std::uniform_real_distribution<double> distribution(start, end);
        static std::random_device randomDevice;
        static std::mt19937 generator(randomDevice());
        return distribution(generator);
    }

    inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

    // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction
    inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + 0.5); }

    std::string toPPM(int32_t width, int32_t height);
    void writeToPPM(const std::string& path, int32_t width, int32_t height, uint8_t* pixelBuffer);
    void writeToPNG(const std::string& path, int32_t width, int32_t height, uint8_t* pixelBuffer);
    void writeToPNG(const std::string& path, int32_t width, int32_t height, float3* pixelBuffer);
    void openImage(const std::wstring& path);

    inline CUDA_HOST_DEVICE void writePixel(uint8_t* pixelBuffer, int32_t index, double r, double g, double b) {
        pixelBuffer[index] = 256 * std::clamp(r, 0.0, 0.999);
        pixelBuffer[index + 1] = 256 * std::clamp(g, 0.0, 0.999);
        pixelBuffer[index + 2] = 256 * std::clamp(b, 0.0, 0.999);
    }

    inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

    inline void reportGPUUsageInfo() {
        size_t freeBytes;
        size_t totalBytes;

        gpuErrorCheck(cudaMemGetInfo(&freeBytes, &totalBytes));

        auto freeDb = (double)freeBytes;

        auto totalDb = (double)totalBytes;

        auto usedDb = totalDb - freeDb;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            usedDb / 1024.0 / 1024.0, freeDb / 1024.0 / 1024.0, totalDb / 1024.0 / 1024.0);
    }

    inline void queryDeviceProperties() {
        int32_t deviceIndex = 0;
        cudaDeviceProp devicePro;
        cudaGetDeviceProperties(&devicePro, deviceIndex);

        std::cout << "使用的GPU device：" << deviceIndex << ": " << devicePro.name << std::endl;
        std::cout << "SM的数量：" << devicePro.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devicePro.sharedMemPerBlock / 1024.0 << "KB\n";
        std::cout << "每个SM的最大线程块数：" << devicePro.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "每个线程块的最大线程数：" << devicePro.maxThreadsPerBlock << std::endl;
        std::cout << "每个SM的最大线程数：" << devicePro.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个SM的最大线程束数：" << devicePro.warpSize << std::endl;
    }
}
