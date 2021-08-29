
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA.h"
#include "Utils.h"
#include "GPUTimer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Ray.h"
#include "Constants.h"

#include <algorithm>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

CUDA_HOST_DEVICE Vec3 rayColor(const Ray& ray) {
    auto unitDirection = unitVector(ray.direction);
    auto t = 0.5f * (unitDirection.y() + 1.0f);
    return lerp(Color::White(), Color::LightCornflower(), t);
}

CUDA_HOST_DEVICE void writePixel(uint8_t* pixelBuffer, int32_t index, Float r, Float g, Float b) {
    Float start = 0.0f;
    Float end = 0.999f;
    pixelBuffer[index] = static_cast<uint8_t>(256 * std::clamp(r, start, end));
    pixelBuffer[index + 1] = static_cast<uint8_t>(256 * std::clamp(g, start, end));
    pixelBuffer[index + 2] = static_cast<uint8_t>(256 * std::clamp(b, start, end));
}


CUDA_GLOBAL void pathTracingKernel(uint8_t* pixelBuffer, int32_t width, int32_t height) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;

    auto index = y * width + x;

    if (index < width * height) {
        Vec3 upperLeftCorner(-2.0f, 1.0f, -1.0f);
        Vec3 horizontal(4.0f, 0.0f, 0.0f);
        Vec3 vertical(0.0f, 2.0f, 0.0f);
        Vec3 origin(0.0f, 0.0f, 0.0f);

        auto u = Float(x) / width;
        auto v = Float(y) / height;

        Ray ray(origin, unitVector(upperLeftCorner + u * horizontal + v * vertical));

        auto color = rayColor(ray);
        writePixel(pixelBuffer, index * 3, color.x(), color.y(), color.z());
    }
}

 void pathTracing(uint8_t* pixelBuffer, int32_t width, int32_t height) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.y,
                  (width + blockSize.y - 1) / blockSize.y);

    //int32_t minGridSize = 0;
    //int32_t blockSize = 0;
    //int32_t gridSize = 0;

    //auto size = width * height;

    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pathTracingKernel, 0, size);

    //// Round up according to array size
    //gridSize = (size + blockSize - 1) / blockSize;

    pathTracingKernel<<<gridSize, blockSize>>>(pixelBuffer, width, height);
    gpuErrorCheck(cudaDeviceSynchronize());
}

int main()
{
    constexpr auto width = 512;
    constexpr auto height = 384;

    uint8_t* pixelBuffer = nullptr;

    gpuErrorCheck(cudaMallocManaged(&pixelBuffer, sizeof(uint8_t) * width * height * 3));

    GPUTimer timer("Rendering start...");

    pathTracing(pixelBuffer, width, height);

    timer.stop("Rendering elapsed time");

    //writeToPPM("render.ppm", pixelBuffer, width, height);
    stbi_write_png("render.png", width, height, 3, pixelBuffer, width * 3);

    Utils::openImage(L"render.png");

    gpuErrorCheck(cudaFree(pixelBuffer));

    return 0;
}
