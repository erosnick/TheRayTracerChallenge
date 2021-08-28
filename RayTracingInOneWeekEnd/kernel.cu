
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
    auto t = 0.5 * (unitDirection.y() + 1.0);
    return lerp(Color::White(), Color::LightCornflower(), t);
}

CUDA_HOST_DEVICE void writePixel(uint8_t* pixelBuffer, int32_t index, double r, double g, double b) {
    pixelBuffer[index] = static_cast<uint8_t>(256 * std::clamp(r, 0.0, 0.999));
    pixelBuffer[index + 1] = static_cast<uint8_t>(256 * std::clamp(g, 0.0, 0.999));
    pixelBuffer[index + 2] = static_cast<uint8_t>(256 * std::clamp(b, 0.0, 0.999));
}


CUDA_GLOBAL void pathTracingKernel(uint8_t* pixelBuffer, int32_t width, int32_t height) {
    auto row = threadIdx.y + blockDim.y * blockIdx.y;
    auto column = threadIdx.x + blockDim.x * blockIdx.x;

    auto index = row * width + column;

    Vec3 upperLeftCorner(-2.0, 1.0, -1.0);
    Vec3 horizontal(4.0, 0.0, 0.0);
    Vec3 vertical(0.0, 2.0, 0.0);
    Vec3 origin(0.0, 0.0, 0.0);

    auto u = double(column) / width;
    auto v = double(row) / height;

    Ray ray(origin, unitVector(upperLeftCorner + u * horizontal + v * vertical));

    auto color = rayColor(ray);
    writePixel(pixelBuffer, index * 3, color.x(), color.y(), color.z());
}

 void pathTracing(uint8_t* pixelBuffer, int32_t width, int32_t height) {
    dim3 blockSize(20, 20);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.y,
                  (width + blockSize.y - 1) / blockSize.y);

    pathTracingKernel<<<gridSize, blockSize>>>(pixelBuffer, width, height);
    gpuErrorCheck(cudaDeviceSynchronize());
}

int main()
{
    constexpr auto width = 200;
    constexpr auto height = 100;

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
