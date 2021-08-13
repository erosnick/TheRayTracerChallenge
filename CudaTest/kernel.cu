
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Timer.h"
#include "Tuple.h"
#include "Ray.h"
#include "Constants.h"
#include "Utils.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>

struct Matrix {
    int32_t width;
    int32_t height;
    double* elements;
};

struct Viewport {
    double scale;
    double fov;
    double imageAspectRatio;
    double width;
    double height;
};

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// 获取矩阵A的(row, column)元素
__device__ double getElement(Matrix* A, int32_t row, int32_t column) {
    return A->elements[row * A->width + column];
}

// 为矩阵A的(row, column)元素赋值
__device__ void setElement(Matrix* A, int32_t row, int32_t column, double value) {
    A->elements[row * A->width + column] = value;
}

// 矩阵相乘kernel，2D，每个线程计算一个元素
__global__ void matrixMulKernel(Matrix* A, Matrix* B, Matrix* C) {
    double value = 0.0;
    int32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t column = threadIdx.x + blockIdx.x * blockDim.x;

    for (auto i = 0; i < A->width; i++) {
        value += getElement(A, row, i) * getElement(B, i, column);
    }

    setElement(C, row, column, value);
}

void matrixMulCuda();

__device__ void writePixel(Tuple* pixelBuffer, int32_t index, const Tuple& color) {
    pixelBuffer[index] = color;
}

__device__ Ray getRay(double x, double y) {
    Ray ray(point(0.0), vector(x, y, -1.0));
    return ray;
}

__global__ void fillBufferKernel(int32_t width, int32_t height, const Viewport& viewport, Tuple* pixelBuffer) {
    int32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t column = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t index = row * width + column;

    double x = (viewport.height * (column + 0.5) / width - 1) * viewport.imageAspectRatio * viewport.scale;
    double y = (1.0 - viewport.height * (row + 0.5) / height) * viewport.scale;

    Ray ray = getRay(x, y);

    Tuple color = { x, y, 0.0 };
    writePixel(pixelBuffer, index, color);
}

void fillBufferCuda();

void queryDeviceProperties() {
    int32_t deviceIndex = 0;
    cudaDeviceProp devicePro;
    cudaGetDeviceProperties(&devicePro, deviceIndex);

    std::cout << "使用的GPU device：" << deviceIndex << ": " << devicePro.name << std::endl;
    std::cout << "SM的数量：" << devicePro.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devicePro.sharedMemPerBlock / 1024.0 << "KB\n";
    std::cout << "每个SM的最大线程块数：" << devicePro.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "每个线程块的最大线程数：" << devicePro.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM的最大线程数：" << devicePro.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << devicePro.maxThreadsPerMultiProcessor / 32 << std::endl;
}

int main()
{
    queryDeviceProperties();

    //matrixMulCuda();
    fillBufferCuda();

    return 0;
}

void matrixMulCuda() {
    // Choose which GPU to run on, change this on a multi-GPU system.
    gpuErrorCheck(cudaSetDevice(0));

    int32_t width = 1 << 10;
    int32_t height = 1 << 10;
    Matrix* A;
    Matrix* B;
    Matrix* C;

    // 申请托管内存
    gpuErrorCheck(cudaMallocManaged((void**)&A, sizeof(Matrix)));
    gpuErrorCheck(cudaMallocManaged((void**)&B, sizeof(Matrix)));
    gpuErrorCheck(cudaMallocManaged((void**)&C, sizeof(Matrix)));

    int32_t bytes = width * height * sizeof(double);
    gpuErrorCheck(cudaMallocManaged((void**)&A->elements, bytes));
    gpuErrorCheck(cudaMallocManaged((void**)&B->elements, bytes));
    gpuErrorCheck(cudaMallocManaged((void**)&C->elements, bytes));

    // 初始化数据
    A->width = width;
    A->height = height;
    B->width = width;
    B->width = width;
    C->height = height;
    C->height = height;

    for (auto i = 0; i < width * height; i++) {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // 执行kernel
    matrixMulKernel << <gridSize, blockSize >> > (A, B, C);

    // 同步device，保证结果能正确访问
    gpuErrorCheck(cudaDeviceSynchronize());

    // 检查执行结果

    cudaFree(A->elements);
    cudaFree(A);
    cudaFree(B->elements);
    cudaFree(B);
    cudaFree(C->elements);
    cudaFree(C);
}

void fillBufferCuda() {
    // Choose which GPU to run on, change this on a multi-GPU system.
    gpuErrorCheck(cudaSetDevice(0));

    constexpr int32_t width = 640;
    constexpr int32_t height = 480;

    Tuple* pixelBuffer = nullptr;

    gpuErrorCheck(cudaMallocManaged((void**)&pixelBuffer, width * height * sizeof(Tuple)));

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    Viewport viewport;

    viewport.fov = 90.0;
    viewport.scale = std::tan(Math::radians(viewport.fov / 2));

    viewport.imageAspectRatio = static_cast<double>(width) / height;

    viewport.height = 2.0 * viewport.scale;
    viewport.width = viewport.height * viewport.imageAspectRatio;
    
    Timer timer;

    fillBufferKernel << <gridSize, blockSize >> > (width, height, viewport, pixelBuffer);

    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop();

    writeToPPM("render.ppm", width, height, pixelBuffer);

    gpuErrorCheck(cudaFree(pixelBuffer));
}