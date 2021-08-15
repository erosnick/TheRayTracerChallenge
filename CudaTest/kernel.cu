
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Intersection.h"
#include "Timer.h"
#include "Tuple.h"
#include "Constants.h"
#include "Ray.h"
//#include "Sphere.h"
#include "Shape.h"
#include "Utils.h"
#include "Types.h"
#include "Material.h"

#include <math.h>

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

struct Payload {
    Intersection* intersections;
    Sphere* object;
    Viewport* viewport;
    Tuple* pixelBuffer;
};

class Sphere : public Shape {
public:
    CUDA_HOST_DEVICE Sphere()
    : origin({ 0.0, 0.0, 0.0 }), radius(1.0) {}

    CUDA_HOST_DEVICE Sphere(const Tuple& inOrigin, double inRadius = 1.0) 
    : origin(inOrigin), radius(inRadius) {}

    CUDA_HOST_DEVICE void foo() override {}

    inline CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position) const override {
        auto normal = (position - origin);
        return  normal.normalize();
    }

    inline CUDA_HOST_DEVICE void intersect(const Ray& ray, Intersection* intersections) override {
        auto oc = (ray.origin - origin);
        auto a = ray.direction.dot(ray.direction);
        auto b = 2.0 * ray.direction.dot(oc);
        auto c = oc.dot(oc) - radius * radius;

        auto discriminant = b * b - 4 * a * c;

        if (discriminant < 0.0) {
            return;
        }

        // 与巨大球体求交的时候，会出现判别式大于0，但是有两个负根的情况，
        // 这种情况出现在射线方向的反向延长线能和球体相交的场合。
        auto t1 = (-b - std::sqrt(discriminant)) / (2 * a);
        auto t2 = (-b + std::sqrt(discriminant)) / (2 * a);

        auto position1 = ray.position(t1);

        auto normal1 = normalAt(position1);

        auto position2 = ray.position(t2);

        auto normal2 = normalAt(position2);

        if ((t1 > 0.0) || (t2 > 0.0)) {
            intersections[0] = { true, false, 1, t1, this, position1, normal1, ray };
            intersections[1] = { true, false, 1, t2, this, position2, normal2, ray };
        }
    }

    Tuple origin;
    double radius;
};

CUDA_DEVICE double magnitudeSquared(const Tuple& v) {
    double lengthSquared = v.x() * v.x() + v.y() * v.y() + v.z() * v.z();
    return lengthSquared;
}

CUDA_DEVICE double magnitude(const Tuple& v) {
    return std::sqrt(magnitudeSquared(v));
}

CUDA_DEVICE Tuple normalize(const Tuple& v) {
    double length = magnitude(v);
    return Tuple(v.x() / length, v.y() / length, v.z() / length);
}

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
CUDA_DEVICE double getElement(Matrix* A, int32_t row, int32_t column) {
    return A->elements[row * A->width + column];
}

// 为矩阵A的(row, column)元素赋值
CUDA_DEVICE void setElement(Matrix* A, int32_t row, int32_t column, double value) {
    A->elements[row * A->width + column] = value;
}

// 矩阵相乘kernel，2D，每个线程计算一个元素
CUDA_GLOBAL void matrixMulKernel(Matrix* A, Matrix* B, Matrix* C) {
    double value = 0.0;
    int32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t column = threadIdx.x + blockIdx.x * blockDim.x;

    for (auto i = 0; i < A->width; i++) {
        value += getElement(A, row, i) * getElement(B, i, column);
    }

    setElement(C, row, column, value);
}

void matrixMulCuda();

CUDA_DEVICE void writePixel(Tuple* pixelBuffer, int32_t index, const Tuple& pixelColor) {
    pixelBuffer[index] = pixelColor;
}

CUDA_GLOBAL void createObject(Sphere** object, Tuple origin, double radius) {
    // It is necessary to create object representing a function
    // directly in global memory of the GPU device for virtual
    // functions to work correctly, i.e. virtual function table
    // HAS to be on GPU as well.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        //auto* t = new Test();
        //t->foo();
        (*object) = new Sphere(origin, radius);
    }
}

class Test {

};

CUDA_GLOBAL void deleteObject(Sphere** object) {
    delete (*object);
}

CUDA_GLOBAL void fillBufferKernel(int32_t width, int32_t height, Payload* payload) {
    int32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t column = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t index = row * width + column;

    auto viewport = payload->viewport;

    auto x = (viewport->height * (column + 0.5) / width - 1) * viewport->imageAspectRatio * viewport->scale;
    auto y = (1.0 - viewport->height * (row + 0.5) / height) * viewport->scale;

    auto direction = vector(x, y, -1.0);

    auto ray = Ray(point(0.0), direction.normalize());

    auto sphere = payload->object;

    Intersection intersections[2];

    //sphere->intersect(ray, intersections);
    sphere->foo();

    //Sphere* sphere = new Sphere();

    //delete sphere;
    //Test* test = new Test();

    //delete test;

    Tuple pixelColor = Color::skyBlue;

    //auto hit = nearestHitCUDA(intersections, 2);

    //if (hit.bHit) {
    //    pixelColor = hit.normal;
    //}


    writePixel(payload->pixelBuffer, index, pixelColor);
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

    constexpr auto width = 640;
    constexpr auto height = 480;

    Payload* payload = nullptr;

    gpuErrorCheck(cudaMallocManaged((void**)&payload, sizeof(Payload)));

    gpuErrorCheck(cudaMallocManaged((void**)&payload->pixelBuffer, width * height * sizeof(Tuple)));

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    gpuErrorCheck(cudaMallocManaged((void**)&payload->viewport, sizeof(Viewport)));

    payload->viewport->fov = 90.0;
    payload->viewport->scale = std::tan(Math::radians(payload->viewport->fov / 2));

    payload->viewport->imageAspectRatio = static_cast<double>(width) / height;

    payload->viewport->height = 2.0 * payload->viewport->scale;
    payload->viewport->width = payload->viewport->height * payload->viewport->imageAspectRatio;

    Sphere** object = nullptr;

    gpuErrorCheck(cudaMallocManaged((void**)&object, sizeof(Sphere**)));

    createObject<<<1, 1>>>(object, point(0.0, 0.0, -3.0), 1.0);
    
    gpuErrorCheck(cudaDeviceSynchronize());
    
    payload->object = *object;

    //gpuErrorCheck(cudaMallocManaged((void**)&payload->object->material, sizeof(Material)));

    gpuErrorCheck(cudaMallocManaged((void**)&payload->intersections, sizeof(Intersection) * 2));
  
    Timer timer;

    fillBufferKernel<<<gridSize, blockSize>>>(width, height, payload);

    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop();

    Intersection i;

    writeToPPM("render.ppm", width, height, payload->pixelBuffer);

    gpuErrorCheck(cudaFree(payload->intersections));
    deleteObject << <1, 1 >> > (object);
    gpuErrorCheck(cudaDeviceSynchronize());
    //gpuErrorCheck(cudaFree(payload->object));
    gpuErrorCheck(cudaFree(payload->viewport));
    gpuErrorCheck(cudaFree(payload->pixelBuffer));
}