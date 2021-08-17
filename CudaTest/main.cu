
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
#include "Camera.h"
#include "World.h"

#include "KernelRandom.h"

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
    World* world;
    Viewport* viewport;
    Tuple* pixelBuffer;
    Camera* camera;
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

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

CUDA_DEVICE void writePixel(Tuple* pixelBuffer, int32_t index, const Tuple& pixelColor) {
    pixelBuffer[index] = pixelColor;
}

CUDA_GLOBAL void createObject(Shape** object, Tuple origin, double radius) {
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

CUDA_GLOBAL void deleteObject(Shape** object) {
    delete (*object);
}

CUDA_GLOBAL void fillBufferKernel(int32_t width, int32_t height, Payload* payload) {
    int32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t column = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t index = row * width + column;

    //auto viewport = payload->viewport;

    Tuple defaultColor = Color::skyBlue;
    Tuple pixelColor = Color::black;

    const int32_t samplesPerPixel = 1;

    for (int i = 0; i < samplesPerPixel; i++) {
        curandState state;
        curand_init((unsigned long long)clock() + column, 0, 0, &state);

        double rx = curand_uniform_double(&state);
        double ry = curand_uniform_double(&state);

        //auto x = (viewport->height * (column + 0.5 + rx) / width - 1) * viewport->imageAspectRatio * viewport->scale;
        //auto y = (1.0 - viewport->height * (row + 0.5 + ry) / height) * viewport->scale;
        auto x = (static_cast<double>(column) + rx) / (width - 1);
        auto y = (static_cast<double>(row) + ry) / (height - 1);
        
        auto ray = payload->camera->getRay(x, y);

        Intersection intersections[MAXELEMENTS];

        int32_t count = 0;
    
        payload->world->intersect(ray, intersections, &count);

        auto hit = nearestHit(intersections, count);

        if (hit.bHit) {
            pixelColor += hit.normal;
        }
        else {
            pixelColor += defaultColor;
        }
    }

    writePixel(payload->pixelBuffer, index, pixelColor / samplesPerPixel);
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
    std::cout << "每个SM的最大线程束数：" << devicePro.warpSize << std::endl;
}

int main()
{
    queryDeviceProperties();

    fillBufferCuda();

    return 0;
}

void fillBufferCuda() {
    // Choose which GPU to run on, change this on a multi-GPU system.
    gpuErrorCheck(cudaSetDevice(0));

    constexpr auto width = 640;
    constexpr auto height = 480;

#if 1
    Payload* payload = nullptr;

    gpuErrorCheck(cudaMallocManaged((void**)&payload, sizeof(Payload)));

    gpuErrorCheck(cudaMallocManaged((void**)&payload->pixelBuffer, width * height * sizeof(Tuple)));

    gpuErrorCheck(cudaMallocManaged((void**)&payload->viewport, sizeof(Viewport)));

    payload->viewport->fov = 90.0;
    payload->viewport->scale = std::tan(Math::radians(payload->viewport->fov / 2));

    payload->viewport->imageAspectRatio = static_cast<double>(width) / height;

    payload->viewport->height = 2.0 * payload->viewport->scale;
    payload->viewport->width = payload->viewport->height * payload->viewport->imageAspectRatio;

    gpuErrorCheck(cudaMallocManaged((void**)&payload->camera, sizeof(Camera)));

    payload->camera->init(width, height);
    payload->camera->computeParameters();

    constexpr int32_t objectCount = 4;

    Shape** objects[objectCount];

    for (auto i = 0; i < objectCount; i++) {
        gpuErrorCheck(cudaMallocManaged((void**)&objects[i], sizeof(Shape**)));
    }

    createObject<<<1, 1>>>(objects[0], point(-1.0, 0.0, -3.0), 1.0);
    createObject<<<1, 1>>>(objects[1], point(1.0, 0.0, -3.0), 1.0);
    createObject<<<1, 1>>>(objects[2], point(-3.0, 0.0, -3.0), 1.0);
    createObject<<<1, 1>>>(objects[3], point(3.0, 0.0, -3.0), 1.0);
    
    gpuErrorCheck(cudaDeviceSynchronize());
    
    gpuErrorCheck(cudaMallocManaged((void**)&payload->world, sizeof(World)));

    for (auto i = 0; i < objectCount; i++) {
        payload->world->addObject(*objects[i]);
    }

    //gpuErrorCheck(cudaMallocManaged((void**)&payload->object->material, sizeof(Material)));
  
    Timer timer;

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    fillBufferKernel<<<gridSize, blockSize>>>(width, height, payload);

    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop();

    writeToPPM("render.ppm", width, height, payload->pixelBuffer);

    gpuErrorCheck(cudaFree(payload->world));

    for (auto i = 0; i < objectCount; i++) {
        deleteObject<<<1, 1>>>(objects[i]);
    }
    
    gpuErrorCheck(cudaDeviceSynchronize());

    for (auto i = 0; i < objectCount; i++) {
        gpuErrorCheck(cudaFree(objects[i]));
    }

    gpuErrorCheck(cudaFree(payload->viewport));
    gpuErrorCheck(cudaFree(payload->pixelBuffer));

#endif
}