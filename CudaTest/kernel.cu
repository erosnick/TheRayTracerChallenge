
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Intersection.h"
#include "Timer.h"
#include "Tuple.h"
#include "Constants.h"
#include "Ray.h"
#include "Shape.h"
#include "Utils.h"
#include "Types.h"
#include "Material.h"
#include "Camera.h"
#include "World.h"
#include "kernel.h"
#include "Shading.h"
#include "Light.h"

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

constexpr int32_t objectCount = 2;
constexpr int32_t lightCount = 1;
constexpr int32_t materialCount = 2;

Payload* payload = nullptr;

Shape** objects[objectCount];
Light** lights[lightCount];
Material** materials[materialCount];

class Sphere : public Shape {
public:
    CUDA_HOST_DEVICE Sphere()
    : origin({ 0.0, 0.0, 0.0 }), radius(1.0) {}

    CUDA_HOST_DEVICE Sphere(const Tuple& inOrigin, double inRadius = 1.0) 
    : origin(inOrigin), radius(inRadius) {}

    CUDA_HOST_DEVICE ~Sphere() {
        if (material) {
            delete material;
        }
    }

    inline CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position) const override {
        auto normal = (position - origin);
        return  normal.normalize();
    }

    inline CUDA_HOST_DEVICE bool intersect(const Ray& ray, Intersection* intersections) override {
        auto oc = (ray.origin - origin);
        auto a = ray.direction.dot(ray.direction);
        auto b = 2.0 * ray.direction.dot(oc);
        auto c = oc.dot(oc) - radius * radius;

        auto discriminant = b * b - 4 * a * c;

        if (discriminant < 0.0) {
            return false;
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
            intersections[0] = { true, true, 1, t1, this, position1, normal1, ray };
            intersections[1] = { true, true, 1, t2, this, position2, normal2, ray };
        }

        return true;
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

CUDA_DEVICE void writePixel(uint8_t* pixelBuffer, int32_t index, const Tuple& pixelColor) {
    pixelBuffer[index] = 256 * std::clamp(pixelColor.x(), 0.0, 0.999);
    pixelBuffer[index + 1] = 256 * std::clamp(pixelColor.y(), 0.0, 0.999);
    pixelBuffer[index + 2] = 256 * std::clamp(pixelColor.z(), 0.0, 0.999);
}

//void createObject(int32_t** objects[]) {
//    auto ptr = objects[0];
//    //auto ptr = (*objects[0]);
//    //(*objects[1]) = new Sphere();
//}

CUDA_GLOBAL void createObject(int32_t*** objects) {
    auto ptr = objects[0];
    //auto ptr = (*objects[0]);
    //(*objects[1]) = new Sphere();
}

CUDA_GLOBAL void createObject(Shape** object, Tuple* origins, double* radius = nullptr, Material** material[] = nullptr) {
    // It is necessary to create object representing a function
    // directly in global memory of the GPU device for virtual
    // functions to work correctly, i.e. virtual function table
    // HAS to be on GPU as well.
    auto index = threadIdx.x;
    //(*object[index]) = new Sphere(origins[index], radius[index]);
    //(*object[index]) = new Sphere();
    (*object) = new Sphere();
    printf("%d\n", index);
    //(*object)->material = material;
}

CUDA_GLOBAL void createLight(Light** light, Tuple inPosition, Tuple inIntensity) {
    // It is necessary to create object representing a function
    // directly in global memory of the GPU device for virtual
    // functions to work correctly, i.e. virtual function table
    // HAS to be on GPU as well.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*light) = new Light(inPosition, inIntensity);
    }
}

CUDA_GLOBAL void createMaterial(Material** material) {
    // It is necessary to create object representing a function
    // directly in global memory of the GPU device for virtual
    // functions to work correctly, i.e. virtual function table
    // HAS to be on GPU as well.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*material) = new Material();
    }
}

template<typename T>
CUDA_GLOBAL void deleteObject(T** object) {
    delete (*object);
}

CUDA_GLOBAL void fillBufferKernel(int32_t width, int32_t height, Payload* payload) {
    int32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t column = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t index = row * width + column;
    
    //row = 159;
    //column = 192;

    //auto viewport = payload->viewport;

    Tuple defaultColor = Color::skyBlue;
    Tuple pixelColor = Color::black;

    const int32_t samplesPerPixel = 1;

    for (int i = 0; i < samplesPerPixel; i++) {
        //curandState state;
        //curand_init((unsigned long long)clock() + column, 0, 0, &state);

        double rx = 0.0; // curand_uniform_double(&state);
        double ry = 0.0; // curand_uniform_double(&state);

        //auto x = (viewport->height * (column + 0.5 + rx) / width - 1) * viewport->imageAspectRatio * viewport->scale;
        //auto y = (1.0 - viewport->height * (row + 0.5 + ry) / height) * viewport->scale;
        auto x = (static_cast<double>(column) + rx) / (width - 1);
        auto y = (static_cast<double>(row) + ry) / (height - 1);
        
        auto ray = payload->camera->getRay(x, y);

        //pixelColor = colorAt(payload->world, ray, 1);
        Intersection intersections[MAXELEMENTS];
        auto count = 0;

        payload->world->intersect(ray, intersections, &count);

        auto hit = nearestHit(intersections, count);

        if (hit.bHit) {
            pixelColor = hit.normal;
        }
        else {
            pixelColor += defaultColor;
        }
    }

    writePixel(payload->pixelBuffer, index * 3, pixelColor / samplesPerPixel);
}

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

void cleanup() {
    gpuErrorCheck(cudaFree(payload->world));

    for (auto i = 0; i < materialCount; i++) {
        deleteObject<<<1, 1>>>(materials[i]);
    }

    for (auto i = 0; i < lightCount; i++) {
        deleteObject<<<1, 1>>>(lights[i]);
    }

    for (auto i = 0; i < objectCount; i++) {
        deleteObject<<<1, 1>>>(objects[i]);
    }

    gpuErrorCheck(cudaDeviceSynchronize());

    for (auto i = 0; i < objectCount; i++) {
        gpuErrorCheck(cudaFree(objects[i]));
    }

    gpuErrorCheck(cudaFree(payload->viewport));
    gpuErrorCheck(cudaFree(payload->pixelBuffer));
}

int32_t size = 0;
std::shared_ptr<ImageData> imageData;

void initialize(int32_t width, int32_t height) {
    // Choose which GPU to run on, change this on a multi-GPU system.
    gpuErrorCheck(cudaSetDevice(0));

    gpuErrorCheck(cudaMallocManaged((void**)&payload, sizeof(Payload)));

    gpuErrorCheck(cudaMallocManaged((void**)&payload->pixelBuffer, width * height * 3 * sizeof(uint8_t)));

    gpuErrorCheck(cudaMallocManaged((void**)&payload->viewport, sizeof(Viewport)));

    payload->viewport->fov = 90.0;
    payload->viewport->scale = std::tan(Math::radians(payload->viewport->fov / 2));

    payload->viewport->imageAspectRatio = static_cast<double>(width) / height;

    payload->viewport->height = 2.0 * payload->viewport->scale;
    payload->viewport->width = payload->viewport->height * payload->viewport->imageAspectRatio;

    gpuErrorCheck(cudaMallocManaged((void**)&payload->camera, sizeof(Camera)));

    payload->camera->init(width, height);
    payload->camera->computeParameters();

    //for (auto i = 0; i < materialCount; i++) {
    //    gpuErrorCheck(cudaMallocManaged((void**)&materials[i], sizeof(Material**)));
    //}

    //createMaterial<<<1, 1>>>(materials[0]);
    //createMaterial<<<1, 1>>>(materials[1]);

    //gpuErrorCheck(cudaDeviceSynchronize());

    for (auto i = 0; i < objectCount; i++) {
        gpuErrorCheck(cudaMallocManaged((void**)&objects[i], sizeof(Shape**)));
    }

    Tuple origins[objectCount];

    origins[0] = point(-1.5, 0.0, -3.0);
    origins[1] = point(1.5, 0.0, -3.0);

    double radiuses[objectCount] = { 1.0, 1.0 };

    createObject<<<1, 1>>>(objects[0], origins, radiuses, materials);
    createObject<<<1, 1>>>(objects[1], origins, radiuses, materials);
    //createObject<<<1, 1>>>(objects[0], point( 1.5, 0.0, -3.0), 1.0, *materials[0]);
    //createObject<<<1, 1>>> (objects[2], point(-4.0, 0.0, -3.0), 1.0);
    //createObject<<<1, 1>>> (objects[3], point( 3.0, 0.0, -3.0), 1.0);

    //for (auto i = 0; i < lightCount; i++) {
    //    gpuErrorCheck(cudaMallocManaged((void**)&lights[i], sizeof(Light**)));
    //}

    //createLight<<<1, 1>>>(lights[0], point(0.0, 1.0, 0.0), Tuple(1.0, 1.0, 1.0));

    gpuErrorCheck(cudaDeviceSynchronize());

    //(*objects[0])->material->ambient = 0.1;
    //(*objects[0])->material->ambient = 0.9;
    //(*objects[0])->material->ambient = 0.2;
    //(*objects[0])->material->shininess = 128.0;
    //(*objects[0])->material->reflective = 0.0;
    //(*objects[0])->material->transparency = 0.0;
    //(*objects[0])->material->refractiveIndex = 1.0;

    gpuErrorCheck(cudaMallocManaged((void**)&payload->world, sizeof(World)));

    for (auto i = 0; i < objectCount; i++) {
        payload->world->addObject(*objects[i]);
    }

    //for (auto i = 0; i < lightCount; i++) {
    //    payload->world->addLight(*lights[i]);
    //}

    size = width * height * 3;
    imageData = std::make_shared<ImageData>();
    imageData->data = new uint8_t[size];
}

//ImageData* launch(int32_t width, int32_t height) {
//    //queryDeviceProperties();
//
//    //Timer timer;
//
//    initialize(width, height);
//
//    dim3 blockSize(32, 32);
//    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
//        (height + blockSize.y - 1) / blockSize.y);
//
//    fillBufferKernel<<<gridSize, blockSize>>>(width, height, payload);
//
//    gpuErrorCheck(cudaDeviceSynchronize());
//
//    //timer.stop();
//
//    writeToPPM("render.ppm", width, height, payload->pixelBuffer);
//
//    imageData->width = width;
//    imageData->height = height;
//    imageData->data = payload->pixelBuffer;
//    imageData->channels = 3;
//    imageData->size = size;
//
//    return imageData.get();
//}

int main() {
    //queryDeviceProperties();

    constexpr int32_t width = 480;
    constexpr int32_t height = 320;

    initialize(width, height);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    Timer timer;

    fillBufferKernel<<<gridSize, blockSize >>>(width, height, payload);

    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop();

    writeToPPM("render.ppm", width, height, payload->pixelBuffer);

    return 0;
}