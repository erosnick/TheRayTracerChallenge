
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Intersection.h"
#include "Timer.h"
#include "Tuple.h"
#include "Constants.h"
#include "Ray.h"
#include "Shape.h"
#include "Sphere.h"
#include "Quad.h"
#include "Utils.h"
#include "Types.h"
#include "Material.h"
#include "Camera.h"
#include "World.h"
#include "kernel.h"
#include "Shading.h"
#include "Light.h"
#include "Pattern.h"
#include "Array.h"

#include "KernelRandom.h"

#include <math.h>

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

struct Viewport {
    double scale;
    double fov;
    double imageAspectRatio;
    double width;
    double height;
};

Payload* payload = nullptr;

Array<Shape**> objects(3);
Array<Light**> lights(1);
Array<Material**> materials(3);
World** world = nullptr;

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

CUDA_HOST_DEVICE void writePixel(uint8_t* pixelBuffer, int32_t index, const Tuple& pixelColor) {
    pixelBuffer[index] = 256 * std::clamp(std::sqrt(pixelColor.x()), 0.0, 0.999);
    pixelBuffer[index + 1] = 256 * std::clamp(std::sqrt(pixelColor.y()), 0.0, 0.999);
    pixelBuffer[index + 2] = 256 * std::clamp(std::sqrt(pixelColor.z()), 0.0, 0.999);
}

CUDA_GLOBAL void createQuad(Shape** object, Material** material, Matrix4 transform) {
    (*object) = new Quad();
    (*object)->setTransformation(transform);
    //(*object)->material = new Material(color(0.0), 0.1, 0.9, 0.9, 128.0, 0.25, 0.0, 1.0);
    (*object)->material = *material;
    (*object)->material->pattern = new CheckerPattern();
    (*object)->material->pattern->transform(scaling(0.25, 1.0, 0.25));
}

template<typename T>
CUDA_GLOBAL void createObject(T** object) {
    (*object) = new T();
}

CUDA_GLOBAL void addObject(World** world, Shape** object) {
    (*world)->addObject(*object);
}

CUDA_GLOBAL void addObjects(World** world, Array<Shape**> objects, int32_t count) {
    //for (auto i = 0; i < count; i++) {
    //    (*world)->addObject(*objects[i]);
    //}
    auto object = objects[0];
    (*world)->addObject(*object);
}

CUDA_GLOBAL void addLight(World** world, Light** light) {
    (*world)->addLight(*light);
}

CUDA_GLOBAL void addLights(World** world, Light** lights[], int32_t count) {
    for (auto i = 0; i < count; i++) {
        (*world)->addLight(*lights[i]);
    }
}

CUDA_GLOBAL void createSphere(Shape** object, Tuple origin, double radius, Material** material, Matrix4 transform) {
    // It is necessary to create object representing a function
    // directly in global memory of the GPU device for virtual
    // functions to work correctly, i.e. virtual function table
    // HAS to be on GPU as well.
    //auto index = threadIdx.x;
    (*object) = new Sphere(origin, radius);
    (*object)->setTransformation(transform);
    (*object)->material = *material;
    //(*object)->material = new Material(color(1.0, 0.0, 0.0), 0.1, 0.9, 0.9, 128.0, 0.25);
}

CUDA_GLOBAL void createLight(Light** light, Tuple inPosition, Tuple inIntensity, Matrix4 transform) {
    // It is necessary to create object representing a function
    // directly in global memory of the GPU device for virtual
    // functions to work correctly, i.e. virtual function table
    // HAS to be on GPU as well.
    (*light) = new Light(inPosition, inIntensity);
    (*light)->transform(transform);
}

CUDA_GLOBAL void createMaterial(Material** material, Tuple color = Color::red, double ambient = 0.1, double diffuse = 0.9, double specular = 0.9, 
                                double shininess = 128.0, double reflective = 0.0, double transparency = 0.0, double refractiveIndex = 1.0) {
    // It is necessary to create object representing a function
    // directly in global memory of the GPU device for virtual
    // functions to work correctly, i.e. virtual function table
    // HAS to be on GPU as well.
    (*material) = new Material(color, ambient, diffuse, specular, shininess, reflective, transparency, refractiveIndex);
}

template<typename T>
CUDA_GLOBAL void deleteObject(T** object) {
    delete (*object);
}

//struct StackLimitTest {
//    double dummy1;
//    double dummy2;
//    double dummy3;
//    double dummy4;
//    double dummy5;
//    double dummy6;
//    double dummy7;
//    double dummy8;
//    double dummy9;
//    double dummy10;
//    double dummy11;
//    double dummy12;
//    double dummy13;
//    double dummy14;
//    double dummy15;
//    double dummy16;
//    double dummy17;
//    double dummy18;
//    double dummy19;
//    double dummy20;
//};

struct StackLimitTest {
    //CUDA_HOST_DEVICE StackLimitTest() {
    //}

    //CUDA_HOST_DEVICE StackLimitTest(double inT, Shape* inShape, const Ray& inRay = Ray())
    //    : t(inT), object(inShape), subObject(nullptr), ray(inRay) {
    //}

    //CUDA_HOST_DEVICE StackLimitTest(bool bInHit, int32_t inCount, double inT, Shape* inSphere)
    //    : bHit(bInHit), count(inCount), t(inT), object(inSphere), subObject(nullptr) {
    //}

    //CUDA_HOST_DEVICE StackLimitTest(bool bInHit, bool bInShading, int32_t inCount, double inT, Shape* inSphere, const Tuple& inPosition, const Tuple& inNormal, const Ray& inRay)
    //    : bHit(bInHit), bShading(bInShading), count(inCount), t(inT), object(inSphere), subObject(nullptr), position(inPosition), normal(inNormal), ray(inRay) {
    //}

    //CUDA_HOST_DEVICE ~StackLimitTest() {}

    bool bHit = false;
    bool bShading = true;
    int32_t count = 0;
    double t = 100000000.0;
    Shape* subObject = nullptr;
    Shape* object = nullptr;
    Tuple position;
    //Tuple normal;
    //Ray ray;
};

CUDA_GLOBAL void rayTracingKernel(int32_t width, int32_t height, Payload* payload) {
    int32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t column = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t index = row * width + column;

    //auto viewport = payload->viewport;

    //printf("%d, %d\n", row, column);

    Tuple defaultColor = Color::skyBlue;
    Tuple pixelColor = defaultColor;

    constexpr int32_t samplesPerPixel = 1;
    constexpr int32_t depth = 5;

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

        // 在kernel里动态分配内存对性能影响很大！！！
        Array<StackLimitTest> intersections(1);
        //Array<Intersection> intersections(1);
        //auto hitInfo = colorAt(payload->world, ray);

        //if (hitInfo.bHit) {
            //pixelColor = hitInfo.surface + computeReflectionAndRefraction(hitInfo, payload->world, depth);
        //}
    }

    writePixel(payload->pixelBuffer, index * 3, (pixelColor ) / samplesPerPixel);
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

int32_t size = 0;
std::shared_ptr<ImageData> imageData;

template<typename T>
CUDA_HOST_DEVICE void foo(T** objects[]) {
    auto object = objects[0];
}

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

    auto viewMatrix = payload->camera->lookAt(60.0, point(0.0, 0.0, 6.0), point(0.0, 0.0, -5.0), vector(0.0, 1.0, 0.0));

    for (auto i = 0; i < materials.size(); i++) {
        gpuErrorCheck(cudaMallocManaged((void**)&materials[i], sizeof(Material**)));
    }

    createMaterial<<<1, 1>>>(materials[0]);
    createMaterial<<<1, 1>>>(materials[1]);
    createMaterial<<<1, 1>>>(materials[2], Color::black, 0.1, 0.9, 0.9, 128.0, 1.0, 1.0, 1.5);
    //createMaterial<<<1, 1>>>(materials[3], Color::red, 0.1, 0.9, 0.9, 128.0, 0.125, 0.0, 1.0);

    for (auto i = 0; i < objects.size(); i++) {
        gpuErrorCheck(cudaMallocManaged((void**)&objects[i], sizeof(Shape**)));
    }

    for (auto i = 0; i < lights.size(); i++) {
        gpuErrorCheck(cudaMallocManaged((void**)&lights[i], sizeof(Light**)));
    }

    gpuErrorCheck(cudaMallocManaged((void**)&world, sizeof(World**)));

    Array<Tuple> origins(3);

    origins[0] = point(-1.5,  0.0, 0.0);
    origins[1] = point( 1.5,  0.0, 0.0);
    origins[2] = point( 0.0, -0.2, 1.5);

    double radiuses[3] = { 1.0, 1.0, 0.8 };

    createSphere<<<1, 1>>>(objects[0], origins[0], radiuses[0], materials[0], viewMatrix);
    createSphere<<<1, 1>>>(objects[1], origins[1], radiuses[1], materials[1], viewMatrix);
    createSphere<<<1, 1>>>(objects[2], origins[2], radiuses[2], materials[2], viewMatrix);

    //auto transformation = viewMatrix * translate(0.0, -1.0, 0.0) * rotateY(Math::pi_2) * scaling(3.0, 1.0, 3.0);
    //createQuad<<<1, 1>>>(objects[3], materials[3], transformation);

    createObject<<<1, 1>>>(world);

    for (auto i = 0; i < objects.size(); i++) {
        addObject<<<1, 1>>>(world, objects[i]);
    }

    createLight<<<1, 1>>>(lights[0], point(0.0, 1.0, 2.0), Tuple(1.0, 1.0, 1.0), viewMatrix);

    addLight<<<1, 1>>>(world, lights[0]);

    gpuErrorCheck(cudaDeviceSynchronize());

    payload->world = *world;

    size = width * height * 3;
    imageData = std::make_shared<ImageData>();
    imageData->data = new uint8_t[size];
}

void cleanup() {
    deleteObject<<<1, 1>>>(world);

    gpuErrorCheck(cudaDeviceSynchronize());

    for (auto i = 0; i < materials.size(); i++) {
        gpuErrorCheck(cudaFree(materials[i]));
    }

    for (auto i = 0; i < lights.size(); i++) {
        gpuErrorCheck(cudaFree(lights[i]));
    }

    for (auto i = 0; i < objects.size(); i++) {
        gpuErrorCheck(cudaFree(objects[i]));
    }

    gpuErrorCheck(cudaFree(payload->viewport));
    gpuErrorCheck(cudaFree(payload->pixelBuffer));
    gpuErrorCheck(cudaFree(payload));
}

//ImageData* launch(int32_t width, int32_t height) {
//    //queryDeviceProperties();
//
//    //Timer timer;
//
//    //initialize(width, height);
//
//    dim3 blockSize(32, 32);
//    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
//        (height + blockSize.y - 1) / blockSize.y);
//
//    //rayTracingKernel<<<gridSize, blockSize>>>(width, height, payload);
//
//    //gpuErrorCheck(cudaDeviceSynchronize());
//
//    //timer.stop();
//
//    //writeToPPM("render.ppm", width, height, payload->pixelBuffer);
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
    gpuErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, sizeof(Intersection) * MAXELEMENTS * 8));

    constexpr int32_t width = 480;
    constexpr int32_t height = 320;

    auto size = sizeof(Intersection);

    initialize(width, height);

#if 1
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    Timer timer;
#if 1
    Camera camera(width, height);

    auto viewMatrix = camera.lookAt(60.0, point(0.0, 0.0, 6.0), point(0.0, 0.0, -5.0), vector(0.0, 1.0, 0.0));

    World* world = new World();

    auto sphere = new Sphere(point(-1.5, 0.0, 0.0));
    sphere->setTransformation(viewMatrix);
    sphere->setMaterial(new Material(Color::red, 0.1, 0.9, 0.9, 128.0, 0.0, 0.0, 1.0));
    world->addObject(sphere);

    sphere = new Sphere(point(1.5, 0.0, 0.0));
    sphere->setTransformation(viewMatrix);
    sphere->setMaterial(new Material(Color::red, 0.1, 0.9, 0.9, 128.0, 0.0, 0.0, 1.0));
    world->addObject(sphere);

    sphere = new Sphere(point(0.0, -0.2, 1.8), 0.8);
    sphere->setTransformation(viewMatrix);
    sphere->setMaterial(new Material(Color::black, 0.1, 0.9, 0.9, 128.0, 1.0, 1.0, 1.5));
    world->addObject(sphere);

    auto quad = new Quad();
    auto transformation = viewMatrix * translate(0.0, -1.0, 0.0) * rotateY(Math::pi_2) * scaling(3.0, 1.0, 3.0);
    quad->setTransformation(transformation);
    quad->material = new Material(color(0.0), 0.1, 0.9, 0.9, 128.0, 0.125, 0.0, 1.0);
    quad->material->pattern = new CheckerPattern();
    quad->material->pattern->transform(scaling(0.25, 1.0, 0.25));
    world->addObject(quad);

    auto light = new Light(point(0.0, 1.0, -2.0), Color::white);
    light->transform(viewMatrix);
    world->addLight(light);

    light = new Light(point(0.0, 1.0, 3.0), Color::white);
    light->transform(viewMatrix);
    world->addLight(light);

    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            auto index = y * width + x;
            auto finalColor = color(0.0, 0.0, 0.0);
            auto dx = (static_cast<double>(x)) / (width - 1);
            auto dy = (static_cast<double>(y)) / (height - 1);

            auto ray = camera.getRay(dx, dy);

            Tuple defaultColor = Color::skyBlue;
            Tuple pixelColor = defaultColor;

            auto hitInfo = colorAt(world, ray);

            if (hitInfo.bHit) {
                pixelColor = hitInfo.surface + computeReflectionAndRefraction(hitInfo, world, 5);
            }

            writePixel(payload->pixelBuffer, index * 3, pixelColor);
        }
    }
#endif

    //rayTracingKernel<<<gridSize, blockSize>>>(width, height, payload);

    //gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop();

    //writeToPPM("render.ppm", width, height, payload->pixelBuffer);
    Utils::writeToPNG("./render.png", width, height, payload->pixelBuffer);
    Utils::openImage(L"./render.png");

    cleanup();
#endif

    return 0;
}