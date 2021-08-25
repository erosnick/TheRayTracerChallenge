
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
#include "Cube.h"
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
World** world = nullptr;

Array<Shape**> objects(4);
Array<Light**> lights(2);
Array<Material**> materials(4);
Shape*** objectPool = nullptr;

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

CUDA_GLOBAL void updateObjectsKernel(Array<Shape**> objects, Matrix4 transformation) {
    for (auto i = 0; i < 4; i++) {
        (*objects[i])->setTransformation(transformation);
    }
}

void updateObjects(const Array<Shape**>& objects, const Matrix4& transformation) {
    updateObjectsKernel<<<1, 1>>>(objects, transformation);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void updateObjectsKernel(World* world, Matrix4 transformation) {
    for (auto i = 0; i < world->objectCount(); i++) {
        world->getObject(i)->transform(transformation);
        world->getObject(i)->updateTransformation();
        //world->getObject(i)->transformNormal(transformation);
        if (world->getObject(i)->material->pattern) {
            //world->getObject(i)->material->pattern->transform(transformation);
        }
    }

    for (auto i = 0; i < world->ligthCount(); i++) {
        world->getLight(i)->transform(transformation);
    }
}

void updateObjects(World* world, const Matrix4& transformation) {
    updateObjectsKernel<<<1, 1>>>(world, transformation);
    gpuErrorCheck(cudaDeviceSynchronize());
}

template<typename T>
CUDA_GLOBAL void createObject(T** object) {
    (*object) = new T();
}

CUDA_GLOBAL void createCube(World** world, Shape** object, Material** material, Matrix4 transformation) {
    (*object) = new Cube();
    (*object)->transform(transformation);
    (*object)->transformNormal(transformation);
    (*object)->material = *material;
    (*world)->addObject(*object);
}

CUDA_GLOBAL void createQuad(World** world, Shape** object, Material** material, Matrix4 transformation) {
    (*object) = new Quad();
    (*object)->setTransformation(transformation);
    (*object)->transformNormal(transformation);
    //(*object)->material = new Material(color(0.0), 0.1, 0.9, 0.9, 128.0, 0.25, 0.0, 1.0);
    (*object)->material = *material;
    (*object)->material->pattern = new CheckerPattern();
    //(*object)->material->pattern->setTransformation(scaling(0.25, 1.0, 0.25));
    (*object)->material->pattern->transform(scaling(0.25, 1.0, 0.25));
    (*world)->addObject(*object);
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

CUDA_GLOBAL void createSphere(World**world, Shape** object, Tuple origin, double radius, Material** material, Matrix4 transformation) {
    // It is necessary to create object representing a function
    // directly in global memory of the GPU device for virtual
    // functions to work correctly, i.e. virtual function table
    // HAS to be on GPU as well.
    //auto index = threadIdx.x;
    (*object) = new Sphere(origin, radius);
    //(*object)->setTransformation(transformation);
    (*object)->transform(transformation);
    (*object)->material = *material;
    //(*object)->material = new Material(color(1.0, 0.0, 0.0), 0.1, 0.9, 0.9, 128.0, 0.25);
    (*world)->addObject(*object);
}

CUDA_GLOBAL void createLight(World** world, Light** light, Tuple inPosition, Tuple inIntensity, Matrix4 transformation) {
    // It is necessary to create object representing a function
    // directly in global memory of the GPU device for virtual
    // functions to work correctly, i.e. virtual function table
    // HAS to be on GPU as well.
    (*light) = new Light(inPosition, inIntensity);
    (*light)->transform(transformation);
    (*world)->addLight(*light);
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

CUDA_GLOBAL void rayTracingKernel(int32_t width, int32_t height, Payload* payload) {
    int32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    int32_t column = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t index = row * width + column;

    //auto viewport = payload->viewport;

    //printf("%d, %d\n", threadIdx.x, threadIdx.y);
    Tuple pixelColor = Color::black;

    constexpr int32_t samplesPerPixel = 1;
    constexpr int32_t depth = 3;

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
        //auto hitInfo = colorAt(payload->world, ray);

        //if (hitInfo.bHit) {
        //    pixelColor = hitInfo.surface + computeReflectionAndRefraction(hitInfo, payload->world, depth);
        //}

        pixelColor += colorAt(payload->world, ray, depth);

        //Array<Intersection> intersections;
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

    // Intersection = 32bits, MAXELEMENTS = 8, 32 * 8 * 16 = 4096
    gpuErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, sizeof(Intersection) * MAXELEMENTS * 1024));

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
        gpuErrorCheck(cudaMallocManaged(&materials[i], sizeof(Material**)));
    }

    gpuErrorCheck(cudaMallocManaged(&objectPool, sizeof(Shape**) * 4));

    for (auto i = 0; i < objects.size(); i++) {
        objects[i] = (Shape**)objectPool + i;
    }

    //for (auto i = 0; i < objects.size(); i++) {
    //    gpuErrorCheck(cudaMallocManaged((void**)&objects[i], sizeof(Shape**)));
    //}

    for (auto i = 0; i < lights.size(); i++) {
        gpuErrorCheck(cudaMallocManaged(&lights[i], sizeof(Light**)));
    }

    gpuErrorCheck(cudaMallocManaged(&world, sizeof(World**)));

    createMaterial<<<1, 1>>>(materials[0], Color::black, 0.1, 0.9, 0.9, 128.0, 0.125, 0.0, 1.0);
    createMaterial<<<1, 1>>>(materials[1]);
    createMaterial<<<1, 1>>>(materials[2], Color::black, 0.1, 0.9, 0.9, 128.0, 1.0, 1.0, 1.5);
    createMaterial<<<1, 1>>>(materials[3], Color::red, 0.1, 0.9, 0.9, 128.0, 0.125, 0.0, 1.0);
    //createMaterial<<<1, 1>>>(materials[4], Color::limeGreen, 0.1, 0.9, 0.9, 128.0, 0.125, 0.0, 1.0);

    Array<Tuple> origins(3);

    origins[0] = point(-1.5,  0.0, 0.0);
    origins[1] = point( 1.5,  0.0, 0.0);
    origins[2] = point( 0.0, -0.2, 1.5);

    double radiuses[3] = { 1.0, 1.0, 0.8 };

    createObject<<<1, 1>>>(world);
    gpuErrorCheck(cudaDeviceSynchronize());

    createSphere<<<1, 1>>>(world, objects[0], origins[0], radiuses[0], materials[0], viewMatrix);
    createSphere<<<1, 1>>>(world, objects[1], origins[1], radiuses[1], materials[1], viewMatrix);
    createSphere<<<1, 1>>>(world, objects[2], origins[2], radiuses[2], materials[2], viewMatrix);

    auto transformation = translate(1.5, 0.0, 1.0) * scaling(0.25, 0.25, 0.25);
    //createCube<<<1, 1>>>(world, objects[4], materials[4], transformation);

    transformation = translate(0.0, -1.0, 0.0) * scaling(3.0, 1.0, 3.0);
    createQuad<<<1, 1>>>(world, objects[3], materials[3], transformation);

    //for (auto i = 0; i < objects.size(); i++) {
    //    addObject<<<1, 1>>>(world, objects[i]);
    //}

    createLight<<<1, 1>>>(world, lights[0], point(0.0, 1.0, 2.0), Tuple(1.0, 1.0, 1.0), viewMatrix);
    createLight<<<1, 1>>>(world, lights[1], point(0.0, 1.0, -2.0), Tuple(1.0, 1.0, 1.0), viewMatrix);

    //addLight<<<1, 1>>>(world, lights[0]);
    //addLight<<<1, 1>>>(world, lights[1]);

    gpuErrorCheck(cudaDeviceSynchronize());

    payload->world = *world;

    size = width * height * 3;
    imageData = std::make_shared<ImageData>();
    imageData->data = new uint8_t[size];
}

void cleanup() {
    for (auto i = 0; i < materials.size(); i++) {
        gpuErrorCheck(cudaFree(materials[i]));
    }

    for (auto i = 0; i < lights.size(); i++) {
        gpuErrorCheck(cudaFree(lights[i]));
    }

    gpuErrorCheck(cudaFree(objectPool));

    deleteObject<<<1, 1>>>(world);

    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaFree(payload->viewport));
    gpuErrorCheck(cudaFree(payload->pixelBuffer));
    gpuErrorCheck(cudaFree(payload));
}

#ifdef GPU_RELEASE
ImageData* launch(int32_t width, int32_t height) {
    //queryDeviceProperties();

    //Timer timer;

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    rayTracingKernel<<<gridSize, blockSize>>>(width, height, payload);

    gpuErrorCheck(cudaDeviceSynchronize());

    //timer.stop();

    //writeToPPM("render.ppm", width, height, payload->pixelBuffer);

    imageData->width = width;
    imageData->height = height;
    imageData->data = payload->pixelBuffer;
    imageData->channels = 3;
    imageData->size = size;

    return imageData.get();
}

#else

int main() {
    //queryDeviceProperties();

    constexpr int32_t width = 480;
    constexpr int32_t height = 320;

    auto size = width * height;

#if 0
    initialize(width, height);

    //int32_t minGridSize = 0;
    //int32_t blockSize = 0;
    //int32_t gridSize = 0;

    //float time = 0.0f;
    //cudaEvent_t start;
    //cudaEvent_t stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start, 0);

    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rayTracingKernel, 0, size);

    //// Round up according to array size
    //gridSize = (size + blockSize - 1) / blockSize;

    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&time, start, stop);
    //printf("Occupancy calculator elapsed time: %3.3f ms \n", time);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    Timer timer;

    rayTracingKernel<<<gridSize, blockSize>>>(width, height, payload);

    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop();

    //writeToPPM("render.ppm", width, height, payload->pixelBuffer);
    Utils::writeToPNG("./render.png", width, height, payload->pixelBuffer);
    Utils::openImage(L"./render.png");

    cleanup();

#else
    Camera camera(width, height);

    auto viewMatrix = camera.lookAt(60.0, point(3.0, 3.0, 6.0), point(0.0, 0.0, -5.0), vector(0.0, 1.0, 0.0));

    World* world = new World();

    auto sphere = new Sphere(point(-1.5, 0.0, 0.0));
    //sphere->setTransformation(viewMatrix);
    sphere->setMaterial(new Material(Color::red, 0.1, 0.9, 0.9, 128.0, 0.0, 0.0, 1.0));

    //world->addObject(sphere);

    sphere = new Sphere(point(1.5, 0.0, 0.0));
    //sphere->setTransformation(viewMatrix);
    sphere->setMaterial(new Material(Color::red, 0.1, 0.9, 0.9, 128.0, 0.0, 0.0, 1.0));

    //world->addObject(sphere);

    sphere = new Sphere(point(0.0, -0.2, 1.8), 0.8);
    //sphere->setTransformation(viewMatrix);
    sphere->setMaterial(new Material(Color::black, 0.1, 0.9, 0.9, 128.0, 1.0, 1.0, 1.5));

    //world->addObject(sphere);

    auto quad = new Quad();
    auto transformation = translate(0.0, -1.0, 0.0) * scaling(3.0, 1.0, 3.0);
    quad->setTransformation(transformation);
    quad->material = new Material(color(0.0), 0.1, 0.9, 0.9, 128.0, 0.125, 0.0, 1.0);
    quad->material->pattern = new CheckerPattern();
    quad->material->pattern->transform(scaling(0.25, 1.0, 0.25));

    world->addObject(quad);

    auto cube = new Cube();
    //cube->transform(translate(2.0, -0.28, 0.0) * scaling(0.125, 0.125, 0.125));
    cube->transform(scaling(0.125, 0.125, 0.125));
    cube->setMaterial(new Material());

    world->addObject(cube);

    auto light = new Light(point(0.0, 1.0, -3.0), Color::white);
    light->transform(viewMatrix);

    //world->addLight(light);

    light = new Light(point(0.0, 3.0, 3.0), Color::white);
    light->transform(viewMatrix);

    world->addLight(light);

    for (auto i = 0; i < world->objectCount(); i++) {
        world->getObject(i)->transform(viewMatrix);
        world->getObject(i)->updateTransformation();
    }

    auto depth = 1;

    auto pixelBuffer = new uint8_t[width * height * 3];
    Timer timer;
    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            auto index = y * width + x;
            auto dx = (static_cast<double>(x)) / (width - 1);
            auto dy = (static_cast<double>(y)) / (height - 1);

            auto ray = camera.getRay(dx, dy);

            Tuple defaultColor = Color::black;
            Tuple pixelColor = defaultColor;

            if (x == 234 && y == 73) {
                auto a = 0;
            }

            //auto hitInfo = colorAt(world, ray);

            //if (hitInfo.bHit) {
            //    pixelColor = hitInfo.surface + computeReflectionAndRefraction(hitInfo, world, depth);
            //}
            pixelColor += colorAt(world, ray, depth);

            writePixel(pixelBuffer, index * 3, pixelColor);
        }
    }
    timer.stop();
    Utils::writeToPNG("./render.png", width, height, pixelBuffer);
    Utils::openImage(L"./render.png");
    delete[] pixelBuffer;
#endif

    return 0;
}

#endif