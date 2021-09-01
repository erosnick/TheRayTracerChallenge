
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Common/CUDA.h"
#include "../Common/Utils.h"
#include "../Common/Canvas.h"
#include "../Common/Array.h"
#include "Timer.h"
#include "GPUTimer.h"
#include "Camera.h"
#include "Object.h"
#include "Sphere.h"

#include <algorithm>

constexpr auto SPHERES = 2;

CUDA_CONSTANT Sphere constantSpheres[SPHERES];

inline CUDA_HOST_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) {
    auto closestSoFar = tMax;
    HitResult tempHitResult;
    bool bHitAnything = false;
    for (auto& sphere : constantSpheres) {
        if (sphere.hit(ray, tMin, closestSoFar, tempHitResult)) {
            closestSoFar = tempHitResult.t;
            bHitAnything = true;
            hitResult = tempHitResult;
        }
    }

    return bHitAnything;
}

inline CUDA_DEVICE Float3 rayColor(const Ray& ray, uint32_t* seed0, uint32_t* seed1, int32_t depth) {
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0) {
        return make_float3(1.0f, 0.0f, 0.0);
    }

    HitResult hitResult;
    if (hit(ray, Math::epsilon, Math::infinity, hitResult)) {
        // Find a random point S in unit sphere tangent to hit point
        auto point = Utils::randomInUnitSphere(seed0, seed1);
        auto target = hitResult.position + hitResult.normal + point;
        // Cast a new ray towards to S
        Ray nextRay(hitResult.position, target - hitResult.position);
        return (0.5f * rayColor(nextRay, seed0, seed1, depth - 1));
    }

    auto unitDirection = normalize(ray.direction);
    auto t = 0.5f * (unitDirection.y + 1.0f);
    return lerp(make_float3(1.0f, 1.0f, 1.0f), make_float3(0.5f, 0.7f, 1.0f), t);
}

//inline CUDA_DEVICE Float3 rayColor(Ray& ray, uint32_t* seed0, uint32_t* seed1, int32_t depth) {
//    // If we've exceeded the ray bounce limit, no more light is gathered.
//    if (depth <= 0) {
//        return make_float3(0.0f, 0.0f, 0.0);
//    }
//
//    HitResult hitResult;
//    if (hit(ray, 0.0f, Math::infinity, hitResult)) {
//        // Find a random point S in unit sphere tangent to hit point
//        auto point = Utils::randomInUnitSphere(seed0, seed1);
//        auto target = hitResult.position + hitResult.normal + point;
//        // Cast a new ray towards to S
//        ray = Ray(hitResult.position, target - hitResult.position);
//        //return (0.5f * rayColor(nextRay, seed0, seed1, depth - 1));
//        depth--;
//        return make_float3(0.0f, 0.0f, 0.0);
//    }
//
//    auto unitDirection = normalize(ray.direction);
//    auto t = 0.5f * (unitDirection.y + 1.0f);
//    return lerp(make_float3(1.0f, 1.0f, 1.0f), make_float3(0.5f, 0.7f, 1.0f), t);
//}

CUDA_GLOBAL void pathTracingKernel(Canvas* canvas, Camera camera, Array<Sphere>* spheres) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;

    auto width = camera.getImageWidth();
    auto height = camera.getImageHeight();
    constexpr auto samplesPerPixel = 1;
    constexpr auto maxDepth = 5;
    auto depth = maxDepth;
    constexpr auto bounces = 1;
    uint32_t seed0 = x;
    uint32_t seed1 = y;

    auto index = y * width + x;

    if (index < width * height) {
        auto color = make_float3(0.0f, 0.0f, 0.0f);
        for (auto i = 0; i < samplesPerPixel; i++) {

            Float rx = Utils::getrandom(&seed0, &seed1);
            Float ry = Utils::getrandom(&seed0, &seed1);

            //Float rx = Utils::random(x);
            //Float ry = Utils::random(x);

            auto dx = Float(x + rx) / (width - 1);
            auto dy = Float(y + ry) / (height - 1);

            Ray ray = camera.getRay(dx, dy);

            //for (auto j = 0; j < bounces; j++) {
                //color += rayColor(ray, &seed0, &seed1, depth);
                //color += rayColor(ray, &seed0, &seed1, depth);
            //}
            //printf("%d, %d\n", seed0, seed1);
            //printf("%f, %f, %f\n", position.x, position.y, position.z);

            color += rayColor(ray, &seed0, &seed1, maxDepth);
        }
        
        canvas->writePixel(index, color / samplesPerPixel);
    }
}

 void pathTracingGPU(int32_t width, int32_t height) {
     Camera camera(width, height);
     Canvas* canvas = nullptr;
     Array<Sphere>* spheres = nullptr;

     gpuErrorCheck(cudaMallocManaged(&spheres, sizeof(Array<Sphere>)));
     spheres->initialize();

     //auto startX = -9.5f;
     //auto startY = -9.5f;
     //for (auto j = 0; j < 20; j++) {
     //    for (auto i = 0; i < 20; i++) {
     //        Float3 color;
     //        color.x = Utils::randomFloat();
     //        color.y = Utils::randomFloat();
     //        color.z = Utils::randomFloat();
     //        Sphere sphere;
     //        sphere.center = make_float3(startX + i, startY + j, -10.0f);
     //        sphere.color = color;
     //        sphere.radius = 0.5f;
     //        spheres->add(sphere);
     //    }
     //}
     gpuErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

     Sphere testSpheres[SPHERES];

     Sphere sphere;
     sphere.center = make_float3(0.0f, 0.0f, -1.0f);
     sphere.radius = 0.5f;

     testSpheres[0] = sphere;
     spheres->add(sphere);

     sphere.center = make_float3(0.0f, -100.5f, -1.0f);
     sphere.radius = 100.0f;

     testSpheres[1] = sphere;
     spheres->add(sphere);

     gpuErrorCheck(cudaMemcpyToSymbol(constantSpheres, testSpheres, sizeof(Sphere) * SPHERES));

     //gpuErrorCheck(cudaMemcpyToSymbol(constantSpheres, spheres->data(), sizeof(Sphere) * SPHERES));

     gpuErrorCheck(cudaMallocManaged(&canvas, sizeof(Canvas*)));
     canvas->initialize(width, height);

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

     GPUTimer timer("Rendering start...");

     pathTracingKernel<<<gridSize, blockSize>>>(canvas, camera, spheres);
     gpuErrorCheck(cudaDeviceSynchronize());

     timer.stop("Rendering elapsed time");

     canvas->writeToPNG("render.png");
     Utils::openImage(L"render.png");

     gpuErrorCheck(cudaFree(spheres));
     
     canvas->uninitialize();
     gpuErrorCheck(cudaFree(canvas));
}

 //void pathTracingCPU(int32_t width, int32_t height) {
 //    Camera camera(width, height);
 //    Canvas canvas(width, height);

 //    Timer timer;
 //    for (auto y = height - 1; y >= 0; y--) {
 //        for (auto x = 0; x < width; x++) {
 //            auto index = y * width + x;

 //            auto dx = Float(x) / (width - 1);
 //            auto dy = Float(y) / (height - 1);

 //            Ray ray = camera.getRay(dx, dy);

 //            auto color = rayColor(ray);
 //            canvas.writePixel(index, color.x, color.y, color.z);
 //        }
 //    }
 //    timer.stop();

 //    canvas.writeToPNG("render.png");
 //    Utils::openImage(L"render.png");
 //}

int main()
{
    //pathTracingCPU(1280, 720);
    pathTracingGPU(1280, 720);

    return 0;
}
