#include "kernels.h"
#include "Utils.h"
#include "Timer.h"
#include "GPUTimer.h"
#include "Canvas.h"

struct cuComplex {
    CUDA_HOST_DEVICE cuComplex(Float a, Float b) : r(a), i(b) {}

    CUDA_HOST_DEVICE Float magnitudeSquared() const { return r * r + i * i; }

    CUDA_HOST_DEVICE cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    CUDA_HOST_DEVICE cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }

    Float r;
    Float i;
};

CUDA_HOST_DEVICE int32_t julia(int32_t x, int32_t y, int32_t width, int32_t height) {
    constexpr Float scale = 1.5f;
    Float jx = scale * (Float)(width / 2 - x) / (width / 2);
    Float jy = scale * (Float)(height / 2 - y) / (height / 2);

    cuComplex c(-0.8f, 0.156f);
    cuComplex a(jx, jy);

    for (auto i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitudeSquared() > 1000.0f) {
            return 0;
        }
    }

    return 1;
}

CUDA_GLOBAL void juliaKernel(Canvas* canvas, int32_t width, int32_t height) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;

    auto index = y * width + x;

    if (index < (width * height)) {
        auto juliaValue = julia(x, y, width, height);
        canvas->writePixel(index, juliaValue, 0.0, 0.0);
    }
}

void juliaKernel(int32_t width, int32_t height) {
    Utils::reportGPUUsageInfo();

    //auto minGridSize = 0;
    //auto blockSize = 0;
    //auto gridSize = 0;

    //auto size = width * height;

    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, size);

    //// Round up according to array size
    //gridSize = (size + blockSize - 1) / blockSize;

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    GPUTimer timer("Rendering start...");

    auto canvas = createCanvas(width, height);

    juliaKernel<<<gridSize, blockSize>>>(canvas, width, height);

    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop("Rendering elapsed time");

    canvas->writeToPNG("render.png");

    // 这样删除还不够
    canvas->uninitialize();
    gpuErrorCheck(cudaFree(canvas));
    
    Utils::openImage(L"render.png");

    Utils::reportGPUUsageInfo();
}

constexpr auto arrayLength = 65536000;
CUDA_GLOBAL void addKernel(int32_t* a, int32_t* b, int32_t* c) {
    auto threadId = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (threadId < arrayLength) {
        c[threadId] = a[threadId] + b[threadId];
    }
}

void addKernel() {
    int32_t* a = nullptr;
    int32_t* b = nullptr;
    int32_t* c = nullptr;

    gpuErrorCheck(cudaMallocManaged(&a, sizeof(int32_t) * arrayLength));
    gpuErrorCheck(cudaMallocManaged(&b, sizeof(int32_t) * arrayLength));
    gpuErrorCheck(cudaMallocManaged(&c, sizeof(int32_t) * arrayLength));

    for (auto i = 0; i < arrayLength; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    auto blockSize = 1024;
    auto gridSize = (arrayLength + blockSize - 1) / blockSize;

    addKernel<<<gridSize, blockSize>>>(a, b, c);

    gpuErrorCheck(cudaDeviceSynchronize());
    
    for (auto i = 0; i < arrayLength; i++) {
        //printf("%d\n", c[i]);
    }

    gpuErrorCheck(cudaFree(c));
    gpuErrorCheck(cudaFree(b));
    gpuErrorCheck(cudaFree(a));
}

CUDA_GLOBAL void waveKernel(Canvas* canvas, int32_t width, int32_t height, int32_t ticks) {
    int32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    auto index = x + y * blockDim.x * gridDim.x;

    if (index < (width * height)) {
        Float fx = x - width / 2.0f;
        Float fy = y - height / 2.0;
        auto d = sqrt(fx * fx + fy * fy);
        auto grey = (128.0 + 127.0 * cos(d / 10.0 - ticks / 7.0) / (d / 10.0 + 1.0));

        auto r = grey * 0.003921;
        canvas->writePixel(index, r, r, r);
    }
}

void waveKernel(int32_t width, int32_t height, int32_t ticks) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    auto canvas = createCanvas(width, height);

    waveKernel<<<gridSize, blockSize>>>(canvas, width, height, ticks);

    gpuErrorCheck(cudaDeviceSynchronize());

    canvas->writeToPNG("render.png");

    // 这样删除还不够
    canvas->uninitialize();
    gpuErrorCheck(cudaFree(canvas));

    Utils::openImage(L"render.png");
}

constexpr auto N = 33 * 1024;
constexpr auto threadsPerBlock = 256;

CUDA_GLOBAL void dotKernel(Float* a, Float* b, Float* c) {
    __shared__ Float cache[threadsPerBlock];

    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;
    auto cacheIndex = threadIdx.x;

    auto temp = 0.0;
    while (threadId < N) {
        temp += a[threadId] * b[threadId];
        threadId += blockDim.x * gridDim.x;
    }

    // 设置cache中相应位置上的值
    cache[cacheIndex] = temp;

    // 对线程块中的线程进行同步
    __syncthreads();

    // 对于归约运算来说，以下代码要求threadPerBlock必须时2的指数
    auto i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

void dotKernel(int32_t width, int32_t height) {
    Float* a = nullptr;
    Float* b = nullptr;
    Float* c = nullptr;

    gpuErrorCheck(cudaMallocManaged(&a, sizeof(Float) * arrayLength));
    gpuErrorCheck(cudaMallocManaged(&b, sizeof(Float) * arrayLength));
    gpuErrorCheck(cudaMallocManaged(&c, sizeof(Float) * arrayLength));

    for (auto i = 0; i < arrayLength; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    auto blockSize = 1024;
    auto gridSize = (arrayLength + blockSize - 1) / blockSize;

    dotKernel<<<gridSize, blockSize>>>(a, b, c);

    gpuErrorCheck(cudaFree(c));
    gpuErrorCheck(cudaFree(b));
    gpuErrorCheck(cudaFree(a));
}

#include "Buffer.h"
constexpr auto PI = 3.1415926535897932f;

CUDA_GLOBAL void sharedMemoryKernel(Canvas* canvas) {
    // 将threadIdx / blockIdx映射到像素位置
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;

    auto index = y * blockDim.x * gridDim.x + x;

    CUDA_SHARED Float shared[16][16];

    // 现在计算这个位置上的值
    const Float period = 128.0;

    shared[threadIdx.x][threadIdx.y] = (sinf(x * 2.0 * PI / period) + 1.0) *
                                       (sinf(y * 2.0 * PI / period) + 1.0) / 4.0;

    __syncthreads();

    auto r = shared[15 - threadIdx.x][15 - threadIdx.y];
    canvas->writePixel(index, 0.0, r, 0.0);
}

void sharedMemoryKernel(int32_t width, int32_t height) {
    dim3 gridSize(width / 16, height / 16);
    dim3 blockSize(16, 16);

    Canvas* canvas = createCanvas(width, height);

    sharedMemoryKernel<<<gridSize, blockSize>>>(canvas);

    gpuErrorCheck(cudaDeviceSynchronize());

    canvas->writeToPNG("render.png");

    // 这样删除还不够
    canvas->uninitialize();
    gpuErrorCheck(cudaFree(canvas));

    Utils::openImage(L"render.png");
}

#include "Constants.h"

constexpr Float INF = 2e10f;
constexpr auto SPHERES = 400;

struct Ray {
    Float3 origin; // ray origin
    Float3 direction;  // ray direction 
    CUDA_HOST_DEVICE Ray(Float3 inOrigin, Float3 inDirection) : origin(inOrigin), direction(inDirection) {}
};

struct Sphere {
    Float3 color;
    Float3 origin;
    Float radius;

    CUDA_HOST_DEVICE Float hit(Float ox, Float oy, Float* n) {
        auto dx = ox - origin.x;
        auto dy = oy - origin.y;
        // 在x-y平面上到球心的距离是否小于等于球体半径
        if ((dx * dx + dy * dy) < radius * radius) {
            // 应用两次勾股定理来求dz
            auto dz = sqrt(radius * radius - dx * dx - dy * dy);
            // 根据距离计算颜色衰减
            *n = dz / sqrt(radius * radius);

            return dz + origin.z;
        }

        return -INF;
    }

    CUDA_HOST_DEVICE Float hit(const Ray& ray, Float* n) {
        Float3 oc = (ray.origin - origin);
        Float a = dot(ray.direction, ray.direction);
        Float b = 2.0f * dot(ray.direction, oc);
        Float c = dot(oc, oc) - radius * radius;
        Float t = -INF;
        Float epsilon = FLT_EPSILON;
        auto discriminant = b * b - 4 * a * c;

        if (discriminant < 0.0f) {
            return -INF;
        }

        auto inverse2a = 1.0f / 2 * a;

        auto d = sqrt(discriminant);

        Float t1 = (-b - d) * inverse2a;
        Float t2 = (-b + d) * inverse2a;
        (t = t1) > epsilon ? t : ((t = t2) > epsilon ? t : -INF);

        Float3 position = ray.origin + t * ray.direction;

        *n = (abs(origin.z) - abs(position.z)) / radius;

        return t;
    }
};

CUDA_CONSTANT Sphere constantSpheres[SPHERES];

CUDA_GLOBAL void rayTracingKernel(Canvas* canvas, Sphere* spheres, int32_t width, int32_t height) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;

    auto index = x + y * blockDim.x * gridDim.x;

    if (index < width * height) {
        auto scale = tan(45.0f * PI / 180.0f);
        auto imageAspectRatio = Float(width) / height;
        Float dx = (2.0f * ((x + 0.5f) / width) - 1.0f) * scale * imageAspectRatio;
        Float dy = (1.0f - 2.0 * ((y + 0.5f) / height)) * scale;

        auto origin = make_float3(0.0f, 0.0f, 0.0f);

        Ray ray(origin, normalize(make_float3(dx, dy, -1.0f)));

        auto tMax = -INF;

        //Float3 color = make_float3(0.5f, 0.7f, 1.0f);
        Float3 color = make_float3(0.0f, 0.0f, 0.0f);
#if 1
        for (auto& sphere : constantSpheres) {
            Float n = 1.0f;
            //auto t = sphere.hit(ox, oy, &n);
            auto t = sphere.hit(ray, &n);
            if (t > tMax) {
                color.x = sphere.color.x * n;
                color.y = sphere.color.y * n;
                color.z = sphere.color.z * n;
            }
        }

#else
        for (auto i = 0; i < SPHERES; i++) {
            auto n = 0.0f;
            auto t = spheres[i].hit(ox, oy, &n);
            if (t > tMax) {
                color.x() = spheres[i].color.x * n;
                color.y() = spheres[i].color.y * n;
                color.z() = spheres[i].color.z * n;
            }
        }
#endif
        canvas->writePixel(index, color.x, color.y, color.z);
    }
}

void rayTracing(Canvas* canvas, Sphere* spheres, int32_t width, int32_t height) {
    Float scale = tan(45.0f * PI / 180.0f);
    auto imageAspectRatio = Float(width) / height;
    auto origin = make_float3(0.0f, 0.0f, 0.0f);
    Float tMax = -INF;

    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            auto index = y * width + x;

            Float dx = (2.0f * ((x + 0.5f) / width) - 1.0f) * scale * imageAspectRatio;
            Float dy = (1.0f - 2.0 * ((y + 0.5f) / height)) * scale;

            Ray ray(origin, normalize(make_float3(dx, dy, -1.0f)));

            Float3 color = make_float3(0.0f, 0.0f, 0.0f);

            for (auto i = 0; i < SPHERES; i++) {
                Float n = 0.0f;
                Float t = spheres[i].hit(ray, &n);
                if (t > tMax) {
                    color.x = spheres[i].color.x * n;
                    color.y = spheres[i].color.y * n;
                    color.z = spheres[i].color.z * n;
                }
            }

            canvas->writePixel(index, color.x, color.y, color.z);
        }
    }
}

void rayTracingKernel(int32_t width, int32_t height) {
    auto canvas = createCanvas(width, height);

    Sphere* spheres = nullptr;

    gpuErrorCheck(cudaMallocManaged(&spheres, sizeof(Sphere) * SPHERES));

    for (auto y = 0; y < 20; y++) {
        for (auto x = 0; x < 20; x++) {
            auto i = y * 20 + x;
            spheres[i].color.x = Utils::randomFloat();
            spheres[i].color.y = Utils::randomFloat();
            spheres[i].color.z = Utils::randomFloat();
            //spheres[i].origin.x = Utils::randomFloat(0.0, 800.0) - 500.0;
            //spheres[i].origin.y = Utils::randomFloat(0.0, 800.0) - 500.0;
            //spheres[i].origin.z = Utils::randomFloat(0.0, 800.0) - 500.0;
            //spheres[i].radius = Utils::randomFloat(0.0, 100.0) + 20.0;
            spheres[i].origin.x = (Float(x - 10 + 0.5f) / 10.0f) * 20.0f;
            spheres[i].origin.y = (Float(y - 10 + 0.5f) / 10.0f) * 20.0f;
            //spheres[i].color.x = 1.0f;
            //spheres[i].color.y = 0.0f;
            //spheres[i].color.z = 0.0f;
            //spheres[i].origin.x = -19.0f;
            //spheres[i].origin.y = 0.0f;
            spheres[i].origin.z = -20.0f;
            spheres[i].radius = 1.0f;
        }
    }

    auto size = sizeof(Sphere);

    gpuErrorCheck(cudaMemcpyToSymbol(constantSpheres, spheres, sizeof(Sphere) * SPHERES));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    GPUTimer timer("Rendering start...");

    rayTracingKernel<<<gridSize, blockSize>>>(canvas, spheres, width, height);

    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop("Rendering elapsed time");

    //Timer timer;
    //rayTracing(canvas, spheres, width, height);
    //timer.stop();

    gpuErrorCheck(cudaFree(spheres));

    canvas->writeToPNG("render.png");

    // 这样删除还不够
    canvas->uninitialize();
    gpuErrorCheck(cudaFree(canvas));

    Utils::openImage(L"render.png");
}

CUDA_GLOBAL void copyConstKernel(Float* iptr, const Float* cptr) {
    // 讲threadIdx / blockIdx映射到像素位置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int index = x + y * blockDim.x * gridDim.x;

    if (cptr[index] != 0) {
        iptr[index] = cptr[index];
    }
}