#include "kernels.h"
#include "Utils.h"
#include "GPUTimer.h"
#include "Canvas.h"

struct cuComplex {
    CUDA_HOST_DEVICE cuComplex(float a, float b) : r(a), i(b) {}

    CUDA_HOST_DEVICE float magnitudeSquared() const { return r * r + i * i; }

    CUDA_HOST_DEVICE cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    CUDA_HOST_DEVICE cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }

    float r;
    float i;
};

CUDA_HOST_DEVICE int32_t julia(int32_t x, int32_t y, int32_t width, int32_t height) {
    constexpr float scale = 1.5f;
    float jx = scale * (float)(width / 2 - x) / (width / 2);
    float jy = scale * (float)(height / 2 - y) / (height / 2);

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

constexpr auto length = 65536000;
CUDA_GLOBAL void addKernel(int32_t* a, int32_t* b, int32_t* c) {
    auto threadId = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (threadId < length) {
        c[threadId] = a[threadId] + b[threadId];
    }
}

void addKernel() {
    int32_t* a = nullptr;
    int32_t* b = nullptr;
    int32_t* c = nullptr;

    gpuErrorCheck(cudaMallocManaged(&a, sizeof(int32_t) * length));
    gpuErrorCheck(cudaMallocManaged(&b, sizeof(int32_t) * length));
    gpuErrorCheck(cudaMallocManaged(&c, sizeof(int32_t) * length));

    for (auto i = 0; i < length; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    auto blockSize = 1024;
    auto gridSize = (length + blockSize - 1) / blockSize;

    addKernel<<<gridSize, blockSize>>>(a, b, c);

    gpuErrorCheck(cudaDeviceSynchronize());
    
    for (auto i = 0; i < length; i++) {
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
        double fx = x - width / 2.0;
        double fy = y - height / 2.0;
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

CUDA_GLOBAL void dotKernel(float* a, float* b, float* c) {
    __shared__ double cache[threadsPerBlock];

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
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;

    gpuErrorCheck(cudaMallocManaged(&a, sizeof(float) * length));
    gpuErrorCheck(cudaMallocManaged(&b, sizeof(float) * length));
    gpuErrorCheck(cudaMallocManaged(&c, sizeof(float) * length));

    for (auto i = 0; i < length; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    auto blockSize = 1024;
    auto gridSize = (length + blockSize - 1) / blockSize;

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

    CUDA_SHARED double shared[16][16];

    // 现在计算这个位置上的值
    const float period = 128.0;

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

constexpr auto INF = 2e10f;
constexpr auto SPHERES = 800;

struct Sphere {
    float3 color;
    float3 origin;
    float radius;

    CUDA_HOST_DEVICE float hit(float ox, float oy, float* n) {
        auto dx = ox - origin.x;
        auto dy = oy - origin.y;
        // 在x-y平面上到球心的距离是否小于等于球体半径
        if ((dx * dx + dy * dy) < radius * radius) {
            // 应用两次勾股定理来求dz
            auto dz = sqrtf(radius * radius - dx * dx - dy * dy);
            // 根据距离计算颜色衰减
            *n = dz / sqrtf(radius * radius);
            return dz + origin.z;
        }

        return -INF;
    }
};

CUDA_CONSTANT Sphere constantSpheres[SPHERES];

CUDA_GLOBAL void rayTracingKernel(Canvas* canvas, Sphere* spheres, int32_t width, int32_t height) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;

    auto index = x + y * blockDim.x * gridDim.x;

    auto ox = (x - width / 2.0);
    auto oy = (y - height / 2.0);

    auto tMax = -INF;

    auto red = 0.0f;
    auto green = 0.0f;
    auto blue = 0.0f;
#if 1
    for (auto& sphere : constantSpheres) {
        auto n = 0.0f;
        auto t = sphere.hit(ox, oy, &n);
        if (t > tMax) {
            red = sphere.color.x * n;
            green = sphere.color.y * n;
            blue = sphere.color.z * n;
        }
    }
#else
    for (auto i = 0; i < SPHERES; i++) {
        auto n = 0.0f;
        auto t = spheres[i].hit(ox, oy, &n);
        if (t > tMax) {
            red = spheres[i].color.x * n;
            green = spheres[i].color.y * n;
            blue = spheres[i].color.z * n;
        }
    }
#endif

    canvas->writePixel(index, red, green, blue);
}

void rayTracingKernel(int32_t width, int32_t height) {
    auto canvas = createCanvas(width, height);

    Sphere* spheres = nullptr;

    gpuErrorCheck(cudaMallocManaged(&spheres, sizeof(Sphere) * SPHERES));

    for (auto i = 0; i < SPHERES; i++) {
        spheres[i].color.x = Utils::randomFloat();
        spheres[i].color.y = Utils::randomFloat();
        spheres[i].color.z = Utils::randomFloat();
        //spheres[i].origin.x = Utils::randomDouble(0.0, 800.0) - 500.0;
        //spheres[i].origin.y = Utils::randomDouble(0.0, 800.0) - 500.0;
        //spheres[i].origin.z = Utils::randomDouble(0.0, 800.0) - 500.0;
        //spheres[i].radius = Utils::randomDouble(0.0, 100.0) + 20.0;
        spheres[i].origin.x = i % 20 * 40.0f - width / 2 + 20.0f;
        spheres[i].origin.y = i / 20 * 40.0f - height / 2 + 20.0f;
        spheres[i].origin.z = 300.0f;
        spheres[i].radius = 20.0f;
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

    gpuErrorCheck(cudaFree(spheres));

    canvas->writeToPNG("render.png");

    // 这样删除还不够
    canvas->uninitialize();
    gpuErrorCheck(cudaFree(canvas));

    Utils::openImage(L"render.png");
}

CUDA_GLOBAL void copyConstKernel(float* iptr, const float* cptr) {
    // 讲threadIdx / blockIdx映射到像素位置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int index = x + y * blockDim.x * gridDim.x;

    if (cptr[index] != 0) {
        iptr[index] = cptr[index];
    }
}