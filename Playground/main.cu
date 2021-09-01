
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA.h"
#include "kernels.h"
#include "../Common/Utils.h"

#include <cstdio>
#include <algorithm>

CUDA_HOST_DEVICE int32_t fibonacci(int32_t n) {
    if (n == 0) {
        return 0;
    }

    if (n == 1) {
        return 1;
    }

    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main()
{
    Utils::reportGPUUsageInfo();

    Utils::queryDeviceProperties();

    constexpr auto width = 800;
    constexpr auto height = 800;

    //juliaKernel(width, height);

    //GPUTiming(addKernel, "aaa", "bbb");

    //waveKernel(width, height, 100);

    //sharedMemoryKernel(width, height);

    rayTracingKernel(width, height);

    return 0;
}
