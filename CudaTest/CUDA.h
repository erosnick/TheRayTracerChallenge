#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__