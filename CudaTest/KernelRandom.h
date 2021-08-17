#pragma once

#include "CUDA.h"
#include <curand_kernel.h>
#include <curand_mrg32k3a.h>
#include <curand_mtgp32_host.h>
#include "device_launch_parameters.h"

//CUDA_GLOBAL void initializeGenerator(curandStateXORWOW_t* states, unsigned long long seed, size_t size) {
//    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
//
//    if (index >= size) {
//        return;
//    }
//
//    curand_init(seed, index, 0, &states[index]);
//}
//
//CUDA_GLOBAL void useGenerator(curandStateMtgp32_t* states, double* data, size_t size) {
//    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
//
//    data[index] = curand_uniform_double(&states[index]);
//}