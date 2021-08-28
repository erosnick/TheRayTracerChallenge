#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA.h"

#include <cstdio>
#include <string>
#include <iostream>

#define GPUTiming(function, startMessage, stopMessage) {\
GPUTimer timer(startMessage);\
function;\
timer.stop(stopMessage);\
}\

class GPUTimer {
public:
    GPUTimer(const std::string& message) {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        start(message);
    }

    ~GPUTimer() {
        cudaEventDestroy(stopEvent);
        cudaEventDestroy(startEvent);
    }

    void start(const std::string& message) {
        cudaEventRecord(startEvent, 0);
        printf("%s\n", message.c_str());
    }

    void stop(const std::string& message) {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        printf("%s: %3.3f ms \n", message.c_str(), time);
    }

    float time = 0.0f;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
};