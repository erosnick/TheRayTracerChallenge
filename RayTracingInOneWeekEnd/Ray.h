#pragma once

#include "Vec3.h"

class Ray {
public:
    CUDA_HOST_DEVICE Ray() {}
    CUDA_HOST_DEVICE Ray(const Vec3& inOrigin, const Vec3& inDirection)
    : origin(inOrigin), direction(inDirection) {
    }

    CUDA_HOST_DEVICE Vec3 at(double t) {
        return origin + t * direction;
    }

    Vec3 origin;
    Vec3 direction;
};