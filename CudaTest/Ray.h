#pragma once

#include "Tuple.h"

class Ray {
public:
    CUDA_HOST_DEVICE Ray()
    : origin({ 0.0, 0.0, 0.0, 1.0 }), direction({ 0.0, 0.0, 0.0, 0.0 }) {}
    CUDA_HOST_DEVICE Ray(const Tuple& inOrigin, const Tuple& inDirection)
    : origin(inOrigin), direction(inDirection) {}

    inline CUDA_HOST_DEVICE Tuple position(Float t) const {
        return origin + direction * t;
    }

    Tuple origin;
    Tuple direction;
};