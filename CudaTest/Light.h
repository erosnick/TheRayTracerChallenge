#pragma once

#include "Tuple.h"
#include "Matrix.h"

class Light {
public:
    CUDA_HOST_DEVICE Light() {
        constant = 1.0;
        linear = 0.045;
        quadratic = 0.0075;

        bAttenuation = true;;
    }

    CUDA_HOST_DEVICE Light(const Tuple& inPosition, const Tuple& inIntensity)
    : position(inPosition), intensity(inIntensity) {
        constant = 1.0;
        linear = 0.045;
        quadratic = 0.0075;

        bAttenuation = true;;
    }

    CUDA_HOST_DEVICE void transform(const Matrix4& matrix) {
        position = matrix * position;
    }

    Tuple position;
    Tuple intensity;

    double constant = 1.0;
    double linear = 0.045;
    double quadratic = 0.0075;

    bool bAttenuation = true;;
};