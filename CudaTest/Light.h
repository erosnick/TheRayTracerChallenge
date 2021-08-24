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
    : position(inPosition), transformedPosition(position), intensity(inIntensity) {
        //transformation[0][3] = position.x();
        //transformation[1][3] = position.y();
        //transformation[2][3] = position.z();
        constant = 1.0;
        linear = 0.045;
        quadratic = 0.0075;

        bAttenuation = true;;
    }

    CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) {
        transformedPosition = inTransformation * transformation * position;
    }

    Matrix4 transformation;
    Tuple position;
    Tuple transformedPosition;
    Tuple intensity;

    double constant = 1.0;
    double linear = 0.045;
    double quadratic = 0.0075;

    bool bAttenuation = true;;
};