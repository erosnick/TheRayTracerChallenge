#pragma once

#include "Tuple.h"
#include "Object.h"
#include "Matrix.h"

class Light : public Object {
public:
    Light() {
    }

    Light(const Tuple& inPosition, const Tuple& inIntensity) 
    : position(inPosition), intensity(inIntensity) {
        id++;
    }

    void transform(const Matrix4& matrix) {
        position = matrix * position;
    }

    Tuple position;
    Tuple intensity;

    double constant = 1.0;
    double linear = 0.045;
    double quadratic = 0.0075;
};