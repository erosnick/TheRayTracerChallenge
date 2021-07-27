#pragma once

#include "Tuple.h"

class Light {
public:
    Light() {}
    Light(const Tuple& inPosition, const Tuple& inIntensity) 
    : position(inPosition), intensity(inIntensity) {
    }

    Tuple position;
    Tuple intensity;

    double constant = 1.0;
    double linear = 0.045;
    double quadratic = 0.0075;
};