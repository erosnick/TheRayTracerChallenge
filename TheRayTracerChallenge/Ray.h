#pragma once

#include "Tuple.h"

class Ray {
public:
    constexpr Ray()
    : origin({ 0.0, 0.0, 0.0, 1.0 }), direction({ 0.0, 0.0, 0.0, 0.0 }) {}
    constexpr Ray(const Tuple& inOrigin, const Tuple& inDirection) 
    : origin(inOrigin), direction(inDirection) {}

    inline Tuple position(double t) const {
        return origin + direction * t;
    }

    Tuple origin = { 0.0, 0.0, 0.0, 1.0 };
    Tuple direction = { 0.0, 0.0, 0.0, 0.0 };
};