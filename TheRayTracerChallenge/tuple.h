#pragma once

#include "utils.h"
#include <limits>

class tuple {
public:
    tuple() {}
    tuple(float inX, float inY, float inZ, float inW = 0.0f) {
        x = inX;
        y = inY;
        z = inZ;
        w = inW;
    }

    tuple operator-() {
        return tuple(-x, -y, -z, -w);
    }

    float magnitude() const {
        float squaredLength = x * x + y * y + z * z;

        float length = std::sqrtf(squaredLength);

        return length;
    }

    tuple normalize() {
        float length = magnitude();

        x /= length;
        y /= length;
        z /= length;
        
        return *this;
    }

    static tuple point(float x, float y, float z) {
        return tuple(x, y, z, 1.0f);
    }

    static tuple vector(float x, float y, float z) {
        return tuple(x, y, z);
    }

    float dot(const tuple& other) {
        return x * other.x + y * other.y + z * other.z;
    }

    tuple cross(const tuple& other) {
        return vector(y * other.z - z * other.y,
                      z * other.x - x * other.z,
                      x * other.y - y * other.x);
    }

    float x;
    float y;
    float z;
    float w;
};

inline bool operator==(const tuple& a, const tuple& b) {
    constexpr float epsilon = 0.0001f;// std::numeric_limits<float>::epsilon();
    if ((std::abs(a.x - b.x) < epsilon)
     && (std::abs(a.y - b.y) < epsilon)
     && (std::abs(a.z - b.z) < epsilon)
     && (std::abs(a.w - b.w) < epsilon)) {
        return true;
    }

    return false;
}

inline tuple operator+(const tuple& a, const tuple& b) {
    return tuple(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline tuple operator-(const tuple& a, const tuple& b) {
    return tuple(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline float operator*(const tuple& a, const tuple& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline tuple operator*(const tuple& v, float scalar) {
    return tuple(v.x * scalar, v.y * scalar, v.z * scalar, v.w * scalar);
}

inline tuple operator*(float scalar, const tuple& v) {
    return v * scalar;
}

inline tuple operator/(const tuple& v, float scalar) {
    return tuple(v.x / scalar, v.y / scalar, v.z / scalar, v.w / scalar);
}