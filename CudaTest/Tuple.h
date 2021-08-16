#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "CUDA.h"
#include <limits>
#include <cstdint>
#include <cmath>

class Tuple {
public:
    constexpr Tuple()
    : data({ 0.0, 0.0, 0.0, 0.0 }) {}
    constexpr Tuple(double inX, double inY, double inZ, double inW = 0.0)
    : data({ inX, inY, inZ, inW }) {
    }

    inline constexpr Tuple operator-() const {
        return Tuple(-x(), -y(), -z(), -w());
    }

    inline CUDA_HOST_DEVICE double magnitude() const {
        return std::sqrt(magnitudeSqured());
    }

    inline CUDA_HOST_DEVICE double magnitudeSqured() const {
        double lengthSquared = x() * x() + y() * y() + z() * z();

        return lengthSquared;
    }

    inline CUDA_HOST_DEVICE Tuple normalize() {
        double length = magnitude();

        x() /= length;
        y() /= length;
        z() /= length;
        
        return *this;
    }

    inline CUDA_HOST_DEVICE constexpr double x() const {
        return data.x;
    }

    inline CUDA_HOST_DEVICE constexpr double& x() {
        return data.x;
    }

    inline CUDA_HOST_DEVICE constexpr double y() const {
        return data.y;
    }

    inline CUDA_HOST_DEVICE constexpr double& y() {
        return data.y;
    }

    inline CUDA_HOST_DEVICE constexpr double z() const {
        return data.z;
    }

    inline CUDA_HOST_DEVICE constexpr double& z() {
        return data.z;
    }

    inline CUDA_HOST_DEVICE constexpr double w() const {
        return data.w;
    }

    inline CUDA_HOST_DEVICE constexpr double& w() {
        return data.w;
    }

    CUDA_HOST_DEVICE double dot(const Tuple& other) const {
        return x() * other.x() + y() * other.y() + z() * other.z() + w() * other.w();
    }

    CUDA_HOST_DEVICE Tuple cross(const Tuple& other) const {
        return Tuple(other.z() * y() - z() * other.y(),
                     other.x() * z() - x() * other.z(),
                     other.y() * x() - y() * other.x(), 0.0);
    }

    CUDA_HOST_DEVICE double operator[](int32_t index) const {
        return data.elements[index];
    }

    CUDA_HOST_DEVICE double& operator[](int32_t index) {
        return data.elements[index];
    }

    CUDA_HOST_DEVICE Tuple& operator+=(const Tuple& other) {
        data.x += other.x();
        data.y += other.y();
        data.z += other.z();
        data.w += other.w();

        return *this;
    }

    union Data {
        struct {
            double x;
            double y;
            double z;
            double w;
        };

        struct {
            double red;
            double green;
            double blue;
            double alpha;
        };

        double elements[4];
    } data;
};

class Vector2 {
public:
    Vector2()
    : x(0.0), y(0.0) {}

    Vector2(double inX, double inY)
    : x(inX), y(inY) {}

    double& operator[](int32_t index) {
        return elements[index];
    }

    union{
        struct {
            double x;
            double y;
        };
        double elements[2];
    };
};

class Vector3 {
public:
    constexpr Vector3()
    : x(0.0), y(0.0), z(0.0) {}

    constexpr Vector3(double inX, double inY, double inZ)
    : x(inX), y(inY), z(inZ) {}

    double operator[](int32_t index) const {
        return elements[index];
    }

    double& operator[](int32_t index) {
        return elements[index];
    }

    union {
        struct {
            double x;
            double y;
            double z;
        };
        double elements[3];
    };
};

inline constexpr Tuple point(double x, double y, double z) {
    return Tuple(x, y, z, 1.0);
}

inline constexpr Tuple point() {
    return Tuple(0.0, 0.0, 0.0, 1.0);
}

inline constexpr Tuple point(double value) {
    return point(value, value, value);
}

inline constexpr Tuple vector(double x, double y, double z) {
    return Tuple(x, y, z);
}

inline constexpr Tuple vector(double value) {
    return vector(value, value, value);
}

inline constexpr Tuple vector() {
    return Tuple();
}

inline constexpr Tuple color(double inRed, double inGreen, double inBlue) {
    return Tuple(inRed, inGreen, inBlue);
}

inline constexpr Tuple color(int32_t inRed, int32_t inGreen, int32_t inBlue) {
    return Tuple(1.0 / 255 * inRed,
                 1.0 / 255 * inGreen,
                 1.0 / 255 * inBlue);
}

inline constexpr Tuple color(double value) {
    return color(value, value, value);
}

inline bool operator==(const Vector2& a, const Vector2& b) {
    constexpr double epsilon = 0.0001;// std::numeric_limits<double>::epsilon();
    auto dx = std::abs(std::abs(a.x) - std::abs(b.x));
    auto dy = std::abs(std::abs(a.y) - std::abs(b.y));
    if ((dx < epsilon)
     && (dy < epsilon)) {
        return true;
    }

    return false;
}

inline bool operator==(const Vector3& a, const Vector3& b) {
    constexpr double epsilon = 0.0001;// std::numeric_limits<double>::epsilon();
    auto dx = std::abs(std::abs(a.x) - std::abs(b.x));
    auto dy = std::abs(std::abs(a.y) - std::abs(b.y));
    auto dz = std::abs(std::abs(a.z) - std::abs(b.z));
    if ((dx < epsilon)
     && (dy < epsilon)
     && (dz < epsilon)) {
        return true;
    }

    return false;
}

inline bool operator==(const Tuple& a, const Tuple& b) {
    constexpr double epsilon = 0.00001;// std::numeric_limits<double>::epsilon();
    auto dx = std::abs(std::abs(a.x()) - std::abs(b.x()));
    auto dy = std::abs(std::abs(a.y()) - std::abs(b.y()));
    auto dz = std::abs(std::abs(a.z()) - std::abs(b.z()));
    auto dw = std::abs(std::abs(a.w()) - std::abs(b.w()));
    if ((dx < epsilon)
     && (dy < epsilon)
     && (dz < epsilon)
     && (dw < epsilon)) {
        return true;
    }

    return false;
}

inline bool operator!=(const Tuple& a, const Tuple& b) {
    return !operator==(a, b);
}

inline constexpr Tuple operator+(const Tuple& a, const Tuple& b) {
    return Tuple(a.x() + b.x(), a.y() + b.y(), a.z() + b.z(), a.w() + b.w());
}

inline constexpr Tuple operator-(const Tuple& a, const Tuple& b) {
    return Tuple(a.x() - b.x(), a.y() - b.y(), a.z() - b.z(), a.w() - b.w());
}

inline constexpr Tuple operator*(const Tuple& a, const Tuple& b) {
    return Tuple(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(), a.w() * b.w());
}

inline constexpr Tuple operator+(const Tuple& v, double scalar) {
    return Tuple(v.x() + scalar, v.y() + scalar, v.z() + scalar, v.w());
}

inline constexpr Tuple operator*(const Tuple& v, double scalar) {
    return Tuple(v.x() * scalar, v.y() * scalar, v.z() * scalar, v.w() * scalar);
}

inline constexpr Tuple operator*(double scalar, const Tuple& v) {
    return v * scalar;
}

inline constexpr Tuple operator/(const Tuple& v, double scalar) {
    return Tuple(v.x() / scalar, v.y() / scalar, v.z() / scalar, v.w() / scalar);
}

inline constexpr Tuple reflect(const Tuple& v, const Tuple& n) {
    return v - 2.0 * n.dot(v) * n;
}