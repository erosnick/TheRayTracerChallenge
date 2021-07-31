#pragma once

#include "utils.h"
#include <limits>
#include <cstdint>
#include <cmath>

class Tuple {
public:
    constexpr Tuple()
    : x(0.0), y(0.0), z(0.0), w(0.0) {}
    constexpr Tuple(double inX, double inY, double inZ, double inW = 0.0)
    : x(inX), y(inY), z(inZ), w(inW) {
    }

    Tuple operator-() const {
        return Tuple(-x, -y, -z, -w);
    }

    double magnitude() const {
        double squaredLength = x * x + y * y + z * z;

        double length = std::sqrt(squaredLength);

        return length;
    }

    Tuple normalize() {
        double length = magnitude();

        x /= length;
        y /= length;
        z /= length;
        
        return *this;
    }

    double dot(const Tuple& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }

    Tuple cross(const Tuple& other) {
        return Tuple(other.z * y - z * other.y,
                     other.x * z - x * other.z,
                     other.y * x - y * other.x, 0.0);
    }

    const double operator[](int32_t index) const {
        return elements[index];
    }

    double& operator[](int32_t index) {
        return elements[index];
    }

    Tuple& operator+=(const Tuple& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;

        return *this;
    }

    union {
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
    };
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
    Vector3()
        : x(0.0), y(0.0), z(0.0) {}

    Vector3(double inX, double inY, double inZ)
        : x(inX), y(inY), z(inZ) {}

    const double operator[](int32_t index) const {
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

inline Tuple point(double x, double y, double z) {
    return Tuple(x, y, z, 1.0);
}

inline Tuple point() {
    return Tuple(0.0, 0.0, 0.0, 1.0);
}

inline Tuple vector(double x, double y, double z) {
    return Tuple(x, y, z);
}

inline Tuple vector() {
    return Tuple();
}

inline constexpr Tuple color(double inRed, double inGreen, double inBlue) {
    return Tuple(inRed, inGreen, inBlue);
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
    auto dx = std::abs(std::abs(a.x) - std::abs(b.x));
    auto dy = std::abs(std::abs(a.y) - std::abs(b.y));
    auto dz = std::abs(std::abs(a.z) - std::abs(b.z));
    auto dw = std::abs(std::abs(a.w) - std::abs(b.w));
    if ((dx < epsilon)
     && (dy < epsilon)
     && (dz < epsilon)
     && (dw < epsilon)) {
        return true;
    }

    return false;
}

inline Tuple operator+(const Tuple& a, const Tuple& b) {
    return Tuple(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline Tuple operator-(const Tuple& a, const Tuple& b) {
    return Tuple(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline Tuple operator*(const Tuple& a, const Tuple& b) {
    return Tuple(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline Tuple operator+(const Tuple& v, double scalar) {
    return Tuple(v.x + scalar, v.y + scalar, v.z + scalar, v.w);
}

inline Tuple operator*(const Tuple& v, double scalar) {
    return Tuple(v.x * scalar, v.y * scalar, v.z * scalar, v.w * scalar);
}

inline Tuple operator*(double scalar, const Tuple& v) {
    return v * scalar;
}

inline Tuple operator/(const Tuple& v, double scalar) {
    return Tuple(v.x / scalar, v.y / scalar, v.z / scalar, v.w / scalar);
}

inline Tuple reflect(const Tuple& v, const Tuple& n) {
    return v - 2.0 * n.dot(v) * n;
}