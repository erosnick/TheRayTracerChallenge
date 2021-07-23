#pragma once

#include "utils.h"
#include <limits>
#include <cstdint>
#include <cmath>

class Tuple {
public:
    Tuple()
    : x(0.0), y(0.0), z(0.0), w(0.0) {}
    Tuple(double inX, double inY, double inZ, double inW = 0.0) {
        x = inX;
        y = inY;
        z = inZ;
        w = inW;
    }

    Tuple operator-() {
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
        return Tuple(y * other.z - z * other.y,
                     z * other.x - x * other.z,
                     x * other.y - y * other.x, 0.0);
    }

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

class Vec2 {
public:
    Vec2()
    : x(0.0), y(0.0) {}

    Vec2(double inX, double inY)
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

class Vec3 {
public:
    Vec3()
        : x(0.0), y(0.0), z(0.0) {}

    Vec3(double inX, double inY, double inZ)
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

inline Tuple vector(double x, double y, double z) {
    return Tuple(x, y, z);
}

inline Tuple color(double inRed, double inGreen, double inBlue) {
    return Tuple(inRed, inGreen, inBlue);
}

inline bool operator==(const Vec2& a, const Vec2& b) {
    constexpr double epsilon = 0.0001;// std::numeric_limits<double>::epsilon();
    if ((std::abs(a.x - b.x) < epsilon)
     && (std::abs(a.y - b.y) < epsilon)) {
        return true;
    }

    return false;
}

inline bool operator==(const Vec3& a, const Vec3& b) {
    constexpr double epsilon = 0.0001;// std::numeric_limits<double>::epsilon();
    if ((std::abs(a.x - b.x) < epsilon)
     && (std::abs(a.y - b.y) < epsilon)
     && (std::abs(a.z - b.z) < epsilon)) {
        return true;
    }

    return false;
}

inline bool operator==(const Tuple& a, const Tuple& b) {
    constexpr double epsilon = 0.0001;// std::numeric_limits<double>::epsilon();
    if ((std::abs(a.x - b.x) < epsilon)
     && (std::abs(a.y - b.y) < epsilon)
     && (std::abs(a.z - b.z) < epsilon)
     && (std::abs(a.w - b.w) < epsilon)) {
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

inline Tuple operator*(const Tuple& v, double scalar) {
    return Tuple(v.x * scalar, v.y * scalar, v.z * scalar, v.w * scalar);
}

inline Tuple operator*(double scalar, const Tuple& v) {
    return v * scalar;
}

inline Tuple operator/(const Tuple& v, double scalar) {
    return Tuple(v.x / scalar, v.y / scalar, v.z / scalar, v.w / scalar);
}