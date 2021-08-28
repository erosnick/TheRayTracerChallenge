#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA.h"
#include "cutil_math.h"

class Vec3 {
public:
    CUDA_HOST_DEVICE constexpr Vec3()
    : elements({ 0.0f, 0.0f, 0.0f }) {
    }

    CUDA_HOST_DEVICE constexpr Vec3(double element0, double element1, double element2)
    : elements({ element0, element1, element2 }) {
    }

    CUDA_HOST_DEVICE Vec3(const double3& inElements) {
        elements = inElements;
    }

    CUDA_HOST_DEVICE Vec3 operator-() const {
        return -elements;
    }

    CUDA_HOST_DEVICE double operator[](int32_t index) const {
        switch (index)
        {
        case 0:
            return elements.x;
            break;
        case 1:
            return elements.y;
            break;
        case 2:
            return elements.y;
            break;
        default:
            return -1.0;
            break;
        }
    }

    CUDA_HOST_DEVICE double& operator[](int32_t index) {
        return (*this)[index];
    }

    CUDA_HOST_DEVICE Vec3& operator+=(const Vec3& v) {
        elements.x += v.x();
        elements.y += v.y();
        elements.z += v.z();

        return *this;
    }

    CUDA_HOST_DEVICE Vec3& operator*=(double t) {
        elements.x *= t;
        elements.y *= t;
        elements.z *= t;

        return *this;
    }

    CUDA_HOST_DEVICE Vec3& operator/=(double t) {
        return *this *= 1.0 / t;
    }

    CUDA_HOST_DEVICE double length() const {
        return sqrt(lengthSquared());
    }

    CUDA_HOST_DEVICE double lengthSquared() const {
        return dot(elements, elements);
    }

    CUDA_HOST_DEVICE double x() const {
        return elements.x;
    }

    CUDA_HOST_DEVICE double y() const {
        return elements.y;
    }

    CUDA_HOST_DEVICE double z() const {
        return elements.z;
    }

    double3 elements;
};

inline CUDA_HOST_DEVICE Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.x() + v.x(), u.x() + v.y(), u.z() + v.z());
}

inline CUDA_HOST_DEVICE Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.x() - v.x(), u.x() - v.y(), u.z() - v.z());
}

inline CUDA_HOST_DEVICE Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.x() * v.x(), u.x() * v.y(), u.z() * v.z());
}

inline CUDA_HOST_DEVICE Vec3 operator*(double t, const Vec3& v) {
    return Vec3(t * v.x(), t * v.y(), t * v.z());
}

inline CUDA_HOST_DEVICE Vec3 operator*(const Vec3& v, double t) {
    return t * v;
}

inline CUDA_HOST_DEVICE Vec3 operator/(Vec3 v, double t) {
    return (1 / t) * v;
}

inline CUDA_HOST_DEVICE double dot(const Vec3& u, const Vec3& v) {
    return u.x() * v.x()
         + u.y() * v.y()
         + u.z() * v.z();
}

inline CUDA_HOST_DEVICE Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.y() * v.z() - u.z() * v.y(),
                u.z() * v.x() - u.x() * v.z(),
                u.x() * v.y() - u.y() * v.x());
}

inline CUDA_HOST_DEVICE Vec3 unitVector(const Vec3& v) {
    return v / v.length();
}

inline CUDA_HOST_DEVICE Vec3 lerp(const Vec3& v0, const Vec3& v1, double t) {
    return (1.0 - t) * v0 + t * v1;
}