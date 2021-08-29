#pragma once

#include "Utils.h"
#include <cmath>
#include "CUDA.h"
#include "cutil_math.h"
#include "Types.h"

#include <cstdint>

class Tuple {
public:
    CUDA_HOST_DEVICE constexpr Tuple()
    : data({ 0.0, 0.0, 0.0, 0.0 }) {}
    CUDA_HOST_DEVICE constexpr Tuple(Float inX, Float inY, Float inZ, Float inW = 0.0)
    : data({ inX, inY, inZ, inW }) {
    }

    inline CUDA_HOST_DEVICE constexpr Tuple operator-() const {
        return Tuple(-x(), -y(), -z(), -w());
    }

    inline CUDA_HOST_DEVICE Float magnitude() const {
        return sqrt(magnitudeSqured());
    }

    inline CUDA_HOST_DEVICE Float magnitudeSqured() const {
        Float lengthSquared = x() * x() + y() * y() + z() * z();

        return lengthSquared;
    }

    inline CUDA_HOST_DEVICE Tuple normalize() {
        Float length = magnitude();

        x() /= length;
        y() /= length;
        z() /= length;
        
        return *this;
    }

    inline CUDA_HOST_DEVICE constexpr Float x() const {
        return data.x;
    }

    inline CUDA_HOST_DEVICE constexpr Float& x() {
        return data.x;
    }

    inline CUDA_HOST_DEVICE constexpr Float y() const {
        return data.y;
    }

    inline CUDA_HOST_DEVICE constexpr Float& y() {
        return data.y;
    }

    inline CUDA_HOST_DEVICE constexpr Float z() const {
        return data.z;
    }

    inline CUDA_HOST_DEVICE constexpr Float& z() {
        return data.z;
    }

    inline CUDA_HOST_DEVICE constexpr Float w() const {
        return data.w;
    }

    inline CUDA_HOST_DEVICE constexpr Float& w() {
        return data.w;
    }

    CUDA_HOST_DEVICE Float dot(const Tuple& other) const {
        return x() * other.x() + y() * other.y() + z() * other.z() + w() * other.w();
    }

    CUDA_HOST_DEVICE Tuple cross(const Tuple& other) const {
        return Tuple(other.z() * y() - z() * other.y(),
                     other.x() * z() - x() * other.z(),
                     other.y() * x() - y() * other.x(), 0.0);
    }

    CUDA_HOST_DEVICE Float operator[](int32_t index) const {
        return data.elements[index];
    }

    CUDA_HOST_DEVICE Float& operator[](int32_t index) {
        return data.elements[index];
    }

    CUDA_HOST_DEVICE Tuple& operator+=(const Tuple& other) {
        data.x += other.x();
        data.y += other.y();
        data.z += other.z();
        data.w += other.w();

        return *this;
    }

    CUDA_HOST_DEVICE Tuple& operator*=(const Tuple& other) {
        data.x *= other.x();
        data.y *= other.y();
        data.z *= other.z();
        data.w *= other.w();

        return *this;
    }

    CUDA_HOST_DEVICE void print() const {
        printf("(%f, %f, %f)\n", x(), y(), z());
    }

    union Data {
        struct {
            Float x;
            Float y;
            Float z;
            Float w;
        };

        struct {
            Float red;
            Float green;
            Float blue;
            Float alpha;
        };

        Float elements[4];
    } data;
};

//class Tuple {
//public:
//    CUDA_HOST_DEVICE constexpr Tuple()
//        : elements({ 0.0, 0.0, 0.0, 0.0 }) {}
//    CUDA_HOST_DEVICE constexpr Tuple(Float inX, Float inY, Float inZ, Float inW = 0.0)
//        : elements({ inX, inY, inZ, inW }) {
//    }
//
//    inline CUDA_HOST_DEVICE constexpr Tuple operator-() const {
//        return Tuple(-x(), -y(), -z(), -w());
//    }
//
//    inline CUDA_HOST_DEVICE Float magnitude() const {
//        return std::sqrt(magnitudeSqured());
//    }
//
//    inline CUDA_HOST_DEVICE Float magnitudeSqured() const {
//        Float lengthSquared = x() * x() + y() * y() + z() * z();
//
//        return lengthSquared;
//    }
//
//    inline CUDA_HOST_DEVICE Tuple normalize() {
//        Float length = magnitude();
//
//        x() /= length;
//        y() /= length;
//        z() /= length;
//
//        return *this;
//    }
//
//    inline CUDA_HOST_DEVICE constexpr Float x() const {
//        return elements.x;
//    }
//
//    inline CUDA_HOST_DEVICE constexpr Float& x() {
//        return elements.x;
//    }
//
//    inline CUDA_HOST_DEVICE constexpr Float y() const {
//        return elements.y;
//    }
//
//    inline CUDA_HOST_DEVICE constexpr Float& y() {
//        return elements.y;
//    }
//
//    inline CUDA_HOST_DEVICE constexpr Float z() const {
//        return elements.z;
//    }
//
//    inline CUDA_HOST_DEVICE constexpr Float& z() {
//        return elements.z;
//    }
//
//    inline CUDA_HOST_DEVICE constexpr Float w() const {
//        return elements.w;
//    }
//
//    inline CUDA_HOST_DEVICE constexpr Float& w() {
//        return elements.w;
//    }
//
//    CUDA_HOST_DEVICE Float dot(const Tuple& other) const {
//        return x() * other.x() + y() * other.y() + z() * other.z() + w() * other.w();
//    }
//
//    CUDA_HOST_DEVICE Tuple cross(const Tuple& other) const {
//        return Tuple(other.z() * y() - z() * other.y(),
//            other.x() * z() - x() * other.z(),
//            other.y() * x() - y() * other.x(), 0.0);
//    }
//
//    CUDA_HOST_DEVICE Float operator[](int32_t index) const {
//        switch (index)
//        {
//        case 0:
//            return elements.x;
//            break;
//        case 1:
//            return elements.y;
//            break;
//        case 2:
//            return elements.z;
//        case 3:
//            return elements.w;
//            break;
//        default:
//            return elements.w;
//            break;
//        }
//    }
//
//    CUDA_HOST_DEVICE Float& operator[](int32_t index) {
//        switch (index)
//        {
//        case 0:
//            return elements.x;
//            break;
//        case 1:
//            return elements.y;
//            break;
//        case 2:
//            return elements.z;
//        case 3:
//            return elements.w;
//            break;
//        default:
//            return elements.w;
//            break;
//        }
//    }
//
//    CUDA_HOST_DEVICE Tuple& operator+=(const Tuple& other) {
//        elements.x += other.x();
//        elements.y += other.y();
//        elements.z += other.z();
//        elements.w += other.w();
//
//        return *this;
//    }
//
//    CUDA_HOST_DEVICE Tuple& operator*=(const Tuple& other) {
//        elements.x *= other.x();
//        elements.y *= other.y();
//        elements.z *= other.z();
//        elements.w *= other.w();
//
//        return *this;
//    }
//
//    CUDA_HOST_DEVICE void print() const {
//        printf("(%f, %f, %f)\n", x(), y(), z());
//    }
//
//    Float4 elements;
//};

class Vector2 {
public:
    //CUDA_HOST_DEVICE Vector2()
    //: x(0.0), y(0.0) {}

    //Vector2(Float inX, Float inY)
    //: x(inX), y(inY) {}

    CUDA_HOST_DEVICE Float& operator[](int32_t index) {
        return elements[index];
    }

    union{
        struct {
            Float x;
            Float y;
        };
        Float elements[2];
    };
};

class Vector3 {
public:
    //constexpr Vector3()
    //: x(0.0), y(0.0), z(0.0) {}

    //constexpr Vector3(Float inX, Float inY, Float inZ)
    //: x(inX), y(inY), z(inZ) {}

    CUDA_HOST_DEVICE Float operator[](int32_t index) const {
        return data.elements[index];
    }

    CUDA_HOST_DEVICE Float& operator[](int32_t index) {
        return data.elements[index];
    }

    CUDA_HOST_DEVICE Float x() const {
        return data.x;
    }

    CUDA_HOST_DEVICE Float& x() {
        return data.x;
    }

    CUDA_HOST_DEVICE Float y() const {
        return data.y;
    }

    CUDA_HOST_DEVICE Float& y() {
        return data.y;
    }

    CUDA_HOST_DEVICE Float z() const {
        return data.z;
    }

    CUDA_HOST_DEVICE Float& z() {
        return data.z;
    }

    union Data {
        struct {
            Float x;
            Float y;
            Float z;
        };
        Float elements[3];
    } data;
};

//class Vector3 {
//public:
//    //constexpr Vector3()
//    //: x(0.0), y(0.0), z(0.0) {}
//
//    //constexpr Vector3(Float inX, Float inY, Float inZ)
//    //: x(inX), y(inY), z(inZ) {}
//
//    CUDA_HOST_DEVICE Float operator[](int32_t index) const {
//        switch (index)
//        {
//        case 0:
//            return elements.x;
//            break;
//        case 1:
//            return elements.y;
//            break;
//        case 2:
//            return elements.z;
//        default:
//            return elements.z;
//            break;
//        }
//    }
//
//    CUDA_HOST_DEVICE Float& operator[](int32_t index) {
//        switch (index)
//        {
//        case 0:
//            return elements.x;
//            break;
//        case 1:
//            return elements.y;
//            break;
//        case 2:
//            return elements.z;
//        default:
//            return elements.z;
//            break;
//        }
//    }
//
//    CUDA_HOST_DEVICE Float x() const {
//        return elements.x;
//    }
//
//    CUDA_HOST_DEVICE Float& x() {
//        return elements.x;
//    }
//
//    CUDA_HOST_DEVICE Float y() const {
//        return elements.y;
//    }
//
//    CUDA_HOST_DEVICE Float& y() {
//        return elements.y;
//    }
//
//    CUDA_HOST_DEVICE Float z() const {
//        return elements.z;
//    }
//
//    CUDA_HOST_DEVICE Float& z() {
//        return elements.z;
//    }
//
//    Float3 elements;
//};

//class Vector4 {
//public:
//    constexpr Vector4()
//    : data({ 0.0, 0.0, 0.0, 0.0 }) {}
//
//    constexpr Vector4(Float inX, Float inY, Float inZ, Float inW)
//    : data({ inX, inY, inZ, inW }) {}
//
//    CUDA_HOST_DEVICE Float operator[](int32_t index) const {
//        return elements.elements[index];
//    }
//
//    CUDA_HOST_DEVICE Float& operator[](int32_t index) {
//        return elements.elements[index];
//    }
//
//    CUDA_HOST_DEVICE operator Tuple () const {
//        return Tuple(elements.x, elements.y, elements.z, elements.w);
//    }
//
//    //CUDA_HOST_DEVICE operator Tuple&() const {
//    //    return (*this);
//    //}
//
//    CUDA_HOST_DEVICE Float x() const {
//        return elements.x;
//    }
//
//    CUDA_HOST_DEVICE Float& x() {
//        return elements.x;
//    }
//
//    CUDA_HOST_DEVICE Float y() const {
//        return elements.y;
//    }
//
//    CUDA_HOST_DEVICE Float& y() {
//        return elements.y;
//    }
//
//    CUDA_HOST_DEVICE Float z() const {
//        return elements.z;
//    }
//
//    CUDA_HOST_DEVICE Float& z() {
//        return elements.z;
//    }
//
//    CUDA_HOST_DEVICE Float w() const {
//        return elements.w;
//    }
//
//    CUDA_HOST_DEVICE Float& w() {
//        return elements.w;
//    }
//
//    union Data {
//        struct {
//            Float x;
//            Float y;
//            Float z;
//            Float w;
//        };
//        Float elements[4];
//    } data;
//};

class Vector4 {
public:
    constexpr Vector4()
        : elements({ 0.0, 0.0, 0.0, 0.0 }) {}

    constexpr Vector4(Float inX, Float inY, Float inZ, Float inW)
        : elements({ inX, inY, inZ, inW }) {}

    CUDA_HOST_DEVICE Float operator[](int32_t index) const {
        switch (index)
        {
        case 0:
            return elements.x;
            break;
        case 1:
            return elements.y;
            break;
        case 2:
            return elements.z;
        case 3:
            return elements.w;
            break;
        default:
            return elements.w;
            break;
        }
    }

    CUDA_HOST_DEVICE Float& operator[](int32_t index) {
        return (*this)[index];
    }

    CUDA_HOST_DEVICE operator Tuple () const {
        return Tuple(elements.x, elements.y, elements.z, elements.w);
    }

    CUDA_HOST_DEVICE Float x() const {
        return elements.x;
    }

    CUDA_HOST_DEVICE Float& x() {
        return elements.x;
    }

    CUDA_HOST_DEVICE Float y() const {
        return elements.y;
    }

    CUDA_HOST_DEVICE Float& y() {
        return elements.y;
    }

    CUDA_HOST_DEVICE Float z() const {
        return elements.z;
    }

    CUDA_HOST_DEVICE Float& z() {
        return elements.z;
    }

    CUDA_HOST_DEVICE Float w() const {
        return elements.w;
    }

    CUDA_HOST_DEVICE Float& w() {
        return elements.w;
    }

    Float4 elements;
};

inline constexpr Tuple point(Float x, Float y, Float z) {
    return Tuple(x, y, z, 1.0);
}

inline constexpr Tuple point() {
    return Tuple(0.0, 0.0, 0.0, 1.0);
}

inline constexpr Tuple point(Float value) {
    return point(value, value, value);
}

inline constexpr Tuple vector(Float x, Float y, Float z) {
    return Tuple(x, y, z);
}

inline constexpr Tuple vector(Float value) {
    return vector(value, value, value);
}

inline constexpr Tuple vector() {
    return Tuple();
}

inline constexpr Tuple color(Float inRed, Float inGreen, Float inBlue) {
    return Tuple(inRed, inGreen, inBlue);
}

inline constexpr Tuple color(int32_t inRed, int32_t inGreen, int32_t inBlue) {
    return Tuple(1.0 / 255 * inRed,
                 1.0 / 255 * inGreen,
                 1.0 / 255 * inBlue);
}

inline constexpr Tuple color(Float value) {
    return color(value, value, value);
}

inline bool operator==(const Vector2& a, const Vector2& b) {
    constexpr Float epsilon = 0.0001;// std::numeric_limits<Float>::epsilon();
    auto dx = std::abs(std::abs(a.x) - std::abs(b.x));
    auto dy = std::abs(std::abs(a.y) - std::abs(b.y));
    if ((dx < epsilon)
     && (dy < epsilon)) {
        return true;
    }

    return false;
}

inline bool operator==(const Vector3& a, const Vector3& b) {
    constexpr Float epsilon = 0.0001;// std::numeric_limits<Float>::epsilon();
    auto dx = std::abs(std::abs(a.x()) - std::abs(b.x()));
    auto dy = std::abs(std::abs(a.y()) - std::abs(b.y()));
    auto dz = std::abs(std::abs(a.z()) - std::abs(b.z()));
    if ((dx < epsilon)
     && (dy < epsilon)
     && (dz < epsilon)) {
        return true;
    }

    return false;
}

inline CUDA_HOST_DEVICE bool operator==(const Tuple& a, const Tuple& b) {
    constexpr Float epsilon = 0.00001;// std::numeric_limits<Float>::epsilon();
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

inline CUDA_HOST_DEVICE bool operator!=(const Tuple& a, const Tuple& b) {
    return !operator==(a, b);
}

inline CUDA_HOST_DEVICE Tuple operator+(const Tuple& a, const Tuple& b) {
    return Tuple(a.x() + b.x(), a.y() + b.y(), a.z() + b.z(), a.w() + b.w());
}

inline constexpr Tuple operator-(const Tuple& a, const Tuple& b) {
    return Tuple(a.x() - b.x(), a.y() - b.y(), a.z() - b.z(), a.w() - b.w());
}

inline constexpr Tuple operator*(const Tuple& a, const Tuple& b) {
    return Tuple(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(), a.w() * b.w());
}

inline constexpr Tuple operator+(const Tuple& v, Float scalar) {
    return Tuple(v.x() + scalar, v.y() + scalar, v.z() + scalar, v.w());
}

inline constexpr Tuple operator*(const Tuple& v, Float scalar) {
    return Tuple(v.x() * scalar, v.y() * scalar, v.z() * scalar, v.w() * scalar);
}

inline constexpr Tuple operator*(Float scalar, const Tuple& v) {
    return v * scalar;
}

inline constexpr Tuple operator/(const Tuple& v, Float scalar) {
    return Tuple(v.x() / scalar, v.y() / scalar, v.z() / scalar, v.w() / scalar);
}

inline constexpr Tuple reflect(const Tuple& v, const Tuple& n) {
    return v - 2.0 * n.dot(v) * n;
}