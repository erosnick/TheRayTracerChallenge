#pragma once

#include "Tuple.h"
#include "Ray.h"
#include "Constants.h"
#include "Array.h"

#include <iostream>
#include <cmath>

class Matrix4;

inline CUDA_HOST_DEVICE Matrix4 translate(double x, double y, double z);
inline CUDA_HOST_DEVICE Matrix4 translate(const Vector3& v);
inline CUDA_HOST_DEVICE Matrix4 scaling(double x, double y, double z);
inline CUDA_HOST_DEVICE Matrix4 scaling(const Vector3& v);
inline CUDA_HOST_DEVICE Matrix4 rotateX(double radian);
inline CUDA_HOST_DEVICE Matrix4 rotateY(double radian);
inline CUDA_HOST_DEVICE Matrix4 rotateZ(double radian);
inline CUDA_HOST_DEVICE Matrix4 shearing(double xy, double xz, double yx, double yz, double zx, double zy);
CUDA_HOST_DEVICE Matrix4 operator*(const Matrix4& a, const Matrix4& b);
CUDA_HOST_DEVICE Tuple operator*(const Matrix4& a, const Tuple& b);

class Matrix2 {
public:
    CUDA_HOST_DEVICE Matrix2() {
        data.row[0] = { 1.0, 0.0 };
        data.row[1] = { 0.0, 1.0 };
    }

    CUDA_HOST_DEVICE Matrix2(const Vector2& row0, const Vector2& row1) {
        data.row[0] = row0;
        data.row[1] = row1;
    }

    union Data {
        struct {
            Vector2 row[2];
        };
        double m[2][2];
    } data;

    CUDA_HOST_DEVICE const Vector2 operator[](int32_t rowIndex) const {
        return data.row[rowIndex];
    }

    CUDA_HOST_DEVICE Vector2& operator[](int32_t rowIndex) {
        return data.row[rowIndex];
    }

    CUDA_HOST_DEVICE double determinant() const {
        return data.m[0][0] * data.m[1][1] - data.m[0][1] * data.m[1][0];
    }
};

class Matrix3 {
public:
    CUDA_HOST_DEVICE Matrix3() {
        data.row[0] = { 1.0, 0.0, 0.0 };
        data.row[1] = { 0.0, 1.0, 0.0 };
        data.row[2] = { 0.0, 0.0, 1.0 };
    }

    Matrix3(const Vector3& row0, const Vector3& row1, const Vector3& row2) {
        data.row[0] = row0;
        data.row[1] = row1;
        data.row[2] = row2;
    }

    CUDA_HOST_DEVICE const Vector3 operator[](int32_t rowIndex) const {
        return data.row[rowIndex];
    }

    CUDA_HOST_DEVICE Vector3& operator[](int32_t rowIndex) {
        return data.row[rowIndex];
    }

    CUDA_HOST_DEVICE Matrix2 submatrix(int32_t row, int32_t column) const;

    CUDA_HOST_DEVICE double minor(int32_t row, int32_t column) const {
        return submatrix(row, column).determinant();
    }

    // if row + column is an odd number, then you negate the minor.Otherwise,
    // you just return the minor as is.
    CUDA_HOST_DEVICE double cofactor(int32_t row, int32_t column) const {
        if ((row + column) % 2 == 0) {
            return minor(row, column);
        }
        else {
            return -minor(row, column);
        }
    }

    CUDA_HOST_DEVICE double determinant() const {
        double result = 0.0;

        // Pick any row(or column), multiply each element by its cofactor,
        // and add the results.
        for (auto column = 0; column < 3; column++) {
            result = result + data.m[0][column] * cofactor(0, column);
        }

        return result;
    }

    union Data {
        struct {
            Vector3 row[3];
        };

        double m[3][3];
    } data;
};

class Matrix4 {
public:
    CUDA_HOST_DEVICE Matrix4() {
        rows[0] = { 1.0, 0.0, 0.0, 0.0 };
        rows[1] = { 0.0, 1.0, 0.0, 0.0 };
        rows[2] = { 0.0, 0.0, 1.0, 0.0 };
        rows[3] = { 0.0, 0.0, 0.0, 1.0 };
    }

    CUDA_HOST_DEVICE Matrix4(const Tuple& row0, const Tuple& row1, const Tuple& row2, const Tuple& row3) {
        rows[0] = { row0.x(), row0.y(), row0.z(), row0.w() };
        rows[1] = { row1.x(), row1.y(), row1.z(), row1.w() };;
        rows[2] = { row2.x(), row2.y(), row2.z(), row2.w() };;
        rows[3] = { row3.x(), row3.y(), row3.z(), row3.w() };;
    }

    CUDA_HOST_DEVICE Matrix4 transpose() const;

    CUDA_HOST_DEVICE const Tuple operator[](int32_t rowIndex) const {
        return rows[rowIndex];
    }

    CUDA_HOST_DEVICE Tuple& operator[](int32_t rowIndex) {
        return rows[rowIndex];
    }

    CUDA_HOST_DEVICE Matrix3 submatrix(int32_t row, int32_t column) const;

    CUDA_HOST_DEVICE double minor(int32_t row, int32_t column) const {
        return submatrix(row, column).determinant();
    }

    // if row + column is an odd number, then you negate the minor.Otherwise,
    // you just return the minor as is.
    CUDA_HOST_DEVICE double cofactor(int32_t row, int32_t column) const {
        if ((row + column) % 2 == 0) {
            return minor(row, column);
        }
        else {
            return -minor(row, column);
        }
    }

    CUDA_HOST_DEVICE Matrix4 inverse() const;

    CUDA_HOST_DEVICE double determinant() const {
        double result = 0.0;
        
        // Pick any row(or column), multiply each element by its cofactor,
        // and add the results.
        for (auto column = 0; column < 4; column++) {
            result = result + rows[0][column] * cofactor(0, column);
        }

        return result;
    }

    CUDA_HOST_DEVICE bool invertible() const {
        return (determinant() != 0.0);
    }

    CUDA_HOST_DEVICE Matrix4& scaling(double x, double y, double z) {
        auto self = *this;
        *this = self * ::scaling(x, y, z);
        return (*this);
    }

    CUDA_HOST_DEVICE Matrix4& scaling(const Vector3& v) {
        return scaling(v.x(), v.y(), v.z());
    }

    CUDA_HOST_DEVICE Matrix4& translate(double x, double y, double z) {
        auto self = *this;
        *this = self * translate(x, y, z);
        return (*this);
    }

    CUDA_HOST_DEVICE Matrix4& translate(const Vector3& v) {
        return translate(v.x(), v.y(), v.z());
    }

    CUDA_HOST_DEVICE Matrix4& rotateX(double radian) {
        auto self = *this;
        *this = self * rotateX(radian);
        return (*this);
    }

    CUDA_HOST_DEVICE Matrix4& rotateY(double radian) {
        auto self = *this;
        *this = self * rotateY(radian);
        return (*this);
    }

    CUDA_HOST_DEVICE Matrix4& rotateZ(double radian) {
        auto self = *this;
        *this = self * rotateZ(radian);
        return (*this);
    }

    CUDA_HOST_DEVICE Matrix4& shearing(double xy, double xz, double yx, double yz, double zx, double zy) {
        auto self = *this;
        *this = self * shearing(xy, xz, yx, yz, zx, zy);
        return (*this);
    }

    //union Data {
    //    struct {
    //        Vector4 row[4];
    //    };

    //    double m[4][4];
    //} data;
    Tuple rows[4];
};

inline bool operator==(const Matrix2& a, const Matrix2& b) {
    return (a[0] == b[0] && a[1] == b[1]);
}

inline bool operator==(const Matrix3& a, const Matrix3& b) {
    return (a[0] == b[0]
         && a[1] == b[1]
         && a[2] == b[2]);
}

inline bool operator==(const Matrix4& a, const Matrix4& b) {
    return (a[0] == b[0] 
         && a[1] == b[1]
         && a[2] == b[2]
         && a[3] == b[3]);
}

inline CUDA_HOST_DEVICE bool operator!=(const Matrix4& a, const Matrix4& b) {
    return !(a == b);
}

inline std::ostream& operator << (std::ostream& os, const Matrix4& value) {
    os << "|" << value[0][0] << "|" << value[0][1] << "|" << value[0][2] << "|" << value[0][3] << "|\n"
       << "|" << value[1][0] << "|" << value[1][1] << "|" << value[1][2] << "|" << value[1][3] << "|\n"
       << "|" << value[2][0] << "|" << value[2][1] << "|" << value[2][2] << "|" << value[2][3] << "|\n"
       << "|" << value[3][0] << "|" << value[3][1] << "|" << value[3][2] << "|" << value[3][3] << "|";
    return os;
}

inline CUDA_HOST_DEVICE Matrix4 translate(double x, double y, double z) {
    auto result = Matrix4();

    result[0][3] = x;
    result[1][3] = y;
    result[2][3] = z;

    return result;
}

inline CUDA_HOST_DEVICE Matrix4 translate(const Vector3& v) {
    return translate(v.x(), v.y(), v.z());
}

inline CUDA_HOST_DEVICE Matrix4 scaling(double x, double y, double z) {
    auto result = Matrix4();

    result[0][0] = x;
    result[1][1] = y;
    result[2][2] = z;

    return result;
}

inline CUDA_HOST_DEVICE Matrix4 scaling(const Vector3& v) {
    return scaling(v.x(), v.y(), v.z());
}

inline CUDA_HOST_DEVICE Matrix4 rotateX(double radian) {
    auto result = Matrix4();

    result[1][1] =  std::cos(radian);
    result[1][2] = -std::sin(radian);
    result[2][1] =  std::sin(radian);
    result[2][2] =  std::cos(radian);

    return result;
}

inline CUDA_HOST_DEVICE Matrix4 rotateY(double radian) {
    auto result = Matrix4();

    result[0][0] =  std::cos(radian);
    result[0][2] =  std::sin(radian);
    result[2][0] = -std::sin(radian);
    result[2][2] =  std::cos(radian);

    return result;
}

inline CUDA_HOST_DEVICE Matrix4 rotateZ(double radian) {
    auto result = Matrix4();

    result[0][0] =  std::cos(radian);
    result[0][1] = -std::sin(radian);
    result[1][0] =  std::sin(radian);
    result[1][1] =  std::cos(radian);

    return result;
}

inline CUDA_HOST_DEVICE Matrix4 shearing(double xy, double xz, double yx, double yz, double zx, double zy) {
    auto result = Matrix4();

    result[0][1] = xy;
    result[0][2] = xz;
    result[1][0] = yx;
    result[1][2] = yz;
    result[2][0] = zx;
    result[2][1] = zy;

    return result;
}

inline CUDA_HOST_DEVICE Ray transformRay(const Ray& ray, const Matrix4& matrix) {
    auto result = Ray();

    result.origin = matrix * ray.origin;
    result.direction = matrix * ray.direction;

    return result;
}

CUDA_HOST_DEVICE Matrix4 viewTransform(const Tuple& eye, const Tuple& center, const Tuple& up);