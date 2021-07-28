#pragma once

#include "Tuple.h"
#include "Ray.h"
#include <iostream>
#include <vector>

constexpr double PI = 3.14159265358979323846;
constexpr double PI_2 = 1.57079632679489661923;
constexpr double PI_4 = 0.785398163397448309616;

class Matrix4;

inline Matrix4 translation(double x, double y, double z);
inline Matrix4 translation(const Vector3& v);
inline Matrix4 scaling(double x, double y, double z);
inline Matrix4 scaling(const Vector3& v);
inline Matrix4 rotationX(double radian);
inline Matrix4 rotationY(double radian);
inline Matrix4 rotationZ(double radian);
inline Matrix4 shearing(double xy, double xz, double yx, double yz, double zx, double zy);
inline Matrix4 operator*(const Matrix4& a, const Matrix4& b);

class Matrix2 {
public:
    Matrix2() {
        row[0] = { 1.0, 0.0 };
        row[1] = { 0.0, 1.0 };
    }

    Matrix2(const Vector2& row0, const Vector2& row1) {
        row[0] = row0;
        row[1] = row1;
    }

    union {
        struct {
            Vector2 row[2];
        };
        double m[2][2];
    };

    const Vector2 operator[](int32_t rowIndex) const {
        return row[rowIndex];
    }

    Vector2& operator[](int32_t rowIndex) {
        return row[rowIndex];
    }

    double determinant() const {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }
};

class Matrix3 {
public:
    Matrix3() {
        row[0] = { 1.0, 0.0, 0.0 };
        row[1] = { 0.0, 1.0, 0.0 };
        row[2] = { 0.0, 0.0, 1.0 };
    }

    Matrix3(const Vector3& row0, const Vector3& row1, const Vector3& row2) {
        row[0] = row0;
        row[1] = row1;
        row[2] = row2;
    }

    const Vector3 operator[](int32_t rowIndex) const {
        return row[rowIndex];
    }

    Vector3& operator[](int32_t rowIndex) {
        return row[rowIndex];
    }

    Matrix2 submatrix(int32_t row, int32_t column) const {
        auto result = Matrix2();

        std::vector<int32_t> rowIndex = { 0, 1, 2 };
        std::vector<int32_t> columnIndex = { 0, 1, 2 };

        for (auto iterator = rowIndex.begin(); iterator != rowIndex.end(); iterator++) {
            if (*iterator == row) {
                rowIndex.erase(iterator);
                break;
            }
        }

        for (auto iterator = columnIndex.begin(); iterator != columnIndex.end(); iterator++) {
            if (*iterator == column) {
                columnIndex.erase(iterator);
                break;
            }
        }

        result[0][0] = m[rowIndex[0]][columnIndex[0]];
        result[0][1] = m[rowIndex[0]][columnIndex[1]];
        result[1][0] = m[rowIndex[1]][columnIndex[0]];
        result[1][1] = m[rowIndex[1]][columnIndex[1]];

        return result;
    }

    double minor(int32_t row, int32_t column) const {
        return submatrix(row, column).determinant();
    }

    // if row + column is an odd number, then you negate the minor.Otherwise,
    // you just return the minor as is.
    double cofactor(int32_t row, int32_t column) const {
        if ((row + column) % 2 == 0) {
            return minor(row, column);
        }
        else {
            return -minor(row, column);
        }
    }

    double determinant() const {
        double result = 0.0;

        // Pick any row(or column), multiply each element by its cofactor,
        // and add the results.
        for (auto column = 0; column < 3; column++) {
            result = result + m[0][column] * cofactor(0, column);
        }

        return result;
    }

    union {
        struct {
            Vector3 row[3];
        };

        double m[3][3];
    };
};

class Matrix4 {
public:
    Matrix4() {
        row[0] = { 1.0, 0.0, 0.0, 0.0 };
        row[1] = { 0.0, 1.0, 0.0, 0.0 };
        row[2] = { 0.0, 0.0, 1.0, 0.0 };
        row[3] = { 0.0, 0.0, 0.0, 1.0 };
    }

    Matrix4(const Tuple& row0, const Tuple& row1, const Tuple& row2, const Tuple& row3) {
        row[0] = row0;
        row[1] = row1;
        row[2] = row2;
        row[3] = row3;
    }

    Matrix4 transpose() {
        auto result = Matrix4();

        result[0][0] = m[0][0];
        result[1][0] = m[0][1];
        result[2][0] = m[0][2];
        result[3][0] = m[0][3];

        result[0][1] = m[1][0];
        result[1][1] = m[1][1];
        result[2][1] = m[1][2];
        result[3][1] = m[1][3];

        result[0][2] = m[2][0];
        result[1][2] = m[2][1];
        result[2][2] = m[2][2];
        result[3][2] = m[2][3];

        result[0][3] = m[3][0];
        result[1][3] = m[3][1];
        result[2][3] = m[3][2];
        result[3][3] = m[3][3];

        return result;
    }

    const Tuple operator[](int32_t rowIndex) const {
        return row[rowIndex];
    }

    Tuple& operator[](int32_t rowIndex) {
        return row[rowIndex];
    }

    Matrix3 submatrix(int32_t row, int32_t column) const {
        auto result = Matrix3();

        std::vector<int32_t> rowIndex = { 0, 1, 2, 3 };
        std::vector<int32_t> columnIndex = { 0, 1, 2, 3 };

        for (auto iterator = rowIndex.begin(); iterator != rowIndex.end(); iterator++) {
            if (*iterator == row) {
                rowIndex.erase(iterator);
                break;
            }
        }

        for (auto iterator = columnIndex.begin(); iterator != columnIndex.end(); iterator++) {
            if (*iterator == column) {
                columnIndex.erase(iterator);
                break;
            }
        }

        result[0][0] = m[rowIndex[0]][columnIndex[0]];
        result[0][1] = m[rowIndex[0]][columnIndex[1]];
        result[0][2] = m[rowIndex[0]][columnIndex[2]];
        result[1][0] = m[rowIndex[1]][columnIndex[0]];
        result[1][1] = m[rowIndex[1]][columnIndex[1]];
        result[1][2] = m[rowIndex[1]][columnIndex[2]];
        result[2][0] = m[rowIndex[2]][columnIndex[0]];
        result[2][1] = m[rowIndex[2]][columnIndex[1]];
        result[2][2] = m[rowIndex[2]][columnIndex[2]];

        return result;
    }

    double minor(int32_t row, int32_t column) const {
        return submatrix(row, column).determinant();
    }

    // if row + column is an odd number, then you negate the minor.Otherwise,
    // you just return the minor as is.
    double cofactor(int32_t row, int32_t column) const {
        if ((row + column) % 2 == 0) {
            return minor(row, column);
        }
        else {
            return -minor(row, column);
        }
    }

    Matrix4 inverse() const {
        auto result = Matrix4();

        if (!invertible()) {
            std::cout << "Matrix is non-invertible.\n";
            return result;
        }

        auto d = determinant();

        for (auto row = 0; row < 4; row++) {
            for (auto column = 0; column < 4; column++) {
                auto c = cofactor(row, column);

                // Note that "column, row" here, instead of "row, column",
                // accomplishes the transpose operation!
                result[column][row] = c / d;
            }
        }

        return result;
    }

    double determinant() const {
        double result = 0.0;
        
        // Pick any row(or column), multiply each element by its cofactor,
        // and add the results.
        for (auto column = 0; column < 4; column++) {
            result = result + m[0][column] * cofactor(0, column);
        }

        return result;
    }

    bool invertible() const {
        return (determinant() != 0.0);
    }

    Matrix4& scaling(double x, double y, double z) {
        auto self = *this;
        *this = self * ::scaling(x, y, z);
        return (*this);
    }

    Matrix4& scaling(const Vector3& v) {
        return scaling(v.x, v.y, v.z);
    }

    Matrix4& translation(double x, double y, double z) {
        auto self = *this;
        *this = self * translation(x, y, z);
        return (*this);
    }

    Matrix4& translation(const Vector3& v) {
        return translation(v.x, v.y, v.z);
    }

    Matrix4& rotationX(double radian) {
        auto self = *this;
        *this = self * rotationX(radian);
        return (*this);
    }

    Matrix4& rotationY(double radian) {
        auto self = *this;
        *this = self * rotationY(radian);
        return (*this);
    }

    Matrix4& rotationZ(double radian) {
        auto self = *this;
        *this = self * rotationZ(radian);
        return (*this);
    }

    Matrix4& shearing(double xy, double xz, double yx, double yz, double zx, double zy) {
        auto self = *this;
        *this = self * shearing(xy, xz, yx, yz, zx, zy);
        return (*this);
    }

    union {
        struct {
            Tuple row[4];
        };

        double m[4][4];
    };
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

inline bool operator!=(const Matrix4& a, const Matrix4& b) {
    return !(a == b);
}

inline Matrix4 operator*(const Matrix4& a, const Matrix4& b) { 
    const auto& aRow0 = a[0];
    const auto& aRow1 = a[1];
    const auto& aRow2 = a[2];
    const auto& aRow3 = a[3];

    const auto& bColumn0 = Tuple(b[0][0], b[1][0], b[2][0], b[3][0]);
    const auto& bColumn1 = Tuple(b[0][1], b[1][1], b[2][1], b[3][1]);
    const auto& bColumn2 = Tuple(b[0][2], b[1][2], b[2][2], b[3][2]);
    const auto& bColumn3 = Tuple(b[0][3], b[1][3], b[2][3], b[3][3]);

    auto result = Matrix4();

    result[0][0] = aRow0.dot(bColumn0);
    result[0][1] = aRow0.dot(bColumn1);
    result[0][2] = aRow0.dot(bColumn2);
    result[0][3] = aRow0.dot(bColumn3);

    result[1][0] = aRow1.dot(bColumn0);
    result[1][1] = aRow1.dot(bColumn1);
    result[1][2] = aRow1.dot(bColumn2);
    result[1][3] = aRow1.dot(bColumn3);

    result[2][0] = aRow2.dot(bColumn0);
    result[2][1] = aRow2.dot(bColumn1);
    result[2][2] = aRow2.dot(bColumn2);
    result[2][3] = aRow2.dot(bColumn3);

    result[3][0] = aRow3.dot(bColumn0);
    result[3][1] = aRow3.dot(bColumn1);
    result[3][2] = aRow3.dot(bColumn2);
    result[3][3] = aRow3.dot(bColumn3);

    return result;
}

inline Tuple operator*(const Matrix4& a, const Tuple& b) {
    auto result = Tuple();

    const auto& aRow0 = a[0];
    const auto& aRow1 = a[1];
    const auto& aRow2 = a[2];
    const auto& aRow3 = a[3];

    result[0] = aRow0.dot(b);
    result[1] = aRow1.dot(b);
    result[2] = aRow2.dot(b);
    result[3] = aRow3.dot(b);

    return result;
}

inline std::ostream& operator << (std::ostream& os, const Matrix4& value) {
    os << "|" << value[0][0] << "|" << value[0][1] << "|" << value[0][2] << "|" << value[0][3] << "|\n"
       << "|" << value[1][0] << "|" << value[1][1] << "|" << value[1][2] << "|" << value[1][3] << "|\n"
       << "|" << value[2][0] << "|" << value[2][1] << "|" << value[2][2] << "|" << value[2][3] << "|\n"
       << "|" << value[3][0] << "|" << value[3][1] << "|" << value[3][2] << "|" << value[3][3] << "|";
    return os;
}

inline Matrix4 translation(double x, double y, double z) {
    auto result = Matrix4();

    result[0][3] = x;
    result[1][3] = y;
    result[2][3] = z;

    return result;
}

inline Matrix4 translation(const Vector3& v) {
    return translation(v.x, v.y, v.z);
}

inline Matrix4 scaling(double x, double y, double z) {
    auto result = Matrix4();

    result[0][0] = x;
    result[1][1] = y;
    result[2][2] = z;

    return result;
}

inline Matrix4 scaling(const Vector3& v) {
    return scaling(v.x, v.y, v.z);
}

inline Matrix4 rotationX(double radian) {
    auto result = Matrix4();

    result[1][1] =  std::cos(radian);
    result[1][2] = -std::sin(radian);
    result[2][1] =  std::sin(radian);
    result[2][2] =  std::cos(radian);

    return result;
}

inline Matrix4 rotationY(double radian) {
    auto result = Matrix4();

    result[0][0] =  std::cos(radian);
    result[0][2] =  std::sin(radian);
    result[2][0] = -std::sin(radian);
    result[2][2] =  std::cos(radian);

    return result;
}

inline Matrix4 rotationZ(double radian) {
    auto result = Matrix4();

    result[0][0] = std::cos(radian);
    result[0][1] = -std::sin(radian);
    result[1][0] = std::sin(radian);
    result[1][1] = std::cos(radian);

    return result;
}

inline Matrix4 shearing(double xy, double xz, double yx, double yz, double zx, double zy) {
    auto result = Matrix4();

    result[0][1] = xy;
    result[0][2] = xz;
    result[1][0] = yx;
    result[1][2] = yz;
    result[2][0] = zx;
    result[2][1] = zy;

    return result;
}

inline Ray transformRay(const Ray& ray, const Matrix4& matrix) {
    auto result = Ray();

    result.origin = matrix * ray.origin;
    result.direction = matrix * ray.direction;

    return result;
}

inline Matrix4 viewTransform(const Tuple& from, const Tuple& to, const Tuple& up) {
    auto viewMatrix = Matrix4();

    auto forward = (to - from).normalize();
    auto right = (forward.cross(up)).normalize();
    auto trueUp = (right.cross(forward)).normalize();

    viewMatrix[0] = right;
    viewMatrix[1] = trueUp;
    viewMatrix[2] = -forward;

    viewMatrix[0][3] = -right.dot(from);
    viewMatrix[1][3] = -trueUp.dot(from);
    viewMatrix[2][3] = forward.dot(from);

    return viewMatrix;
}