#pragma once

#include "Tuple.h"
#include <iostream>
#include <vector>

class Matrix2 {
public:
    Matrix2() {
        row[0] = { 1.0, 0.0 };
        row[1] = { 0.0, 1.0 };
    }

    Matrix2(const Vec2& row0, const Vec2& row1) {
        row[0] = row0;
        row[1] = row1;
    }

    union {
        struct {
            Vec2 row[2];
        };
        double m[2][2];
    };

    const Vec2 operator[](int32_t rowIndex) const {
        return row[rowIndex];
    }

    Vec2& operator[](int32_t rowIndex) {
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

    Matrix3(const Vec3& row0, const Vec3& row1, const Vec3& row2) {
        row[0] = row0;
        row[1] = row1;
        row[2] = row2;
    }

    const Vec3 operator[](int32_t rowIndex) const {
        return row[rowIndex];
    }

    Vec3& operator[](int32_t rowIndex) {
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

    union {
        struct {
            Vec3 row[3];
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