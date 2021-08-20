#include "Matrix.h"

CUDA_HOST_DEVICE Matrix2 Matrix3::submatrix(int32_t row, int32_t column) const {
    auto result = Matrix2();

    Array<int32_t> rowIndex;
    rowIndex.add(0);
    rowIndex.add(1);
    rowIndex.add(2);

    Array<int32_t> columnIndex;
    columnIndex.add(0);
    columnIndex.add(1);
    columnIndex.add(2);

    for (auto i = 0; i < 3; i++) {
        if (rowIndex[i] == row) {
            rowIndex.remove(row);
            break;
        }
    }

    for (auto i = 0; i < 3; i++) {
        if (columnIndex[i] == column) {
            columnIndex.remove(column);
            break;
        }
    }

    result[0][0] = data.m[rowIndex[0]][columnIndex[0]];
    result[0][1] = data.m[rowIndex[0]][columnIndex[1]];
    result[1][0] = data.m[rowIndex[1]][columnIndex[0]];
    result[1][1] = data.m[rowIndex[1]][columnIndex[1]];

    return result;
}

CUDA_HOST_DEVICE Matrix4 Matrix4::transpose() const {
    auto result = Matrix4();

    result[0][0] = rows[0][0];
    result[1][0] = rows[0][1];
    result[2][0] = rows[0][2];
    result[3][0] = rows[0][3];

    result[0][1] = rows[1][0];
    result[1][1] = rows[1][1];
    result[2][1] = rows[1][2];
    result[3][1] = rows[1][3];

    result[0][2] = rows[2][0];
    result[1][2] = rows[2][1];
    result[2][2] = rows[2][2];
    result[3][2] = rows[2][3];

    result[0][3] = rows[3][0];
    result[1][3] = rows[3][1];
    result[2][3] = rows[3][2];
    result[3][3] = rows[3][3];

    return result;
}

CUDA_HOST_DEVICE Matrix4 Matrix4::inverse() const {
    auto result = Matrix4();

    if (!invertible()) {
        printf("Matrix is non-invertible.\n");
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

CUDA_HOST_DEVICE Matrix3 Matrix4::submatrix(int32_t row, int32_t column) const {
    auto result = Matrix3();

    Array<int32_t> rowIndex;
    rowIndex.add(0);
    rowIndex.add(1);
    rowIndex.add(2);
    rowIndex.add(3);

    Array<int32_t> columnIndex;
    columnIndex.add(0);
    columnIndex.add(1);
    columnIndex.add(2);
    columnIndex.add(3);

    for (auto i = 0; i < 4; i++) {
        if (rowIndex[i] == row) {
            rowIndex.remove(row);
            break;
        }
    }

    for (auto i = 0; i < 4; i++) {
        if (columnIndex[i] == column) {
            columnIndex.remove(column);
            break;
        }
    }

    result[0][0] = rows[rowIndex[0]][columnIndex[0]];
    result[0][1] = rows[rowIndex[0]][columnIndex[1]];
    result[0][2] = rows[rowIndex[0]][columnIndex[2]];
    result[1][0] = rows[rowIndex[1]][columnIndex[0]];
    result[1][1] = rows[rowIndex[1]][columnIndex[1]];
    result[1][2] = rows[rowIndex[1]][columnIndex[2]];
    result[2][0] = rows[rowIndex[2]][columnIndex[0]];
    result[2][1] = rows[rowIndex[2]][columnIndex[1]];
    result[2][2] = rows[rowIndex[2]][columnIndex[2]];

    return result;
}

CUDA_HOST_DEVICE Tuple operator*(const Matrix4& a, const Tuple& b) {
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

CUDA_HOST_DEVICE Matrix4 operator*(const Matrix4& a, const Matrix4& b) {
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

CUDA_HOST_DEVICE Matrix4 viewTransform(const Tuple& eye, const Tuple& center, const Tuple& up) {
    auto viewMatrix = Matrix4();

    auto forward = (center - eye).normalize();
    auto right = (forward.cross(up)).normalize();
    auto trueUp = (right.cross(forward)).normalize();

    viewMatrix[0] = right;
    viewMatrix[1] = trueUp;
    viewMatrix[2] = -forward;

    viewMatrix[0][3] = -right.dot(eye);
    viewMatrix[1][3] = -trueUp.dot(eye);
    viewMatrix[2][3] = forward.dot(eye);

    return viewMatrix;
}