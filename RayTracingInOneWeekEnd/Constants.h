#pragma once

#include "Vec3.h"
#include <cstdint>
#include <limits>

constexpr int32_t MAXELEMENTS = 8;

namespace Math {
    constexpr double pi = 3.1415926535897932;
    constexpr double pi_3 = 1.0471975511965976;
    constexpr double pi_2 = 1.5707963267948966;
    constexpr double pi_4 = 0.7853981633974483;
    constexpr double pi_6 = 0.5235987755982988;
    constexpr double sqrt_2 = 1.414214;
    constexpr double sqrt_3 = 1.732051;
    constexpr double cos30d = 0.866025;
    constexpr double cos45d = 0.707107;
    constexpr double cos60d = 0.5;
    constexpr double sin30d = 0.5;
    constexpr double sin45d = 0.707107;
    constexpr double sin60D = 0.866025;
    constexpr double epsilon = 0.000001;
    constexpr Vec3 xAxis = Vec3(1.0, 0.0, 0.0);
    constexpr Vec3 yAxis = Vec3(0.0, 1.0, 0.0);
    constexpr Vec3 zAxis = Vec3(0.0, 0.0, -1.0);
    //constexpr double infinityd = std::numeric_limits<double>::infinity();
    constexpr double infinityd = 10000000.0;
    constexpr double infinityf = std::numeric_limits<float>::infinity();
    constexpr double infinityi = std::numeric_limits<int32_t>::infinity();

    inline CUDA_HOST_DEVICE double radians(double degree) {
        return pi / 180.0 * degree;
    }

    inline CUDA_HOST_DEVICE double degrees(double radian) {
        return radian * 180.0 / pi;
    }
}

namespace Color {
    constexpr double oneOver255 = 1.0 / 255;
    constexpr Vec3 black = Vec3(0.0, 0.0, 0.0);
    constexpr Vec3 dawn = Vec3(0.1, 0.1, 0.1);
    constexpr Vec3 white = Vec3(1.0, 1.0, 1.0);
    constexpr Vec3 grey = Vec3(0.5, 0.5, 0.5);
    constexpr Vec3 gray = Vec3(0.7, 0.7, 0.7);
    constexpr Vec3 red = Vec3(1.0, 0.0, 0.0);
    constexpr Vec3 green = Vec3(0.0, 1.0, 0.0);
    constexpr Vec3 yellow = Vec3(1.0, 1.0, 0.0);
    constexpr Vec3 purple = Vec3(1.0, 0.0, 1.0);
    constexpr Vec3 blue = Vec3(0.0, 0.0, 1.0);
    constexpr Vec3 pink = Vec3(1.0, 0.55, 0.55);
    constexpr Vec3 skyBlue = Vec3(134, 203, 237);
    constexpr Vec3 moonstone = Vec3(60, 162, 200);
    constexpr Vec3 turquoise = Vec3(64, 224, 208);
    constexpr Vec3 limeGreen = Vec3(110, 198, 175);
    constexpr Vec3 roseRed = Vec3(0.76, 0.12, 0.34);
    constexpr Vec3 crimsonRed = Vec3(0.86, 0.08, 0.24);
    constexpr Vec3 lightGreen = Vec3(0.38, 1.0, 0.18);
    constexpr Vec3 orange = Vec3(0.85, 0.49, 0.32);
    constexpr Vec3 cornflower = Vec3(0.4, 0.6, 0.9);
    constexpr Vec3 background = Vec3(0.235294, 0.67451, 0.843137);
    constexpr Vec3 lightCornflower = Vec3(0.5, 0.7, 1.0);
    constexpr Vec3 White() { return white; }
    constexpr Vec3 LightCornflower() { return lightCornflower; }
}