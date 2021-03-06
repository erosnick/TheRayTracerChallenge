#pragma once

#include "Tuple.h"

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
    constexpr Tuple xAxis = vector(1.0, 0.0, 0.0);
    constexpr Tuple yAxis = vector(0.0, 1.0, 0.0);
    constexpr Tuple zAxis = vector(0.0, 0.0, -1.0);
    constexpr double infinityd = std::numeric_limits<double>::infinity();
    constexpr double infinityf = std::numeric_limits<float>::infinity();
    constexpr double infinityi = std::numeric_limits<int32_t>::infinity();

    inline double radians(double degree) {
        return pi / 180.0 * degree;
    }

    inline double degrees(double radian) {
        return radian * 180.0 / pi;
    }
}

namespace Color {
    constexpr double oneOver255 = 1.0 / 255;
    constexpr Tuple black = color(0.0, 0.0, 0.0);
    constexpr Tuple dawn = color(0.1, 0.1, 0.1);
    constexpr Tuple white = color(1.0, 1.0, 1.0);
    constexpr Tuple grey = color(0.5, 0.5, 0.5);
    constexpr Tuple gray = color(0.7, 0.7, 0.7);
    constexpr Tuple red = color(1.0, 0.0, 0.0);
    constexpr Tuple green = color(0.0, 1.0, 0.0);
    constexpr Tuple yellow = color(1.0, 1.0, 0.0);
    constexpr Tuple purple = color(1.0, 0.0, 1.0);
    constexpr Tuple blue = color(0.0, 0.0, 1.0);
    constexpr Tuple pink = color(1.0, 0.55, 0.55);
    constexpr Tuple pinkBlue = pink + blue;
    constexpr Tuple skyBlue = color(134, 203, 237);
    constexpr Tuple moonstone = color(60, 162, 200);
    constexpr Tuple turquoise = color(64, 224, 208);
    constexpr Tuple limeGreen = color(110, 198, 175);
    constexpr Tuple roseRed = color(0.76, 0.12, 0.34);
    constexpr Tuple crimsonRed = color(0.86, 0.08, 0.24);
    constexpr Tuple lightGreen = color(0.38, 1.0, 0.18);
    constexpr Tuple orange = color(0.85, 0.49, 0.32);
    constexpr Tuple cornflower = color(0.4, 0.6, 0.9);
    constexpr Tuple background = color(0.235294, 0.67451, 0.843137);
}