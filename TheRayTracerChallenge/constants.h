#pragma once

#include "Tuple.h"

namespace Math {
    constexpr double pi = 3.14159265358979323846;
    constexpr double pi_2 = 1.57079632679489661923;
    constexpr double pi_4 = 0.785398163397448309616;
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
}

namespace Color {
    constexpr Tuple black = color(0.0, 0.0, 0.0);
    constexpr Tuple white = color(1.0, 1.0, 1.0);
    constexpr Tuple red = color(1.0, 0.0, 0.0);
    constexpr Tuple green = color(0.0, 1.0, 0.0);
    constexpr Tuple blue = color(0.0, 0.0, 1.0);
}