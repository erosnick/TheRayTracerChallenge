#pragma once

#include "Tuple.h"

constexpr int32_t MAXELEMENTS = 8;

namespace Math {
    constexpr Float pi = 3.1415926535897932f;
    constexpr Float pi_3 = 1.0471975511965976f;
    constexpr Float pi_2 = 1.5707963267948966f;
    constexpr Float pi_4 = 0.7853981633974483f;
    constexpr Float pi_6 = 0.5235987755982988f;
    constexpr Float sqrt_2 = 1.414214f;
    constexpr Float sqrt_3 = 1.732051f;
    constexpr Float cos30d = 0.866025f;
    constexpr Float cos45d = 0.707107f;
    constexpr Float cos60d = 0.5f;
    constexpr Float sin30d = 0.5f;
    constexpr Float sin45d = 0.707107f;
    constexpr Float sin60D = 0.866025f;
    constexpr Float epsilon = 0.000001f;
    constexpr Tuple xAxis = vector(1.0f, 0.0f, 0.0f);
    constexpr Tuple yAxis = vector(0.0f, 1.0f, 0.0f);
    constexpr Tuple zAxis = vector(0.0f, 0.0f, -1.0f);
    //constexpr Float infinityd = std::numeric_limits<Float>::infinity();
    constexpr Float infinity = 10000000.0f;
    //constexpr Float infinity = std::numeric_limits<Float>::infinity();
    //constexpr int32_t infinityi = std::numeric_limits<int32_t>::infinity();

    inline CUDA_HOST_DEVICE Float radians(Float degree) {
        return pi / 180.0f * degree;
    }

    inline CUDA_HOST_DEVICE Float degrees(Float radian) {
        return radian * 180.0f / pi;
    }
}

namespace Color {
    constexpr Float oneOver255 = 1.0f / 255;
    constexpr Tuple black = Tuple(0.0f, 0.0f, 0.0f);
    constexpr Tuple dawn = Tuple(0.1f, 0.1f, 0.1f);
    constexpr Tuple white = Tuple(1.0f, 1.0f, 1.0f);
    constexpr Tuple grey = Tuple(0.5f, 0.5f, 0.5f);
    constexpr Tuple gray = Tuple(0.7f, 0.7f, 0.7f);
    constexpr Tuple red = Tuple(1.0f, 0.0f, 0.0f);
    constexpr Tuple green = Tuple(0.0f, 1.0f, 0.0f);
    constexpr Tuple yellow = Tuple(1.0f, 1.0f, 0.0f);
    constexpr Tuple purple = Tuple(1.0f, 0.0f, 1.0f);
    constexpr Tuple blue = Tuple(0.0f, 0.0f, 1.0f);
    constexpr Tuple pink = Tuple(1.0f, 0.55f, 0.55f);
    constexpr Tuple skyBlue = Tuple(134, 203, 237);
    constexpr Tuple lightCornflower = Tuple(0.5f, 0.7f, 1.0f);
    constexpr Tuple moonstone = Tuple(60, 162, 200);
    constexpr Tuple turquoise = Tuple(64, 224, 208);
    constexpr Tuple limeGreen = Tuple(110, 198, 175);
    constexpr Tuple roseRed = Tuple(0.76f, 0.12f, 0.34f);
    constexpr Tuple crimsonRed = Tuple(0.86f, 0.08f, 0.24f);
    constexpr Tuple lightGreen = Tuple(0.38f, 1.0f, 0.18f);
    constexpr Tuple orange = Tuple(0.85f, 0.49f, 0.32f);
    constexpr Tuple cornflower = Tuple(0.4f, 0.6f, 0.9f);
    constexpr Tuple background = Tuple(0.235294f, 0.67451f, 0.843137f);
    constexpr Tuple White() { return white; }
    constexpr Tuple LightCornflower() { return lightCornflower; }
}