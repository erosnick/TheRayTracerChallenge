#pragma once

#include "Tuple.h"
#include "Matrix.h"
#include "constants.h"
#include "types.h"

#include <cmath>

class Pattern {
public:
    virtual Tuple patternAt(const Tuple& position) const { return Tuple(); };
    virtual Tuple patternAtShape(const ShapePtr& shape, const Tuple& position) const;

    virtual inline void transform(const Matrix4& inTransformation) {
        transformation = inTransformation * transformation;
    }

    virtual inline void setTransformation(const Matrix4& inTransformation) {
        transformation = inTransformation;
    }

    Matrix4 transformation;
    Tuple color1 = Color::white;
    Tuple color2 = Color::black;
};

class TestPattern : public Pattern {
public:
    Tuple patternAt(const Tuple& position) const override {
        return color(position.x, position.y, position.z);
    }
};

class StripePattern : public Pattern {
public:
    StripePattern() {}
    StripePattern(const Tuple& inColor1, const Tuple& inColor2) {
        color1 = inColor1;
        color2 = inColor2;
    }

    inline Tuple patternAt(const Tuple& position) const override {
        if ((std::fmod(std::floor(position.x), 2.0) == 0.0)) {
            return color1;
        }
        else {
            return color2;
        }
    }
};

class CheckerPattern : public Pattern {
public:
    CheckerPattern() {}
    CheckerPattern(const Tuple& inColor1, const Tuple& inColor2) {
        color1 = inColor1;
        color2 = inColor2;
    }

    inline Tuple patternAt(const Tuple& position) const override {
        auto sum = std::floor(position.x) + std::floor(position.y) + std::floor(position.z);
        //auto sum = std::floor(position.x) + std::floor(position.z);
        if (std::fmod(sum, 2.0) == 0.0) {
            return color1;
        }
        else {
            return color2;
        }
    }
};

class GradientPattern : public Pattern {
public:
    GradientPattern() {}
    GradientPattern(const Tuple& inColor1, const Tuple& inColor2) {
        color1 = inColor1;
        color2 = inColor2;
    }

    // color(p, ca, cb) = ca + (cb - ca) * (px - floor(px))
    inline Tuple patternAt(const Tuple& position) const override {
        auto distance = color2 - color1;
        auto fraction = (position.x - std::floor(position.x));
        //auto fraction = position.x + 0.5;
        //return color1 + distance * fraction;
        return color1 * (1.0 - fraction) + color2 * fraction;
    }
};

class RingPattern : public Pattern {
public:
    RingPattern() {}
    RingPattern(const Tuple& inColor1, const Tuple& inColor2) {
        color1 = inColor1;
        color2 = inColor2;
    }

    inline Tuple patternAt(const Tuple& position) const override {
        auto distance = std::sqrt(position.x * position.x + position.z * position.z);
    
        if (std::fmod(std::floor(distance), 2.0) == 0.0) {
            return color1;
        }

        return color2;
    }
};