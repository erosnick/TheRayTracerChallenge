#pragma once

#include "Tuple.h"
#include "Matrix.h"
#include "constants.h"
#include "types.h"

#include <cmath>

class Pattern {
public:
    virtual Tuple patternAt(const Tuple& position)  const { return Tuple(); };
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

class StripePattern : public Pattern {
public:
    StripePattern() {}
    StripePattern(const Tuple& inColor1, const Tuple& inColor2) {
        color1 = inColor1;
        color2 = inColor2;
    }

    inline Tuple patternAt(const Tuple& position)  const override {
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

    inline Tuple patternAt(const Tuple& position)  const override {
        if ((std::abs(std::fmod(std::floor(position.x), 2.0)))
         == (std::abs(std::fmod(std::floor(position.z), 2.0)))) {
            return color1;
        }
        else {
            return color2;
        }
    }
};

