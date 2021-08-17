#pragma once

#include "Tuple.h"
#include "Matrix.h"
#include "Constants.h"
#include "Types.h"

#include <cmath>

class Pattern {
public:
    virtual Tuple patternAt(const Tuple& position) const { return Tuple(); };
    virtual Tuple patternAtShape(const ShapePtr& shape, const Tuple& position) const;

    virtual void transform(const Matrix4& inTransformation) {
        transformation = inTransformation * transformation;
    }

    virtual void setTransformation(const Matrix4& inTransformation) {
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
    CheckerPattern(const Tuple& inColor1, const Tuple& inColor2, 
                   const PlaneOrientation& inPlaneOrientation = PlaneOrientation::XZ, double inScale = 1.0) {
        color1 = inColor1;
        color2 = inColor2;
        planeOrientation = inPlaneOrientation;
        scale = 1.0 / inScale;
    }

    inline Tuple patternAt(const Tuple& position) const override {
        double sum = 0.0;
        switch (planeOrientation)
        {
        case PlaneOrientation::XY:
            sum = std::floor(position.x * scale) + std::floor(position.y * scale);
            break;
        case PlaneOrientation::YZ:
            sum = std::floor(position.y * scale) + std::floor(position.z * scale);
            break;
        case PlaneOrientation::XZ:
            sum = std::floor(position.x * scale) + std::floor(position.z * scale);
            break;
        default:
            break;
        }
        //auto sum = std::floor(position.x) + std::floor(position.y) + std::floor(position.z);
        
        if (std::fmod(sum, 2.0) == 0.0) {
            return color1;
        }
        else {
            return color2;
        }
    }

    double scale = 1.0;
    Tuple transformedPosition;
    PlaneOrientation planeOrientation = PlaneOrientation::XZ;
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