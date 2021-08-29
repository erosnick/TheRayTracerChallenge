#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Matrix.h"

class Pattern {
public:
    virtual CUDA_HOST_DEVICE Tuple patternAt(const Tuple& position) const { return Tuple(); };
    virtual CUDA_HOST_DEVICE Tuple patternAtShape(Shape* shape, const Tuple& position) const;

    virtual CUDA_HOST_DEVICE void setTransformation(const Matrix4& inTransformation) {
        worldTransformation = inTransformation;
    }

    virtual CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) {
        transformation = inTransformation * worldTransformation;
    }

    Matrix4 worldTransformation;
    Matrix4 transformation;
    Tuple color1 = Color::white;
    Tuple color2 = Color::black;
};

class TestPattern : public Pattern {
public:
    CUDA_HOST_DEVICE Tuple patternAt(const Tuple& position) const override {
        return color(position.x(), position.y(), position.z());
    }
};

class StripePattern : public Pattern {
public:
    CUDA_HOST_DEVICE StripePattern() {}
    CUDA_HOST_DEVICE StripePattern(const Tuple& inColor1, const Tuple& inColor2) {
        color1 = inColor1;
        color2 = inColor2;
    }

    inline CUDA_HOST_DEVICE Tuple patternAt(const Tuple& position) const override {
        if ((std::fmod(std::floor(position.x()), 2.0) == 0.0)) {
            return color1;
        }
        
        return color2;
    }
};

class CheckerPattern : public Pattern {
public:
    CUDA_HOST_DEVICE CheckerPattern() {}
    CUDA_HOST_DEVICE CheckerPattern(const Tuple& inColor1, const Tuple& inColor2,
                   const PlaneOrientation& inPlaneOrientation = PlaneOrientation::XZ, Float inScale = 1.0) {
        color1 = inColor1;
        color2 = inColor2;
        planeOrientation = inPlaneOrientation;
        scale = 1.0 / inScale;
    }

    inline CUDA_HOST_DEVICE Tuple patternAt(const Tuple& position) const override {
        Float sum = 0.0;
        switch (planeOrientation)
        {
        case PlaneOrientation::XY:
            sum = floor(position.x() * scale) + floor(position.y() * scale);
            break;
        case PlaneOrientation::YZ:
            sum = floor(position.y() * scale) + floor(position.z() * scale);
            break;
        case PlaneOrientation::XZ:
            sum = floor(position.x() * scale) + floor(position.z() * scale);
            break;
        default:
            break;
        }
        
        if (fmod(sum, 2.0) == 0.0) {
            return color1;
        }

        return color2;
    }

    Float scale = 1.0;
    Tuple transformedPosition;
    PlaneOrientation planeOrientation = PlaneOrientation::XZ;
};

class GradientPattern : public Pattern {
public:
    CUDA_HOST_DEVICE GradientPattern() {}
    CUDA_HOST_DEVICE GradientPattern(const Tuple& inColor1, const Tuple& inColor2) {
        color1 = inColor1;
        color2 = inColor2;
    }

    // color(p, ca, cb) = ca + (cb - ca) * (px - floor(px))
    inline CUDA_HOST_DEVICE Tuple patternAt(const Tuple& position) const override {
        auto distance = color2 - color1;
        auto fraction = (position.x() - std::floor(position.x()));
        //auto fraction = position.x + 0.5;
        //return color1 + distance * fraction;
        return color1 * (1.0 - fraction) + color2 * fraction;
    }
};

class RingPattern : public Pattern {
public:
    CUDA_HOST_DEVICE RingPattern() {}
    CUDA_HOST_DEVICE RingPattern(const Tuple& inColor1, const Tuple& inColor2) {
        color1 = inColor1;
        color2 = inColor2;
    }

    inline Tuple CUDA_HOST_DEVICE patternAt(const Tuple& position) const override {
        auto distance = std::sqrt(position.x() * position.x() + position.z() * position.z());
    
        if (std::fmod(std::floor(distance), 2.0) == 0.0) {
            return color1;
        }

        return color2;
    }
};