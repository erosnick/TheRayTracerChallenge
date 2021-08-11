#pragma once

#include "Shape.h"

class Triangle : public Shape {
public:
    Triangle() {
        computeNormal();
    }

    Triangle(const Tuple& inV0, const Tuple& inV1, const Tuple& inV2) 
    : v0(inV0), v1(inV1), v2(inV2) {
        computeNormal();
    }

    inline void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) override {
        Shape::setTransformation(inTransformation);

        v0 = transformation * v0;
        v1 = transformation * v1;
        v2 = transformation * v2;
    }

    inline void transformNormal(const Matrix4& worldMatrix) {
        normal = worldMatrix * normal;
    }

    inline void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);

        v0 = inTransformation * v0;
        v1 = inTransformation * v1;
        v2 = inTransformation * v2;
    };

    inline void computeNormal() {
        auto v0v1 = v1 - v0;
        auto v0v2 = v2 - v0;

        normal = v0v1.cross(v0v2).normalize();
    }

    inline Tuple normalAt(const Tuple& position) const override {
        return normal;
    };

    InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) override;

    Tuple v0 = point(0.0, 1.0, 0.0);
    Tuple v1 = point(-1.0, 0.0, 0.0);
    Tuple v2 = point(1.0, 0.0, 0.0);
    Tuple normal = vector(0.0);
};