#pragma once

#include "Shape.h"

class Triangle : public Shape {
public:
    Triangle() {}
    Triangle(const Tuple& inV0, const Tuple& inV1, const Tuple& inV2) 
    : v0(inV0), v1(inV1), v2(inV2) {}

    void setTransformation(const Matrix4& inTransformation) override;

    Tuple normalAt(const Tuple& position) const override;

    std::vector<Intersection> intersect(const Ray& ray, bool bTransformRay = false) override;

    Tuple v0 = point(0.0, 1.0, 0.0);
    Tuple v1 = point(-1.0, 0.0, 0.0);
    Tuple v2 = point(1.0, 0.0, 0.0);
};