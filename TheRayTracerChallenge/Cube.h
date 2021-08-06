#pragma once

#include "Shape.h"
#include "types.h"

class Cube : public Shape {
public:
    Cube() {
        initPlanes();
    }
    virtual  ~Cube() {}

    void setTransformation(const Matrix4& inTransformation) override;

    void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);
    }

    Tuple normalAt(const Tuple& position) const override;

    InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) override;

    std::tuple<double, double> checkAxis(double origin, double direction);

    void initPlanes();

    std::vector<std::shared_ptr<Plane>> planes;

    Tuple center = point(0.0, 0.0, 0.0);
};