#pragma once

#include "Shape.h"
#include "types.h"

class Cube : public Shape {
public:
    Cube() {
    }

    virtual  ~Cube() {}

    void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) override;

    void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);
    }

    Tuple normalAt(const Tuple& position) const override;

    InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) override;

    std::tuple<double, double> checkAxis(double origin, double direction);

    void initQuads();

    std::vector<std::shared_ptr<Quad>> planes;

    Tuple center = point(0.0, 0.0, 0.0);
};