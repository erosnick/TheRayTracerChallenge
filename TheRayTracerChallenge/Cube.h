#pragma once

#include "Shape.h"
#include "types.h"

class Cube : public Shape {
public:
    Cube();

    virtual  ~Cube() {}

    void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) override;

    void transform(const Matrix4& inTransformation) override;

    void transformNormal(const Matrix4& worldMatrix);

    Tuple normalAt(const Tuple& position) const override;

    InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) override;

    std::tuple<double, double> checkAxis(double origin, double direction);

    void initQuads();

    void setMaterial(const MaterialPtr& inMaterial);

    void setMaterial(const MaterialPtr& inMaterial, int32_t quadIndex);

    std::vector<std::shared_ptr<Quad>> quads;

    Tuple center = point(0.0, 0.0, 0.0);
};