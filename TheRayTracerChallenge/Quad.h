#pragma once

#include "Shape.h"
#include "types.h"

class Quad : public Shape {
public:

    Quad();
    virtual ~Quad() {}

    void setTransformation(const Matrix4& inTransformation, bool bTransformPosition  = false ) override;

    Tuple normalAt(const Tuple& position) const override;

    InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) override;

private:
    std::vector<TrianglePtr> triangles;
};