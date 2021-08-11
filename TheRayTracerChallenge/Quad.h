#pragma once

#include "Shape.h"
#include "types.h"

#include <string>

class Quad : public Shape {
public:

    Quad(const std::string& inName = "Quad");
    virtual ~Quad() {}

    void setTransformation(const Matrix4& inTransformation, bool bTransformPosition  = false ) override;

    void transform(const Matrix4& inTransformation) override;

    void transformNormal(const Matrix4& worldMatrix);

    Tuple normalAt(const Tuple& position) const override;

    InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) override;

    bool onQuad(const Tuple& inPosition, Tuple& normal);

    void setMaterial(const MaterialPtr& inMaterial) override;

private:
    std::vector<TrianglePtr> triangles;
    std::string name;
};