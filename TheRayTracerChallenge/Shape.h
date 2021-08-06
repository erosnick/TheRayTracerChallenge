#pragma once

#include <vector>
#include <memory>
#include "Matrix.h"
#include "Material.h"

struct Intersection;

class Shape : public std::enable_shared_from_this<Shape> {
public:
    virtual void setTransformation(const Matrix4& inTransformation);

    virtual void transform(const Matrix4& inTransformation) {
        transformation = transformation * inTransformation;
    }

    virtual Tuple normalAt(const Tuple& position) const = 0;

    virtual InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) = 0;

    std::shared_ptr<Shape> GetPtr() {
        return shared_from_this();
    }

    Matrix4 transformation;
    Material material;

    bool bIsLight = false;
};

class TestShape : public Shape
{
public:
    virtual Tuple normalAt(const Tuple& position) const override;
    InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) override;
};

inline TestShape testShape() {
    auto shape = TestShape();
    return shape;
}