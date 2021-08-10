#pragma once

#include <vector>
#include <memory>
#include "Matrix.h"
#include "types.h"

struct Intersection;

class Shape : public std::enable_shared_from_this<Shape> {
public:
    Shape();

    virtual void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false);

    virtual void transform(const Matrix4& inTransformation) {
        transformation = inTransformation * transformation;
    }

    virtual Tuple normalAt(const Tuple& position) const = 0;

    virtual InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) = 0;

    virtual void setMaterial(const MaterialPtr& inMaterial) {
        material = inMaterial;
    }

    std::shared_ptr<Shape> GetPtr() {
        return shared_from_this();
    }

    Matrix4 transformation;
    MaterialPtr material;

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