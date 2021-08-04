#pragma once

#include <vector>
#include <memory>
#include "Matrix.h"
#include "Material.h"

struct Intersection;

class Shape : public std::enable_shared_from_this<Shape> {
public:
    virtual inline void setTransformation(const Matrix4& inTransformation) {
        transformation = inTransformation;
    }

    virtual inline void transform(const Matrix4& inTransformation) {
        transformation = transformation * inTransformation;
    }

    virtual inline Tuple normalAt(const Tuple& position) const = 0;

    virtual std::vector<Intersection> intersect(const Ray& ray, bool bTransformRay = false) = 0;

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
    std::vector<Intersection> intersect(const Ray& ray, bool bTransformRay = false) override;
};

inline TestShape testShape() {
    auto shape = TestShape();
    return shape;
}