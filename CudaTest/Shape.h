#pragma once

#include "CUDA.h"

#include <vector>
#include <memory>
#include "Tuple.h"
#include "Matrix.h"
#include "Types.h"

struct Intersection;

class Shape {
public:
    CUDA_HOST_DEVICE Shape() {}

    virtual CUDA_HOST_DEVICE ~Shape() {}

    virtual void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) {
        transformation = inTransformation;
    }

    virtual CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) {
        transformation = inTransformation * transformation;
    }

    virtual inline CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position = point(0.0)) const { return Tuple(); }

    virtual inline CUDA_HOST_DEVICE bool intersect(const Ray& ray, Intersection* intersections) { return false; }

    virtual void setMaterial(Material* inMaterial) {
        material = inMaterial;
    }

    Matrix4 transformation;
    Material* material;

    bool bIsLight = false;
};

class TestShape : public Shape
{
public:
    CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position = point(0.0)) const override;
    CUDA_HOST_DEVICE bool intersect(const Ray& ray, Intersection* intersections) override { return false; }
};

inline TestShape testShape() {
    auto shape = TestShape();
    return shape;
}