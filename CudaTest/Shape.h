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

    virtual CUDA_HOST_DEVICE void foo() {}

    virtual inline CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position = point(0.0)) const { return Tuple(); }

    virtual inline CUDA_HOST_DEVICE void intersect(const Ray& ray, Intersection* intersections) {}

    virtual void setMaterial(Material* inMaterial) {
        //material = inMaterial;
    }

    Matrix4 transformation;
    Material* material;

    bool bIsLight = false;
};

class TestShape : public Shape
{
public:
    virtual CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position = point(0.0)) const override;
    //CUDA_HOST_DEVICE IntersectionSet intersect(const Ray& ray, bool bTransformRay = false) override;
    //CUDA_HOST_DEVICE Intersection* intersectCUDA(const Ray& ray, bool bTransformRay = false) override { return nullptr; };
};

inline TestShape testShape() {
    auto shape = TestShape();
    return shape;
}