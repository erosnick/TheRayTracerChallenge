#pragma once

#include "CUDA.h"

#include "Tuple.h"
#include "Matrix.h"
#include "Types.h"
#include "Material.h"
#include "Array.h"

struct Intersection;

class Shape {
public:
    CUDA_HOST_DEVICE Shape() {
    }

    virtual CUDA_HOST_DEVICE ~Shape() {
        if (material) {
            delete material;
        }
    }

    virtual CUDA_HOST_DEVICE void setPosition(const Tuple& inPosition) {
        position = inPosition;
    }

    virtual CUDA_HOST_DEVICE void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) {
        transformation = worldTransformation = inTransformation;
    }

    virtual CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) {
        transformation = inTransformation * worldTransformation;
    }

    virtual CUDA_HOST_DEVICE void updateTransformation() {}

    virtual CUDA_HOST_DEVICE void transformNormal(const Matrix4& worldMatrix) {}

    virtual inline CUDA_HOST_DEVICE Tuple normalAt(const Tuple& inPosition = point(0.0)) const { return Tuple(); }

    virtual inline CUDA_HOST_DEVICE bool intersect(const Ray& ray, Array<Intersection>& intersections) { return false; }

    virtual inline CUDA_HOST_DEVICE bool intersect(const Ray& ray, Intersection intersections[]) { return false; }

    virtual CUDA_HOST_DEVICE void setMaterial(Material* inMaterial) {
        material = inMaterial;
    }

    Tuple position;
    Tuple transformedPosition;
    Matrix4 worldTransformation;
    Matrix4 transformation;
    Material* material = nullptr;

    bool bIsLight = false;
};