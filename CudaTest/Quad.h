#pragma once

#include "Types.h"
#include "Shape.h"
#include "Triangle.h"

class Quad : public Shape {
public:
    CUDA_HOST_DEVICE Quad(const char* inName = "Quad");
    CUDA_HOST_DEVICE virtual ~Quad();

    CUDA_HOST_DEVICE void setTransformation(const Matrix4& inTransformation, bool bTransformPosition  = false ) override;

    CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) override;

    CUDA_HOST_DEVICE void transformNormal(const Matrix4& worldMatrix);

    CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position = point(0.0)) const override;

    CUDA_HOST_DEVICE bool intersect(const Ray& ray, Intersection* intersections) override;

    CUDA_HOST_DEVICE bool onQuad(const Tuple& inPosition, Tuple& normal);

    CUDA_HOST_DEVICE void setMaterial(Material* inMaterial) override;

    bool bCube = false;

//private:
    Triangle* triangles[2];
    const char* name;
};