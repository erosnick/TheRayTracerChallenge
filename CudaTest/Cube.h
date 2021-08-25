#pragma once

#include "Shape.h"
#include "Types.h"
#include "Array.h"

class Cube : public Shape {
public:
    CUDA_HOST_DEVICE Cube();

    virtual CUDA_HOST_DEVICE  ~Cube() {}

    CUDA_HOST_DEVICE void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) override;

    CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) override;

    CUDA_HOST_DEVICE void updateTransformation() override;

    CUDA_HOST_DEVICE void transformNormal(const Matrix4& worldMatrix);

    CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position) const override;

    CUDA_HOST_DEVICE bool intersect(const Ray& ray, Array<Intersection>& intersections) override;

    //CUDA_HOST_DEVICE std::tuple<double, double> checkAxis(double origin, double direction);

    CUDA_HOST_DEVICE void initQuads();

    CUDA_HOST_DEVICE void setMaterial(Material* inMaterial);

    CUDA_HOST_DEVICE void setMaterial(Material* inMaterial, int32_t quadIndex);

    Array<Quad*> quads;

    Tuple normal = vector(0.0);

    Tuple center = point(0.0);
};