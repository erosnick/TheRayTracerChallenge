#pragma once

#include "Tuple.h"
#include "Ray.h"
#include "Matrix.h"
#include "Material.h"
#include "Shape.h"
#include <tuple>
#include <vector>

class Sphere : public Shape {
public:
    CUDA_HOST_DEVICE Sphere()
        : origin({ 0.0, 0.0, 0.0 }), radius(1.0) {
        position = origin;
    }

    CUDA_HOST_DEVICE Sphere(const Tuple& inOrigin, double inRadius = 1.0)
        : origin(inOrigin), radius(inRadius) {
        position = origin;
    }

    CUDA_HOST_DEVICE virtual ~Sphere() {
    }

    inline CUDA_HOST_DEVICE Tuple normalAt(const Tuple& inPosition) const override {
        auto normal = (inPosition - transformedPosition);
        return  normal.normalize();
    }

    CUDA_HOST_DEVICE bool intersect(const Ray& ray, Array<Intersection>& intersections) override;

    CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);
    }

    CUDA_HOST_DEVICE void updateTransformation() override {
        transformedPosition = transformation * origin;
    }

    CUDA_HOST_DEVICE void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) {
        Shape::setTransformation(inTransformation, bTransformPosition);
    }

    Tuple origin;;
    double radius;
};