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
        : origin(point(0.0)), position(origin), radius(1.0) {}

    CUDA_HOST_DEVICE Sphere(const Tuple& inOrigin, double inRadius = 1.0)
        : origin(inOrigin), position(origin), radius(inRadius) {
        //transformation[0][3] = origin.x();
        //transformation[1][3] = origin.y();
        //transformation[2][3] = origin.z();
    }

    CUDA_HOST_DEVICE virtual ~Sphere() {
    }

    inline CUDA_HOST_DEVICE Tuple normalAt(const Tuple& inPosition) const override {
        auto normal = (inPosition - position);
        return  normal.normalize();
    }

    CUDA_HOST_DEVICE bool intersect(const Ray& ray, Array<Intersection>& intersections) override;

    CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);

        //origin.x() = transformation[0][3];
        //origin.y() = transformation[1][3];
        //origin.z() = transformation[2][3];
    }

    CUDA_HOST_DEVICE void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) {
        Shape::setTransformation(inTransformation, bTransformPosition);
        position = transformation * origin;
    }

    Tuple origin;;
    Tuple position;
    double radius;
};