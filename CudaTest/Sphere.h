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
    CUDA_HOST_DEVICE Sphere() {}

    CUDA_HOST_DEVICE Sphere(const Tuple& inOrigin, double inRadius = 1.0)
        : origin(inOrigin), radius(inRadius) {
        //transformation[0][3] = origin.x();
        //transformation[1][3] = origin.y();
        //transformation[2][3] = origin.z();
    }

    CUDA_HOST_DEVICE virtual ~Sphere() {}

    CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position) const override {
        auto normal = (position - origin);
        normal.w() = 0.0;
        return  normal.normalize();
    }

    CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);

        //origin.x() = transformation[0][3];
        //origin.y() = transformation[1][3];
        //origin.z() = transformation[2][3];
    }

    void scaling(double x, double y, double z) {
        scale.x() = x;
        scale.y() = y;
        scale.z() = z;

        radius = scale.x();
    }

    void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) override;

    CUDA_HOST_DEVICE void intersect(const Ray& ray, Intersection* intersections) override;

    Tuple origin = { 0.0, 0.0, 0.0, 1.0 };
    double radius = 1.0;

    Tuple scale = { 1.0, 1.0, 1.0, 0.0 };
};

inline bool operator==(const Sphere& a, const Sphere& b) {
    return ((a.origin == b.origin)
         && (a.radius == b.radius)
         ////&& (a.transformation == b.transformation)
         /*&& (a.material == b.material)*/);
}

inline SpherePtr glassSphere() {
    auto sphere = std::make_shared<Sphere>();

    //sphere->material->transparency = 1.0;
    //sphere->material->refractiveIndex = 1.5;

    return sphere;
}