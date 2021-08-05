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
    Sphere() {
    }

    Sphere(const Tuple& inOrigin, double inRadius = 1.0)
        : origin(inOrigin), radius(inRadius) {
        scaling(radius, radius, radius);
        transformation[0][3] = origin.x;
        transformation[1][3] = origin.y;
        transformation[2][3] = origin.z;
    }

    //inline std::tuple<bool, int32_t, double, double> intersect(const Ray& ray) {
    //    auto oc = (ray.origin - origin);
    //    auto a = ray.direction.dot(ray.direction);
    //    auto b = 2.0 * ray.direction.dot(oc);
    //    auto c = oc.dot(oc) - radius * radius;

    //    auto discriminant = b * b - 4 * a * c;

    //    if (discriminant < 0.0) {
    //        return { false, 0, std::numeric_limits<double>::infinity(),
    //                        std::numeric_limits<double>::infinity() };
    //    }

    //    auto t1 = (-b - std::sqrt(discriminant)) / (2 * a);
    //    auto t2 = (-b + std::sqrt(discriminant)) / (2 * a);

    //    if (t1 > t2) {
    //        return { true, 2, t2, t1 };
    //    }
    //    else {
    //        return { true, 2, t1, t2 };
    //    }
    //}

    Tuple normalAt(const Tuple& position) const override {
        auto normal = (position - origin);
        //normal = (transform.inverse()).transpose() * normal;
        normal.w = 0.0;
        return  normal.normalize();
    }

    void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);

        origin.x = transformation[0][3];
        origin.y = transformation[1][3];
        origin.z = transformation[2][3];
    }

    void scaling(double x, double y, double z) {
        scale.x = x;
        scale.y = y;
        scale.z = z;

        radius = scale.x;
    }

    void setTransformation(const Matrix4& inTransformation) override {
        transformation = inTransformation;

        origin.x = transformation[0][3];
        origin.y = transformation[1][3];
        origin.z = transformation[2][3];
    }

    std::vector<Intersection> intersect(const Ray& ray, bool bTransformRay = false)  override;

    Tuple origin = { 0.0, 0.0, 0.0, 1.0 };
    double radius = 1.0;

    Tuple scale = { 1.0, 1.0, 1.0, 0.0 };
};

inline bool operator==(const Sphere& a, const Sphere& b) {
    return ((a.origin == b.origin)
         && (a.radius == b.radius)
         && (a.transformation == b.transformation)
         && (a.material == b.material));
}

inline SpherePtr glassSphere() {
    auto sphere = std::make_shared<Sphere>();

    sphere->material.transparency = 1.0;
    sphere->material.refractiveIndex = 1.5;

    return sphere;
}