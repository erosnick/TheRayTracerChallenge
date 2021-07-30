#pragma once

#include "Tuple.h"
#include "Ray.h"
#include "Matrix.h"
#include "Material.h"

#include <tuple>
#include <vector>

struct Intersection;

class Sphere : public Object {
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

    inline Tuple normalAt(const Tuple& position) const {
        auto normal = (position - origin);
        //normal = (transform.inverse()).transpose() * normal;
        normal.w = 0.0;
        return  normal.normalize();
    }

    inline void transform(const Matrix4& inTransformation) {
        transformation = inTransformation * transformation;

        origin.x = transformation[0][3];
        origin.y = transformation[1][3];
        origin.z = transformation[2][3];
    }

    inline void scaling(double x, double y, double z) {
        scale.x = x;
        scale.y = y;
        scale.z = z;

        radius = scale.x;
    }

    inline void setTransform(const Matrix4& inTransformation) {
        transformation = inTransformation;

        origin.x = transformation[0][3];
        origin.y = transformation[1][3];
        origin.z = transformation[2][3];
    }

    std::vector<Intersection> intersect(const Ray& ray, bool bTransformRay = false) const;

    Tuple origin = { 0.0, 0.0, 0.0, 1.0 };
    double radius = 1.0;

    Matrix4 transformation = Matrix4();

    Tuple scale = { 1.0, 1.0, 1.0, 0.0 };

    Material material;

    bool bIsLight = false;
};

inline bool operator==(const Sphere& a, const Sphere& b) {
    return ((a.origin == b.origin)
         && (a.radius == b.radius)
         && (a.transformation == b.transformation)
         && (a.material == b.material));
}