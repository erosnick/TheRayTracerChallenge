#pragma once

#include "Tuple.h"
#include "Ray.h"
#include "Matrix.h"
#include "Material.h"

#include <tuple>
#include <vector>

struct Intersection;

class Sphere {
public:
    Sphere() {
        id++;
    }

    Sphere(const Tuple& inOrigin, double inRadius = 1.0) 
    : origin(inOrigin), radius(inRadius) {
        id++;
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
        return (position - origin).normalize();
    }

    inline void setTransform(const Matrix4& inTransform) {
        transform = inTransform;

        origin.x = transform[0][3];
        origin.y = transform[1][3];
        origin.z = transform[2][3];

        scale.x = transform[0][0];
        scale.y = transform[1][1];
        scale.z = transform[2][2];
    }

    std::vector<Intersection> intersect(const Ray& ray, bool bTransformRay = false) const;

    Tuple origin = { 0.0, 0.0, 0.0, 1.0 };
    double radius = 1.0;

    Matrix4 transform = Matrix4();

    Tuple scale = { 1.0, 1.0, 1.0, 0.0 };

    Material material;
    
    static int32_t id;
};

inline bool operator==(const Sphere& s1, const Sphere& s2) {
    return (s1.id == s2.id);
}