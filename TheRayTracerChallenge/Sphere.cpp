#include "Sphere.h"
#include "Intersection.h"

std::vector<Intersection> Sphere::intersect(const Ray& ray, bool bTransformRay) const {
    auto transformedRay = ray;

    if (bTransformRay) {
        transformedRay = transformRay(ray, transform.inverse());
    }

    auto oc = (transformedRay.origin - origin);
    auto a = transformedRay.direction.dot(transformedRay.direction);
    auto b = 2.0 * transformedRay.direction.dot(oc);
    auto c = oc.dot(oc) - radius * radius;

    auto discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0) {
        return std::vector<Intersection>();
    }

    auto t1 = (-b - std::sqrt(discriminant)) / (2 * a);
    auto t2 = (-b + std::sqrt(discriminant)) / (2 * a);

    auto position1 = transformedRay.position(t1);

    //position1.x *= scale.x;
    //position1.y *= scale.y;
    //position1.z *= scale.z;

    auto normal1 = normalAt(position1);

    auto position2 = transformedRay.position(t2);

    //position2.x *= scale.x;
    //position2.y *= scale.y;
    //position2.z *= scale.z;

    auto normal2 = normalAt(position2);

    if (t1 < t2) {
        return { {true, 1, t1, *this, position1, normal1}, {true, 1, t2, *this, position2, normal2 } };
    }
    else {
        return { {true, 1, t2, *this, position2, normal2 }, {true, 1, t1, *this, position1, normal1 } };
    }
}