#include "Sphere.h"
#include "Intersection.h"

int32_t Sphere::id = 0;

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
    auto normal1 = (position1 - origin).normalize();

    auto position2 = transformedRay.position(t2);
    auto normal2 = (position2 - origin).normalize();

    if (t1 < t2) {
        return { {true, 1, t1, *this, position1, normal1}, {true, 1, t2, *this, position2, normal2 } };
    }
    else {
        return { {true, 1, t2, *this, position2, normal2 }, {true, 1, t1, *this, position1, normal1 } };
    }
}