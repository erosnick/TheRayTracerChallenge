#include "Sphere.h"
#include "Intersection.h"

std::vector<Intersection> Sphere::intersect(const Ray& ray, bool bTransformRay) const {
    auto intersections = std::vector<Intersection>();
    auto transformedRay = ray;

    if (bTransformRay) {
        transformedRay = transformRay(ray, transformation.inverse());
    }

    auto oc = (transformedRay.origin - origin);
    auto a = transformedRay.direction.dot(transformedRay.direction);
    auto b = 2.0 * transformedRay.direction.dot(oc);
    auto c = oc.dot(oc) - radius * radius;

    auto discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0) {
        return intersections;
    }

    // 与巨大球体求交的时候，会出现判别式大于0，但是有两个负根的情况，
    // 这种情况出现在射线方向的反向延长线能和球体相交的场合。
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

    if (t1 > 0.0 && t2 > 0.0) {
        if (t1 < t2) {
            intersections.push_back({ true, 1, t1, *this, position1, normal1, transformedRay });
            intersections.push_back({ true, 1, t2, *this, position2, normal2, transformedRay });
        }
        else {
            intersections.push_back({ true, 1, t2, *this, position2, normal2, transformedRay });
            intersections.push_back({ true, 1, t1, *this, position1, normal1, transformedRay });
        }
    }

    return intersections;
}