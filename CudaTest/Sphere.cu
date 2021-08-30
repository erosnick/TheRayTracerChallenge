#include "Sphere.h"
#include "Intersection.h"

CUDA_HOST_DEVICE bool Sphere::intersect(const Ray& ray, Array<Intersection>& intersections) {
    auto oc = (ray.origin - transformedPosition);
    auto a = ray.direction.dot(ray.direction);
    auto b = 2.0f * ray.direction.dot(oc);
    auto c = oc.dot(oc) - radius * radius;

    auto discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0) {
        return false;
    }

    // 与巨大球体求交的时候，会出现判别式大于0，但是有两个负根的情况，
    // 这种情况出现在射线方向的反向延长线能和球体相交的场合。
    Float t1 = (-b - sqrt(discriminant)) / (2 * a);
    Float t2 = (-b + sqrt(discriminant)) / (2 * a);

    if ((t1 > Math::epsilon) || (t2 > Math::epsilon)) {
        intersections.add({ true, true, t1, this });
        intersections.add({ true, true, t2, this });
        return true;
    }

    return false;
}

CUDA_HOST_DEVICE bool Sphere::intersect(const Ray& ray, Intersection intersections[]) {
    auto oc = (ray.origin - transformedPosition);
    auto a = ray.direction.dot(ray.direction);
    auto b = 2.0 * ray.direction.dot(oc);
    auto c = oc.dot(oc) - radius * radius;

    auto discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0) {
        return false;
    }

    // 与巨大球体求交的时候，会出现判别式大于0，但是有两个负根的情况，
    // 这种情况出现在射线方向的反向延长线能和球体相交的场合。
    Float t1 = (-b - sqrt(discriminant)) / (2 * a);
    Float t2 = (-b + sqrt(discriminant)) / (2 * a);

    if ((t1 > Math::epsilon) || (t2 > Math::epsilon)) {
        intersections[0] = { true, true, t1, this };
        intersections[1] = { true, true, t2, this };
        return true;
    }

    return false;
}
