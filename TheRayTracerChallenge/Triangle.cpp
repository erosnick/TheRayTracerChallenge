#include "Triangle.h"
#include "Intersection.h"

InsersectionSet Triangle::intersect(const Ray& ray, bool bTransformRay) {
    // TODO: Implement this function that tests whether the triangle
    // that's specified by v0, v1 and v2 intersects with the ray (whose
    // origin is *orig* and direction is *dir*)
    // Also don't forget to update tnear, u and v.
    auto intersections = InsersectionSet();
    Tuple E1 = v1 - v0;
    Tuple E2 = v2 - v0;
    Tuple S = ray.origin - v0;
    Tuple S1 = ray.direction.cross(E2);
    Tuple S2 = S.cross(E1);
    double coefficient = 1.0f / S1.dot(E1);
    double t = coefficient * S2.dot(E2);
    double b1 = coefficient * S1.dot(S);
    double b2 = coefficient * S2.dot(ray.direction);

    // Constrains:
    // 1 ~ 4 must be satisfied at the same time
    // 1.t must be greater than or equal to 0
    // 2.u must be a non-negative number with a 
    // value less than or equal to 1.
    // 3.v must be a non-negative number with a 
    // value less than or equal to 1.
    // 4.v + u must be a number less than or equal to 1
    // 理论上来说只返回射线前进方向上的交点(t > 0)，这里
    // 返回特殊情况下t < 0的交点(射线在Cube内时产生的两个交点)，
    // 计算折射率时候需要用到，和Sphere的原理类似
    if ((t > -Math::epsilon * 2
        && t < 0.0)
        || (t >= Math::epsilon)
        && (b1 >= Math::epsilon)
        && (b2 >= Math::epsilon)
        && ((1.0f - b1 - b2) >= Math::epsilon)) {
        auto intersection = Intersection();
        intersection.t = t;
        intersection.object = GetPtr();
        intersection.ray = ray;
        intersection.position = ray.position(t);
        intersections.push_back(intersection);
    }

    //if ((t >= Math::epsilon)
    //    && (b1 >= Math::epsilon)
    //    && (b2 >= Math::epsilon)
    //    && ((1.0f - b1 - b2) >= Math::epsilon)) {
    //    auto intersection = Intersection();
    //    intersection.t = t;
    //    intersection.object = GetPtr();
    //    intersection.ray = ray;
    //    intersection.position = ray.position(t);
    //    intersections.push_back(intersection);
    //}

    return intersections;
}
