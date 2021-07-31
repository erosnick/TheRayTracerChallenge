#pragma once

#include "Intersection.h"
#include "Shape.h"
#include "Sphere.h"

HitInfo prepareComputations(const Intersection& intersection, const Ray& ray) {
    // Instantiate a data structure for storing some precomputed values
    HitInfo hitInfo;

    // Copy the intersection's properties for convenience
    hitInfo.t = intersection.t;
    hitInfo.object = intersection.object;

    // Precompute some useful values
    hitInfo.position = ray.position(hitInfo.t);
    hitInfo.viewDirection = -ray.direction;
    hitInfo.normal = hitInfo.object->normalAt(hitInfo.position);

    if (hitInfo.normal.dot(hitInfo.viewDirection) < 0.0) {
        hitInfo.inside = true;
        hitInfo.normal = -hitInfo.normal;
    }

    hitInfo.overPosition = hitInfo.position + hitInfo.normal * Math::epsilon;

    return hitInfo;
}