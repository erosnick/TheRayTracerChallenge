#pragma once

#include "Sphere.h"
#include "constants.h"
#include <initializer_list>
#include <vector>

struct HitInfo {
    double t;
    Sphere object;
    Tuple position;
    Tuple viewDirection;
    Tuple normal;
    Tuple overPosition;
    bool inside = false;
};

struct Intersection {
    Intersection() {}

    Intersection(double inT, const Sphere& inSphere)
        : t(inT), object(inSphere) {
    }

    Intersection(bool bInHit, int32_t inCount, double inT, const Sphere& inSphere)
        : bHit(bInHit), count(inCount), t(inT), object(inSphere) {
    }

    Intersection(bool bInHit, bool bInShading, int32_t inCount, double inT, const Sphere& inSphere, const Tuple& inPosition, const Tuple& inNormal, const Ray& inRay)
        : bHit(bInHit), bShading(bInShading), count(inCount), t(inT), object(inSphere), position(inPosition), normal(inNormal), ray(inRay) {
    }

    bool bHit = false;
    bool bShading = true;
    int32_t count = 0;
    double t = std::numeric_limits<double>::infinity();
    Sphere object;
    Tuple position;
    Tuple normal;
    Ray ray;
};

inline bool operator==(const Intersection& a, const Intersection& b) {
    return ((a.bHit == b.bHit)
         && (a.count == b.count)
         && (a.t == b.t)
         && (a.object == b.object));
}

inline bool operator<(const Intersection& a, const Intersection& b) {
    return (a.t < b.t);
}

inline std::vector<Intersection> intersections(const std::initializer_list<Intersection>& args) {
    auto records = std::vector<Intersection>();
    for (const auto& element : args) {
        records.push_back(element);
    }
    return records;
}

inline Intersection hit(const std::vector<Intersection>& records) {
    auto result = Intersection();

    for (const auto& record : records) {
        if ((record.t > 0.0) && (record.t < result.t)) {
            result = record;
        }
    }

    return result;
}

inline HitInfo prepareComputations(const Intersection& intersection, const Ray& ray) {
    // Instantiate a data structure for storing some precomputed values
    HitInfo hitInfo;

    // Copy the intersection's properties for convenience
    hitInfo.t = intersection.t;
    hitInfo.object = intersection.object;

    // Precompute some useful values
    hitInfo.position = ray.position(hitInfo.t);
    hitInfo.viewDirection = -ray.direction;
    hitInfo.normal = hitInfo.object.normalAt(hitInfo.position);

    if (hitInfo.normal.dot(hitInfo.viewDirection) < 0.0) {
        hitInfo.inside = true;
        hitInfo.normal = -hitInfo.normal;
    }

    hitInfo.overPosition = hitInfo.position + hitInfo.normal * EPSILON;

    return hitInfo;
}