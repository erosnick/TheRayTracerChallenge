#pragma once

#include "types.h"
#include "constants.h"
#include "Ray.h"
#include <initializer_list>
#include <vector>
#include <memory>
#include <tuple>

struct HitInfo {
    double t;
    ShapePtr object = nullptr;
    Tuple position;
    Tuple viewDirection;
    Tuple normal;
    Tuple overPosition;
    Tuple underPosition;
    Tuple reflectVector;
    double n1 = 1.0;
    double n2 = 1.0;
    bool bInside = false;
};

struct Intersection {
    Intersection() {}

    Intersection(double inT, const ShapePtr& inSphere, const Ray& inRay = Ray())
        : t(inT), object(inSphere), ray(inRay) {
    }

    Intersection(bool bInHit, int32_t inCount, double inT, const ShapePtr& inSphere)
        : bHit(bInHit), count(inCount), t(inT), object(inSphere) {
    }

    Intersection(bool bInHit, bool bInShading, int32_t inCount, double inT, const ShapePtr& inSphere, const Tuple& inPosition, const Tuple& inNormal, const Ray& inRay)
        : bHit(bInHit), bShading(bInShading), count(inCount), t(inT), object(inSphere), position(inPosition), normal(inNormal), ray(inRay) {
    }

    bool bHit = false;
    bool bShading = true;
    int32_t count = 0;
    double t = std::numeric_limits<double>::infinity();
    ShapePtr object;
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

inline InsersectionSet intersections(const std::initializer_list<Intersection>& args) {
    auto records = InsersectionSet();
    for (const auto& element : args) {
        records.push_back(element);
    }
    return records;
}

inline Intersection nearestHit(const InsersectionSet& records) {
    auto result = Intersection();

    for (const auto& record : records) {
        if ((record.t > 0.0) && (record.t < result.t)) {
            result = record;
        }
    }

    return result;
}

HitInfo prepareComputations(const Intersection& hit, const Ray& ray,
                            const InsersectionSet& intersections = InsersectionSet());