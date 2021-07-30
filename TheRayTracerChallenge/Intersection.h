#pragma once

#include "types.h"
#include "constants.h"
#include "Ray.h"
#include <initializer_list>
#include <vector>
#include <memory>

struct HitInfo {
    double t;
    ShapePtr object;
    Tuple position;
    Tuple viewDirection;
    Tuple normal;
    Tuple overPosition;
    bool inside = false;
};

struct Intersection {
    Intersection() {}

    Intersection(double inT, const ShapePtr& inSphere)
        : t(inT), object(inSphere) {
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

HitInfo prepareComputations(const Intersection& intersection, const Ray& ray);