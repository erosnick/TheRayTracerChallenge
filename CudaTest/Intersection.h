#pragma once

#include "CUDA.h"

#include "Types.h"
#include "Constants.h"
#include "Ray.h"
#include <initializer_list>
#include <vector>
#include <memory>
#include <tuple>

struct HitInfo {
    double t;
    Shape* object = nullptr;
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
    CUDA_HOST_DEVICE Intersection()
    : t(0.0), object(nullptr), subObject(nullptr) {}

    CUDA_HOST_DEVICE ~Intersection() {}
     
    CUDA_DEVICE Intersection(double inT, Shape* inSphere, const Ray& inRay = Ray())
    : t(inT), object(inSphere), subObject(nullptr), ray(inRay) {
    }

    CUDA_DEVICE Intersection(bool bInHit, int32_t inCount, double inT, Shape* inSphere)
    : bHit(bInHit), count(inCount), t(inT), object(inSphere), subObject(nullptr) {
    }

    CUDA_DEVICE Intersection(bool bInHit, bool bInShading, int32_t inCount, double inT, Shape* inSphere, const Tuple& inPosition, const Tuple& inNormal, const Ray& inRay)
    : bHit(bInHit), bShading(bInShading), count(inCount), t(inT), object(inSphere), subObject(nullptr), position(inPosition), normal(inNormal), ray(inRay) {
    }

    bool bHit = false;
    bool bShading = true;
    int32_t count = 0;
    double t = std::numeric_limits<double>::infinity();
    Shape* subObject;
    Shape* object;
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

inline IntersectionSet intersections(const std::initializer_list<Intersection>& args) {
    auto records = IntersectionSet();
    for (const auto& element : args) {
        records.push_back(element);
    }
    return records;
}

inline Intersection nearestHit(const IntersectionSet& records) {
    auto result = Intersection();

    for (const auto& record : records) {
        if ((record.t > 0.0) && (record.t < result.t)) {
            result = record;
        }
    }

    return result;
}

inline CUDA_HOST_DEVICE Intersection nearestHitCUDA(Intersection* intersections, int32_t count) {
    auto result = Intersection();

    for (auto i = 0; i < count; i++) {
        if ((intersections[i].t > 0.0) && (intersections[i].t < result.t)) {
            result = intersections[i];
        }
    }

    return result;
}

HitInfo prepareComputations(const Intersection& hit, const Ray& ray,
                            const IntersectionSet& intersections = IntersectionSet());