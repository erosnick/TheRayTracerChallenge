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
    CUDA_HOST_DEVICE Intersection() {
        t = 100000000.0;
        object = nullptr;
        subObject = nullptr;
    }

    CUDA_HOST_DEVICE Intersection(double inT, Shape* inShape, const Ray& inRay = Ray()) {
        t = inT;
        object = inShape;
        subObject = nullptr;
        ray = inRay;
    }

    CUDA_DEVICE Intersection(bool bInHit, int32_t inCount, double inT, Shape* inSphere)
    : bHit(bInHit), count(inCount), t(inT), object(inSphere), subObject(nullptr) {
    }

    CUDA_DEVICE Intersection(bool bInHit, bool bInShading, int32_t inCount, double inT, Shape* inSphere, const Tuple& inPosition, const Tuple& inNormal, const Ray& inRay)
    : bHit(bInHit), bShading(bInShading), count(inCount), t(inT), object(inSphere), subObject(nullptr), position(inPosition), normal(inNormal), ray(inRay) {
    }

    CUDA_HOST_DEVICE ~Intersection() {}

    bool bHit = false;
    bool bShading = true;
    int32_t count = 0;
    double t;
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

inline CUDA_HOST_DEVICE Intersection nearestHit(Intersection* intersections, int32_t count) {
    Intersection intersection(10000.0, nullptr);
    for (auto i = 0; i < count; i++) {
        if ((intersections[i].t > 0.0) && (intersections[i].t < intersection.t)) {
            intersection = intersections[i];
        }
    }

    return intersection;
}

HitInfo prepareComputations(const Intersection& hit, const Ray& ray,
                            const IntersectionSet& intersections = IntersectionSet());