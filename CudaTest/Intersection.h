#pragma once

#include "CUDA.h"

#include "Types.h"
#include "Constants.h"
#include "Ray.h"
#include <initializer_list>
#include "Array.h"

struct HitInfo {
    bool bHit = false;
    Float t;
    Shape* object = nullptr;
    Tuple position;
    Tuple viewDirection;
    Tuple normal;
    Tuple overPosition;
    Tuple underPosition;
    Tuple reflectVector;
    Tuple surface;
    Float n1 = 1.0;
    Float n2 = 1.0;
    bool bInside = false;
};

struct Intersection {
    CUDA_HOST_DEVICE Intersection() {
    }

    CUDA_HOST_DEVICE Intersection(Float inT, Shape* inShape)
    : t(inT), object(inShape), subObject(nullptr) {
    }

    CUDA_HOST_DEVICE Intersection(bool bInHit, Float inT, Shape* inSphere)
    : bHit(bInHit), t(inT), object(inSphere), subObject(nullptr) {
    }

    CUDA_HOST_DEVICE Intersection(bool bInHit, bool bInShading, Float inT, Shape* inSphere)
    : bHit(bInHit), bShading(bInShading), t(inT), object(inSphere), subObject(nullptr) {
    }

    CUDA_HOST_DEVICE ~Intersection() {}

    bool bHit = false;
    bool bShading = true;
    Float t = 100000000.0;
    Shape* subObject = nullptr;
    Shape* object = nullptr;
};

inline CUDA_HOST_DEVICE bool operator==(const Intersection& a, const Intersection& b) {
    return ((a.bHit == b.bHit)
         && (a.t == b.t)
         && (a.object == b.object));
}

inline CUDA_HOST_DEVICE bool operator<(const Intersection& a, const Intersection& b) {
    return (a.t < b.t);
}

inline CUDA_HOST_DEVICE Intersection nearestHit(const Array<Intersection>& intersections) {
    Intersection intersection;
    for (auto i = 0; i < intersections.size(); i++) {
        if ((intersections[i].t > 0.0) && (intersections[i].t < intersection.t)) {
            intersection = intersections[i];
        }
    }

    return intersection;
}

inline CUDA_HOST_DEVICE Intersection nearestHit(Intersection intersections[], int32_t count) {
    Intersection intersection;
    for (auto i = 0; i < count; i++) {
        if ((intersections[i].t > 0.0) && (intersections[i].t < intersection.t)) {
            intersection = intersections[i];
        }
    }

    return intersection;
}

CUDA_HOST_DEVICE HitInfo prepareComputations(const Intersection& hit, const Ray& ray, const Array<Intersection>& intersections);
CUDA_HOST_DEVICE HitInfo prepareComputations(const Intersection& hit, const Ray& ray, Intersection intersections[], int32_t count);