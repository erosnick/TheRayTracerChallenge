#pragma once

#include "Sphere.h"
#include <initializer_list>
#include <vector>

struct Intersection {
    Intersection() {}
    Intersection(bool hit, int32_t inCount, double inT, const Sphere& sphere, const Tuple& inPosition = point(), const Tuple& inNormal = vector())
    : bHit(hit), count(inCount), t(inT), object(sphere), position(inPosition), normal(inNormal) {}

    Intersection(double inT, const Sphere& sphere)
        : t(inT), object(sphere) {}

    bool bHit = false;
    int32_t count = 0;
    double t = std::numeric_limits<double>::infinity();
    Sphere object;
    Tuple position = point();
    Tuple normal = vector();
};

inline bool operator==(const Intersection& a, const Intersection& b) {
    return ((a.bHit == b.bHit)
         && (a.count == b.count)
         && (a.t == b.t)
         && (a.object == b.object)
         && (a.position == b.position)
         && (a.normal == b.normal));
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