#pragma once

#include "Tuple.h"
#include "Material.h"
#include "Light.h"
#include "World.h"
#include "Intersection.h"

Tuple lighting(const Material& material, const Light& light, 
               const Tuple& position, const Tuple& viewDirection, 
               const Tuple& normal, bool bHalfLambert = false, bool bBlinnPhong = false);

Tuple lighting(const Material& material, const Light& light,
               const HitInfo& hitInfo, bool bHalfLambert = false, bool bBlinnPhong = false);

inline Tuple shadeHit(const World& world, const HitInfo& hitInfo, bool bHalfLambert = false, bool bBlinnPhong = false) {
    auto finalColor = color(0.0, 0.0, 0.0);

    for (const auto& light : world.getLights()) {
        return lighting(hitInfo.object.material, light, hitInfo, bHalfLambert, bBlinnPhong);
    }
    
    return finalColor;
}

inline Tuple colorAt(const World& world, const Ray& ray) {
    auto finalColor = Tuple();

    auto intersections = world.intersect(ray);

    if (intersections.size() == 0) {
        return color(0.0, 0.0, 0.0);
    }

    // Nearest intersection
    const auto& intersection = intersections[0];

    auto hitInfo = prepareComputations(intersection, ray);

    finalColor = shadeHit(world, hitInfo, false, true);

    return finalColor;
}
