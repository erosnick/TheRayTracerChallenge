#pragma once

#include "Tuple.h"
#include "Material.h"
#include "Light.h"
#include "World.h"
#include "Intersection.h"

Tuple lighting(const Material& material, const Light& light, 
               const Tuple& position, const Tuple& viewDirection, 
               const Tuple& normal, bool bInShadow = false, 
               bool bHalfLambert = false, bool bBlinnPhong = false);

Tuple lighting(const Material& material, const Light& light,
               const HitInfo& hitInfo, bool bInShadow = false,
               bool bHalfLambert = false, bool bBlinnPhong = false);

inline bool isShadow(const World& world, const Light& light, const Tuple& position) {
    auto toLight = light.position - position;
    const auto distance = toLight.magnitude();

    auto ray = Ray(position, toLight.normalize());
    auto intersections = world.intersect(ray);

    if (intersections.size() > 0) {
        const auto& intersection = intersections[0];

        if (!intersection.object->bIsLight && intersection.t < distance) {
            return true;
        }
    }

    return false;
}

inline Tuple shadeHit(const World& world, const HitInfo& hitInfo, 
                      bool bHalfLambert = false, bool bBlinnPhong = false) {
    auto finalColor = color(0.0, 0.0, 0.0);

    for (const auto& light : world.getLights()) {
        //auto transformedLight = light;
        //transformedLight.transform(hitInfo.object.transform.inverse());
        auto inShadow = isShadow(world, light, hitInfo.overPosition);
        finalColor += lighting(hitInfo.object->material, light, hitInfo, inShadow, bHalfLambert, bBlinnPhong);
    }
    
    return finalColor;
}

inline Tuple colorAt(const World& world, Ray& ray) {
    auto finalColor = Tuple();

    auto intersections = world.intersect(ray);

    if (intersections.size() == 0) {
        auto t = 0.5 * (ray.direction.y + 1.0);
        auto missColor = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
        return missColor;
    }

    // Nearest intersection
    const auto& intersection = intersections[0];

    if (!intersection.bShading) {
        finalColor = color(1.0, 1.0, 1.0);
        return finalColor;
    }

    auto hitInfo = prepareComputations(intersection, intersection.ray);

    finalColor = shadeHit(world, hitInfo, false, true);

    return finalColor;
}