#pragma once

#include "Tuple.h"
#include "Material.h"
#include "Light.h"
#include "World.h"
#include "Intersection.h"


inline Tuple shadeHit(const World& world, const HitInfo& hitInfo,
    int32_t remaining = 5, bool bHalfLambert = false, bool bBlinnPhong = false);
inline Tuple colorAt(const World& world, Ray& ray, int32_t remaining = 5);
inline Tuple reflectedColor(const World& world, const HitInfo& hitInfo, int32_t remaining = 5);

Tuple lighting(const Material& material, const ShapePtr& object, const Light& light, 
               const Tuple& position, const Tuple& viewDirection, 
               const Tuple& normal, bool bInShadow = false, 
               bool bHalfLambert = false, bool bBlinnPhong = false);

Tuple lighting(const Material& material, const ShapePtr& object, const Light& light,
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
                      int32_t remaining, bool bHalfLambert, bool bBlinnPhong) {
    auto surface = Color::black;

    for (const auto& light : world.getLights()) {
        //auto transformedLight = light;
        //transformedLight.transform(hitInfo.object.transform.inverse());
        auto inShadow = isShadow(world, light, hitInfo.overPosition);
        surface += lighting(hitInfo.object->material, hitInfo.object, light, hitInfo, inShadow, bHalfLambert, bBlinnPhong);
    }

    auto reflected = reflectedColor(world, hitInfo, remaining);
    
    return surface + reflected;
}

// colorat() 
//  - world.intersect()
//  - prepareComputations()
//  - shadeHit() -> lighting()
inline Tuple colorAt(const World& world, Ray& ray, int32_t remaining) {
    auto surface = Color::black;

    auto intersections = world.intersect(ray);

    if (intersections.size() == 0) {
        auto t = 0.5 * (ray.direction.y + 1.0);
        auto missColor = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
        return missColor;
    }

    // Nearest intersection
    const auto& intersection = intersections[0];

    if (!intersection.bShading) {
        surface = Color::white;
        return surface;
    }

    auto hitInfo = prepareComputations(intersection, intersection.ray);

    surface = shadeHit(world, hitInfo, remaining, false, true);

    return surface;
}

inline Tuple reflectedColor(const World& world, const HitInfo& hitInfo, int32_t remaining) {
    if (hitInfo.object->material.reflective == 0.0 || remaining == 0) {
        return Color::black;
    }

    auto reflectedRay = Ray(hitInfo.overPosition, hitInfo.reflectVector);
    auto color = colorAt(world, reflectedRay, remaining - 1);

    return color * hitInfo.object->material.reflective;
}