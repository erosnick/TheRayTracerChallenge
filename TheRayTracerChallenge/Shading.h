#pragma once

#include "types.h"

Tuple shadeHit(const World& world, const HitInfo& hitInfo,
    int32_t remaining = 5, bool bHalfLambert = false, bool bBlinnPhong = false);
Tuple colorAt(const World& world, Ray& ray, int32_t remaining = 5);
Tuple reflectedColor(const World& world, const HitInfo& hitInfo, int32_t remaining = 5);
Tuple refractedColor(const World& world, const HitInfo& hitInfo, int32_t remaining = 5);
double schlick(const HitInfo& hitInfo);

Tuple lighting(const MaterialPtr& material, const ShapePtr& object, const Light& light, 
               const Tuple& position, const Tuple& viewDirection, 
               const Tuple& normal, bool bInShadow = false, 
               bool bHalfLambert = false, bool bBlinnPhong = false);

Tuple lighting(const MaterialPtr& material, const ShapePtr& object, const Light& light,
               const HitInfo& hitInfo, bool bInShadow = false,
               bool bHalfLambert = false, bool bBlinnPhong = false);

bool isShadow(const World& world, const Light& light, const Tuple& position);