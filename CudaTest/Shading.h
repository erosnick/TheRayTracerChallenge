#pragma once

#include "CUDA.h"
#include "Types.h"

CUDA_DEVICE Tuple shadeHit(World* world, const HitInfo& hitInfo, int32_t remaining = 5, bool bHalfLambert = false, bool bBlinnPhong = false);
CUDA_DEVICE Tuple colorAt(World* world, Ray& ray, int32_t remaining = 5);
CUDA_DEVICE Tuple reflectedColor(World* world, const HitInfo& hitInfo, int32_t reflectionRemaining = 5);
CUDA_DEVICE Tuple refractedColor(World* world, const HitInfo& hitInfo, int32_t refractionRemaining = 5);
CUDA_DEVICE double schlick(const HitInfo& hitInfo);

CUDA_DEVICE Tuple lighting(Material* material, Shape* object, Light* light,
               const Tuple& position, const Tuple& viewDirection, 
               const Tuple& normal, bool bInShadow = false, 
               bool bHalfLambert = false, bool bBlinnPhong = false);

CUDA_DEVICE Tuple lighting(Material* material, Shape* object, Light* light,
               const HitInfo& hitInfo, bool bInShadow = false,
               bool bHalfLambert = false, bool bBlinnPhong = false);

CUDA_DEVICE bool isShadow(World* world, Light* light, const Tuple& position);