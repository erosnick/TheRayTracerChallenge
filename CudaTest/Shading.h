#pragma once

#include "CUDA.h"
#include "Types.h"

CUDA_HOST_DEVICE Tuple computeReflectionAndRefraction(const HitInfo& hitInfo, World* world, int32_t depth);
CUDA_HOST_DEVICE Tuple shadeHit(World* world, const HitInfo& hitInfo, bool bHalfLambert = false, bool bBlinnPhong = false);
CUDA_HOST_DEVICE HitInfo colorAt(World* world, const Ray& ray);
CUDA_HOST_DEVICE Tuple reflectedColor(World* world, HitInfo& inHitInfo);
CUDA_HOST_DEVICE Tuple refractedColor(World* world, HitInfo& inHitInfo);

CUDA_HOST_DEVICE Tuple shadeHit(World* world, const HitInfo& hitInfo, int32_t remaining, bool bHalfLambert = false, bool bBlinnPhong = false);
CUDA_HOST_DEVICE Tuple colorAt(World* world, const Ray& ray, int32_t remaining);
CUDA_HOST_DEVICE Tuple reflectedColor(World* world, const HitInfo& inHitInfo, int32_t remaining);
CUDA_HOST_DEVICE Tuple refractedColor(World* world, const HitInfo& inHitInfo, int32_t remaining);
CUDA_HOST_DEVICE Float schlick(const HitInfo& hitInfo);

CUDA_HOST_DEVICE Tuple lighting(Material* material, Shape* object, Light* light,
               const Tuple& position, const Tuple& viewDirection, 
               const Tuple& normal, bool bInShadow = false, 
               bool bHalfLambert = false, bool bBlinnPhong = false);

CUDA_HOST_DEVICE Tuple lighting(Material* material, Shape* object, Light* light,
               const HitInfo& hitInfo, bool bInShadow = false,
               bool bHalfLambert = false, bool bBlinnPhong = false);

CUDA_HOST_DEVICE bool isShadow(World* world, Light* light, const Tuple& position);