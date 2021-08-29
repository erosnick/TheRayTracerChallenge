#pragma once

#include "Tuple.h"
#include "Types.h"

struct Material {
    CUDA_HOST_DEVICE Material()
    : pattern(nullptr) {
        color = { 1.0, 0.0, 0.0, 0.0 };
        ambient = 0.1;
        diffuse = 0.9;
        specular = 0.9;
        shininess = 128.0;
        reflective = 0.0;
        transparency = 0.0;
        refractiveIndex = 1.0;
        bCastShadow = true;
        pattern = nullptr;
    }

    CUDA_HOST_DEVICE Material(const Tuple& inColor, Float inAmbient, Float inDiffse, Float inSpecular, Float inShininess, 
                              Float inReflective = 0.0, Float inTransparency = 0.0, Float inRefractiveIndex = 1.0)
    : color(inColor), ambient(inAmbient), diffuse(inDiffse), specular(inSpecular), shininess(inShininess), 
      reflective(inReflective), transparency(inTransparency), refractiveIndex(inRefractiveIndex) {
    }

    Tuple color = { 1.0, 0.0, 0.0, 0.0 };
    Float ambient = 0.1;
    Float diffuse = 0.9;
    Float specular = 0.9;
    Float shininess = 128.0;
    Float reflective = 0.0;
    Float transparency = 0.0;
    Float refractiveIndex = 1.0;
    bool bCastShadow = true;
    Pattern* pattern = nullptr;
};

inline bool operator==(const Material& a, const Material& b) {
    return ((a.color == b.color)
         && (a.ambient == b.ambient)
         && (a.diffuse == b.diffuse)
         && (a.specular == b.specular)
         && (a.shininess == b.shininess));
}