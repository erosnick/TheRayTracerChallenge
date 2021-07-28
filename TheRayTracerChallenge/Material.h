#pragma once

#include "Tuple.h"
#include "Object.h"

struct Material : public Object {
    Material() {

    }

    Material(const Tuple& inColor, double inAmbient, double inDiffse, double inSpecular, double inShininess)
    : color(inColor), ambient(inAmbient), diffuse(inDiffse), specular(inSpecular), shininess(inShininess) {
    }

    Tuple color = { 1.0, 0.0, 0.0, 0.0 };
    double ambient = 0.1;
    double diffuse = 0.9;
    double specular = 0.9;
    double shininess = 128.0;
};

inline bool operator==(const Material& a, const Material& b) {
    return ((a.color == b.color)
         && (a.ambient == b.ambient)
         && (a.diffuse == b.diffuse)
         && (a.specular == b.specular)
         && (a.shininess == b.shininess));
}