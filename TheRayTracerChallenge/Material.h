#pragma once

#include "Tuple.h"

struct Material {
    Tuple color;
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