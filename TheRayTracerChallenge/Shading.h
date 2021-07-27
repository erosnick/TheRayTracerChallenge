#pragma once

#include "Tuple.h"
#include "Material.h"
#include "Light.h"

Tuple Lighting(const Material& material, const Tuple& position, const Light& light, const Tuple& eye, const Tuple& normal);