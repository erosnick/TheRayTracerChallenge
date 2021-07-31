#pragma once

#include <memory>

class Shape;

using ShapePtr = std::shared_ptr<Shape>;

class Sphere;

using SpherePtr = std::shared_ptr<Sphere>;

class Plane;

using PlanePtr = std::shared_ptr<Plane>;

class Pattern;

using PatternPtr = std::shared_ptr<Pattern>;
