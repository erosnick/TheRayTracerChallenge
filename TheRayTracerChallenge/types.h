#pragma once

#include <memory>

enum class PlaneOrientation : uint8_t {
    XY,
    YZ,
    XZ
};

class Tuple;

class World;

class Ray;

class Light;

struct HitInfo;

struct Material;

struct Intersection;

class Shape;

using ShapePtr = std::shared_ptr<Shape>;

class Sphere;

using SpherePtr = std::shared_ptr<Sphere>;

class Plane;

using PlanePtr = std::shared_ptr<Plane>;

class Pattern;

using PatternPtr = std::shared_ptr<Pattern>;
