#pragma once

#include <vector>
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

using MaterialPtr = std::shared_ptr<Material>;

struct Intersection;

using InsersectionSet = std::vector<Intersection>;

class Shape;

using ShapePtr = std::shared_ptr<Shape>;

class Sphere;

using SpherePtr = std::shared_ptr<Sphere>;

class Plane;

using PlanePtr = std::shared_ptr<Plane>;

class Pattern;

using PatternPtr = std::shared_ptr<Pattern>;

class Triangle;

using TrianglePtr = std::shared_ptr<Triangle>;

class Quad;

using QuadPtr = std::shared_ptr<Quad>;

class Cube;

using CubePtr = std::shared_ptr<Cube>;