#pragma once

#include <cstdint>

enum class PlaneOrientation : uint8_t {
    XY,
    YZ,
    XZ
};

class Tuple;

class World;

struct Payload;

class Ray;

class Light;

struct HitInfo;

struct Material;

struct Intersection;

class Shape;

class Sphere;

class Plane;

class Pattern;

class Triangle;

class Quad;

class Cube;