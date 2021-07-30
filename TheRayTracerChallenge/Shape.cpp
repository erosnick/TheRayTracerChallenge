#include "Shape.h"
#include "Intersection.h"

Tuple TestShape::normalAt(const Tuple& position) const {
    return vector(0.0, 0.0, 0.0);
}

std::vector<Intersection> TestShape::intersect(const Ray& ray, bool bTransformRay) {
    return std::vector<Intersection>();
}
