#include "Shape.h"
#include "Intersection.h"

Tuple TestShape::normalAt(const Tuple& position) const {
    return vector(0.0, 0.0, 0.0);
}

InsersectionSet TestShape::intersect(const Ray& ray, bool bTransformRay) {
    return InsersectionSet();
}

void Shape::setTransformation(const Matrix4& inTransformation) {
    transformation = inTransformation;
}
