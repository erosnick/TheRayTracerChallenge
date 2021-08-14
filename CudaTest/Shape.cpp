#include "Shape.h"
#include "Intersection.h"
#include "Material.h"

Tuple TestShape::normalAt(const Tuple& position) const {
    return vector(0.0, 0.0, 0.0);
}

void Shape::setTransformation(const Matrix4& inTransformation, bool bTransformPosition) {
    transformation = inTransformation;
}
