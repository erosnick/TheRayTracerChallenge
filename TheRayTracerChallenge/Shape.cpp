#include "Shape.h"
#include "Intersection.h"
#include "Material.h"

Shape::Shape() {
    material = std::make_shared<Material>();
}

Tuple TestShape::normalAt(const Tuple& position) const {
    return vector(0.0, 0.0, 0.0);
}

InsersectionSet TestShape::intersect(const Ray& ray, bool bTransformRay) {
    return InsersectionSet();
}


void Shape::setTransformation(const Matrix4& inTransformation, bool bTransformPosition) {
    transformation = inTransformation;
}
