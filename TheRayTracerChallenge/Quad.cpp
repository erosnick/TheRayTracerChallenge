#include "Quad.h"
#include "Triangle.h"
#include "Intersection.h"

Quad::Quad() {
    auto triangle = std::make_shared<Triangle>(point(-1.0, 0.0, -1.0), point(-1.0, 0.0, 1.0), point(1.0, 0.0, 1.0));
    triangles.push_back(triangle);

    triangle = std::make_shared<Triangle>(point(-1.0, 0.0, -1.0), point(1.0, 0.0, 1.0), point(1.0, 0.0, -1.0));
    triangles.push_back(triangle);
}

void Quad::setTransformation(const Matrix4& inTransformation, bool bTransformPosition) {
    Shape::setTransformation(inTransformation);

    for (auto& triangle : triangles) {
        triangle->setTransformation(transformation);
    }
}

Tuple Quad::normalAt(const Tuple& position) const {
    return triangles[0]->normalAt(position);
}

InsersectionSet Quad::intersect(const Ray& ray, bool bTransformRay) {
    auto intersections1 = triangles[0]->intersect(ray, bTransformRay);
    auto intersections2 = triangles[1]->intersect(ray, bTransformRay);

    auto result = InsersectionSet();

    if (intersections1.size() > 0) {
        result = intersections1;
    }
    else if (intersections2.size() > 0) {
        result = intersections2;
    }

    if (result.size() > 0) {
        result[0].object = GetPtr();
    }

    return result;
}
