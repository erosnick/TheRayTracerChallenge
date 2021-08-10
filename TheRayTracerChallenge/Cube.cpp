#include "Cube.h"
#include "Intersection.h"
#include "Plane.h"
#include "Quad.h"

#include <algorithm>

Cube::Cube() {
    initQuads();
}

void Cube::setTransformation(const Matrix4& inTransformation, bool bTransformPosition) {
    Shape::setTransformation(inTransformation, bTransformPosition);

    for (auto& quad : quads) {
        quad->setTransformation(transformation);
    }
}

void Cube::transform(const Matrix4& inTransformation) {
    Shape::transform(inTransformation);

    for (auto& quad : quads) {
        quad->setTransformation(transformation);
    }
}

Tuple Cube::normalAt(const Tuple& position) const {
    auto normal = vector(0.0);
    for (const auto& quad : quads) {
        if (quad->onQuad(position, normal)) {
            return normal;
        }
    }

    return Tuple();
}

InsersectionSet Cube::intersect(const Ray& ray, bool bTransformRay) {
    auto results = InsersectionSet();
    for (const auto& quad : quads) {
        if (auto result = quad->intersect(ray); result.size() > 0) {
            //result[0].object = GetPtr();
            results.push_back(result[0]);
        }
    }

    std::sort(results.begin(), results.end());

    return results;
}

std::tuple<double, double> Cube::checkAxis(double origin, double direction) {
    auto tminNumerator = -1.0 - origin;
    auto tmaxNumerator = 1.0 - origin;

    auto tmin = Math::infinityd;
    auto tmax = Math::infinityd;

    if (std::abs(direction) >= Math::epsilon) {
        tmin = tminNumerator / direction;
        tmax = tmaxNumerator / direction;
    }
    else {
        tmin = tminNumerator * Math::infinityd;
        tmax = tmaxNumerator * Math::infinityd;
    }
    if (tmin > tmax) {
        std::swap(tmin, tmax);
    }

    return { tmin, tmax };
}

void Cube::initQuads() {
    auto top = std::make_shared<Quad>("Top");
    top->setTransformation(translate(0.0, 1.0, 0.0));

    quads.push_back(top);

    auto bottom = std::make_shared<Quad>();
    bottom->setTransformation(translate(0.0, -1.0, 0.0));

    quads.push_back(bottom);

    auto back = std::make_shared<Quad>();
    back->setTransformation(translate(0.0, 0.0, -1.0) * rotateX(Math::pi_2));

    quads.push_back(back);

    auto front = std::make_shared<Quad>();
    front->setTransformation(translate(0.0, 0.0, 1.0) * rotateX(Math::pi_2));

    quads.push_back(front);

    auto left = std::make_shared<Quad>();
    left->setTransformation(translate(-1.0, 0.0, 0.0) * rotateZ(Math::pi_2));

    quads.push_back(left);

    auto right = std::make_shared<Quad>();
    right->setTransformation(translate(1.0, 0.0, 0.0) * rotateZ(Math::pi_2));

    quads.push_back(right);
}

void Cube::setMaterial(const MaterialPtr& inMaterial) {
    material = inMaterial;
    for (auto& quad : quads) {
        quad->setMaterial(inMaterial);
    }
}

void Cube::setMaterial(const MaterialPtr& inMaterial, int32_t quadIndex) {
    if (quadIndex > quads.size() - 1) {
        return;
    }

    quads[quadIndex]->setMaterial(inMaterial);
}
