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

void Cube::transformNormal(const Matrix4& worldMatrix) {
    for (auto& quad : quads) {
        quad->transformNormal(worldMatrix);
    }
}

Tuple Cube::normalAt(const Tuple& position) const {
    auto normal = vector(0.0);
    for (const auto& quad : quads) {
        if (quad->onQuad(position, normal)) {
            // 如果不变换法线，会导致多线程情况下随机噪点？
            // 可能是两种返回法线的方式之间存在差异，带排查
            return normal;
        }
    }

    return normal;
}

InsersectionSet Cube::intersect(const Ray& ray, bool bTransformRay) {
    auto results = InsersectionSet();
    for (const auto& quad : quads) {
        if (auto result = quad->intersect(ray); result.size() > 0) {
            result[0].subObject = result[0].object;
            result[0].object = GetPtr();
            results.push_back(result[0]);
        }
    }

    if (results.size() > 0) {
        if (results.size() == 1 && results[0].t < Math::epsilon) {
            results.clear();
        }
        else {
            auto hit = nearestHit(results);

            normal = hit.subObject->normalAt();
        }
    }

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
    top->bCube = true;
    top->setTransformation(translate(0.0, 1.0, 0.0));

    quads.push_back(top);

    auto bottom = std::make_shared<Quad>("Bottom");
    bottom->bCube = true;
    bottom->transformNormal(rotateX(Math::pi));
    bottom->setTransformation(translate(0.0, -1.0, 0.0) * rotateX(Math::pi));

    quads.push_back(bottom);

    auto back = std::make_shared<Quad>("Back");
    back->bCube = true;
    back->transformNormal(rotateX(-Math::pi_2));
    back->setTransformation(translate(0.0, 0.0, -1.0) * rotateX(-Math::pi_2));

    quads.push_back(back);

    auto front = std::make_shared<Quad>("Front");
    front->bCube = true;
    front->transformNormal(rotateX(Math::pi_2));
    front->setTransformation(translate(0.0, 0.0, 1.0) * rotateX(Math::pi_2));

    quads.push_back(front);

    auto left = std::make_shared<Quad>("Left");
    left->bCube = true;
    left->transformNormal(rotateZ(Math::pi_2));
    left->setTransformation(translate(-1.0, 0.0, 0.0) * rotateZ(Math::pi_2));

    quads.push_back(left);

    auto right = std::make_shared<Quad>("Right");
    right->bCube = true;
    right->transformNormal(rotateZ(-Math::pi_2));
    right->setTransformation(translate(1.0, 0.0, 0.0) * rotateZ(-Math::pi_2));

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
