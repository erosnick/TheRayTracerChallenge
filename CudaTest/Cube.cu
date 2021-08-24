#include "Cube.h"
#include "Intersection.h"
#include "Quad.h"

#include <algorithm>

Cube::Cube() {
    initQuads();
}

void Cube::setTransformation(const Matrix4& inTransformation, bool bTransformPosition) {
    Shape::setTransformation(inTransformation, bTransformPosition);

    for (auto& quad : quads) {
        quad->setTransformation(inTransformation);
    }
}

void Cube::transform(const Matrix4& inTransformation) {
    Shape::transform(inTransformation);

    for (auto& quad : quads) {
        quad->transform(inTransformation);
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

bool Cube::intersect(const Ray& ray, Array<Intersection>& intersections) {
    auto size = intersections.size();
    for (const auto& quad : quads) {
        if (quad->intersect(ray, intersections)) {
            intersections[intersections.size() - 1].subObject = intersections[intersections.size() - 1].object;
            intersections[intersections.size() - 1].object = this;
        }
    }
    
    auto bHit = intersections.size() > size;
    if (bHit) {
        if ((intersections.size() - size) == 1 && intersections[intersections.size() - 1].t < Math::epsilon) {
            intersections.remove(intersections[intersections.size() - 1]);
        }
        else {
            Array<Intersection> results;
            results.add(intersections[intersections.size() - 2]);
            results.add(intersections[intersections.size() - 1]);
            auto hit = nearestHit(results);
            normal = hit.subObject->normalAt();
            //printf("%f, %f, %f\n", normal.x(), normal.y(), normal.z());
        }
    }

    return bHit;
}

//std::tuple<double, double> Cube::checkAxis(double origin, double direction) {
//    auto tminNumerator = -1.0 - origin;
//    auto tmaxNumerator = 1.0 - origin;
//
//    auto tmin = Math::infinityd;
//    auto tmax = Math::infinityd;
//
//    if (std::abs(direction) >= Math::epsilon) {
//        tmin = tminNumerator / direction;
//        tmax = tmaxNumerator / direction;
//    }
//    else {
//        tmin = tminNumerator * Math::infinityd;
//        tmax = tmaxNumerator * Math::infinityd;
//    }
//    if (tmin > tmax) {
//        std::swap(tmin, tmax);
//    }
//
//    return { tmin, tmax };
//}

void Cube::initQuads() {
    auto top = new Quad("Top", true);
    top->setTransformation(translate(0.0, 1.0, 0.0));

    quads.add(top);

    auto bottom = new Quad("Bottom", true);
    bottom->transformNormal(rotateX(Math::pi));
    bottom->setTransformation(translate(0.0, -1.0, 0.0) * rotateX(Math::pi));

    quads.add(bottom);

    auto back = new Quad("Back", true);
    back->transformNormal(rotateX(-Math::pi_2));
    back->setTransformation(translate(0.0, 0.0, -1.0) * rotateX(-Math::pi_2));

    quads.add(back);

    auto front = new Quad("Front", true);
    front->transformNormal(rotateX(Math::pi_2));
    front->setTransformation(translate(0.0, 0.0, 1.0) * rotateX(Math::pi_2));

    quads.add(front);

    auto left = new Quad("Left", true);
    left->transformNormal(rotateZ(Math::pi_2));
    left->setTransformation(translate(-1.0, 0.0, 0.0) * rotateZ(Math::pi_2));

    quads.add(left);

    auto right = new Quad("Right", true);
    right->transformNormal(rotateZ(-Math::pi_2));
    right->setTransformation(translate(1.0, 0.0, 0.0) * rotateZ(-Math::pi_2));

    quads.add(right);
}

void Cube::setMaterial(Material* inMaterial) {
    material = inMaterial;
    for (auto& quad : quads) {
        quad->setMaterial(inMaterial);
    }
}

void Cube::setMaterial(Material* inMaterial, int32_t quadIndex) {
    if (quadIndex > quads.size() - 1) {
        return;
    }

    quads[quadIndex]->setMaterial(inMaterial);
}
