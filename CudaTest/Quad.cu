#include "Quad.h"
#include "Triangle.h"
#include "Intersection.h"

CUDA_HOST_DEVICE Quad::Quad(const char* inName)
: name(inName) {
    triangles[0] = new Triangle(point(-1.0, 0.0, -1.0), point(-1.0, 0.0, 1.0), point(1.0, 0.0, 1.0));
    triangles[1] = new Triangle(point(-1.0, 0.0, -1.0), point(1.0, 0.0, 1.0), point(1.0, 0.0, -1.0));
}

CUDA_HOST_DEVICE Quad::~Quad() {
    for (auto i = 0; i < 2; i++) {
        if (triangles[i]) {
            delete triangles[i];
        }
    }
}

CUDA_HOST_DEVICE void Quad::setTransformation(const Matrix4& inTransformation, bool bTransformPosition) {
    Shape::setTransformation(inTransformation);

    for (auto& triangle : triangles) {
        triangle->setTransformation(inTransformation);
    }
}

CUDA_HOST_DEVICE void Quad::transform(const Matrix4& inTransformation) {
    Shape::transform(inTransformation);

    for (auto& triangle : triangles) {
        triangle->transform(inTransformation);
    }
}

CUDA_HOST_DEVICE void Quad::transformNormal(const Matrix4& worldMatrix) {
    for (auto& triangle : triangles) {
        triangle->transformNormal(worldMatrix);
    }
}

CUDA_HOST_DEVICE Tuple Quad::normalAt(const Tuple& position) const {
    return triangles[0]->normalAt();
}

CUDA_HOST_DEVICE bool Quad::intersect(const Ray& ray, Array<Intersection>& intersections) {
    auto size = intersections.size();
    
    // 如果和四边形其中一个三角形相交，则认为相交，因为两个三角形共面
    triangles[0]->intersect(ray, intersections);
    
    if (intersections.size() == size) {
        triangles[1]->intersect(ray, intersections);
    }

    auto bHit = intersections.size() > size;

    // 过滤掉Quad单独使用(不是Cube的部分)时t < 0的情况
    if (bHit) {
        intersections[intersections.size() - 1].object = this;

        auto intersection = intersections[intersections.size() - 1];
        if (!bCube && intersection.t < Math::epsilon) {
            intersections.remove(intersection);
        }
    }

    return bHit;
}

CUDA_HOST_DEVICE bool Quad::onQuad(const Tuple& inPosition, Tuple& normal) {
    auto p0p1 = (inPosition - triangles[0]->v0).normalize();
    auto p0p2 = (inPosition - triangles[1]->v2).normalize();

    auto normal1 = triangles[0]->normalAt(inPosition);
    auto normal2 = triangles[1]->normalAt(inPosition);
    
    // 这里要用绝对值，因为点积的值可能为负(钝角的情况)
    if (std::abs(p0p1.dot(normal1)) <= Math::epsilon) {
        normal = normal1;
    }
    else if (std::abs(p0p2.dot(normal2)) <= Math::epsilon) {
        normal = normal2;
    }

    return (normal != vector(0.0));
}

CUDA_HOST_DEVICE void Quad::setMaterial(Material* inMaterial) {
    material = inMaterial;
    for (auto& triangle : triangles) {
        triangle->setMaterial(inMaterial);
    }
}
