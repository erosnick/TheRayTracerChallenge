#include "Quad.h"
#include "Triangle.h"
#include "Intersection.h"

CUDA_HOST_DEVICE Quad::Quad(const char* inName, bool isCube)
: name(inName), bIsCube(isCube) {
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

CUDA_HOST_DEVICE void Quad::updateTransformation() {
    for (auto& triangle : triangles) {
        triangle->updateTransformation();
    }
}

CUDA_HOST_DEVICE void Quad::transformNormal(const Matrix4& inTransformation) {
    for (auto& triangle : triangles) {
        triangle->transformNormal(inTransformation);
    }
}

CUDA_HOST_DEVICE Tuple Quad::normalAt(const Tuple& inPosition) const {
    return triangles[0]->normalAt();
}

CUDA_HOST_DEVICE bool Quad::intersect(const Ray& ray, Array<Intersection>& intersections) {
    Array<Intersection> triangleIntersection;
    
    // 如果和四边形其中一个三角形相交，则认为相交，因为两个三角形共面
    triangles[0]->intersect(ray, triangleIntersection);
    
    if (triangleIntersection.size() == 0) {
        triangles[1]->intersect(ray, triangleIntersection);
    }

    auto bHit = triangleIntersection.size() > 0;

    // 过滤掉Quad单独使用(不是Cube的部分)时t < 0的情况
    if (bHit) {
        triangleIntersection[0].object = this;
        triangleIntersection[0].subObject = this;

        auto intersection = triangleIntersection[0];

        if (!bIsCube && intersection.t < Math::epsilon) {
            triangleIntersection.remove(intersection);
        }
        else {
            intersections.add(intersection);
        }
    }

    return bHit;
}

CUDA_HOST_DEVICE bool Quad::onQuad(const Tuple& inPosition, Tuple& normal) {
    auto p0p1 = (inPosition - triangles[0]->transformedv0).normalize();
    auto p0p2 = (inPosition - triangles[1]->transformedv2).normalize();

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
