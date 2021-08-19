#include "Quad.h"
#include "Triangle.h"
#include "Intersection.h"

CUDA_HOST_DEVICE Quad::Quad(char* inName)
: name(inName) {
    auto triangle = new Triangle(point(-1.0, 0.0, -1.0), point(-1.0, 0.0, 1.0), point(1.0, 0.0, 1.0));
    triangles[0] = triangle;

    triangle = new Triangle(point(-1.0, 0.0, -1.0), point(1.0, 0.0, 1.0), point(1.0, 0.0, -1.0));
    triangles[1] = triangle;
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
    return triangles[0]->normalAt(position);
}

CUDA_HOST_DEVICE bool Quad::intersect(const Ray& ray, Intersection* intersections) {
    Intersection intersections1;
    
    triangles[0]->intersect(ray, &intersections1);
    Intersection intersections2;
    
    if (intersections1.bHit) {
        triangles[1]->intersect(ray, &intersections2);
    }

    int32_t count = 0;

    // 如果和四边形其中一个三角形相交，则认为相交，因为两个三角形共面
    if (intersections1.bHit) {
        intersections[0] = intersections1;
        count++;
    }
    else if (intersections2.bHit) {
        intersections[0] = intersections2;
        count++;
    }

    // 过滤掉Quad单独使用(不是Cube的部分)时t < 0的情况
    if (count > 0) {
        intersections[0].object = this;

        if (!bCube && intersections[0].t < Math::epsilon) {
            //result.clear();
        }
    }

    return count > 0;
}

bool Quad::onQuad(const Tuple& inPosition, Tuple& normal) {
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

void Quad::setMaterial(Material* inMaterial) {
    material = inMaterial;
    for (auto& triangle : triangles) {
        triangle->setMaterial(inMaterial);
    }
}
