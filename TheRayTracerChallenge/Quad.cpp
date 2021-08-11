#include "Quad.h"
#include "Triangle.h"
#include "Intersection.h"

Quad::Quad(const std::string& inName) 
: name(inName) {
    auto triangle = std::make_shared<Triangle>(point(-1.0, 0.0, -1.0), point(-1.0, 0.0, 1.0), point(1.0, 0.0, 1.0));
    triangles.push_back(triangle);

    triangle = std::make_shared<Triangle>(point(-1.0, 0.0, -1.0), point(1.0, 0.0, 1.0), point(1.0, 0.0, -1.0));
    triangles.push_back(triangle);
}

void Quad::setTransformation(const Matrix4& inTransformation, bool bTransformPosition) {
    Shape::setTransformation(inTransformation);

    for (auto& triangle : triangles) {
        triangle->setTransformation(inTransformation);
    }
}

void Quad::transform(const Matrix4& inTransformation) {
    Shape::transform(inTransformation);

    for (auto& triangle : triangles) {
        triangle->transform(inTransformation);
    }
}

void Quad::transformNormal(const Matrix4& worldMatrix) {
    for (auto& triangle : triangles) {
        triangle->transformNormal(worldMatrix);
    }
}

Tuple Quad::normalAt(const Tuple& position) const {
    return triangles[0]->normalAt(position);
}

InsersectionSet Quad::intersect(const Ray& ray, bool bTransformRay) {
    auto intersections1 = triangles[0]->intersect(ray, bTransformRay);
    auto intersections2 = InsersectionSet();
    
    if (intersections1.size() == 0) {
        intersections2 = triangles[1]->intersect(ray, bTransformRay);
    }

    auto result = InsersectionSet();

    // ������ı�������һ���������ཻ������Ϊ�ཻ����Ϊ���������ι���
    if (intersections1.size() > 0) {
        result = intersections1;
    }
    else if (intersections2.size() > 0) {
        result = intersections2;
    }

    // ���˵�Quad����ʹ��(����Cube�Ĳ���)ʱt < 0�����
    if (result.size() > 0 && result[0].t > Math::epsilon) {
        result[0].object = GetPtr();
    }
    else {
        result.clear();
    }

    return result;
}

bool Quad::onQuad(const Tuple& inPosition, Tuple& normal) {
    auto p0p1 = (inPosition - triangles[0]->v0).normalize();
    auto p0p2 = (inPosition - triangles[1]->v2).normalize();

    auto normal1 = triangles[0]->normalAt(inPosition);
    auto normal2 = triangles[1]->normalAt(inPosition);
    
    // ����Ҫ�þ���ֵ����Ϊ�����ֵ����Ϊ��(�۽ǵ����)
    if (std::abs(p0p1.dot(normal1)) <= Math::epsilon) {
        normal = normal1;
    }
    else if (std::abs(p0p2.dot(normal2)) <= Math::epsilon) {
        normal = normal2;
    }

    return (normal != vector(0.0));
}

void Quad::setMaterial(const MaterialPtr& inMaterial) {
    material = inMaterial;
    for (auto& triangle : triangles) {
        triangle->setMaterial(inMaterial);
    }
}
