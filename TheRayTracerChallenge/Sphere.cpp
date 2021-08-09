#include "Sphere.h"
#include "Intersection.h"

void Sphere::setTransformation(const Matrix4& inTransformation, bool bTransformPosition){
    Shape::setTransformation(inTransformation, bTransformPosition);

    origin = transformation * origin;
}

InsersectionSet Sphere::intersect(const Ray& ray, bool bTransformRay) {
    auto intersections = InsersectionSet();
    auto transformedRay = ray;

    if (bTransformRay) {
        transformedRay = transformRay(ray, transformation.inverse());
    }

    auto oc = (transformedRay.origin - origin);
    auto a = transformedRay.direction.dot(transformedRay.direction);
    auto b = 2.0 * transformedRay.direction.dot(oc);
    auto c = oc.dot(oc) - radius * radius;

    auto discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0) {
        return intersections;
    }

    // ��޴������󽻵�ʱ�򣬻�����б�ʽ����0�����������������������
    // ����������������߷���ķ����ӳ����ܺ������ཻ�ĳ��ϡ�
    auto t1 = (-b - std::sqrt(discriminant)) / (2 * a);
    auto t2 = (-b + std::sqrt(discriminant)) / (2 * a);

    auto position1 = transformedRay.position(t1);

    //position1.x *= scale.x;
    //position1.y *= scale.y;
    //position1.z *= scale.z;

    auto normal1 = normalAt(position1);

    auto position2 = transformedRay.position(t2);

    //position2.x *= scale.x;
    //position2.y *= scale.y;
    //position2.z *= scale.z;

    auto normal2 = normalAt(position2);

    if ((t1 > 0.0) || (t2 > 0.0)) {
        intersections.push_back({ true, !bIsLight, 1, t1, GetPtr(), position1, normal1, transformedRay });
        intersections.push_back({ true, !bIsLight, 1, t2, GetPtr(), position2, normal2, transformedRay });
    }
    else {
        int a = 0;
    }

    return intersections;
}