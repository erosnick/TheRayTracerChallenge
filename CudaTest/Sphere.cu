#include "Sphere.h"
#include "Intersection.h"

CUDA_HOST_DEVICE bool Sphere::intersect(const Ray& ray, Intersection* intersections) {
    auto oc = (ray.origin - origin);
    auto a = ray.direction.dot(ray.direction);
    auto b = 2.0 * ray.direction.dot(oc);
    auto c = oc.dot(oc) - radius * radius;

    auto discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0) {
        return false;
    }

    // ��޴������󽻵�ʱ�򣬻�����б�ʽ����0�����������������������
    // ����������������߷���ķ����ӳ����ܺ������ཻ�ĳ��ϡ�
    auto t1 = (-b - std::sqrt(discriminant)) / (2 * a);
    auto t2 = (-b + std::sqrt(discriminant)) / (2 * a);

    auto position1 = ray.position(t1);

    auto normal1 = normalAt(position1);

    auto position2 = ray.position(t2);

    auto normal2 = normalAt(position2);

    if ((t1 > 0.0) || (t2 > 0.0)) {
        intersections[0] = { true, true, 1, t1, this, position1, normal1, ray };
        intersections[1] = { true, true, 1, t2, this, position2, normal2, ray };
    }

    return true;
}