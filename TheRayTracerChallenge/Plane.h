#pragma once

#include "Shape.h"
#include "Intersection.h"

class Plane : public Shape {
public:
    Plane() {}
    Plane(const Tuple& inPosition, const Tuple& inNormal = Math::yAxis, int32_t inWidth = 100, int32_t inHeight = 100)
    : position(inPosition), normal(inNormal), width(inWidth), height(inHeight) {}

    virtual ~Plane() {}

    inline void setTransformation(const Matrix4& inTransformation, bool bTransformPosition = false) override {
        Shape::setTransformation(inTransformation, bTransformPosition);

        if (bTransformPosition) {
            position = transformation * position;
        }

        normal = transformation * normal;
        horizontal = transformation * horizontal;
        vertical = transformation * vertical;
    }

    inline void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);

        position = transformation * position;
        normal = transformation * normal;
        horizontal = transformation * horizontal;
        vertical = transformation * vertical;
    }

    inline Tuple normalAt(const Tuple& position) const override {
        return normal;
    }

    bool checkAxis(const Tuple& inPosition, Tuple& transformedPosition) {
        auto horizontalMin = (position - horizontal * width / 2);
        auto horizontalMax = (position + horizontal * width / 2);
        auto verticalMin = (position - vertical * height / 2);
        auto verticalMax = (position + vertical * height / 2);

        if (planeOrientation == PlaneOrientation::XZ) {
            // 对平面的四个角进行变换，变换到horizontal和vertical组成的2D坐标系内
            auto transformedHorizontalMin = horizontalMin - position;
            transformedHorizontalMin = point(transformedHorizontalMin.dot(horizontal), position.y, transformedHorizontalMin.dot(vertical));

            auto transformedHorizontalMax = horizontalMax - position;
            transformedHorizontalMax = point(transformedHorizontalMax.dot(horizontal), position.y, transformedHorizontalMax.dot(vertical));

            auto transformedVerticalMin = verticalMin - position;
            transformedVerticalMin = point(transformedVerticalMin.dot(horizontal), position.y, transformedVerticalMin.dot(vertical));

            auto transformedVerticalMax = verticalMax - position;
            transformedVerticalMax = point(transformedVerticalMax.dot(horizontal), position.y, transformedVerticalMax.dot(vertical));


            // 同样将命中点也变换到horizontal和vertical组成的2D坐标系内
            transformedPosition = inPosition - position;
            transformedPosition = point(transformedPosition.dot(horizontal), position.y, transformedPosition.dot(vertical));

            // 找出平面的坐标范围
            auto minX = std::min(transformedHorizontalMin.x, transformedHorizontalMax.x);
            auto maxX = std::max(transformedHorizontalMin.x, transformedHorizontalMax.x);
            auto minZ = std::min(transformedVerticalMin.z, transformedVerticalMax.z);
            auto maxZ = std::max(transformedVerticalMin.z, transformedVerticalMax.z);

            // 判断命中点是否在平面内
            if ((transformedPosition.x >= minX)
             && (transformedPosition.x <= maxX)
             && (transformedPosition.z >= minZ)
             && (transformedPosition.z <= maxZ)) {
                return true;
            }
        }
        else if (planeOrientation == PlaneOrientation::XY) {
            auto minX = horizontalMin.x;
            auto maxX = horizontalMax.x;
            auto minY = verticalMin.y;
            auto maxY = verticalMax.y;
            if ((inPosition.x >= minX)
             && (inPosition.x <= maxX)
             && (inPosition.y >= minY)
             && (inPosition.y <= maxY)) {
                return true;
            }
        }
        else {
            auto minY = -horizontalMin.y;
            auto maxY = -horizontalMax.y;
            auto minZ = verticalMax.z;
            auto maxZ = verticalMin.z;
            if ((inPosition.y >= minY)
             && (inPosition.y <= maxY)
             && (inPosition.z >= minZ)
             && (inPosition.z <= maxZ)) {
                return true;
            }
        }

        return false;
    }

    inline InsersectionSet intersect(const Ray& ray, bool bTransformRay = false) override {
        auto intersections = InsersectionSet();
        
        //if (std::abs(ray.direction.y) < EPSILON) {
        //    return intersections;
        //}

        //auto t = -ray.origin.y / ray.direction.y;
        //intersections.push_back(Intersection(t, GetPtr()));
        
        // p = p0 + tu    ray equation   (1)
        // n·(p - p0) = 0 plane equation (2)
        // p0 in equation (1) and (2) are different
        // Let p0 in (2) = p1
        // t = (n·p1 - n·p0) / (n·u)
        auto denominator = normal.dot(ray.direction);

        if (denominator != 0.0) {
            auto t = (normal.dot(position) - normal.dot(ray.origin)) / denominator;

            if (t >= Math::epsilon) {                
                auto intersection = Intersection();
                auto hitPosition = ray.position(t);
                auto transformedPosition = point(0.0);
                if (checkAxis(hitPosition, transformedPosition)) {
                    intersection.bHit = true;
                    intersection.count = 1;
                    intersection.t = t;
                    intersection.object = GetPtr();
                    intersection.ray = ray;
                    intersection.position = ray.position(t);
                    intersections.push_back(intersection);
                }
            }
        }

        return intersections;
    }

    bool onPlane(const Tuple& inPosition) {
        if ((inPosition - position).dot(normal) <= Math::epsilon) {
            return true;
        }

        return false;
    }

    Tuple position = point(0.0, -1.0, 0.0);
    Tuple normal = vector(0.0, 1.0, 0.0);
    Tuple horizontal = vector(1.0, 0.0, 0.0);
    Tuple vertical = vector(0.0, 0.0, -1.0);
    double width = 100.0;
    double height = 100.0;
    PlaneOrientation planeOrientation = PlaneOrientation::XZ;
};