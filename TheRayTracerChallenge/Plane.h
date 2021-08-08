#pragma once

#include "Shape.h"
#include "Intersection.h"

class Plane : public Shape {
public:
    Plane() {}
    Plane(const Tuple& inPosition, const Tuple& inNormal = Math::yAxis, int32_t inWidth = 100, int32_t inHeight = 100)
    : position(inPosition), normal(inNormal), width(inWidth), height(inHeight) {}

    virtual ~Plane() {}

    inline void setTransformation(const Matrix4& inTransformation) override {
        Shape::setTransformation(inTransformation);

        //position = transformation * position;
        normal = transformation * normal;
        horizontal = transformation * horizontal;
        vertical = transformation * vertical;
    }

    inline void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);

        position = transformation * position;
        normal = transformation * normal;
        position = transformation * position;
        horizontal = transformation * horizontal;
        vertical = transformation * vertical;
    }

    inline Tuple normalAt(const Tuple& position) const override {
        return normal;
    }

    bool checkAxis(const Tuple& inPosition) {
        //position = point(0.0);
        auto horizontalMin = (position - horizontal * width / 2);
        auto horizontalMax = (position + horizontal * width / 2);
        auto verticalMin = (position - vertical * height / 2);
        auto verticalMax = (position + vertical * height / 2);

        //auto transformedHorizontalMin = horizontalMin;
        //transformedHorizontalMin.x = horizontal.x * horizontalMin.x + horizontal.z * horizontalMin.z;
        //transformedHorizontalMin.z = vertical.x * horizontalMin.x + vertical.z * horizontalMin.z;

        //transformedHorizontalMin.x -= position.x;
        //transformedHorizontalMin.z -= position.z;

        //auto transformedHorizontalMax = horizontalMax;
        //transformedHorizontalMax.x = horizontal.x * horizontalMax.x + horizontal.z * horizontalMax.z;
        //transformedHorizontalMax.z = vertical.x * horizontalMax.x + vertical.z * horizontalMax.z;

        //transformedHorizontalMax.x -= position.x;
        //transformedHorizontalMax.z -= position.z;

        //auto transformedVerticalMin = verticalMin;
        //transformedVerticalMin.x = horizontal.x * verticalMin.x + horizontal.z * verticalMin.z;
        //transformedVerticalMin.z = vertical.x * verticalMin.x + vertical.z * verticalMin.z;

        //transformedVerticalMin.x -= position.x;
        //transformedVerticalMin.z -= position.z;

        //auto transformedVerticalMax = verticalMax;
        //transformedVerticalMax.x = horizontal.x * verticalMax.x + horizontal.z * verticalMax.z;
        //transformedVerticalMax.z = vertical.x * verticalMax.x + vertical.z * verticalMax.z;

        //transformedVerticalMax.x -= position.x;
        //transformedVerticalMax.z -= position.z;

        auto transformedPosition = inPosition;

        auto testPosition = point(0.70710678118654757, 0.0, -0.70710678118654757);

        testPosition = point(0.0, -1.0, -5.0);

        //transformedPosition = testPosition;

        //transformedPosition.x = horizontal.x * testPosition.x + horizontal.z * testPosition.z;
        //transformedPosition.z = vertical.x * testPosition.x - vertical.z * testPosition.z;

        // 将命中点转换到horizontal和vertical形成的坐标系中再进行判断
        transformedPosition.x = horizontal.x * inPosition.x + horizontal.z * inPosition.z;
        // 因为是右手坐标系，所以vertical.z要取反
        transformedPosition.z = vertical.x * inPosition.x - vertical.z * inPosition.z;

        if (transformedPosition.z > 0.0) {
            int a = 0;
        }

        if (planeOrientation == PlaneOrientation::XZ) {
            auto lowerLeft = position - (horizontal * (width / 2.0)) - (vertical * (height / 2.0));
            auto upperRight = position + (horizontal * (width / 2.0)) + (vertical * (height / 2.0));

            auto testPosition = point(1.76, -1.0, -6.77);
            testPosition = testPosition - position;

            auto a = point(horizontal.x * testPosition.x + horizontal.z * testPosition.z, testPosition.y, vertical.x * testPosition.x + vertical.z * testPosition.z);

            // 对平面的四个角进行变换，变换到horizontal和vertical组成的2D坐标系内
            auto transformedMax = horizontalMax - position;
            transformedMax = point(horizontal.x * transformedMax.x + horizontal.z * transformedMax.z, transformedMax.y, vertical.x * transformedMax.x + vertical.z * transformedMax.z);

            auto transformedHorizontalMin = horizontalMin - position;
            transformedHorizontalMin = point(horizontal.x * transformedHorizontalMin.x + horizontal.z * transformedHorizontalMin.z,
                                             transformedHorizontalMin.y,
                                             vertical.x * transformedHorizontalMin.x + vertical.z * transformedHorizontalMin.z);

            auto transformedHorizontalMax = horizontalMax - position;
            transformedHorizontalMax = point(horizontal.x * transformedHorizontalMax.x + horizontal.z * transformedHorizontalMax.z, 
                                             transformedHorizontalMax.y, 
                                             vertical.x * transformedHorizontalMax.x + vertical.z * transformedHorizontalMax.z);

            auto transformedVerticalMin = verticalMin - position;
            transformedVerticalMin = point(horizontal.x * transformedVerticalMin.x + horizontal.z * transformedVerticalMin.z,
                                           transformedVerticalMin.y,
                                           vertical.x * transformedVerticalMin.x + vertical.z * transformedVerticalMin.z);

            auto transformedVerticalMax = verticalMax - position;
            transformedVerticalMax = point(horizontal.x * transformedVerticalMax.x + horizontal.z * transformedVerticalMax.z,
                                           transformedVerticalMax.y,
                                           vertical.x * transformedVerticalMax.x + vertical.z * transformedVerticalMax.z);


            // 同样将命中点也变换到horizontal和vertical组成的2D坐标系内
            auto transformedPosition = inPosition - position;
            transformedPosition = point(horizontal.x * transformedPosition.x + horizontal.z * transformedPosition.z, 
                                        transformedPosition.y, 
                                        vertical.x * transformedPosition.x + vertical.z * transformedPosition.z);

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

                if (checkAxis(hitPosition)) {
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