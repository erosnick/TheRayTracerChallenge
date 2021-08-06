#pragma once

#include "Shape.h"
#include "Intersection.h"

class Plane : public Shape {
public:
    Plane() {}
    Plane(const Tuple& inPosition, const Tuple& inNormal, int32_t inWidth = 100, int32_t inHeight = 100)
    : position(inPosition), normal(inNormal), width(inWidth), height(inHeight) {}

    virtual ~Plane() {}

    inline void setTransformation(const Matrix4& inTransformation) override {
        Shape::setTransformation(inTransformation);
        position.x = transformation[0][3];
        position.y = transformation[1][3];
        position.z = transformation[2][3];
        normal = transformation * normal;
        horizontal = transformation * horizontal;
        vertical = transformation * vertical;
    }

    inline void transform(const Matrix4& inTransformation) override {
        Shape::transform(inTransformation);

        normal = transformation * normal;
        position = transformation * position;
        horizontal = transformation * horizontal;
        vertical = transformation * vertical;
    }

    inline Tuple normalAt(const Tuple& position) const override {
        return normal;
    }

    bool checkAxis(const Tuple& inPosition) {
        if (planeOrientation == PlaneOrientation::XZ) {
           if ((inPosition.x >= -(position + horizontal * width / 2).x)
            && (inPosition.x <=  (position + horizontal * width / 2).x)
            && (inPosition.z >=  (position + vertical * height / 2).z)
            && (inPosition.z <= -(position + vertical * height / 2).z)) {
               return true;
           }
        }
        else if (planeOrientation == PlaneOrientation::XY) {
            if ((inPosition.x >= -(position + horizontal * width / 2).x)
             && (inPosition.x <=  (position + horizontal * width / 2).x)
             && (inPosition.y >= -(position + vertical * height / 2).y)
             && (inPosition.y <=  (position + vertical * height / 2).y)) {
                return true;
            }
        }
        else {
            if ((inPosition.y >=  (position + horizontal * width / 2).y)
             && (inPosition.y <= -(position + horizontal * width / 2).y)
             && (inPosition.z >=  (position + vertical * height / 2).z)
             && (inPosition.z <= -(position + vertical * height / 2).z)) {
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
        // n，(p - p0) = 0 plane equation (2)
        // p0 in equation (1) and (2) are different
        // Let p0 in (2) = p1
        // t = (n，p1 - n，p0) / (n，u)
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