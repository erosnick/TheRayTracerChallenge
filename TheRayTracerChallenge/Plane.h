#pragma once

#include "Shape.h"
#include "Intersection.h"

class Plane : public Shape {
public:
    Plane() {}
    Plane(const Tuple& inPosition, const Tuple& inNormal)
    : position(inPosition), normal(inNormal) {}

    inline void setTransformation(const Matrix4& inTransformation) override {
        Shape::setTransformation(inTransformation);
        position.x = transformation[0][3];
        position.y = transformation[1][3];
        position.z = transformation[2][3];

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

    inline std::vector<Intersection> intersect(const Ray& ray, bool bTransformRay = false) override {
        auto intersections = std::vector<Intersection>();
        
        //if (std::abs(ray.direction.y) < EPSILON) {
        //    return intersections;
        //}

        //auto t = -ray.origin.y / ray.direction.y;
        //intersections.push_back(Intersection(t, GetPtr()));
        
        // p = p0 + tu  ray equation   (1)
        // n��(p - p0)   plane equation (2)
        // p0 in equation (1) and (2) are different
        // Let p0 in (2) = p1
        // t = (n��p1 - n��p0) / (n��u)
        auto denominator = normal.dot(ray.direction);

        if (denominator != 0.0) {
            auto t = (normal.dot(position) - normal.dot(ray.origin)) / denominator;

            if (t >= 0.0) {                
                auto intersection = Intersection();
                auto hitPosition = ray.position(t);

                if ((hitPosition.x >= -(horizontal * width / 2).x)
                    && (hitPosition.x <= (horizontal * width / 2).x)
                    && (hitPosition.z >= (vertical * height / 2).z)
                    && (hitPosition.z <= -(vertical * height / 2).z)) {
                    intersection.bHit = true;
                    intersection.count = 1;
                    intersection.t = t;
                    intersection.object = GetPtr();
                    intersection.ray = ray;
                    intersections.push_back(intersection);
                }
            }
        }

        return intersections;
    }

    Tuple position = point(0.0, -1.0, 0.0);
    Tuple normal = vector(0.0, 1.0, 0.0);
    Tuple horizontal = vector(1.0, 0.0, 0.0);
    Tuple vertical = vector(0.0, 0.0, -1.0);
    double width = 100.0;
    double height = 100.0;
};