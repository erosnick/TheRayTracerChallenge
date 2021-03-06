#pragma once

#include "Tuple.h"
#include "Ray.h"
#include "Matrix.h"

class Camera {
public:
    Camera(int32_t inImageWidth, int32_t inImageHeight, double inFov = Math::pi_2) {
        imageWidth = inImageWidth;
        imageHeight = inImageHeight;
        fov = inFov;

        aspect = static_cast<double>(imageWidth) / imageHeight;

        origin = point(0.0, 0.0, 0.0);
    }

    inline void setPosition(const Tuple& inPosition) {
        position = inPosition;
    }

    inline Ray getRay(double dx, double dy) {
        auto pixelPosition = lowerLeftCorner + horizontal * dx + vertical * dy;

        auto direction = (pixelPosition - origin).normalize();

        //direction = viewMatrix * direction;

        auto ray = Ray(origin, direction);

        return ray;
    }

    inline Matrix4 viewTransform(const Tuple& eye, const Tuple& center, const Tuple& up) {
        viewMatrix = Matrix4();

        auto forward = (center - eye).normalize();
        auto right = (forward.cross(up)).normalize();
        auto trueUp = (right.cross(forward)).normalize();

        //viewMatrix[0][0] = right.x;
        //viewMatrix[1][0] = right.y;
        //viewMatrix[2][0] = right.z;
        //viewMatrix[0][1] = trueUp.x;
        //viewMatrix[1][1] = trueUp.y;
        //viewMatrix[2][1] = trueUp.z;
        //viewMatrix[0][2] = forward.x;
        //viewMatrix[1][2] = forward.y;
        //viewMatrix[2][3] = forward.z;

        //viewMatrix[0][3] = -right.dot(eye);
        //viewMatrix[1][3] = -trueUp.dot(eye);
        //viewMatrix[2][3] = -forward.dot(eye);

        viewMatrix[0] = right;
        viewMatrix[1] = trueUp;
        viewMatrix[2] = -forward;

        viewMatrix[0][3] = -right.dot(eye);
        viewMatrix[1][3] = -trueUp.dot(eye);
        viewMatrix[2][3] = forward.dot(eye);

        return viewMatrix;
    }

    inline Matrix4 lookAt(double inFov, const Tuple& inFrom, const Tuple& inTo, const Tuple& inUp) {
        fov = inFov;

        auto theta = fov * Math::pi / 180.0;
        auto height = std::tan(theta / 2);

        viewportHeight = 2.0 * height;
        viewportWidth = aspect * viewportHeight;

        horizontal = vector(viewportWidth, 0.0, 0.0);
        vertical = vector(0.0, viewportHeight, 0.0);

        //focalLength = std::sqrt(viewportWidth * viewportWidth + viewportHeight * viewportHeight) / viewportHeight;

        // Compute lower-left corner of projection plane
        lowerLeftCorner = origin - horizontal / 2 - vertical / 2 - vector(0.0, 0.0, focalLength);

        auto viewMatrix = viewTransform(inFrom, inTo, inUp);

        return viewMatrix;
    }

    int32_t imageWidth;
    int32_t imageHeight;

    Tuple position;

    Tuple horizontal;
    Tuple vertical;

    Tuple origin;

    Tuple lowerLeftCorner;

    Matrix4 viewMatrix;

    double viewportWidth = 0.0;
    double viewportHeight = 0.0;
    double focalLength = 1.0;

    double aspect;
    double fov = 90;

    Tuple from;
    Tuple to;
    Tuple right;
    Tuple up;
    Tuple forward;
};