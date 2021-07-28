#pragma once

#include "Tuple.h"
#include "Ray.h"

class Camera {
public:
    Camera(int32_t inImageWidth, int32_t inImageHeight) {
        imageWidth = inImageWidth;
        imageHeight = inImageHeight;

        aspect = static_cast<double>(imageWidth) / imageHeight;

        viewportHeight = 2.0;
        viewportWidth = aspect * viewportHeight;

        horizontal = vector(viewportWidth, 0.0, 0.0);
        vertical = vector(0.0, viewportHeight, 0.0);

        origin = point(0.0, 0.0, 0.0);

        // Compute lower-left corner of projection plane
        lowerLeftCorner = origin - horizontal / 2 - vertical / 2 - vector(0.0, 0.0, focalLength);
    }

    inline void setPosition(const Tuple& inPosition) {
        position = inPosition;
    }

    inline Ray getRay(double dx, double dy) {
        auto pixelPosition = lowerLeftCorner + horizontal * dx + vertical * dy;

        auto direction = (pixelPosition - origin).normalize();

        auto ray = Ray(origin, direction);

        return ray;
    }

    int32_t imageWidth;
    int32_t imageHeight;

    Tuple position;

    Tuple horizontal;
    Tuple vertical;

    Tuple origin;

    Tuple lowerLeftCorner;

    double viewportWidth;
    double viewportHeight;
    double focalLength = 1.0;

    double aspect;
};