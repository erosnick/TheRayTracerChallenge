#pragma once

#include "Tuple.h"
#include "Ray.h"

class Camera {
public:
    Camera(int32_t inImageWidth, int32_t inImageHeight) {
        position = point(0.0, 0.0, 1.0);

        imageWidth = inImageWidth;
        imageHeight = inImageHeight;

        aspect = static_cast<double>(imageWidth) / imageHeight;

        horizontal = vector(1.0, 0.0, 0.0) * aspect;
        vertical = vector(0.0, 1.0, 0.0);

        origin = point(0.0, 0.0, 0.0);

        // Compute lower-left corner of projection plane
        lowerLeftCorner = origin - horizontal / 2 - vertical / 2;
    }

    inline void setPosition(const Tuple& inPosition) {
        position = inPosition;
    }

    inline Ray getRay(double dx, double dy) {
        auto pixelPosition = lowerLeftCorner + horizontal * dx + vertical * dy;

        auto direction = (pixelPosition - position).normalize();

        auto ray = Ray(position, direction);

        return ray;
    }

    int32_t imageWidth;
    int32_t imageHeight;

    Tuple position;

    Tuple horizontal;
    Tuple vertical;

    Tuple origin;

    Tuple lowerLeftCorner;

    double aspect;
};