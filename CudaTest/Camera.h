#pragma once

#include "Tuple.h"
#include "Ray.h"
#include "Constants.h"
#include "Matrix.h"

class Camera {
public:
    Camera(int32_t inImageWidth, int32_t inImageHeight, Float inFov = 90.0) {
        init(inImageWidth, inImageHeight, inFov);
    }

    void init(int32_t inImageWidth, int32_t inImageHeight, Float inFov = 90.0) {
        imageWidth = inImageWidth;
        imageHeight = inImageHeight;
        fov = inFov;
        aspectRatio = static_cast<Float>(imageWidth) / imageHeight;
        focalLength = 1.0;
        cameraSpeed = 6.0f;
        origin = point(0.0, 0.0, 0.0);
    }

    inline void setPosition(const Tuple& inPosition) {
        position = inPosition;
    }

    inline CUDA_HOST_DEVICE Ray getRay(Float dx, Float dy) {
        auto pixelPosition = upperLeftCorner + horizontal * dx - vertical * dy;

        auto direction = (pixelPosition - origin).normalize();

        return Ray(origin, direction);
    }

    inline Matrix4 viewTransform(const Tuple& inEye, const Tuple& inCenter, const Tuple& inUp) {
        viewMatrix = Matrix4();

        eye = inEye;
        center = inCenter;

        forward = (center - eye).normalize();
        right = (forward.cross(inUp)).normalize();
        up = (right.cross(forward)).normalize();

        viewMatrix[0] = right;
        viewMatrix[1] = up;
        viewMatrix[2] = -forward;

        viewMatrix[0][3] = -right.dot(eye);
        viewMatrix[1][3] = -up.dot(eye);
        viewMatrix[2][3] = forward.dot(eye);

        return viewMatrix;
    }

    inline void computeParameters() {
        auto theta = Math::radians(fov);
        auto height = tan(theta / 2);

        viewportHeight = 2.0 * height;
        viewportWidth = aspectRatio * viewportHeight;

        horizontal = vector(viewportWidth, 0.0, 0.0);
        vertical = vector(0.0, viewportHeight, 0.0);

        // Compute lower-left corner of projection plane
        upperLeftCorner = origin - horizontal / 2 + vertical / 2 - vector(0.0, 0.0, focalLength);
    }

    inline Matrix4 lookAt(Float inFov, const Tuple& inEye, const Tuple& inCenter, const Tuple& inUp) {
        fov = inFov;

        computeParameters();

        auto viewMatrix = viewTransform(inEye, inCenter, inUp);

        return viewMatrix;
    }

    void walk(Float delta) {
        eye += forward * cameraSpeed * delta;
        center += forward * cameraSpeed * delta;
        bIsDirty = true;
    }

    void strafe(Float delta) {
        eye += right * cameraSpeed * delta;
        center += right * cameraSpeed * delta;
        bIsDirty = true;
    }

    void raise(Float delta) {
        eye += up * cameraSpeed * delta;
        center += up * cameraSpeed * delta;
        bIsDirty = true;
    }

    void yaw(Float delta) {
        // Should rotate around up vector
        auto rotation = rotateY(Math::radians(delta));
        forward = (rotation * forward).normalize();
        right = (rotation * right).normalize();

        //up = glm::cross(right, forward);

        center = eye + forward;

        bIsDirty = true;
    }

    void pitch(Float delta) {
        // Should rotate around right vector
        auto rotation = rotateX(Math::radians(delta));
        forward = (rotation * forward).normalize();
        //up = glm::normalize(rotation * up);
        center = eye + forward;

        bIsDirty = true;
    }

    void updateViewMatrix() {
        if (bIsDirty) {
            viewMatrix = lookAt(fov, eye, center, up);
            right = { viewMatrix[0][0], viewMatrix[0][1], viewMatrix[0][2] };
            up = { viewMatrix[1][0], viewMatrix[1][1], viewMatrix[1][2] };
            forward = { viewMatrix[2][0], viewMatrix[2][1], viewMatrix[2][2] };
            forward = -forward;
            bIsDirty = false;
        }
    }

    Matrix4& getViewMatrix() {
        return viewMatrix;
    }

private:
    int32_t imageWidth;
    int32_t imageHeight;

    Tuple position;

    Tuple horizontal;
    Tuple vertical;

    Tuple origin;

    Tuple upperLeftCorner;

    Matrix4 viewMatrix;

    Float viewportWidth = 0.0;
    Float viewportHeight = 0.0;
    Float focalLength = 1.0;
    Float cameraSpeed = 6.0f;
    Float aspectRatio;
    Float fov = 90;

    Tuple eye;
    Tuple center;
    Tuple right;
    Tuple up;
    Tuple forward;

    bool bIsDirty = false;
};