#pragma once

#include "Shape.h"

#include <string>

class Quad : public Shape {
public:

    CUDA_HOST_DEVICE Quad(char* inName = "Quad");
    CUDA_HOST_DEVICE virtual ~Quad() {
        for (auto i = 0; i < 2; i++) {
            if (triangles[i]) {
                delete triangles[i];
            }
        }
    }

    CUDA_HOST_DEVICE void setTransformation(const Matrix4& inTransformation, bool bTransformPosition  = false ) override;

    CUDA_HOST_DEVICE void transform(const Matrix4& inTransformation) override;

    CUDA_HOST_DEVICE void transformNormal(const Matrix4& worldMatrix);

    CUDA_HOST_DEVICE Tuple normalAt(const Tuple& position) const override;

    CUDA_HOST_DEVICE bool intersect(const Ray& ray, Intersection* intersections) override;

    CUDA_HOST_DEVICE bool onQuad(const Tuple& inPosition, Tuple& normal);

    CUDA_HOST_DEVICE void setMaterial(Material* inMaterial) override;

    bool bCube = false;

private:
    class Triangle* triangles[2];
    char* name;
};