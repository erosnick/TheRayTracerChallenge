#pragma once

#include "Constants.h"
#include "CUDA.h"
#include "Types.h"
#include <vector>
#include <algorithm>
#include "Light.h"

class World {
public:
    CUDA_HOST_DEVICE void foo(const Ray& ray, int32_t* count);
    CUDA_HOST_DEVICE void intersect(const Ray& ray, Intersection* totalIntersections, int32_t* count);

    void addLight(Light* light) {
        if (lightIndex < MAXELEMENTS - 1) {
            lights[lightIndex + 1] = light;
        }
    }

    void addObject(Shape* object) {
        if (objectIndex < MAXELEMENTS - 1) {
            objects[objectIndex + 1] = object;
        }
    }

    bool contains(Shape* object) const {
        for (auto i = 0; i < objectCount(); i++) {
            if (object == objects[i]) {
                return true;
            }
        }
        return false;
    }

    CUDA_HOST_DEVICE Shape* getObject(int32_t index) const {
        return objects[index];
    }

    CUDA_HOST_DEVICE Light* getLight(int32_t index) const {
        return lights[index];
    }

    int32_t ligthCount() const {
        return lightIndex + 1;
    }

    int32_t objectCount() const {
        return objectIndex + 1;
    }

private:
    int32_t objectIndex;
    int32_t lightIndex;
    Shape* objects[MAXELEMENTS];
    Light* lights[MAXELEMENTS];
};