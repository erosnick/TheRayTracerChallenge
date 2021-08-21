#pragma once

#include "Constants.h"
#include "CUDA.h"
#include "Types.h"
#include "Light.h"
#include "Array.h"

#include <vector>
#include <algorithm>

class World {
public:
    CUDA_HOST_DEVICE World() {}
    CUDA_HOST_DEVICE ~World();

    CUDA_HOST_DEVICE void intersect(const Ray& ray, Array<Intersection>& totalIntersections);

    CUDA_HOST_DEVICE void addLight(Light* light) {
        lights.add(light);
    }

    CUDA_HOST_DEVICE void addObject(Shape* object) {
        objects.add(object);
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

    CUDA_HOST_DEVICE int32_t ligthCount() const {
        return lights.size();
    }

    CUDA_HOST_DEVICE int32_t objectCount() const {
        return objects.size();
    }

private:
    Array<Shape*> objects;
    Array<Light*> lights;
};