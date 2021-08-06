#pragma once

#include "types.h"

#include <vector>
#include <algorithm>
#include "Light.h"

class World {
public:
    InsersectionSet intersect(const Ray& ray) const;

    void addLight(const Light& light) {
        lights.push_back(light);
    }

    void addObject(const ShapePtr& object) {
        objects.push_back(object);
    }

    bool contains(const ShapePtr& object) const {
        if (std::find(objects.begin(), objects.end(), object) != objects.end()) {
            return true;
        }
        
        return false;
    }

    const std::vector<Light>& getLights() const {
        return lights;
    }

    const std::vector<ShapePtr>& getObjects() const {
        return objects;
    }

    const Light& getLight(int32_t index) const {
        return lights[index];
    }

    int32_t ligthCount() const {
        return static_cast<int32_t>(lights.size());
    }

    int32_t objectCount() const {
        return static_cast<int32_t>(objects.size());
    }

private:
    std::vector<ShapePtr> objects;
    std::vector<Light> lights;
};

World defaultWorld();
World defaultWorld1();
World defaultWorld2();
World defaultWorld3();