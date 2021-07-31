#pragma once

#include "Sphere.h"
#include "Light.h"
#include "Ray.h"
#include "Intersection.h"
#include "types.h"

#include <vector>
#include <algorithm>

class World {
public:
    std::vector<Intersection> intersect(const Ray& ray) const {
        std::vector<Intersection> intersections;

        for (const auto& object : objects) {
            auto intersection = object->intersect(ray, false);
            intersections.insert(intersections.end(), intersection.begin(), intersection.end());
        }

        std::sort(intersections.begin(), intersections.end());

        return intersections;
    }

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
        return lights.size();
    }

    int32_t objectCount() const {
        return objects.size();
    }

private:
    std::vector<ShapePtr> objects;
    std::vector<Light> lights;
};

inline World defaultWorld() {
    auto world = World();

    auto sphere = std::make_shared<Sphere>();
    sphere->setTransformation(translate(-1.0, 0.0, -3.0));
    sphere->material = { { 1.0, 0.0, 0.0}, 0.1, 1.0, 0.9, 128.0 };

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>();
    sphere->setTransformation(translate(1.0, 0.0, -3.0));
    sphere->material = { { 1.0, 0.2, 1.0}, 0.1, 1.0, 0.9, 128.0 };

    world.addObject(sphere);

    auto light = Light({ point(-2.0, 2.0, 0.0) }, { 1.0, 1.0, 1.0 });

    world.addLight(light);

    return world;
}

inline World defaultWorld1() {
    auto world = World();

    auto sphere = std::make_shared<Sphere>();
    sphere->material = { { 0.8, 1.0, 0.6}, 0.1, 0.7, 0.2, 128.0 };

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>();
    sphere->setTransformation(scaling(0.5, 0.5, 0.5));

    world.addObject(sphere);

    auto light = Light({ point(-2.0, 2.0, -2.0) }, { 1.0, 1.0, 1.0 });

    world.addLight(light);

    return world;
}

inline World defaultWorld2() {
    auto world = World();

    auto sphere = std::make_shared<Sphere>();
    sphere->material = { { 0.8, 1.0, 0.6}, 0.1, 0.7, 0.2, 128.0 };

    world.addObject(sphere);

    auto light = Light({ point(-10.0, 10.0, 0.0) }, { 1.0, 1.0, 1.0 });

    world.addLight(light);

    return world;
}