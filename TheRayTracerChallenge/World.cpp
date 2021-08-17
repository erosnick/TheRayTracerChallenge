#include "World.h"
#include "Sphere.h"
#include "Ray.h"
#include "Intersection.h"

InsersectionSet World::intersect(const Ray& ray) const {
    InsersectionSet intersections;

    for (const auto& object : objects) {
        auto intersection = object->intersect(ray, false);
        intersections.insert(intersections.end(), intersection.begin(), intersection.end());
    }

    std::sort(intersections.begin(), intersections.end());

    return intersections;
}

World defaultWorld() {
    auto world = World();

    auto sphere = std::make_shared<Sphere>();
    sphere->setTransformation(translate(-1.0, 0.0, -3.0));
    sphere->material = std::make_shared<Material>(color(1.0, 0.0, 0.0), 0.1, 1.0, 0.9, 128.0);

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>();
    sphere->setTransformation(translate(1.0, 0.0, -3.0));
    sphere->material = std::make_shared<Material>(color(1.0, 0.2, 1.0), 0.1, 1.0, 0.9, 128.0);

    world.addObject(sphere);

    auto light = Light({ point(-2.0, 2.0, 0.0) }, { 1.0, 1.0, 1.0 });

    world.addLight(light);

    return world;
}

World defaultWorld1() {
    auto world = World();

    auto sphere = std::make_shared<Sphere>();
    sphere->material = std::make_shared<Material>(color(0.8, 1.0, 0.6), 0.1, 0.7, 0.2, 128.0);

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>();
    sphere->setTransformation(scaling(0.5, 0.5, 0.5));

    world.addObject(sphere);

    auto light = Light({ point(-2.0, 2.0, -2.0) }, { 1.0, 1.0, 1.0 });

    world.addLight(light);

    return world;
}

World defaultWorld2() {
    auto world = World();

    auto sphere = std::make_shared<Sphere>();
    sphere->material = std::make_shared<Material>(color(0.8, 1.0, 0.6), 0.1, 0.7, 0.2, 128.0);

    world.addObject(sphere);

    auto light = Light({ point(-10.0, 10.0, 0.0) }, { 1.0, 1.0, 1.0 });

    world.addLight(light);

    return world;
}

World defaultWorld3() {
    auto world = World();

    auto sphere = std::make_shared<Sphere>();
    sphere->material = std::make_shared<Material>(color(0.8, 1.0, 0.6), 0.1, 0.7, 0.2, 128.0);

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>();
    sphere->setTransformation(scaling(0.5, 0.5, 0.5));

    world.addObject(sphere);

    auto light = Light({ point(-2.0, 2.0, 2.0) }, { 1.0, 1.0, 1.0 });

    world.addLight(light);

    return world;
}

