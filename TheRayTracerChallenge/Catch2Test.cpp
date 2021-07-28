// Catch2Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define CATCH_CONFIG_MAIN

#include <iostream>
#include <array>
#include <vector>
#include <tuple>
#include "catch.hpp"
#include "Tuple.h"
#include "Canvas.h"
#include "Matrix.h"
#include "Sphere.h"
#include "Intersection.h"
#include "Camera.h"
#include "Shading.h"

class Test : public Object {
public:
    virtual void foo() const {
        std::cout << "Test\n";
    }
};

class A : public Test {
public:
    A() {
        std::cout << "A constructor\n";
    }
    virtual void foo() const override {
        std::cout << "A\n";
    }
};

void foo(const Test& object) {
    object.foo();
}

int main(int argc, char* argv[]) {
#if 1
    auto canvas = createCanvas(800, 600);

    auto imageWidth = canvas.getWidth();
    auto imageHeight = canvas.getHeight();

    auto from = point(0.0, 0.0, 0.0);
    auto to = point(0.0, 0.0, 0.0);
    auto up = vector(0.0, 1.0, 0.0);

    auto viewMatrix = viewTransform(from, to, up);

    World world;

    auto sphere = Sphere();
    sphere.setTransform(viewMatrix * translation(-1.0, 0.0, -2.5));
    sphere.material = { { 1.0, 0.0, 0.0}, 0.1, 1.0, 0.9, 128.0 };

    world.addObject(sphere);

    sphere = Sphere();
    sphere.setTransform(viewMatrix * translation(1.0, 0.0, -2.5));
    sphere.material = { { 1.0, 0.2, 1.0}, 0.1, 1.0, 0.9, 128.0 };

    world.addObject(sphere);

    Camera camera(imageWidth, imageHeight);

    auto light = Light({ point(-2.0, 2.0, 0.0) }, { 1.0, 1.0, 1.0 });

    world.addLight(light);
    
    //world = defaultWorld();

    for (auto y = 0; y < imageHeight; y++) {
        for (auto x = 0; x < imageWidth; x++) {
            auto dx = static_cast<double>(x) / (imageWidth - 1);
            auto dy = static_cast<double>(y) / (imageHeight - 1);

            auto ray = camera.getRay(dx, dy);

            auto finalColor = colorAt(world, ray);

            canvas.writePixel(x, y, finalColor);
        }
    }

    canvas.writeToPPM();

#endif

    A a;

    std::vector<A> objects;

    objects.push_back(a);
    objects.push_back(a);

    for (const auto& object : objects) {
        foo(object);
    }

    auto tuple = std::make_tuple<bool, double, double>(true, 10.0, 20.0);

    auto size = std::tuple_size<decltype(tuple)>::value;

    int result = Catch::Session().run(argc, argv);
    return result;
}
