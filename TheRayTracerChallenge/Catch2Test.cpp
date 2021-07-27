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
 
int main(int argc, char* argv[]) {
    auto canvas = createCanvas(800, 600);

    auto imageWidth = canvas.getWidth();
    auto imageHeight = canvas.getHeight();
    
    std::vector<Sphere> scene;

    auto sphere = Sphere();
    sphere.setTransform(translation(-1.0, 0.0, -3.0) * scaling(1.0, 0.5, 1.0));
    sphere.material = { { 1.0, 0.0, 0.0} };

    scene.push_back(sphere);

    sphere = Sphere();
    sphere.setTransform(translation(1.0, 0.0, -3.0));
    sphere.material = { { 1.0, 0.0, 0.0} };

    scene.push_back(sphere);

    Camera camera(imageWidth, imageHeight);

    auto light = Light({ point(-1.25, 0.75, 1.0) }, { 1.0, 1.0, 1.0 });

#if 1
    for (auto y = 0; y < imageHeight; y++) {
        for (auto x = 0; x < imageWidth; x++) {
            auto dx = static_cast<double>(x) / (imageWidth - 1);
            auto dy = static_cast<double>(y) / (imageHeight - 1);

            auto ray = camera.getRay(dx, dy);

            for (const auto& object : scene) {
                auto result = hit(object.intersect(ray));

                if (result.bHit) {
                    auto finalColor = Lighting(object.material, result.position, light, camera.position, result.normal);

                    canvas.writePixel(x, y, finalColor);
                }
            }
        }
    }

    canvas.writeToPPM();

#endif

    auto tuple = std::make_tuple<bool, double, double>(true, 10.0, 20.0);

    auto size = std::tuple_size<decltype(tuple)>::value;

    int result = Catch::Session().run(argc, argv);
    return result;
}
