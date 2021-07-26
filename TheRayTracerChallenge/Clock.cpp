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
 
int main(int argc, char* argv[]) {
    auto canvas = createCanvas(400, 400);

    while (true) {
        projectile = tick(enviroment, projectile);
        projectile.reportStatus();
        if (projectile.position.y <= 0.0) {
            break;
        }
        canvas.writePixel(projectile.position.x, canvas.getHeight() - projectile.position.y, { 1.0, 0.0, 0.0 });
        Sleep(500);
    }

    auto zeroClock = point(150.0, 0.0, 0.0);

    canvas.writePixel(canvas.getWidth() / 2, canvas.getHeight() / 2, { 1.0, 1.0, 1.0, 1.0 });

    for (auto i = 0; i < 12; i++) {
        auto clock = rotationZ(i * PI / 6) * zeroClock;
        canvas.writePixel(clock.x - canvas.getWidth() / 2, canvas.getHeight() / 2 - clock.y, { 1.0, 1.0, 1.0, 1.0 });
    }

    canvas.writeToPPM();

    auto tuple = std::make_tuple<bool, double, double>(true, 10.0, 20.0);

    auto size = std::tuple_size<decltype(tuple)>::value;

    int result = Catch::Session().run(argc, argv);
    return result;
}
