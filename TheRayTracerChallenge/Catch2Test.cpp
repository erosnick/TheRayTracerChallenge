// Catch2Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define CATCH_CONFIG_MAIN

#include <iostream>
#include <array>
#include <vector>
#include "catch.hpp"
#include "Tuple.h"
#include "Canvas.h"
#include "Matrix.h"

class Projectile {
public:
    Projectile() {
        position = point(0.0, 0.0, 0.0);
        velocity = vector(0.0, 0.0, 0.0);
    }

    Projectile(const Tuple& inPosition, const Tuple& inVelocity) {
        position = inPosition;
        velocity = inVelocity;
    }

    void reportStatus() const {
        std::cout << "Location: (" << position.x << ", " << position.y << ", " << position.z << ")" << std::endl;
        std::cout << "Velocity: (" << velocity.x << ", " << velocity.y << ", " << velocity.z << ")" << std::endl;
    }

    Tuple position;
    Tuple velocity;
};

using aaa = Projectile;

class Environment {
public:
    Environment() {
        gravity = vector(0.0, -9.8, 0.0);
        wind = vector(0.0, 1.0, 0.0);
    }

    Environment(const Tuple& inGravity, const Tuple& inWind) {
        gravity = inGravity;
        wind = inWind;
    }

    Tuple gravity;
    Tuple wind;
};

Projectile tick(const Environment& enviroment, const Projectile& projectile) {
    auto position = projectile.position + projectile.velocity;
    auto velocity = projectile.velocity + enviroment.gravity + enviroment.wind;
    return Projectile(position, velocity);
}
 
int main(int argc, char* argv[]) {
    //auto position = point(0.0, 1.0, 0.0);
    //auto velocity = vector(1.0, 1.8, 0.0);
    //velocity = velocity.normalize() * 11.25;
    //Projectile projectile = { position, velocity };
    //Environment enviroment = { {0.0, -0.1, 0.0}, {-0.01, 0.0, 0.0} };

    //auto canvas = createCanvas(900, 550);

    //while (true){
    //    projectile = tick(enviroment, projectile);
    //    projectile.reportStatus();
    //    if (projectile.position.y <= 0.0) {
    //        break;
    //    }
    //    canvas.writePixel(projectile.position.x, canvas.getHeight() - projectile.position.y, { 1.0, 0.0, 0.0 });
    //    Sleep(500);
    //}

    //canvas.writeToPPM();

    Matrix4 matrix;

    std::cout << matrix[3][3] << std::endl;

    int result = Catch::Session().run(argc, argv);
    return result;
}
