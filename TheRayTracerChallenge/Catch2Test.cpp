// Catch2Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define CATCH_CONFIG_MAIN

#include <iostream>
#include <array>
#include <vector>
#include "catch.hpp"
#include "tuple.h"

class Projectile {
public:
    Projectile() {
        position = tuple::point(0.0f, 0.0f, 0.0f);
        velocity = tuple::vector(0.0f, 0.0f, 0.0f);
    }

    Projectile(const tuple& inPosition, const tuple& inVelocity) {
        position = inPosition;
        velocity = inVelocity;
    }

    void reportStatus() const {
        std::cout << "Location: (" << position.x << ", " << position.y << ", " << position.z << ")" << std::endl;
        std::cout << "Velocity: (" << velocity.x << ", " << velocity.y << ", " << velocity.z << ")" << std::endl;
    }

    tuple position;
    tuple velocity;
};

class Environment {
public:
    Environment() {
        gravity = tuple::vector(0.0f, -9.8f, 0.0f);
        wind = tuple::vector(0.0f, 1.0f, 0.0f);
    }

    Environment(const tuple& inGravity, const tuple& inWind) {
        gravity = inGravity;
        wind = inWind;
    }

    tuple gravity;
    tuple wind;
};

Projectile tick(const Environment& enviroment, const Projectile& projectile) {
    auto position = projectile.position + projectile.velocity;
    auto velocity = projectile.velocity + enviroment.gravity + enviroment.wind;
    return Projectile(position, velocity);
}
 
int main(int argc, char* argv[]) {
    //std::cout << 1.0f / Q_rsqrt(100) << std::endl;
    Projectile projectile = { {0.0f, 10.0f, 0.0f}, {1.0f, 0.0f, 0.0f} };
    Environment enviroment = { {0.0f, -0.1f, 0.0f}, {-0.01f, 0.0f, 0.0f} };

    while (true){
        projectile = tick(enviroment, projectile);
        projectile.reportStatus();
        if (projectile.position.y <= 0.0f) {
            break;
        }
        Sleep(1000);
    }

    int result = Catch::Session().run(argc, argv);
    return result;
}
