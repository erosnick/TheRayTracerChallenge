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
    auto canvas = createCanvas(800, 600);

    auto imageWidth = canvas.getWidth();
    auto imageHeight = canvas.getHeight();
    
    std::vector<Sphere> scene;

    scene.push_back({ point(-1.0, 0.0, -3.0), 1.0 });
    scene.push_back({ point(1.0, 0.0, -3.0), 0.8 });

    Sphere sphere(point(0.0, 0.0, 3.0), 1.0);

    Camera camera(imageWidth, imageHeight);

    auto lightPosition = point(-1.25, 0.75, 1.0);

    auto ambientColor = color(0.1, 0.1, 0.1);
    auto diffuseColor = color(1.0, 0.0, 0.0);
    auto specularColor = color(1.0, 1.0, 1.0);

    //auto constant = 1.0;
    //auto linear = 0.09;
    //auto quadratic = 0.032;
    auto constant = 1.0;
    auto linear = 0.045;
    auto quadratic = 0.0075;

    for (auto y = 0; y < imageHeight; y++) {
        for (auto x = 0; x < imageWidth; x++) {
            auto dx = static_cast<double>(x) / (imageWidth - 1);
            auto dy = static_cast<double>(y) / (imageHeight - 1);

            auto ray = camera.getRay(dx, dy);

            for (const auto& object : scene) {
                auto result = hit(object.intersect(ray));

                if (result.bHit) {
                    auto lightDirection = (lightPosition - result.position);
                    auto distance = lightDirection.magnitude();
                    auto attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));
                    lightDirection = lightDirection / distance;
                    auto diffuseTerm = result.normal.dot(lightDirection);
                    auto diffuse = std::max(diffuseTerm, 0.0) * attenuation;

                    auto specular = 0.0;

                    if (diffuseTerm > 0) {
                        auto reflectVector = 2.0 * (diffuseTerm) * result.normal - lightDirection;
                        auto viewDirection = (camera.position - result.position).normalize();
                        specular = std::pow(std::max(lightDirection.dot(reflectVector), 0.0), 128.0) * attenuation;
                    }

                    auto finalColor = ambientColor + diffuseColor * diffuse + specularColor * specular;

                    canvas.writePixel(x, y, finalColor);
                }
            }
        }
    }

    canvas.writeToPPM();

    auto tuple = std::make_tuple<bool, double, double>(true, 10.0, 20.0);

    auto size = std::tuple_size<decltype(tuple)>::value;

    int result = Catch::Session().run(argc, argv);
    return result;
}
