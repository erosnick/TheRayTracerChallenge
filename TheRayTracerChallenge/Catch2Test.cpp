// Catch2Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define CATCH_CONFIG_MAIN

#include <iostream>
#include <array>
#include <vector>
#include <tuple>
#include <windows.h>
#include <omp.h>

#include "catch.hpp"
#include "Tuple.h"
#include "Canvas.h"
#include "Matrix.h"
#include "Sphere.h"
#include "Intersection.h"
#include "Camera.h"
#include "Shading.h"
#include "utils.h"
#include "Timer.h"

void openImage(const std::wstring& path) {

    auto lastSlashPosition = path.find_last_of('/');
    auto imageName = path.substr(lastSlashPosition + 1);

    SHELLEXECUTEINFO execInfo = { 0 };
    execInfo.cbSize = sizeof(SHELLEXECUTEINFO);
    execInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
    execInfo.hwnd = nullptr;
    execInfo.lpVerb = L"open";
    execInfo.lpFile = L"C:\\Windows\\System32\\mspaint.exe";
    execInfo.lpParameters = imageName.c_str();
    execInfo.lpDirectory = path.c_str();
    execInfo.nShow = SW_SHOW;
    execInfo.hInstApp = nullptr;

    ShellExecuteEx(&execInfo);

    WaitForSingleObject(execInfo.hProcess, INFINITE);
}

class Base : public std::enable_shared_from_this<Base> {
public:
    std::shared_ptr<Base> GetPtr() {
        return shared_from_this();
    }
};

class Widget : public Base {
public:

};

int main(int argc, char* argv[]) {
#if 1
    auto canvas = createCanvas(640, 360);

    auto imageWidth = canvas.getWidth();
    auto imageHeight = canvas.getHeight();

    Camera camera(imageWidth, imageHeight);

    auto viewMatrix = camera.lookAt(60.0, point(0.0, 0.0, 8.0), point(0.0, 0.0, -1.0), vector(0.0, 1.0, 0.0));

    World world;

    auto sphere = std::make_shared<Sphere>(point(-1.0, 0.0, -3.0), 1.0);
    sphere->transform(viewMatrix);
    sphere->material = { { 1.0, 0.0, 0.0}, 0.1, 1.0, 0.9, 128.0 };

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(1.0, 0.0, -3.0), 1.0);
    sphere->transform(viewMatrix);
    sphere->material = { { 1.0, 0.2, 1.0}, 0.1, 1.0, 0.9, 128.0 };

    world.addObject(sphere);

    auto floor = std::make_shared<Sphere>(point(0.0, -1001.0, -3.0), 1000.0);
    floor->transform(viewMatrix);
    floor->material.color = color(0.4, 1.0, 0.4);

    world.addObject(floor);

    auto ceiling = std::make_shared<Sphere>(point(0.0, 1005.0, -3.0), 1000.0);
    ceiling->transform(viewMatrix);
    ceiling->material.color = color(0.4, 0.8, 0.9);

    world.addObject(ceiling);

    auto fronWall = std::make_shared<Sphere>(point(0.0, 0.0, -1005.0), 1000.0);
    fronWall->transform(viewMatrix);
    fronWall->material.color = color(0.4, 0.8, 0.9);

    world.addObject(fronWall);

    auto leftWall = std::make_shared<Sphere>(point(-1005.0, 0.0, -3.0), 1000.0);
    leftWall->transform(viewMatrix);
    leftWall->material.color = color(0.4, 0.8, 0.9);

    world.addObject(leftWall);

    auto rightWall = std::make_shared<Sphere>(point(1005.0, 0.0, -3.0), 1000.0);
    rightWall->transform(viewMatrix);
    rightWall->material.color = color(0.4, 0.8, 0.9);

    world.addObject(rightWall);

    auto lightPosition = point(-3.0, 2.0, -3.0);

    auto light = Light(viewMatrix * lightPosition, { 1.0, 1.0, 1.0 });

    world.addLight(light);

    auto lightSphere = std::make_shared<Sphere>(lightPosition, 0.25);
    lightSphere->transform(viewMatrix);
    lightSphere->bIsLight = true;
    
    world.addObject(lightSphere);

    //world = defaultWorld();

    auto samplesPerPixel = 8;

    Timer timer;
    #pragma omp parallel for schedule(dynamic, 4)       // OpenMP
    for (auto y = 0; y < imageHeight; y++) {
        auto percentage = (double)y / (imageHeight - 1) * 100;
        fprintf(stderr, "\rRendering: (%i samples) %.2f%%", samplesPerPixel, percentage);
        for (auto x = 0; x < imageWidth; x++) {
            //std::cout << "Hello, World!, ThreadId = " << omp_get_thread_num();
            auto finalColor = color(0.0, 0.0, 0.0);

            for (auto sample = 0; sample < samplesPerPixel; sample++) {
                auto rx = randomDouble();
                auto ry = randomDouble();
                auto dx = (static_cast<double>(x) + rx) / (imageWidth - 1);
                auto dy = (static_cast<double>(y) + ry) / (imageHeight - 1);

                auto ray = camera.getRay(dx, dy);

                finalColor += colorAt(world, ray);
            }

            canvas.writePixel(x, y, finalColor / samplesPerPixel);
        }
    }

    timer.stop();

    //canvas.writeToPPM("./render.ppm");
    canvas.writeToPNG("./render.png");
    openImage(L"./render.png");
#endif

    auto widget = std::make_shared<Widget>();

    auto widgetPtr = widget->GetPtr();

    auto tuple = std::make_tuple<bool, double, double>(true, 10.0, 20.0);

    auto size = std::tuple_size<decltype(tuple)>::value;

    int result = Catch::Session().run(argc, argv);
    return result;
}
