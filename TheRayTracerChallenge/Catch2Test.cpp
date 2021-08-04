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
#include "Plane.h"
#include "Intersection.h"
#include "Camera.h"
#include "Shading.h"
#include "utils.h"
#include "Timer.h"
#include "Pattern.h"
#include "World.h"
#include "Light.h"
#include "Triangle.h"

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

#define RESOLUTION 1

int main(int argc, char* argv[]) {
#if 1

#if RESOLUTION == 1
    auto canvas = createCanvas(640, 360);
#elif RESOLUTION == 2
    auto canvas = createCanvas(1280, 720);
#elif RESOLUTION == 3
    auto canvas = createCanvas(1920, 1080);
#endif

    constexpr auto samplesPerPixel = 8;

    constexpr auto remaining = 5;

    auto imageWidth = canvas.getWidth();
    auto imageHeight = canvas.getHeight();

    Camera camera(imageWidth, imageHeight);

    // 摄像机和射线起点位置重合会导致渲染瑕疵(屏幕左上角和右上角出现噪点)，具体原因还待排查
    auto viewMatrix = camera.lookAt(60.0, point(0.0, 1.0, 0.1), point(0.0, 1.0, -1.0), vector(0.0, 1.0, 0.0));

    World world;

    auto sphere = std::make_shared<Sphere>(point(-1.0, 0.0, -6.0), 1.0);
    sphere->transform(viewMatrix);
    sphere->material = { { 1.0, 0.0, 0.0}, 0.1, 1.0, 0.9, 128.0 };
    //sphere->material.pattern = std::make_shared<StripePattern>(Color::blue, Color::red);
    //sphere->material.pattern.value()->transform(rotateZ(Math::pi_4));

    //world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(-1.0, 1.3, -10.0), 1.3);
    sphere->transform(viewMatrix);
    sphere->material = { { 0.6, 0.7, 0.8 }, 0.1, 1.0, 0.9, 128.0 };
    //sphere->material.reflective = 0.25;
    //sphere->material.transparency = 1.0;
    //sphere->material.refractiveIndex = 1.5;
    //sphere->material.pattern = std::make_shared<GradientPattern>(Color::red, Color::green);

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(0.25, 1.0, -6.0), 1.0);
    sphere->transform(viewMatrix);
    sphere->material = { { 0.0, 0.0, 0.0 }, 0.1, 1.0, 0.9, 128.0 };
    sphere->material.reflective = 0.05;
    sphere->material.transparency = 1.0;
    sphere->material.refractiveIndex = 1.5;
    //sphere->material.pattern = std::make_shared<GradientPattern>(Color::red, Color::green);

    world.addObject(sphere);

    auto triangle = std::make_shared<Triangle>(point(-1.0, 0.0, -1.0), point(-1.0, 0.0, 1.0), point(1.0, 0.0, 1.0));
    triangle->setTransformation(viewMatrix * translate(0.0, 0.0, -6.0) * scaling(3.0, 1.0, 3.0));
    triangle->material.color = color(0.4, 1.0, 0.4);
    //triangle->material.pattern = std::make_shared<CheckerPattern>();

    world.addObject(triangle);

    triangle = std::make_shared<Triangle>(point(-1.0, 0.0, -1.0), point(1.0, 0.0, 1.0), point(1.0, 0.0, -1.0));
    triangle->setTransformation(viewMatrix * translate(0.0, 0.0, -6.0) * scaling(3.0, 1.0, 3.0));
    triangle->material.color = color(0.4, 1.0, 0.4);
    //triangle->material.pattern = std::make_shared<CheckerPattern>();

    world.addObject(triangle);

    sphere = std::make_shared<Sphere>(point(1.0, 0.0, -3.0), 1.0);
    sphere->transform(viewMatrix);
    sphere->material = { { 0.4, 0.6, 0.9 }, 0.1, 1.0, 0.9, 128.0 };
    //sphere->material.pattern = std::make_shared<RingPattern>(Color::red, Color::green);
    //sphere->material.pattern.value()->transform(scaling(0.125, 0.125, 0.125));

    //world.addObject(sphere);

    //auto floor = std::make_shared<Sphere>(point(0.0, -201.0, -3.0), 200.0);
    //floor->transform(viewMatrix);
    //floor->material.color = color(0.4, 1.0, 0.4);

    //world.addObject(floor);

    //auto ceiling = std::make_shared<Sphere>(point(0.0, 1005.0, -3.0), 1000.0);
    //ceiling->transform(viewMatrix);
    //ceiling->material.color = color(0.4, 0.8, 0.9);

    //world.addObject(ceiling);

    //auto background = std::make_shared<Sphere>(point(0.0, 0.0, -1005.0), 1000.0);
    //background->transform(viewMatrix);
    //background->material.color = color(0.4, 0.8, 0.9);

    //world.addObject(background);

    //auto leftWall = std::make_shared<Sphere>(point(-1005.0, 0.0, -3.0), 1000.0);
    //leftWall->transform(viewMatrix);
    //leftWall->material.color = color(0.4, 0.8, 0.9);

    //world.addObject(leftWall);

    //auto rightWall = std::make_shared<Sphere>(point(1005.0, 0.0, -3.0), 1000.0);
    //rightWall->transform(viewMatrix);
    //rightWall->material.color = color(0.4, 0.8, 0.9);

    //world.addObject(rightWall);

    auto floor = std::make_shared<Plane>(viewMatrix * point(0.0, 0.0, 0.0), viewMatrix * vector(0.0, 1.0, 0.0));
    floor->material.color = color(0.4, 1.0, 0.4);
    floor->material.reflective = 0.125;
    floor->material.pattern = std::make_shared<CheckerPattern>();
    //floor->material.pattern = std::make_shared<CheckerPattern>(color(0.67, 0.67, 0.14), color(0.58, 0.14, 0.0));

    auto ceiling = std::make_shared<Plane>(viewMatrix * point(0.0, 6.0, 0.0), viewMatrix * vector(0.0, -1.0, 0.0));
    ceiling->material.color = color(0.4, 0.8, 0.9);
    //ceiling->material.reflective = 0.25;
    ceiling->material.pattern = std::make_shared<CheckerPattern>();

    auto frontWall = std::make_shared<Plane>(viewMatrix * point(0.0, 0.0, -15.0), viewMatrix * rotateX(Math::pi_2) * vector(0.0, 1.0, 0.0));
    frontWall->material.color = color(0.4, 0.8, 0.9);
    //frontWall->material.specular = 0.1;
    frontWall->material.pattern = std::make_shared<CheckerPattern>();

    auto backWall = std::make_shared<Plane>(viewMatrix * point(0.0, 0.0, 5.0), viewMatrix * vector(0.0, 0.0, -1.0));
    backWall->material.color = color(0.4, 0.8, 0.9);
    backWall->material.pattern = std::make_shared<CheckerPattern>();

    auto leftWall = std::make_shared<Plane>(viewMatrix * point(-5.0, 0.0, 0.0), viewMatrix * vector(1.0, 0.0, 0.0));
    leftWall->material.color = color(0.4, 0.8, 0.9);
    leftWall->material.specular = 0.1;
    //leftWall->material.pattern = std::make_shared<CheckerPattern>();

    auto rightWall = std::make_shared<Plane>(viewMatrix * point(5.0, 0.0, 0.0), viewMatrix * vector(-1.0, 0.0, 0.0));
    rightWall->material.color = color(0.4, 0.8, 0.9);
    rightWall->material.specular = 0.1;
    //rightWall->material.pattern = std::make_shared<CheckerPattern>();

    //world.addObject(floor);
    world.addObject(ceiling);
    world.addObject(frontWall);
    world.addObject(frontWall);
    world.addObject(leftWall);
    world.addObject(rightWall);

    auto light = Light(point(-3.0, 2.0, -10.0), { 0.8, 0.8, 0.8 });
    light.transform(viewMatrix);

    world.addLight(light);

    auto lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    //world.addObject(lightSphere);

    light = Light(point(3.0, 2.0, -7.0), { 0.8, 0.8, 0.8 });
    light.transform(viewMatrix);

    world.addLight(light);

    lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    //world.addObject(lightSphere);

    //world = defaultWorld();

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

                finalColor += colorAt(world, ray, remaining);
            }

            canvas.writePixel(x, y, finalColor / samplesPerPixel);
        }
    }

    timer.stop();

    //canvas.writeToPPM("./render.ppm");
    canvas.writeToPNG("./render.png");
    openImage(L"./render.png");
#endif
    auto tuple = std::make_tuple<bool, double, double>(true, 10.0, 20.0);

    auto size = std::tuple_size<decltype(tuple)>::value;

    int result = Catch::Session().run(argc, argv);
    return result;
}
