// Catch2Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define CATCH_CONFIG_MAIN

#include <iostream>
#include <array>
#include <vector>
#include <tuple>
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
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
#include "Cube.h"
#include "Quad.h"

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

World mainScene(const Matrix4& viewMatrix) {
    World world;

    auto sphere = std::make_shared<Sphere>(point(2.5, 0.0, -11.0), 1.0);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(color(1.0, 0.0, 0.0), 0.1, 1.0, 0.9, 128.0);
    //sphere->material->pattern = std::make_shared<StripePattern>(Color::blue, Color::red);
    //sphere->material->pattern.value()->transform(rotateZ(Math::pi_4));

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(2.0, -0.5, -9.0), 0.5);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(color(0.0, 1.0, 0.0), 0.1, 1.0, 0.9, 128.0);
    //sphere->material->pattern = std::make_shared<StripePattern>(Color::blue, Color::red);
    //sphere->material->pattern.value()->transform(rotateZ(Math::pi_4));

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(-2.5, 0.0, -11.0), 1.0);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(color(0.0, 0.0, 1.0), 0.1, 1.0, 0.9, 128.0);
    //sphere->material->pattern = std::make_shared<StripePattern>(Color::blue, Color::red);
    //sphere->material->pattern.value()->transform(rotateZ(Math::pi_4));

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(0.0, 0.0, -11.0), 1.0);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(color(0.6, 0.7, 0.8), 0.1, 1.0, 0.9, 128.0);
    //sphere->material->reflective = 0.25;
    //sphere->material->transparency = 1.0;
    //sphere->material->refractiveIndex = 1.5;
    //sphere->material->pattern = std::make_shared<GradientPattern>(Color::red, Color::green);

    world.addObject(sphere);

    auto glassSphere = std::make_shared<Sphere>(point(1.3, 0.3, -7.0), 1.3);
    glassSphere->transform(viewMatrix);
    glassSphere->material = std::make_shared<Material>(color(0.0, 0.0, 0.0), 0.1, 1.0, 0.9, 128.0);
    glassSphere->material->reflective = 0.9;
    glassSphere->material->transparency = 1.0;
    glassSphere->material->refractiveIndex = 1.5;
    //sphere->material->pattern = std::make_shared<GradientPattern>(Color::red, Color::green);

    world.addObject(glassSphere);

    glassSphere = std::make_shared<Sphere>(point(-1.0, 0.0, -7.0), 1.0);
    glassSphere->transform(viewMatrix);
    glassSphere->material = std::make_shared<Material>(color(0.0, 0.0, 0.0), 0.1, 1.0, 0.9, 128.0);
    glassSphere->material->reflective = 0.9;
    glassSphere->material->transparency = 1.0;
    glassSphere->material->refractiveIndex = 2.417;
    //sphere->material->pattern = std::make_shared<GradientPattern>(Color::red, Color::green);

    world.addObject(glassSphere);

    auto steelSphere = std::make_shared<Sphere>(point(0.0, -0.4, -5.0), 0.6);
    steelSphere->transform(viewMatrix);
    steelSphere->material = std::make_shared<Material>(color(0.0, 0.0, 0.0), 0.1, 1.0, 0.9, 128.0);
    steelSphere->material->reflective = 1.0;
    steelSphere->material->transparency = 0.0;

    //world.addObject(steelSphere);

    sphere = std::make_shared<Sphere>(point(-0.7, -0.6, -5.0), 0.4);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(color(0.0, 0.0, 0.0), 0.1, 1.0, 0.9, 128.0);
    sphere->material->reflective = 1.0;

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(0.0, -0.7, -5.0), 0.3);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(Color::crimsonRed, 0.1, 1.0, 0.9, 128.0);

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(0.7, -0.6, -5.0), 0.4);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(color(0.0, 0.0, 0.0), 0.1, 1.0, 0.9, 128.0);
    sphere->material->reflective = 1.0;

    world.addObject(sphere);

    auto triangle = std::make_shared<Triangle>(color(-1.0, -1.0, -1.0), point(-1.0, 0.0, 1.0), point(1.0, 0.0, 1.0));
    triangle->setTransformation(viewMatrix * translate(0.0, 0.0, -6.0) * scaling(3.0, 1.0, 3.0));
    triangle->material->color = color(0.4, 1.0, 0.4);
    triangle->material->pattern = std::make_shared<CheckerPattern>();

    //world.addObject(triangle);

    triangle = std::make_shared<Triangle>(point(-1.0, -1.0, -1.0), point(1.0, 0.0, 1.0), point(1.0, 0.0, -1.0));
    triangle->setTransformation(viewMatrix * translate(0.0, 0.0, -6.0) * scaling(3.0, 1.0, 3.0));
    triangle->material->color = color(0.4, 1.0, 0.4);
    triangle->material->pattern = std::make_shared<CheckerPattern>();

    //world.addObject(triangle);

    sphere = std::make_shared<Sphere>(point(1.0, -1.0, -3.0), 1.0);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(point(0.4, 0.6, 0.9), 0.1, 1.0, 0.9, 128.0);
    //sphere->material->pattern = std::make_shared<RingPattern>(Color::red, Color::green);
    //sphere->material->pattern.value()->transform(scaling(0.125, 0.125, 0.125));

    //world.addObject(sphere);

    //auto floor = std::make_shared<Sphere>(point(0.0, -201.0, -3.0), 200.0);
    //floor->transform(viewMatrix);
    //floor->material->color = color(0.4, 1.0, 0.4);

    //world.addObject(floor);

    //auto ceiling = std::make_shared<Sphere>(point(0.0, 1005.0, -3.0), 1000.0);
    //ceiling->transform(viewMatrix);
    //ceiling->material->color = color(0.4, 0.8, 0.9);

    //world.addObject(ceiling);

    //auto background = std::make_shared<Sphere>(point(0.0, 0.0, -1005.0), 1000.0);
    //background->transform(viewMatrix);
    //background->material->color = color(0.4, 0.8, 0.9);

    //world.addObject(background);

    //auto leftWall = std::make_shared<Sphere>(point(-1005.0, 0.0, -3.0), 1000.0);
    //leftWall->transform(viewMatrix);
    //leftWall->material->color = color(0.4, 0.8, 0.9);

    //world.addObject(leftWall);

    //auto rightWall = std::make_shared<Sphere>(point(1005.0, 0.0, -3.0), 1000.0);
    //rightWall->transform(viewMatrix);
    //rightWall->material->color = color(0.4, 0.8, 0.9);

    //world.addObject(rightWall);

    auto floor = std::make_shared<Plane>(viewMatrix * point(0.0, -1.0, 0.0), viewMatrix * vector(0.0, 1.0, 0.0));
    floor->material->color = color(0.4, 1.0, 0.4);
    floor->material->reflective = 0.125;
    floor->material->pattern = std::make_shared<CheckerPattern>();
    //floor->material->pattern = std::make_shared<CheckerPattern>(color(0.67, 0.67, 0.14), color(0.58, 0.14, 0.0));

    auto ceiling = std::make_shared<Plane>(viewMatrix * point(0.0, 5.0, 0.0), viewMatrix * vector(0.0, -1.0, 0.0));
    ceiling->material->color = color(0.4, 0.8, 0.9);
    //ceiling->material->reflective = 0.25;
    ceiling->material->pattern = std::make_shared<CheckerPattern>();

    auto frontWall = std::make_shared<Plane>(viewMatrix * point(0.0, 0.0, -15.0), viewMatrix * rotateX(Math::pi_2) * vector(0.0, 1.0, 0.0));
    frontWall->material->color = color(0.4, 0.8, 0.9);
    //frontWall->material->specular = 0.1;
    frontWall->material->pattern = std::make_shared<CheckerPattern>(Color::white, Color::black, PlaneOrientation::XY);

    auto backWall = std::make_shared<Plane>(viewMatrix * point(0.0, 0.0, 3.0), viewMatrix * vector(0.0, 0.0, -1.0));
    backWall->material->color = color(0.4, 0.8, 0.9);
    backWall->material->pattern = std::make_shared<CheckerPattern>(Color::white, Color::black, PlaneOrientation::XY);

    auto leftWall = std::make_shared<Plane>(viewMatrix * point(-5.0, 0.0, 0.0), viewMatrix * vector(1.0, 0.0, 0.0));
    leftWall->material->color = color(0.4, 0.8, 0.9);
    leftWall->material->specular = 0.1;
    //leftWall->material->pattern = std::make_shared<CheckerPattern>();

    auto rightWall = std::make_shared<Plane>(viewMatrix * point(5.0, 0.0, 0.0), viewMatrix * vector(-1.0, 0.0, 0.0));
    rightWall->material->color = color(0.4, 0.8, 0.9);
    rightWall->material->specular = 0.1;
    //rightWall->material->pattern = std::make_shared<CheckerPattern>();

    world.addObject(floor);
    world.addObject(ceiling);
    world.addObject(frontWall);
    world.addObject(backWall);
    world.addObject(leftWall);
    world.addObject(rightWall);

    auto light = Light(point(-3.0, 1.0, -10.0), { 0.8, 0.8, 0.8 });
    light.transform(viewMatrix);

    world.addLight(light);

    auto lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    //world.addObject(lightSphere);

    light = Light(point(3.0, 1.0, -7.0), { 0.8, 0.8, 0.8 });
    light.transform(viewMatrix);

    world.addLight(light);

    lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    //world.addObject(lightSphere);

    light = Light(point(0.0, 1.0, -1.0), { 0.8, 0.8, 0.8 });
    light.transform(viewMatrix);

    world.addLight(light);

    lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    //world.addObject(lightSphere);

    return world;
}

World pondScene(const Matrix4& viewMatrix) {
    auto world = World();

    auto background = std::make_shared<Plane>(viewMatrix * point(0.0, 0.0, -40.0), viewMatrix * vector(0.0, 0.0, 1.0));
    background->material->pattern = std::make_shared<CheckerPattern>(Color::white, Color::gray, PlaneOrientation::XY, 0.2);
    background->material->specular = 0.1;

    //world.addObject(background);

    auto water = std::make_shared<Plane>(viewMatrix * point(0.0, -1.0, 0.0), viewMatrix * vector(0.0, 1.0, 0.0));
    water->material->color = Color::black;
    water->material->bCastShadow = false;
    water->material->refractiveIndex = 1.333;
    water->material->reflective = 1.0;
    water->material->transparency = 1.0;

    world.addObject(water);

    auto underwater = std::make_shared<Plane>(viewMatrix * point(0.0, -15.0, 0.0), viewMatrix * vector(0.0, 1.0, 0.0));
    underwater->material->bCastShadow = false;
    underwater->material->pattern = std::make_shared<CheckerPattern>();

    world.addObject(underwater);

    auto sphere = std::make_shared<Sphere>(point(0.0, -0.25, -15.0), 0.5);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(Color::crimsonRed, 0.1, 1.0, 0.9, 128.0);

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(0.0, -3.0, -8.0), 0.5);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(Color::blue, 0.1, 1.0, 0.9, 128.0);

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(2.0, -5.0, -11.0), 0.5);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(Color::pink, 0.1, 1.0, 0.9, 128.0);

    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(-2.0, -5.0, -11.0), 0.5);
    sphere->transform(viewMatrix);
    sphere->material = std::make_shared<Material>(Color::green, 0.1, 1.0, 0.9, 128.0);

    world.addObject(sphere);

    auto light = Light(point(5.0, 3.5, -20.0), { 0.4, 0.4, 0.4 });
    light.transform(viewMatrix);
    light.bAttenuation = false;

    world.addLight(light);

    auto lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    world.addObject(lightSphere);

    light = Light(point(-5.0, 3.5, -20.0), { 0.4, 0.4, 0.4 });
    light.transform(viewMatrix);
    light.bAttenuation = false;

    world.addLight(light);

    lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    world.addObject(lightSphere);

    return world;
}

World houseScene(const Matrix4& viewMatrix) {
    auto world = World();

    auto glassSphere = std::make_shared<Sphere>(point(2.0, -0.55, -6.0), 0.25);
    glassSphere->setTransformation(viewMatrix);
    glassSphere->material->color = Color::black;
    glassSphere->material->shininess = 256.0;
    glassSphere->material->reflective = 0.9;
    glassSphere->material->transparency = 1.0;
    glassSphere->material->refractiveIndex = 1.53;

    world.addObject(glassSphere);

    auto steelSphere = std::make_shared<Sphere>(point(3.5, -0.55, -8.0), 0.25);
    steelSphere->material->color = Color::black;
    steelSphere->material->reflective = 0.9;
    steelSphere->setTransformation(viewMatrix);

    world.addObject(steelSphere);

    auto colorSphere = std::make_shared<Sphere>(point(2.0, -0.55, -8.0), 0.25);
    colorSphere->material->color = Color::green;
    colorSphere->setTransformation(viewMatrix);

    world.addObject(colorSphere);

    //auto floor = std::make_shared<Plane>(point(0.0, -1.0, -8.0));
    //floor->width = 6;
    //floor->height = 6;
    ////floor->setTransformation(viewMatrix);
    ////floor->setTransformation(viewMatrix * rotateY(Math::pi_6));
    //floor->setTransformation(viewMatrix * rotateY(Math::pi_4));
    ////floor->setTransformation(viewMatrix, true);
    //floor->material->color = color(0.4, 1.0, 0.4);
    //floor->material->reflective = 0.125;
    //floor->material->pattern = std::make_shared<CheckerPattern>();
    ////auto transformation = viewMatrix * rotateY(Math::pi_4);
    ////floor->material->pattern.value()->setTransformation(viewMatrix);
    ////floor->material->pattern.value()->setTransformation(rotateY(Math::pi_4));
    ////floor->material->pattern.value()->setTransformation(viewMatrix * rotateY(Math::pi_4));
    //auto transformation = viewMatrix;
    //transformation[3][0] = 0.0f;
    //transformation[3][1] = 0.0f;
    //transformation[3][2] = 0.0;
    ////floor->material->pattern.value()->setTransformation(transformation * rotateY(Math::pi_4));

    //world.addObject(floor);

    //auto top = std::make_shared<Plane>(point(0.0, -1.0, -8.0));
    //top->width = 2;
    //top->height = 2;
    //top->setTransformation(viewMatrix, true);
    //top->material->color = color(0.4, 0.8, 0.9);

    //world.addObject(top);

    //auto front = std::make_shared<Plane>(point(0.0, -1.0, -7.0));
    //front->width = 2;
    //front->height = 2;
    //front->planeOrientation = PlaneOrientation::XZ;
    //front->setTransformation(rotateX(Math::pi_2));
    //front->setTransformation(viewMatrix, true);
    //front->material->color = color(0.4, 0.8, 0.9);

    ////world.addObject(front);

    //auto right = std::make_shared<Plane>(point(0.0, -2.0, -7.0));
    //right->width = 1;
    //right->height = 1;
    //right->planeOrientation = PlaneOrientation::YZ;
    //right->setTransformation(viewMatrix, true);
    //right->material->color = color(0.4, 0.8, 0.9);

    //world.addObject(right);

    auto floor = std::make_shared<Quad>();
    auto transformation = translate(0.0, -2.0, -6.0) * scaling(3.0, 1.0, 3.0);
    floor->setTransformation(transformation);
    floor->transform(viewMatrix);
    floor->material->reflective = 0.125;
    floor->material->pattern = std::make_shared<CheckerPattern>();
    floor->material->pattern.value()->setTransformation(scaling(0.25, 1.0, 0.25));
    //floor->material->pattern = std::make_shared<StripePattern>();
    //floor->material->pattern.value()->setTransformation(scaling(0.25, 1.0, 0.25));

    //world.addObject(floor);

    auto top = std::make_shared<Quad>();
    top->setTransformation(viewMatrix * translate(0.0, 0.0, -6.0));

    //world.addObject(top);

    auto bottom = std::make_shared<Quad>();
    bottom->setTransformation(viewMatrix * translate(0.0, -2.0, -6.0));

    //world.addObject(bottom);

    auto back = std::make_shared<Quad>();
    back->setTransformation(viewMatrix * translate(0.0, -1.0, -7.0) * rotateX(Math::pi_2));

    //world.addObject(back);

    auto front = std::make_shared<Quad>();
    front->setTransformation(viewMatrix * translate(0.0, -1.0, -5.0) * rotateX(Math::pi_2));

    //world.addObject(front);

    auto left = std::make_shared<Quad>();
    left->setTransformation(viewMatrix * translate(-1.0, -1.0, -6.0) * rotateZ(Math::pi_2));

    //world.addObject(left);

    auto right = std::make_shared<Quad>();
    right->setTransformation(viewMatrix * translate(1.0, -1.0, -6.0) * rotateZ(Math::pi_2));

    //world.addObject(right);

    auto cube = std::make_shared<Cube>();
    cube->setTransformation(viewMatrix * translate(-0.5, -0.28, -7.0) * scaling(0.5, 0.5, 0.5));
    auto material = std::make_shared<Material>();
    material->color = Color::black;
    material->reflective = 0.9;
    material->transparency = 0.9;
    material->refractiveIndex = 1.33;
    //material->color = Color::cornflower;
    cube->setMaterial(material);

    world.addObject(cube);
    
    cube = std::make_shared<Cube>();
    cube->setTransformation(viewMatrix * translate(-4.0, -0.55, -5.0) * rotateY(Math::pi_4) * scaling(0.25, 0.25, 0.25));
    material = std::make_shared<Material>();
    material->color = Color::yellow;
    cube->setMaterial(material);

    world.addObject(cube);

    auto block = std::make_shared<Cube>();
    block->setTransformation(viewMatrix * translate(0.0, -0.7, -8.0) * rotateY(-Math::pi_6) * scaling(0.4, 0.08, 0.08));
    material = std::make_shared<Material>();
    material->color = Color::pink;
    block->setMaterial(material);

    world.addObject(block);

    block = std::make_shared<Cube>();
    block->setTransformation(viewMatrix * translate(-3.0, -0.5, -8.0) * rotateZ(-Math::pi_2) * scaling(0.3, 0.08, 0.08));
    material = std::make_shared<Material>();
    material->color = Color::lightGreen;
    block->setMaterial(material);

    world.addObject(block);

    block = std::make_shared<Cube>();
    block->setTransformation(viewMatrix * translate(1.0, -0.38, -8.0) * rotateZ(-Math::pi_2) * scaling(0.4, 0.08, 0.08));
    material = std::make_shared<Material>();
    material->color = Color::pinkBlue;
    block->setMaterial(material);

    world.addObject(block);

    block = std::make_shared<Cube>();
    block->setTransformation(viewMatrix* translate(-2.0, -0.725, -6.0) * rotateY(-Math::pi_6) * scaling(0.4, 0.08, 0.08));
    material = std::make_shared<Material>();
    material->color = Color::background;
    block->setMaterial(material);

    world.addObject(block);

    auto house = std::make_shared<Cube>();
    house->setTransformation(viewMatrix * translate(-0.5, 0.0, -7.0) * scaling(20.0, 6.0, 20.0));
    material = std::make_shared<Material>();
    material->specular = 0.1;
    material->color = color(0.3, 0.6, 1.0);
    house->setMaterial(material);
    material = std::make_shared<Material>();
    material->reflective = 0.5;
    material->specular = 0.5;
    material->pattern = std::make_shared<CheckerPattern>();
    material->pattern.value()->setTransformation(scaling(0.1, 1.0, 0.1));
    house->setMaterial(material, 1);

    world.addObject(house);

    auto desktop = std::make_shared<Cube>();
    desktop->setTransformation(viewMatrix * translate(-0.5, -1.0, -8.0) * scaling(5.0, 0.2, 4.0));
    material = std::make_shared<Material>();
    //material->reflective = 1.0;
    material->pattern = std::make_shared<StripePattern>(Color::cornflower, Color::cornflower * 0.7);
    material->pattern.value()->setTransformation(scaling(0.1, 1.0, 0.1));
    desktop->setMaterial(material);

    world.addObject(desktop);

    auto mirror = std::make_shared<Quad>();
    //mirror->material->bCastShadow = false;
    mirror->material->color = Color::dawn;
    mirror->material->reflective = 1.0;
    mirror->setTransformation(viewMatrix * translate(-4.0, 0.0, -26) * rotateX(Math::pi_2) * scaling(8.0, 1.0, 4.0));

    world.addObject(mirror);

    auto light = Light(point(-18.0, 3.0, 25.0), { 1.0, 1.0, 1.0 });
    light.transform(viewMatrix);
    //light.bAttenuation = false;

    world.addLight(light);

    auto lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    world.addObject(lightSphere);

    light = Light(point(-18.0, 3.0, -25.0), { 1.0, 1.0, 1.0 });
    light.transform(viewMatrix);
    //light.bAttenuation = false;

    world.addLight(light);

    lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    world.addObject(lightSphere);

    light = Light(point(-1.5, 3.0, -0.0), { 1.0, 1.0, 1.0 });
    light.transform(viewMatrix);
    //light.bAttenuation = false;

    world.addLight(light);

    lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    world.addObject(lightSphere);

    return world;
}

World cubeScene(const Matrix4& viewMatrix) {
    auto world = World();

    auto floor = std::make_shared<Quad>("Floor");
    auto transformation = translate(0.0, -2.0, -6.0) * scaling(5.0, 1.0, 5.0);
    floor->setTransformation(transformation);
    floor->transform(viewMatrix);
    floor->transformNormal(viewMatrix);
    floor->material->reflective = 0.125;
    floor->material->pattern = std::make_shared<CheckerPattern>();
    floor->material->pattern.value()->setTransformation(scaling(0.25, 1.0, 0.25));

    world.addObject(floor);

    auto wall = std::make_shared<Quad>("Wall");
    transformation = translate(0.0, -2.0, -10.0) * rotateX(Math::pi_2) * scaling(5.0, 1.0, 5.0);
    wall->setTransformation(transformation);
    wall->transform(viewMatrix);
    wall->material->reflective = 0.125;
    wall->material->pattern = std::make_shared<CheckerPattern>();
    wall->material->pattern.value()->setTransformation(scaling(0.25, 1.0, 0.25));

    //world.addObject(wall);

    // 法线不应该用view matrix进行变换！！！
    // Left和Right因为全内反射是黑色？
    auto cube = std::make_shared<Cube>();
    cube->setTransformation(viewMatrix * translate(0.0, -1.1, -3.0) * scaling(1.0, 1.0, 1.0));
    cube->transformNormal(viewMatrix);
    auto material = std::make_shared<Material>();
    material->color = Color::dawn;
    material->reflective = 0.9;
    material->transparency = 0.9;
    material->refractiveIndex = 1.55;
    //cube->material->bCastShadow = false;
    cube->setMaterial(material);

    //world.addObject(cube);

    cube = std::make_shared<Cube>();
    cube->setTransformation(viewMatrix * translate(0.0, -1.9, -5.0) * scaling(3.0, 0.1, 0.1));
    material = std::make_shared<Material>();
    cube->setMaterial(material);

    //world.addObject(cube);

    auto sphere = std::make_shared<Sphere>(point(2.0, -1.1, -3.0), 0.8);
    sphere->material->color = Color::black;
    sphere->material->reflective = 1.0;
    sphere->material->transparency = 1.0;
    sphere->material->refractiveIndex = 1.55;
    sphere->setTransformation(viewMatrix);

    world.addObject(sphere);

    auto light = Light(point(0.0, 2.0, -6.0), { 0.4, 0.4, 0.4 });
    light.transform(viewMatrix);

    world.addLight(light);

    auto lightSphere = std::make_shared<Sphere>(light.position, 0.25);
    lightSphere->bIsLight = true;

    world.addObject(lightSphere);

    return world;
}

World testScene(const Matrix4& viewMatrix) {
    auto world = World();

    auto sphere = std::make_shared<Sphere>(point(-1.5, 0.0, 0.0));
    sphere->setTransformation(viewMatrix);
    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(1.5, 0.0, 0.0));
    sphere->setTransformation(viewMatrix);
    sphere->setMaterial(std::make_shared<Material>(Color::red, 0.1, 0.9, 0.9, 128.0, 0.0, 0.0, 1.0));
    world.addObject(sphere);

    sphere = std::make_shared<Sphere>(point(0.0, -0.2, 1.8), 0.8);
    sphere->setTransformation(viewMatrix);
    sphere->setMaterial(std::make_shared<Material>(Color::black, 0.1, 0.9, 0.9, 128.0, 1.0, 1.0, 1.5));
    world.addObject(sphere);

    auto quad = std::make_shared<Quad>();
    auto transformation = viewMatrix * translate(0.0, -1.0, 0.0) * rotateY(Math::pi_2) * scaling(3.0, 1.0, 3.0);
    quad->setTransformation(transformation);
    quad->material = std::make_shared<Material>(color(0.0), 0.1, 0.9, 0.9, 128.0, 0.125, 0.0, 1.0);
    quad->material->pattern = std::make_shared<CheckerPattern>();
    quad->material->pattern.value()->transform(scaling(0.25, 1.0, 0.25));
    world.addObject(quad);

    auto light = Light(point(0.0, 1.0, -2.0), Color::white);
    light.transform(viewMatrix);
    world.addLight(light);

    light = Light(point(0.0, 1.0, 3.0), Color::white);
    light.transform(viewMatrix);
    world.addLight(light);

    return world;
}

#define RESOLUTION 1

#include "Array.h"

int main(int argc, char* argv[]) {
#if 1

#if RESOLUTION == 1
    auto canvas = createCanvas(640, 360);
#elif RESOLUTION == 2
    auto canvas = createCanvas(1280, 720);
#elif RESOLUTION == 3
    auto canvas = createCanvas(1920, 1080);
#endif

    constexpr auto samplesPerPixel = 1;

    // 反射次数
    constexpr auto reflectionRemaining = 5;

    // 折射次数
    constexpr auto refractionRemaining = 2;

    auto imageWidth = canvas.getWidth();
    auto imageHeight = canvas.getHeight();

    Camera camera(imageWidth, imageHeight);

    // 摄像机和射线起点位置重合会导致渲染瑕疵(屏幕左上角和右上角出现噪点)，具体原因还待排查(已解决，CheckerPattern算法的问题)
    //auto viewMatrix = camera.lookAt(60.0, point(5.0, 3.0, 6.0), point(0.0, 0.0, -5.0), vector(0.0, 1.0, 0.0));
    //auto viewMatrix = camera.lookAt(60.0, point(0.0, 1.0, 3.0), point(0.0, 0.0, -3.0), vector(0.0, 1.0, 0.0));
    auto viewMatrix = camera.lookAt(60.0, point(0.0, 0.0, 6.0), point(0.0, 0.0, -5.0), vector(0.0, 1.0, 0.0));

    auto world = testScene(viewMatrix);
     //world = cubeScene(viewMatrix);
    
     //world = houseScene(viewMatrix);
        
     //world = pondScene(viewMatrix);

     //world = mainScene(viewMatrix);

     //world = defaultWorld();

    Timer timer;
    double percentage = 0.0;
    #pragma omp parallel for schedule(dynamic, 1)       // OpenMP
    for (auto y = 0; y < imageHeight; y++) {
        percentage = (double)y / (imageHeight - 1) * 100;
        fprintf(stderr, "\rRendering: (%i samples) %.2f%%", samplesPerPixel, percentage);
        for (auto x = 0; x < imageWidth; x++) {
            //std::cout << "Hello, World!, ThreadId = " << omp_get_thread_num();
            auto finalColor = color(0.0, 0.0, 0.0);

            for (auto sample = 0; sample < samplesPerPixel; sample++) {
                auto rx = randomDouble() * 0.0;
                auto ry = randomDouble() * 0.0;
                auto dx = (static_cast<double>(x) + rx) / (imageWidth - 1);
                auto dy = (static_cast<double>(y) + ry) / (imageHeight - 1);

                auto ray = camera.getRay(dx, dy);

                //if (x == 195 && y == 6) {
                    //finalColor += Color::green;
                //}

                //if (x == 0 && y == 76) {
                //    finalColor += Color::green;
                //}

                //if (x == 0 && y == 75) {
                //    finalColor += Color::green;
                //}

                //if (x == 238 && y == 126) {
                //    finalColor += Color::green;
                //}

                finalColor += colorAt(world, ray, reflectionRemaining);
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

    int result = 0; // Catch::Session().run(argc, argv);

    return result;
}
