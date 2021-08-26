#pragma once

#include "CUDA.h"
#include "Array.h"
#include "Intersection.h"
#include <cstdint>

struct Payload {
    class World* world;
    struct Viewport* viewport;
    uint8_t* pixelBuffer;
    class Camera* camera;
    Array<Intersection>* intersections;
};

struct ImageData {
    ImageData()
    : data(nullptr) {}
    ~ImageData() {
    }

    uint8_t* data;
    int32_t size;
    int32_t width;
    int32_t height;
    int32_t channels;
};

extern Payload* payload;
extern class World** world;

extern Array<class Shape**> objects;
extern Array<class Light**> lights;
extern Array<struct Material**> materials;

void initialize(int32_t width, int32_t height);

ImageData* launch(int32_t width, int32_t height);

void updateObjects(const Array<Shape**>& objects, const class Matrix4& transformation);
CUDA_GLOBAL void updateObjectsKernel(Array<Shape**> objects, Matrix4 transformation);

void updateObjects(class World* world, const class Matrix4& transformation);
CUDA_GLOBAL void updateObjectsKernel(class World* world, Matrix4 transformation);

void cleanup();
