#pragma once

#include <cstdint>

struct Payload {
    class World* world;
    struct Viewport* viewport;
    uint8_t* pixelBuffer;
    class Camera* camera;
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

void initialize(int32_t width, int32_t height);

ImageData* launch(int32_t width, int32_t height);

void cleanup();
