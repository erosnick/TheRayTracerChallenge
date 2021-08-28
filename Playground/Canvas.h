#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Buffer.h"

class Canvas {
public:
    Canvas(int32_t inWidth, int32_t inHeight) {
        initialize(inWidth, inHeight);
    }

    ~Canvas() {
        uninitialize();
    }

    void initialize(int32_t inWidth, int32_t inHeight) {
        width = inWidth;
        height = inHeight;
        auto size = static_cast<size_t>(width * height * 3);

        gpuErrorCheck(cudaMallocManaged(&pixelBuffer, sizeof(Buffer*)));
        pixelBuffer->initialize(size);
    }

    void uninitialize() {
        pixelBuffer->uninitialize();
        gpuErrorCheck(cudaFree(pixelBuffer));
    }

    inline int32_t getWidth() const {
        return width;
    }

    inline int32_t getHeight() const {
        return height;
    }

    inline CUDA_HOST_DEVICE void writePixel(int32_t x, int32_t y, double red, double green, double blue) {
        auto index = y * width + x;
        writePixel(index, red, green, blue);
    }

    inline CUDA_HOST_DEVICE void writePixel(int32_t index, double red, double green, double blue) {
        (*pixelBuffer)[index * 3] = 256 * std::clamp(sqrt(red), 0.0, 0.999);
        (*pixelBuffer)[index * 3 + 1] = 256 * std::clamp(sqrt(green), 0.0, 0.999);
        (*pixelBuffer)[index * 3 + 2] = 256 * std::clamp(sqrt(blue), 0.0, 0.999);
    }

    //inline Tuple pixelAt(int32_t x, int32_t y) {
    //    return pixelData[y * width + x];
    //}

    inline uint8_t* getPixelBuffer() {
        return pixelBuffer->get();
    }

    inline std::string toPPM() const {
        auto ppm = std::string();
        ppm.append("P3\n");
        ppm.append(std::to_string(width) + " " + std::to_string(height) + "\n");
        ppm.append(std::to_string(255) + "\n");
        return ppm;
    }

    inline void writeToPPM(const std::string& path) {
        auto ppm = std::ofstream(path);

        if (!ppm.is_open()) {
            std::cout << "Open file image.ppm failed.\n";
        }

        std::stringstream ss;
        ss << toPPM();

        for (auto y = height - 1; y >= 0; y--) {
            for (auto x = 0; x < width; x++) {
                auto index = y * width + x;
                auto r = (*pixelBuffer)[index * 3];
                auto g = (*pixelBuffer)[index * 3 + 1];
                auto b = (*pixelBuffer)[index * 3 + 2];
                ss << r << ' ' << g << ' ' << b << '\n';
            }
        }

        ppm.write(ss.str().c_str(), ss.str().size());

        ppm.close();
    }

    inline void writeToPNG(const std::string& path) {
        stbi_write_png("render.png", width, height, 3, pixelBuffer->get(), width * 3);
    }

private:
    Buffer* pixelBuffer;

    int32_t width;
    int32_t height;
};

inline Canvas* createCanvas(int32_t width, int32_t height) {
    Canvas* canvas = nullptr;
    gpuErrorCheck(cudaMallocManaged(&canvas, sizeof(Canvas*)));
    canvas->initialize(width, height);
    return canvas;
}