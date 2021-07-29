#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "Tuple.h"
#include "lodepng.h"

class Canvas {
public:
    Canvas(int32_t inWidth, int32_t inHeight)
        : width(inWidth), height(inHeight) {
        auto size = static_cast<size_t>(width * height);
        pixelData.resize(size, { 0.0f, 0.0f, 0.0f });
    }

    inline int32_t getWidth() const {
        return width;
    }

    inline int32_t getHeight() const {
        return height;
    }

    inline void writePixel(int32_t x, int32_t y, Tuple color) {
        pixelData[y * width + x] = color;
    }

    inline Tuple pixelAt(int32_t x, int32_t y) {
        return pixelData[y * width + x];
    }

    inline std::vector<Tuple>& getPixelData() {
        return pixelData;
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
                const auto& pixelColor = pixelData[index];
                auto r = static_cast<uint8_t>(256 * std::clamp(pixelColor.red, 0.0, 0.999));
                auto g = static_cast<uint8_t>(256 * std::clamp(pixelColor.green, 0.0, 0.999));
                auto b = static_cast<uint8_t>(256 * std::clamp(pixelColor.blue, 0.0, 0.999));
                ss << r << ' ' << g << ' ' << b << '\n';
            }
        }

        //for (int32_t i = pixelData.size() - 1; i >= 0; i--) {
        //    const auto& pixelColor = pixelData[i];
        //    ss << static_cast<int>(256 * std::clamp(pixelColor.red, 0.0, 0.999)) << ' '
        //       << static_cast<int>(256 * std::clamp(pixelColor.green, 0.0, 0.999)) << ' '
        //       << static_cast<int>(256 * std::clamp(pixelColor.blue, 0.0, 0.999)) << '\n';
        //}

        ppm.write(ss.str().c_str(), ss.str().size());

        ppm.close();
    }

    inline void writeToPNG(const std::string& path) {
        std::vector<uint8_t> pixelBuffer;

        for (auto y = height - 1; y >= 0; y--) {
            for (auto x = 0; x < width; x++) {
                auto index = y * width + x;
                const auto& pixelColor = pixelData[index];
                auto r = static_cast<uint8_t>(256 * std::clamp(std::sqrt(pixelColor.red), 0.0, 0.999));
                auto g = static_cast<uint8_t>(256 * std::clamp(std::sqrt(pixelColor.green), 0.0, 0.999));
                auto b = static_cast<uint8_t>(256 * std::clamp(std::sqrt(pixelColor.blue), 0.0, 0.999));
                pixelBuffer.push_back(r);
                pixelBuffer.push_back(g);
                pixelBuffer.push_back(b);
                pixelBuffer.push_back(255);
            }
        }

        //Encode the image
        unsigned error = lodepng::encode(path, pixelBuffer, width, height);
        //if there's an error, display it
        if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }

private:
    std::vector<Tuple> pixelData;

    int32_t width;
    int32_t height;
};

inline Canvas createCanvas(int32_t width, int32_t height) {
    return Canvas(width, height);
}