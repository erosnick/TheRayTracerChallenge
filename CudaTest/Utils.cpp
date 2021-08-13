#include "Utils.h"
#include "Tuple.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

std::string toPPM(int32_t width, int32_t height) {
    auto ppm = std::string();
    ppm.append("P3\n");
    ppm.append(std::to_string(width) + " " + std::to_string(height) + "\n");
    ppm.append(std::to_string(255) + "\n");
    return ppm;
}

void writeToPPM(const std::string& path, int32_t width, int32_t height, Tuple* pixelBuffer) {
    auto ppm = std::ofstream(path);

    if (!ppm.is_open()) {
        std::cout << "Open file image.ppm failed.\n";
    }

    std::stringstream ss;
    ss << toPPM(width, height);

    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            auto index = y * width + x;
            const auto& pixelColor = pixelBuffer[index];
            auto r = static_cast<uint32_t>(256 * std::clamp(pixelColor.red, 0.0, 0.999));
            auto g = static_cast<uint32_t>(256 * std::clamp(pixelColor.green, 0.0, 0.999));
            auto b = static_cast<uint32_t>(256 * std::clamp(pixelColor.blue, 0.0, 0.999));
            ss << r << ' ' << g << ' ' << b << '\n';
        }
    }

    ppm.write(ss.str().c_str(), ss.str().size());

    ppm.close();
}