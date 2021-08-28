#pragma once

#include <string>

namespace Utils {
    std::string toPPM(int32_t width, int32_t height);
    void writeToPPM(const std::string& path, int32_t width, int32_t height, uint8_t* pixelBuffer);
    void writeToPNG(const std::string& path, int32_t width, int32_t height, uint8_t* pixelBuffer);
    void openImage(const std::wstring& path);
}