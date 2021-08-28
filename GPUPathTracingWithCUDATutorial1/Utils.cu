#include "Utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#define NOMINMAX
#include <windows.h>
#undef NOMINMAX

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace Utils {
    std::string toPPM(int32_t width, int32_t height) {
        auto ppm = std::string();
        ppm.append("P3\n");
        ppm.append(std::to_string(width) + " " + std::to_string(height) + "\n");
        ppm.append(std::to_string(255) + "\n");
        return ppm;
    }

    void writeToPPM(const std::string& path, uint8_t* pixelBuffer, int32_t width, int32_t height) {
        auto ppm = std::ofstream(path);

        if (!ppm.is_open()) {
            std::cout << "Open file image.ppm failed.\n";
        }

        std::stringstream ss;
        ss << toPPM(width, height);

        for (auto y = height - 1; y >= 0; y--) {
            for (auto x = 0; x < width; x++) {
                auto index = y * width + x;
                auto r = static_cast<uint32_t>(pixelBuffer[index * 3]);
                auto g = static_cast<uint32_t>(pixelBuffer[index * 3 + 1]);
                auto b = static_cast<uint32_t>(pixelBuffer[index * 3 + 2]);
                ss << r << ' ' << g << ' ' << b << '\n';
            }
        }

        ppm.write(ss.str().c_str(), ss.str().size());

        ppm.close();
    }

    void writeToPNG(const std::string& path, int32_t width, int32_t height, uint8_t* pixelBuffer) {
        stbi_write_png("render.png", width, height, 3, pixelBuffer, width * 3);
    }

    void writeToPNG(const std::string& path, int32_t width, int32_t height, float3* pixelBuffer) {
        auto size = width * height;
        auto* pixelData = new uint8_t[size * 3];

        for (auto i = 0; i < size; i++) {
            pixelData[i * 3] = toInt(pixelBuffer[i].x);
            pixelData[i * 3 + 1] = toInt(pixelBuffer[i].y);
            pixelData[i * 3 + 2] = toInt(pixelBuffer[i].z);
        }

        stbi_write_png("render.png", width, height, 3, pixelData, width * 3);
        delete[] pixelData;
    }

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
}