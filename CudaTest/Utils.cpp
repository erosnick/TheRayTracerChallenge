#include "Utils.h"
#include "Tuple.h"
#include "lodepng.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#define NOMINMAX
#include <windows.h>
#undef NOMINMAX

namespace Utils {
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
                auto r = static_cast<uint32_t>(256 * std::clamp(pixelColor.x(), 0.0, 0.999));
                auto g = static_cast<uint32_t>(256 * std::clamp(pixelColor.y(), 0.0, 0.999));
                auto b = static_cast<uint32_t>(256 * std::clamp(pixelColor.z(), 0.0, 0.999));
                ss << r << ' ' << g << ' ' << b << '\n';
            }
        }

        ppm.write(ss.str().c_str(), ss.str().size());

        ppm.close();
    }

    void writeToPPM(const std::string& path, int32_t width, int32_t height, uint8_t* pixelBuffer) {
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

    void writeToPNG(const std::string& path, int32_t width, int32_t height, uint8_t* pixelBuffer) {
        std::vector<uint8_t> pixelData;

        for (auto y = height - 1; y >= 0; y--) {
            for (auto x = 0; x < width; x++) {
                auto index = y * width + x;
                pixelData.push_back(pixelBuffer[index * 3]);
                pixelData.push_back(pixelBuffer[index * 3 + 1]);
                pixelData.push_back(pixelBuffer[index * 3 + 2]);
                pixelData.push_back(255);
            }
        }

        //Encode the image
        unsigned error = lodepng::encode(path, pixelData, width, height);
        //if there's an error, display it
        if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
}