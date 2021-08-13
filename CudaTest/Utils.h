#pragma once

#include <string>

class Tuple;

std::string toPPM(int32_t width, int32_t height);
void writeToPPM(const std::string& path, int32_t width, int32_t height, Tuple* pixelBuffer);