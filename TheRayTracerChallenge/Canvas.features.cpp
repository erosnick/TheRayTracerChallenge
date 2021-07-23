#include "catch.hpp"
#include "canvas.h"

SCENARIO("Creating a canvas", "[Canvas]") {
    GIVEN("canvas ¡û createCanvas(10, 20)") {
        auto canvas = createCanvas(10, 20);
        REQUIRE(canvas.getWidth() == 10);
        REQUIRE(canvas.getHeight() == 20);
        auto black = color(0.0f, 0.0f, 0.0f);
        for (auto pixelColor : canvas.getPixelData()) {
            REQUIRE(pixelColor == black);
        }
    }
}

SCENARIO("Writing pixels to a canvas", "[Canvas]") {
    GIVEN("canvas ¡û createCanvas(10, 20)") {
        auto canvas = createCanvas(10, 20);
        AND_GIVEN("red") {
            auto red = color(1.0f, 0.0f, 0.0f);
            WHEN("canvas.writePixel(2, 3, red)") {
                canvas.writePixel(2, 3, red);
                REQUIRE(canvas.pixelAt(2, 3) == red);
            }
        }
    }
}

SCENARIO("Constructing the PPM header", "[Canvas]") {
    GIVEN("canvas ¡û createCanvas(5, 3)") {
        auto canvas = createCanvas(5, 3);
        WHEN("ppm = canvas.toPPM()") {
            auto ppm = canvas.toPPM();
            REQUIRE(ppm == "P3\n5 3\n255\n");
        }
    }
}

SCENARIO("Constructing the PPM pixel data", "[Canvas]") {
    GIVEN("canvas ¡û createCanvas(5, 3)") {
        auto canvas = createCanvas(5, 3);
        AND_GIVEN("c1 ¡û color( 1.5f, 0.0f, 0.0f"
                  "c2 ¡û color( 0.0f, 0.5f, 0.0f"
                  "c3 ¡û color(-0.5f, 0.0f, 1.0f") {
            auto c1 = color( 1.5f, 0.0f, 0.0f);
            auto c2 = color( 0.0f, 0.5f, 0.0f);
            auto c3 = color(-0.5f, 0.0f, 1.0f);

            WHEN("canvas.writePixel(0, 0, c1)"
                 "canvas.writePixel(0, 0, c2)"
                 "canvas.writePixel(0, 0, c3"
                 "ppm = canvas.toPPM()") {
                canvas.writePixel(0, 0, c1);
                canvas.writePixel(2, 1, c2);
                canvas.writePixel(4, 2, c3);
                auto ppm = canvas.toPPM();
                REQUIRE(ppm == "P3\n5 3\n255\n");
            }
        }
    }
}