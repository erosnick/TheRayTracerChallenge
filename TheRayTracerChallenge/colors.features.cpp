#include "catch.hpp"
#include "tuple.h"

SCENARIO("Colors are (red, green, blue) tuples", "[color]") {
    GIVEN("c ¡û color(-0.5f, 0.4f, 1.7f)") {
        auto c = color(-0.5f, 0.4f, 1.7f);
        REQUIRE(c.red == -0.5f);
        REQUIRE(c.green == 0.4f);
        REQUIRE(c.blue == 1.7f);
    }
}

SCENARIO("Adding colors", "[color]") {
    GIVEN("c1 ¡û color(0.9f, 0.6f, 0.75f") {
        auto c1 = color(0.9f, 0.6f, 0.75f);
        AND_GIVEN("c2 ¡û color(0.7f, 0.1f, 0.25f") {
            auto c2 = color(0.7f, 0.1f, 0.25f);
            REQUIRE(c1 + c2 == color(1.6f, 0.7f, 1.0f));
        }
    }
}

SCENARIO("Substracting colors", "[color]") {
    GIVEN("c1 ¡û color(0.9f, 0.6f, 0.75f") {
        auto c1 = color(0.9f, 0.6f, 0.75f);
        AND_GIVEN("c2 ¡û color(0.7f, 0.1f, 0.25f") {
            auto c2 = color(0.7f, 0.1f, 0.25f);
            REQUIRE(c1 - c2 == color(0.2f, 0.5f, 0.5f));
        }
    }
}

SCENARIO("Multiplying a color by a scalar", "[color]") {
    GIVEN("c1 ¡û color(0.2f, 0.3f, 0.4f") {
        auto c = color(0.2f, 0.3f, 0.4f);
        REQUIRE(c * 2.0f == color(0.4f, 0.6f, 0.8f));
    }
}

SCENARIO("Multiplying colors", "[color]") {
    GIVEN("c1 ¡û color(0.2f, 0.3f, 0.4f") {
        auto c1 = color(1.0f, 0.2f, 0.4f);
        AND_GIVEN("c2 ¡û color(0.9f, 1.0f, 0.1f") {
            auto c2 = color(0.9f, 1.0f, 0.1f);
            REQUIRE(c1 * c2 == color(0.9f, 0.2f, 0.04f));
        }
    }
}