#include "catch.hpp"
#include "constants.h"
#include "Pattern.h"
#include "Material.h"
#include "Light.h"
#include "Shading.h"
#include "Sphere.h"

auto black = Color::black;
auto white = Color::white;

SCENARIO("Creating a stripe pattern", "[Pattern]") {
    GIVEN("pattern = stripePattern(white, black)") {
        auto pattern = StripePattern(white, black);
        THEN("pattern.color1 == white") {
            REQUIRE(pattern.color1 == white);
            AND_THEN("pattern.color2 == black")
                REQUIRE(pattern.color2 == black);
        }
    }
}

SCENARIO("A stripe pattern is constant in y", "[Pattern]") {
    GIVEN("pattern = stripePattern(white, black)") {
        auto pattern = StripePattern(white, black);
        THEN("stripAt(pattern, point(0.0, 0.0, 0.0)) == white") {
            REQUIRE(pattern.patternAt(point(0.0, 0.0, 0.0)) == white);
            AND_THEN("stripAt(pattern, point(0.0, 1.0, 0.0)) == white")
                REQUIRE(pattern.patternAt(point(0.0, 1.0, 0.0)) == white);
            AND_THEN("stripAt(pattern, point(0.0, 2.0, 0.0)) == white")
                REQUIRE(pattern.patternAt(point(0.0, 2.0, 0.0)) == white);
        }
    }
}

SCENARIO("A stripe pattern is constant in z", "[Pattern]") {
    GIVEN("pattern = stripePattern(white, black)") {
        auto pattern = StripePattern(white, black);
        THEN("stripAt(pattern, point(0.0, 0.0, 0.0)) == white") {
            REQUIRE(pattern.patternAt(point(0.0, 0.0, 0.0)) == white);
            AND_THEN("stripAt(pattern, point(0.0, 0.0, 1.0)) == white")
                REQUIRE(pattern.patternAt(point(0.0, 0.0, 1.0)) == white);
            AND_THEN("stripAt(pattern, point(0.0, 0.0, 2.0)) == white")
                REQUIRE(pattern.patternAt(point(0.0, 0.0, 2.0)) == white);
        }
    }
}

SCENARIO("A stripe pattern alternates in x", "[Pattern]") {
    GIVEN("pattern = stripePattern(white, black)") {
        auto pattern = StripePattern(white, black);
        THEN("stripAt(pattern, point(0.0, 0.0, 0.0)) == white") {
            REQUIRE(pattern.patternAt(point(0.0, 0.0, 0.0)) == white);
            AND_THEN("stripAt(pattern, point(0.9, 0.0, 0.0)) == white")
                REQUIRE(pattern.patternAt(point(0.9, 0.0, 0.0)) == white);
            AND_THEN("stripAt(pattern, point(1.0, 0.0, 0.0)) == black")
                REQUIRE(pattern.patternAt(point(1.0, 0.0, 0.0)) == black);
            AND_THEN("stripAt(pattern, point(-0.1, 0.0, 0.0)) == black")
                REQUIRE(pattern.patternAt(point(-0.1, 0.0, 0.0)) == black);
            AND_THEN("stripAt(pattern, point(-1.0, 0.0, 0.0)) == black")
                REQUIRE(pattern.patternAt(point(-1.0, 0.0, 0.0)) == black);
            AND_THEN("stripAt(pattern, point(-0=1.1, 0.0, 0.0)) == white")
                REQUIRE(pattern.patternAt(point(-1.1, 0.0, 0.0)) == white);
        }
    }
}

SCENARIO("Lighting with a pattern applied", "[Plane]") {
    GIVEN("m.pattern = StripePattern(color(1.0, 1.0, 1.0), color(0.0, 0.0, 0.0))") {
        auto m = std::make_shared<Material>();
        m->pattern = std::make_shared<StripePattern>(color(1.0, 1.0, 1.0), color(0.0, 0.0, 0.0));
        AND_GIVEN("m.ambient = 1.0"
                  "m.diffuse = 0.0"
                  "m.specular = 0.0"
                  "viewDirection = vector(0.0, 0.0, -1.0)"
                  "normal = vector(0.0, 0.0, -1.0)"
                  "light = Light(point(0.0, 0.0, -10.0), vector(1.0, 1.0, 1.0))") {
            m->ambient = 1.0;
            m->diffuse = 0.0;
            m->specular = 0.0;
            auto viewDirection = vector(0.0, 0.0, -1.0);
            auto normal = vector(0.0, 0.0, -1.0);
            auto light = Light(point(0.0, 0.0, -10.0), vector(1.0, 1.0, 1.0));
            WHEN("c1 = lighting(m, light, point(0.9, 0.0, 0.0), viewDirection, normal, false)") {
                auto object = std::make_shared<Sphere>();
                auto c1 = lighting(m, object, light, point(0.9, 0.0, 0.0), viewDirection, normal, false);
                AND_WHEN("c2 = lighting(m, light, point(1.1, 0.0, 0.0), viewDirection, normal, false)") {
                    auto c2 = lighting(m, object, light, point(1.1, 0.0, 0.0), viewDirection, normal, false);
                    THEN("c1 == color(1.0, 1.0, 1.0)") {
                        REQUIRE(c1 == color(1.0, 1.0, 1.0));
                        AND_THEN("c2 == color(0.0, 0.0, 0.0)")
                            REQUIRE(c2 == color(0.0, 0.0, 0.0));
                    }
                }
            }
        }
    }
}

SCENARIO("The default pattern transformation", "[Pattern]") {
    GIVEN("pattern = Pattern()") {
        auto pattern = Pattern();
        THEN("pattern.transformation == identity_matrix") {
            REQUIRE(pattern.transformation == Matrix4());
        }
    }
}

SCENARIO("Assigning a transformation to pattern", "[Pattern]") {
    GIVEN("pattern = Pattern()") {
        auto pattern = Pattern();
        WHEN("pattern.setTransformation(translate(1.0, 2.0, 3.0))") {
            pattern.setTransformation(translate(1.0, 2.0, 3.0));
            THEN("pattern.transformation == translate(1.0, 2.0, 3.0)") {
                REQUIRE(pattern.transformation == translate(1.0, 2.0, 3.0));
            }
        }
    }
}

// The following 3 tests were replaced by the following tests.
//SCENARIO("Stripes with an object transformation", "[Pattern]") {
//    GIVEN("object = Sphere()") {
//        auto object = std::make_shared<Sphere>();
//        AND_GIVEN("object.setTransformation(scaling(2.0, 2.0, 2.0)") {
//            object->setTransformation(scaling(2.0, 2.0, 2.0));
//            AND_GIVEN("pattern = StripPattern(white, black)") {
//                auto pattern = StripePattern(white, black);
//                WHEN("c = pattern.stripeAtObject(object, point(1.5, 0.0, 0.0)") {
//                    auto c = pattern.stripeAtObject(object, point(1.5, 0.0, 0.0));
//                    THEN("c == white") {
//                        REQUIRE(c == white);
//                    }
//                }
//            }
//        }
//    }
//}
//
//SCENARIO("Stripes with a pattern transformation", "[Pattern]") {
//    GIVEN("object = Sphere()") {
//        auto object = std::make_shared<Sphere>();
//        AND_GIVEN("pattern = StripPattern(white, black)") {
//            auto pattern = StripePattern(white, black);
//            AND_GIVEN("pattern.setTransformation(scaling(2.0, 2.0, 2.0))") {
//                pattern.setTransformation(scaling(2.0, 2.0, 2.0));
//                WHEN("c = pattern.stripeAtObject(object, point(1.5, 0.0, 0.0)") {
//                    auto c = pattern.stripeAtObject(object, point(1.5, 0.0, 0.0));
//                    THEN("c == white") {
//                        REQUIRE(c == white);
//                    }
//                }
//            }
//        }
//    }
//}
//
//SCENARIO("Stripes with both an object and a pattern transformation", "[Pattern]") {
//    GIVEN("object = Sphere()") {
//        auto object = std::make_shared<Sphere>();
//        AND_GIVEN("object->setTransformation(scaling(2.0, 2.0, 2.0))") {
//            object->setTransformation(scaling(2.0, 2.0, 2.0));
//            AND_GIVEN("pattern = StripPattern(white, black)") {
//                auto pattern = StripePattern(white, black);
//                AND_GIVEN("pattern.setTransformation(scaling(0.5, 0.0, 0.0))") {
//                    pattern.setTransformation(translate(0.5, 0.0, 0.0));
//                    WHEN("c = pattern.stripeAtObject(object, point(2.5, 0.0, 0.0)") {
//                        auto c = pattern.stripeAtObject(object, point(2.5, 0.0, 0.0));
//                        THEN("c == white") {
//                            REQUIRE(c == white);
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

SCENARIO("A pattern with an object transformation", "[Pattern]") {
    GIVEN("object = Sphere()") {
        auto shape = std::make_shared<Sphere>();
        AND_GIVEN("object.setTransformation(scaling(2.0, 2.0, 2.0)") {
            shape->setTransformation(scaling(2.0, 2.0, 2.0));
            AND_GIVEN("pattern = TestPattern()") {
                auto pattern = TestPattern();
                WHEN("c = pattern.patternAtShape(shape, point(2.0, 3.0, 4.0)") {
                    auto c = pattern.patternAtShape(shape, point(2.0, 3.0, 4.0));
                    THEN("c == color(1.0, 1.5, 2.0)") {
                        REQUIRE(c == color(1.0, 1.5, 2.0));
                    }
                }
            }
        }
    }
}

SCENARIO("A pattern with a pattern transformation", "[Pattern]") {
    GIVEN("object = Sphere()") {
        auto shape = std::make_shared<Sphere>();
        AND_GIVEN("pattern = TestPattern()") {
            auto pattern = TestPattern();
            AND_GIVEN("pattern.setTransformation(scaling(2.0, 2.0, 2.0))") {
                pattern.setTransformation(scaling(2.0, 2.0, 2.0));
                WHEN("c = pattern.patternAtShape(shape, point(2.0, 3.0, 4.0)") {
                    auto c = pattern.patternAtShape(shape, point(2.0, 3.0, 4.0));
                    THEN("c == color(1.0, 1.5, 2.0)") {
                        REQUIRE(c == color(1.0, 1.5, 2.0));
                    }
                }
            }
        }
    }
}

SCENARIO("A pattern with both an object and a pattern transformation", "[Pattern]") {
    GIVEN("object = Sphere()") {
        auto shape = std::make_shared<Sphere>();
        AND_GIVEN("shape->setTransformation(scaling(2.0, 2.0, 2.0))") {
            shape->setTransformation(scaling(2.0, 2.0, 2.0));
            AND_GIVEN("pattern = TestPattern()") {
                auto pattern = TestPattern();
                AND_GIVEN("pattern.setTransformation(translate(0.5, 1.0, 1.5))") {
                    pattern.setTransformation(translate(2.0, 2.0, 2.0));
                    WHEN("c = pattern.patternAtShape(shape, point(2.5, 3.0, 3.5)") {
                        auto c = pattern.patternAtShape(shape, point(2.5, 3.0, 3.5));
                        THEN("c == color(0.75, 0.5, 0.25)") {
                            REQUIRE(c == color(0.75, 0.5, 0.25));
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("A gradient linearly interpolates between colors", "[Pattern]") {
    GIVEN("pattern = GradientPattern()") {
        auto pattern = GradientPattern();
        THEN("pattern.patternAt(point(0.0, 0.0, 0.0)) == white") {
            REQUIRE(pattern.patternAt(point(0.0, 0.0, 0.0)) == Color::white);
            AND_THEN("pattern.patternAt(point(0.25, 0.0, 0.0)) == color(0.75, 0.75, 0.75)")
                REQUIRE(pattern.patternAt(point(0.25, 0.0, 0.0)) == color(0.75, 0.75, 0.75));
            AND_THEN("pattern.patternAt(point(0.5, 0.0, 0.0)) == color(0.5, 0.5, 0.5)")
                REQUIRE(pattern.patternAt(point(0.5, 0.0, 0.0)) == color(0.5, 0.5, 0.5));
            AND_THEN("pattern.patternAt(point(0.75, 0.0, 0.0)) == color(0.25, 0.25, 0.25)")
                REQUIRE(pattern.patternAt(point(0.75, 0.0, 0.0)) == color(0.25, 0.25, 0.25));
        }
    }
}

SCENARIO("A ring should extend in both x and z", "[Pattern]") {
    GIVEN("pattern = RingPattern()") {
        auto pattern = RingPattern();
        THEN("pattern.patternAt(point(0.0, 0.0, 0.0)) == white") {
            REQUIRE(pattern.patternAt(point(0.0, 0.0, 0.0)) == Color::white);
            AND_THEN("pattern.patternAt(point(1.0, 0.0, 0.0)) == Color::black")
                REQUIRE(pattern.patternAt(point(1.0, 0.0, 0.0)) == Color::black);
            AND_THEN("pattern.patternAt(point(0.0, 0.0, 1.0)) == Color::black")
                REQUIRE(pattern.patternAt(point(0.0, 0.0, 1.0)) == Color::black);
            // 0.708 = just slightly more than ¡Ì2 / 2
            AND_THEN("pattern.patternAt(point(0.75, 0.0, 0.0)) == Color::black")
                REQUIRE(pattern.patternAt(point(0.708, 0.0, 0.708)) == Color::black);
        }
    }
}

SCENARIO("Checkers should repeat in x", "[Pattern]") {
    GIVEN("pattern = CheckerPattern()") {
        auto pattern = CheckerPattern();
        THEN("pattern.patternAt(point(0.0, 0.0, 0.0)) == white") {
            REQUIRE(pattern.patternAt(point(0.0, 0.0, 0.0)) == Color::white);
            AND_THEN("pattern.patternAt(point(0.99, 0.0, 0.0)) == Color::white")
                REQUIRE(pattern.patternAt(point(0.99, 0.0, 0.0)) == Color::white);
            AND_THEN("pattern.patternAt(point(1.01, 0.0, 0.0)) == Color::black")
                REQUIRE(pattern.patternAt(point(1.01, 0.0, 0.0)) == Color::black);
        }
    }
}