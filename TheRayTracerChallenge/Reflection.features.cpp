#include "catch.hpp"
#include "Material.h"
#include "Plane.h"
#include "Intersection.h"
#include "World.h"
#include "Shading.h"
#include "Light.h"

SCENARIO("Reflectivity for the default material", "[Reflection]") {
    GIVEN("m = Material()") {
        auto m = Material();
        THEN("m.reflective == 0.0") {
            REQUIRE(m.reflective == 0.0);
        }
    }
}

SCENARIO("Pre-computing the reflection vector", "[Reflection]") {
    GIVEN("shape = Plane()") {
        auto shape = std::make_shared<Plane>();
        AND_GIVEN("r = Ray(point(0.0, 1.0, -1.0), vector(0.0, -¡Ì2/2, ¡Ì2/2))") {
            auto r = Ray(point(0.0, 1.0, -1.0), vector(0.0, -Math::cos45d, Math::cos45d));
            AND_GIVEN("i = Intersection(¡Ì2, shape)") {
                auto i = Intersection(Math::sqrt_2, shape);
                WHEN("comps = prepareComputations(i, r)") {
                    auto comps = prepareComputations(i, r);
                    THEN("comps.reflectVector = vector(0.0, ¡Ì2/2, ¡Ì2/2)") {
                        REQUIRE(comps.reflectVector == vector(0.0, Math::cos45d, Math::cos45d));
                    }
                }
            }
        }
    }
}

SCENARIO("The reflected color for a non-reflective material", "[Reflection]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld3() ;
        AND_GIVEN("r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 0, 1.0))") {
            auto r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 0, 1.0));
            AND_GIVEN("shape = the second object in w") {
                auto shape = w.getObjects()[1];
                AND_GIVEN("shape.material.ambient = 1.0") {
                    shape->material->ambient = 1.0;
                    AND_GIVEN("i = Intersection(1.0, shape)") {
                        auto i = Intersection(1.0, shape);
                        WHEN("comps = prepareComputations(i, r)") {
                            auto comps = prepareComputations(i, r);
                            AND_WHEN("color = w.reflectedColor(comps)") {
                                auto color = reflectedColor(w, comps);
                                THEN("color = color(0.0, 0.0, 0.0)") {
                                    REQUIRE(color == Color::black);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The reflected color for a reflective material", "[Reflection]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld3();
        AND_GIVEN("shape = Plane() wieth:"
                "| material.reflective | 0.5 |"
                "| transform | translation(0.0, -1.0, 0.0) |") {
            auto shape = std::make_shared<Plane>();
            shape->material->reflective = 0.5;
            shape->setTransformation(translate(0.0, -1.0, 0.0));
            AND_GIVEN("shape is added to w") {
                w.addObject(shape);
                AND_GIVEN("r = Ray(point(0.0, 0.0, 3.0), vector(0.0, -¡Ì2/2, -¡Ì2/2))") {
                    auto r = Ray(point(0.0, 0.0, 3.0), vector(0.0, -Math::cos45d, -Math::cos45d));
                    AND_GIVEN("i = Intersection(Math::sqrt_2, shape)") {
                        auto i = Intersection(Math::sqrt_2, shape);
                        WHEN("comps = prepareComputations(i, r)") {
                            auto comps = prepareComputations(i, r);
                            AND_WHEN("color = reflectedColor(w, comps, 1)") {
                                auto color = reflectedColor(w, comps, 1);
                                THEN("color = color(0.117616, 0.147020, 0.088212)") {
                                    REQUIRE(color == ::color(0.117616, 0.147020, 0.088212));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("shade_hit() with a reflective material", "[Reflection]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld3();
        AND_GIVEN("shape = Plane() wieth:"
            "| material.reflective | 0.5 |"
            "| transform | translation(0.0, -1.0, 0.0) |") {
            auto shape = std::make_shared<Plane>();
            shape->material->reflective = 0.5;
            shape->material->color = { 1.0, 1.0, 1.0 };
            shape->setTransformation(translate(0.0, -1.0, 0.0));
            AND_GIVEN("shape is added to w") {
                w.addObject(shape);
                AND_GIVEN("r = Ray(point(0.0, 0.0, 3.0), vector(0.0, -¡Ì2/2, -¡Ì2/2))") {
                    auto r = Ray(point(0.0, 0.0, 3.0), vector(0.0, -Math::cos45d, -Math::cos45d));
                    AND_GIVEN("i = Intersection(Math::sqrt_2, shape)") {
                        auto i = Intersection(Math::sqrt_2, shape);
                        WHEN("comps = prepareComputations(i, r)") {
                            auto comps = prepareComputations(i, r);
                            AND_WHEN("color = shadeHit(w, comps, 1)") {
                                auto color = shadeHit(w, comps, 1);
                                THEN("color = color(0.812056, 0.841460, 0.782652)") {
                                    REQUIRE(color == ::color(0.812056, 0.841460, 0.782652));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("colorAt() with mutually reflective surfaces", "[Reflection]") {
    GIVEN("w = World()") {
        auto w = World();
        AND_GIVEN("w.lights[0] = Light(point(0.0, 0.0, 0.0), color(1.0, 1.0, 1.0))") {
            auto light = Light(point(0.0, 0.0, 0.0), color(1.0, 1.0, 1.0));
            w.addLight(light);
            AND_GIVEN("lower = Plane() with:"
                    "| material.reflective | 1 |"
                    "| transform | translation(0, -1, 0) |") {
                auto lower = std::make_shared<Plane>();
                lower->material->reflective = 1.0;
                lower->setTransformation(translate(0.0, -1.0, 0.0));
                AND_GIVEN("lower is added to w") {
                    w.addObject(lower);
                    AND_GIVEN("upper = Plane() with:"
                        "| material.reflective | 1 |"
                        "| transform | translation(0, 1, 0) |") {
                        auto upper = std::make_shared<Plane>();
                        upper->material->reflective = 1.0;
                        upper->setTransformation(translate(0.0, 1.0, 0.0));
                        AND_GIVEN("upper is added to w") {
                            w.addObject(upper);
                            AND_GIVEN("r = Ray(point(0, 0, 0), vector(0, 1, 0))") {
                                auto r = Ray(point(0, 0, 0), vector(0, 1, 0));
                                THEN("colorAt(w, r) should terminate successfully") {

                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO(" The reflected color at the maximum recursive depth", "[Reflection]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld3();
        AND_GIVEN("shape = Plane() wieth:"
            "| material.reflective | 0.5 |"
            "| transform | translation(0.0, -1.0, 0.0) |") {
            auto shape = std::make_shared<Plane>();
            shape->material->reflective = 0.5;
            shape->material->color = { 1.0, 1.0, 1.0 };
            shape->setTransformation(translate(0.0, -1.0, 0.0));
            AND_GIVEN("shape is added to w") {
                w.addObject(shape);
                AND_GIVEN("r = Ray(point(0.0, 0.0, 3.0), vector(0.0, -¡Ì2/2, -¡Ì2/2))") {
                    auto r = Ray(point(0.0, 0.0, 3.0), vector(0.0, -Math::cos45d, -Math::cos45d));
                    AND_GIVEN("i = Intersection(Math::sqrt_2, shape)") {
                        auto i = Intersection(Math::sqrt_2, shape);
                        WHEN("comps = prepareComputations(i, r)") {
                            auto comps = prepareComputations(i, r);
                            AND_WHEN("color = w.reflectedColor(comps, 0)") {
                                auto color = reflectedColor(w, comps, 0);
                                THEN("color = color(0.0, 0.0, 0.0)") {
                                    REQUIRE(color == ::color(0.0, 0.0, 0.0));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}