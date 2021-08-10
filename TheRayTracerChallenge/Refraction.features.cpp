#include "catch.hpp"
#include "Sphere.h"
#include "Intersection.h"
#include "World.h"
#include "Shading.h"
#include "Pattern.h"
#include "Plane.h"

SCENARIO("A helper for producing a sphere with a glassy material", "[Refraction]") {
    GIVEN("s = glassSphere()") {
        auto s = glassSphere();
        THEN("s.transformation == identity_matrix") {
            REQUIRE(s->transformation == Matrix4());
            AND_THEN("s.material.transparency == 1.0")
                REQUIRE(s->material->transparency == 1.0);
            AND_THEN("s.material.refractiveIndex == 1.5")
                REQUIRE(s->material->refractiveIndex == 1.5);
        }
    }
}

SCENARIO("Finding n1 and n2 at various intersections", "[Refraction]") {
    GIVEN("A = glassSphere() with:"
        "| transform                 | scaling(2, 2, 2) |"
        "| material.refractive_index | 1.5              |") {
        auto A = glassSphere();
        A->scaling(2.0, 2.0, 2.0);
        A->material->refractiveIndex = 1.5;
        AND_GIVEN("B = glassSphere() with:"
            "| transform                 | translation(0, 0, 0.25) |"
            "| material.refractive_index | 2.0                      |") {
            auto B = glassSphere();
            B->setTransformation(translate(0.0, 0.0, 0.25));
            B->material->refractiveIndex = 2.0;
            AND_GIVEN("C = glassSphere() with:"
                "| transform                 | translation(0, 0, -0.25) |"
                "| material.refractive_index | 2.5                     |") {
                auto C = glassSphere();
                C->setTransformation(translate(0.0, 0.0, -0.25));
                C->material->refractiveIndex = 2.5;
                AND_GIVEN("r = Ray(point(0.0, 0.0, 4.0), vector(0.0, 0.0, -1.0))") {
                    auto r = Ray(point(0.0, 0.0, 4.0), vector(0.0, 0.0, -1.0));
                    AND_GIVEN("xs = Intersections({{2, A}, {2.75, B}, {3.25, C}, {4.75, B}, {5.25, C}, {6, A}") {
                        auto xs = intersections({ { 2.0, A }, { 2.75, B }, { 3.25, C },
                                                  { 4.75, B }, { 5.25, C }, { 6.0, A } });
                        WHEN("comps = prepareComputations(xs[<index>], r, xs)") {
                            auto comps = prepareComputations(xs[0], r, xs);
                            THEN("comps.n1 == <n1>") {
                                REQUIRE(comps.n1 == 1.0);
                                AND_THEN("comps.n2 == <n2>")
                                    REQUIRE(comps.n2 == 1.5);
                            }
                        }

                        WHEN("comps = prepareComputations(xs[<index>], r, xs)") {
                            auto comps = prepareComputations(xs[1], r, xs);
                            THEN("comps.n1 == <n1>") {
                                REQUIRE(comps.n1 == 1.5);
                                AND_THEN("comps.n2 == <n2>")
                                    REQUIRE(comps.n2 == 2.0);
                            }
                        }

                        WHEN("comps = prepareComputations(xs[<index>], r, xs)") {
                            auto comps = prepareComputations(xs[2], r, xs);
                            THEN("comps.n1 == <n1>") {
                                REQUIRE(comps.n1 == 2.0);
                                AND_THEN("comps.n2 == <n2>")
                                    REQUIRE(comps.n2 == 2.5);
                            }
                        }

                        WHEN("comps = prepareComputations(xs[<index>], r, xs)") {
                            auto comps = prepareComputations(xs[3], r, xs);
                            THEN("comps.n1 == <n1>") {
                                REQUIRE(comps.n1 == 2.5);
                                AND_THEN("comps.n2 == <n2>")
                                    REQUIRE(comps.n2 == 2.5);
                            }
                        }

                        WHEN("comps = prepareComputations(xs[<index>], r, xs)") {
                            auto comps = prepareComputations(xs[4], r, xs);
                            THEN("comps.n1 == <n1>") {
                                REQUIRE(comps.n1 == 2.5);
                                AND_THEN("comps.n2 == <n2>")
                                    REQUIRE(comps.n2 == 1.5);
                            }
                        }

                        WHEN("comps = prepareComputations(xs[<index>], r, xs)") {
                            auto comps = prepareComputations(xs[5], r, xs);
                            THEN("comps.n1 == <n1>") {
                                REQUIRE(comps.n1 == 1.5);
                                AND_THEN("comps.n2 == <n2>")
                                    REQUIRE(comps.n2 == 1.0);
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The under point is offset below the surface", "[Refraction]") {
    GIVEN("r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0))") {
        auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0));
        AND_GIVEN("shape = glassSphere() with"
                "| transform | translation(0, 0, 1) |") {
            auto shape = glassSphere();
            shape->setTransformation(translate(0.0, 0.0, -1.0));
            AND_GIVEN("i = Intersection(5.0, shape)") {
                auto i = Intersection(5.0, shape);
                AND_GIVEN("xs = intersections(i)") {
                    auto xs = intersections({ i });
                    WHEN("comps = prepareComputations(i, r, xs)") {
                        auto comps = prepareComputations(i, r, xs);
                        THEN("comps.underPosition.z < Math::epsilon / 2") {
                            REQUIRE(comps.underPosition.z < -Math::epsilon / 2);
                            AND_THEN("comps.position.z > comps.underPosition.z)")
                                REQUIRE(comps.position.z > comps.underPosition.z);
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The refracted color with an opaque surface", "[Refraction]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld1();
        AND_GIVEN("shape = the first object in w") {
            auto shape = w.getObjects()[0];
            AND_GIVEN("r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0))") {
                auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0));
                AND_GIVEN("xs = intersections({{4, shape}, {6, shape}})") {
                    auto xs = intersections({ { 4, shape }, { 6, shape } });
                    WHEN("comps = prepareComputations(xs[0], r, xs)") {
                        auto comps = prepareComputations(xs[0], r, xs);
                        AND_WHEN("c = refractedColor(w, comps, 5") {
                            auto c = refractedColor(w, comps, 5);
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The refracted color at the maximum recursive depth", "[Refraction]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld1();
        AND_GIVEN("shape = the first object in w") {
            auto shape = w.getObjects()[0];
            AND_GIVEN("shape has:"
            "| material.transparency     | 1.0 |"
            "| material.refractive_index | 1.5 |") {
                shape->material->transparency = 1.0;
                shape->material->refractiveIndex = 1.5;
                AND_GIVEN("r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0))") {
                    auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0));
                    AND_GIVEN("xs = intersections({{4, shape}, {6, shape}})") {
                        auto xs = intersections({ { 4, shape }, { 6, shape } });
                        WHEN("comps = prepareComputations(xs[0], r, xs)") {
                            auto comps = prepareComputations(xs[0], r, xs);
                            AND_WHEN("c = refractedColor(w, comps, 0") {
                                auto c = refractedColor(w, comps, 0);
                                THEN("c == color(0.0, 0.0, 0.0)") {
                                    REQUIRE(c == Color::black);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The refracted color under total internal reflection", "[Refraction]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld1();
        AND_GIVEN("shape = the first object in w") {
            auto shape = w.getObjects()[0];
            AND_GIVEN("shape has:"
                "| material.transparency     | 1.0 |"
                "| material.refractive_index | 1.5 |") {
                shape->material->transparency = 1.0;
                shape->material->refractiveIndex = 1.5;
                AND_GIVEN("r = Ray(point(0.0, 0.0, ¡Ì2/2), vector(0.0, 1.0, 0.0))") {
                    auto r = Ray(point(0.0, 0.0, Math::cos45d), vector(0.0, 1.0, 0.0));
                    AND_GIVEN("xs = intersections({{-¡Ì2/2, shape}, {¡Ì2/2, shape}})") {
                        auto xs = intersections({ { -Math::cos45d, shape }, { Math::cos45d, shape } });
                        // NOTE: this time you're inside the sphere, so you need
                        // to look at the second intersection, xs[1], not xs[0]
                        WHEN("comps = prepareComputations(xs[1], r, xs)") {
                            auto comps = prepareComputations(xs[1], r, xs);
                            AND_WHEN("c = refractedColor(w, comps, 5") {
                                auto c = refractedColor(w, comps, 5);
                                THEN("c == color(0.0, 0.0, 0.0)") {
                                    REQUIRE(c == Color::black);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The refracted color with a refracted ray", "[Refraction]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld1();
        AND_GIVEN("A = the first object in w") {
            auto A = w.getObjects()[0];
            AND_GIVEN("A has:"
            "| material.ambient | 1.0            |"
            "| material.pattern | test_pattern() |") {
                A->material->ambient = 1.0;
                A->material->pattern = std::make_shared<TestPattern>();
                AND_GIVEN("B = the second object in w") {
                    auto B = w.getObjects()[1];
                    AND_GIVEN("B has:"
                        "| material.transparency     | 1.0 |"
                        "| material.refractive_index | 1.5 |") {
                        B->material->transparency = 1.0;
                        B->material->refractiveIndex = 1.5;
                        AND_GIVEN("r = Ray(point(0.0, 0.0, 0.1), vector(0.0, 1.0, 0.0))") {
                            auto r = Ray(point(0.0, 0.0, 0.1), vector(0.0, 1.0, 0.0));
                            AND_GIVEN("xs = intersections(-0.9899:A, -0.4899:B, 0.4899:B, 0.9899:A)") {
                                auto xs = intersections({ { -0.9899, A }, { -0.4899, B }, { 0.4899, B }, { 0.9899, A } });
                                WHEN("comps = prepareComputations(xs[2], r, xs)") {
                                    auto comps = prepareComputations(xs[2], r, xs);
                                    AND_WHEN("c = refractedColor(w, comps, 5)") {
                                        auto c = refractedColor(w, comps, 5);
                                        THEN("c == color(0.0, 0.998885, 0.047217") {
                                            REQUIRE(c == color(0.0, 0.998885, 0.047217));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("shade_hit() with a transparent material", "[Refraction]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld1();
        AND_GIVEN("floor = Plane() with:"
        "| transform                 | translation(0, -1, 0) |"
        "| material.transparency     | 0.5                   |"
        "| material.refractive_index | 1.5                   |") {
            auto floor = std::make_shared<Plane>();
            floor->setTransformation(translate(0.0, -1.0, 0.0));
            floor->material->transparency = 0.5;
            floor->material->refractiveIndex = 1.5;
            AND_GIVEN("floor is added to w") {
                w.addObject(floor);
                AND_GIVEN("ball = Sphere() with:"
                        "| material.color | (1, 0, 0) |"
                        "| material.ambient | 0.5 |"
                        "| transform | translation(0, -3.5, -0.5) |") {
                    auto ball = std::make_shared<Sphere>();
                    ball->material->color = Color::red;
                    ball->material->ambient = 0.5;
                    ball->setTransformation(translate(0.0, -3.5, -0.5));
                    AND_GIVEN("ball is added to w") {
                        w.addObject(ball);
                        AND_GIVEN("r = Ray(point(0.0, 0.0, -3.0), vector(0.0, -¡Ì2/2, ¡Ì2/2))") {
                            auto r = Ray(point(0.0, 0.0, -3.0), vector(0.0, -Math::cos45d, Math::cos45d));
                            AND_GIVEN("xs = intersections(¡Ì2:floor)") {
                                auto xs = intersections({ {Math::sqrt_2, floor } });
                                WHEN("comps = prepareComputations(xs[0], r, xs)") {
                                    auto comps = prepareComputations(xs[0], r, xs);
                                    AND_WHEN("color = shadeHit(w, comps, 5)") {
                                        auto color = shadeHit(w, comps, 5);
                                        THEN("color == color(0.944440, 0.0, 0.0)") {
                                            REQUIRE(color == ::color(0.944440, 0.0, 0.0));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The Schlick approximation under total internal reflection", "[Refraction]") {
    GIVEN("shape = glassSphere()") {
        auto shape = glassSphere();
        AND_GIVEN("r = Ray(point(0.0, 0.0, ¡Ì2/2), vector(0.0, 1.0, 0.0))") {
            auto r = Ray(point(0.0, 0.0, Math::cos45d), vector(0.0, 1.0, 0.0));
            AND_GIVEN("xs = intersections(-¡Ì2/2:shape, ¡Ì2/2:shape)") {
                auto xs = intersections({ { -Math::cos45d, shape },
                                           { Math::cos45d, shape } });
                WHEN("comps = prepareComputations(xs[1], r, xs)") {
                    auto comps = prepareComputations(xs[1], r, xs);
                    AND_WHEN("reflectance = schlick(comps)") {
                        auto reflectance = schlick(comps);
                        THEN("reflectance == 1.0") {
                            REQUIRE(reflectance == 1.0);
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The Schlick approximation with a perpendicular viewing angle", "[Refraction]") {
    GIVEN("shape = glassSphere()") {
        auto shape = glassSphere();
        AND_GIVEN("r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 1.0, 0.0))") {
            auto r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 1.0, 0.0));
            AND_GIVEN("xs = intersections(-1:shape, 1:shape)") {
                auto xs = intersections({ { -1.0, shape },
                                           { 1.0, shape } });
                WHEN("comps = prepareComputations(xs[1], r, xs)") {
                    auto comps = prepareComputations(xs[1], r, xs);
                    AND_WHEN("reflectance = schlick(comps)") {
                        auto reflectance = schlick(comps);
                        THEN("reflectance == 0.04") {
                            REQUIRE(reflectance == Approx(0.04));
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The Schlick approximation with small angle and n2 > n1", "[Refraction]") {
    GIVEN("shape = glassSphere()") {
        auto shape = glassSphere();
        AND_GIVEN("r = Ray(point(0.0, 0.99, -2.0), vector(0.0, 0.0, 1.0))") {
            auto r = Ray(point(0.0, 0.99, -2.0), vector(0.0, 0.0, 1.0));
            AND_GIVEN("xs = intersections(1.8589:shape)") {
                auto xs = intersections({ { 1.8589, shape } });
                WHEN("comps = prepareComputations(xs[1], r, xs)") {
                    auto comps = prepareComputations(xs[0], r, xs);
                    AND_WHEN("reflectance = schlick(comps)") {
                        auto reflectance = schlick(comps);
                        THEN("reflectance == 0.48873") {
                            REQUIRE(reflectance == Approx(0.48873));
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("shade_hit() with a reflective, transparent material", "[Refraction]") {
    GIVEN("w = defaultWorld1()") {
        auto w = defaultWorld1();
        AND_GIVEN("r = Ray(point(0.0, 0.0, -3.0), vector(0.0, -¡Ì2/2, ¡Ì2/2))") {
            auto r = Ray(point(0.0, 0.0, -3.0), vector(0.0, -Math::cos45d, Math::cos45d));
            AND_GIVEN("floor = Plane() with:"
            "| transform                 | translation(0, -1, 0) |"
            "| material.reflective       | 0.5                   |"
            "| material.transparency     | 0.5                   |"
            "| material.refractive_index | 1.5                   |") {
                auto floor = std::make_shared<Plane>();
                floor->material->reflective = 0.5;
                floor->material->transparency = 0.5;
                floor->material->refractiveIndex = 1.5;
                AND_GIVEN("floor is added to w") {
                    w.addObject(floor);
                    AND_GIVEN("ball = Sphere() with:"
                        "| material.color   | (1, 0, 0)                  |"
                        "| material.ambient | 0.5                        |"
                        "| transform        | translation(0, -3.5, -0.5) |") {
                        auto ball = std::make_shared<Sphere>();
                        ball->setTransformation(translate(0.0, -3.5, -0.5));
                        ball->material->color = color(1.0, 0.0, 0.0);
                        ball->material->ambient = 0.5;
                        AND_GIVEN("ball is added to w") {
                            w.addObject(ball);
                            AND_GIVEN("xs = intersectoins(¡Ì2:floor)") {
                                auto xs = intersections({ { Math::sqrt_2, floor } });
                                WHEN("comps = prepareComputations(xs[0], r, xs)") {
                                    auto comps = prepareComputations(xs[0], r, xs);
                                    AND_WHEN("color = shadeHit(w, comps, 5)") {
                                        auto color = shadeHit(w, comps, 5);
                                        THEN("color == color(0.938870, 0.0061850, 0.003711)") {
                                            REQUIRE(color == ::color(0.938870, 0.0061850, 0.003711));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}