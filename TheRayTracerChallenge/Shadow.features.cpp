#include "catch.hpp"
#include "World.h"
#include "Shading.h"

SCENARIO("There is no shadow when nothing is collinear with point and light", "[Shadow]") {
    GIVEN("w = defaultWorld2()") {
        auto w = defaultWorld2();
        AND_GIVEN("p = point(0.0, 10.0, 0.0)") {
            auto p = point(0.0, 10.0, 0.0);
            THEN("isShadowed(w, p) is false") {
                REQUIRE(!isShadow(w, w.getLight(0), p));
            }
        }
    }
}

SCENARIO("The shadow when an object is between the point and the light", "[Shadow]") {
    GIVEN("w = defaultWorld2()") {
        auto w = defaultWorld2();
        AND_GIVEN("p = point(10.0, -10.0, 0.0)") {
            auto p = point(10.0, -10.0, 0.0);
            THEN("isShadowed(w, p) is true") {
                REQUIRE(isShadow(w, w.getLight(0), p));
            }
        }
    }
}

SCENARIO("There is no shadow when an object is behind the light", "[Shadow]") {
    GIVEN("w = defaultWorld2()") {
        auto w = defaultWorld2();
        AND_GIVEN("p = point(-20.0, 20.0, -20.0)") {
            auto p = point(-20.0, 20.0, -20.0);
            THEN("isShadowed(w, p) is false") {
                REQUIRE(!isShadow(w, w.getLight(0), p));
            }
        }
    }
}

SCENARIO("There is no shadow when an object is behind the point", "[Shadow]") {
    GIVEN("w = defaultWorld2()") {
        auto w = defaultWorld2();
        AND_GIVEN("p = point(-2.0, 2.0, -2.0)") {
            auto p = point(-2.0, 2.0, -2.0);
            THEN("isShadowed(w, p) is false") {
                REQUIRE(!isShadow(w, w.getLight(0), p));
            }
        }
    }
}

SCENARIO("shade_hit() is given an intersection in shadow", "[Shadow]") {
    GIVEN("w = world()") {
        auto w = World();
        AND_GIVEN("w.addLight(Light(point(0.0, 0.0, -10.0), color(1.0, 1.0, 1.0)))") {
            w.addLight(Light(point(0.0, 0.0, -10.0), color(1.0, 1.0, 1.0)));
            AND_GIVEN("s1 = std::make_shared<Sphere>()") {
                auto s1 = std::make_shared<Sphere>();
                AND_GIVEN("s1 is added to w") {
                    w.addObject(s1);
                    AND_GIVEN("s2 = Sphere with:"
                        "| transform | translate(0.0, 0.0, 10.0) |") {
                        auto s2 = std::make_shared<Sphere>();
                        s2->setTransformation(translate(0.0, 0.0, 10.0));
                        AND_GIVEN("s2 is add to w") {
                            w.addObject(s2);
                            AND_GIVEN("r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0)") {
                                auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0));
                                AND_GIVEN("i = Intersection(4.0, s2)") {
                                    auto i = Intersection(4.0, s2);
                                    WHEN("comps = prepareComputations(i, r)") {
                                        auto comps = prepareComputations(i, r);
                                        AND_WHEN("c = shadeHit(w, comps)") {
                                            auto c = shadeHit(w, comps);
                                            THEN("c == color(0.1, 0.0, 0.0") {
                                                REQUIRE(c == color(0.1, 0.0, 0.0));
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
}

SCENARIO("The hit should offset the point", "[Shadow]") {
    GIVEN("r = Ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0))") {
        auto r = Ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        AND_GIVEN("shape = std::make_shared<Sphere>() with"
            "| transform | translate(0.0, 0.0, 1.0) |") {
            auto shape = std::make_shared<Sphere>();
            shape->setTransformation(translate(0.0, 0.0, 1.0));
            AND_GIVEN("i = Intersection(5.0, shape)") {
                auto i = Intersection(5.0, shape);
                WHEN("comps = prepareComputations(i, r)") {
                    auto comps = prepareComputations(i, r);
                    THEN("comps.overPosition.z < -EPSILON / 2") {
                        REQUIRE(comps.overPosition.z < -Math::epsilon / 2);
                        AND_THEN("comps.position.z > comps.overPosition.z") {
                            REQUIRE(comps.position.z > comps.overPosition.z);
                        }
                    }
                }
            }
        }
    }
}