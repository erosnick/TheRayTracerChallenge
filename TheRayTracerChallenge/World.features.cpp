#include "catch.hpp"
#include "World.h"
#include "Material.h"
#include "Shading.h"

SCENARIO("Creating a world", "[World]") {
    GIVEN("w = World()") {
        auto w = World();
        THEN("w contains no objects") {
            REQUIRE(w.objectCount() == 0);
            THEN("w has no light source") {
                REQUIRE(w.ligthCount() == 0);
            }
        }
    }
}

SCENARIO("The default world", "[World]") {
    GIVEN("light = Light(point(-2.0, 2.0, 0.0), color(1.0, 1.0, 1.0)") {
        AND_GIVEN("s1 = Sphere()"
            "| material.color    | (1.0, 0.0, 0.0) |"
            "| material.diffuse  | 1.0             |"
            "| material.specular | 0.9             |") {
            auto s1 = Sphere();
            s1.setTransform(translation(-1.0, 0.0, -3.0));
            auto material = Material();
            material.color = color(1.0, 0.0, 0.0);
            material.diffuse = 1.0;
            material.specular = 0.9;
            material.shininess = 128.0;
            s1.material = material;
            AND_GIVEN("s2 = Sphere()") {
                auto s2 = Sphere();
                s2.setTransform(translation(1.0, 0.0, -3.0));
                material = Material();
                material.color = color(1.0, 0.2, 1.0);
                material.diffuse = 1.0;
                material.specular = 0.9;
                material.shininess = 128.0;
                s2.material = material;
                WHEN("w = defaultWorld()") {
                    auto w = defaultWorld();
                    THEN("w.light == light") {
                        AND_THEN("w contains s1")
                            REQUIRE(w.contains(s1));
                        AND_THEN("w contains s2")
                            REQUIRE(w.contains(s2));
                    }
                }
            }
        }
    }
}

SCENARIO("Intersect a world with a ray", "[World]") {
    GIVEN("w = defaultWorld()") {
        auto w = defaultWorld();
        AND_GIVEN("r = Ray(point(0.0, 0.0, 5.0)") {
            auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0));
            WHEN("xs = w.intersect(r)") {
                auto xs = w.intersect(r);
                THEN("xs.size() == 4") {
                    REQUIRE(xs.size() == 4);
                    AND_THEN("xs[0].t == 8.0")
                        REQUIRE(xs[0].t == 8.0);
                    AND_THEN("xs[1].t == 8.0")
                        REQUIRE(xs[1].t == 8.0);
                    AND_THEN("xs[2].t == 8.0")
                        REQUIRE(xs[2].t == 8.0);
                    AND_THEN("xs[3].t == 8.0")
                        REQUIRE(xs[3].t == 8.0);
                }
            }
        }
    }
}

SCENARIO("Precomputing the state of an intersection", "[World]") {
    GIVEN("r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0)") {
        auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0));
        AND_GIVEN("shape = Sphere()") {
            auto shape = Sphere();
            AND_GIVEN("i = intersection(4.0, shape)") {
                auto i = Intersection(4.0, shape);
                WHEN("comps = prepareComputations(i, r)") {
                    auto comps = prepareComputations(i, r);
                    THEN("comps.t == i.t") {
                        REQUIRE(comps.t == i.t);
                        AND_THEN("comps.object == i.object")
                            REQUIRE(comps.object == i.object);
                        AND_THEN("comps.position == point(0.0, 0.0, 1.0)")
                            REQUIRE(comps.position == point(0.0, 0.0, 1.0));
                        AND_THEN("comps.viewDirection == vector(0.0, 0.0, -1.0)")
                            REQUIRE(comps.viewDirection == vector(0.0, 0.0, 1.0));
                        AND_THEN("comps.normal == vector(0.0, 0.0, -1.0)")
                            REQUIRE(comps.normal == vector(0.0, 0.0, 1.0));
                    }
                }
            }
        }
    }
}

SCENARIO("The hit, when an intersection occurs on the outside", "[World]") {
    GIVEN("r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0)") {
        auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, -1.0));
        AND_GIVEN("shape = Sphere()") {
            auto shape = Sphere();
            AND_GIVEN("i = Intersection(4.0, shape)") {
                auto i = Intersection(4.0, shape);
                WHEN("comps = prepareComputatons(i, r)") {
                    auto comps = prepareComputations(i, r);
                    THEN("comps.inside == false") {
                        REQUIRE(comps.inside == false);
                    }
                }
            }
        }
    }
}

SCENARIO("The hit, when an intersection occurs on the inside", "[World]") {
    GIVEN("r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, -1.0)") {
        auto r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, -1.0));
        AND_GIVEN("shape = Sphere()") {
            auto shape = Sphere();
            AND_GIVEN("i = Intersection(1.0, shape)") {
                auto i = Intersection(1.0, shape);
                WHEN("comps = prepareComputatons(i, r)") {
                    auto comps = prepareComputations(i, r);
                    THEN("comps.position == point(0, 0, 1)") {
                        REQUIRE(comps.position == point(0.0, 0.0, 1.0));
                        AND_THEN("comps.viewDirection == vector(0.0, 0.0, -1.0)")
                            REQUIRE(comps.viewDirection == vector(0.0, 0.0, -1.0));
                        AND_THEN("comps.inside == true")
                            REQUIRE(comps.inside == true);
                    }
                }
            }
        }
    }
}

SCENARIO("Shading an intersection", "[World]") {
    GIVEN("w = defaultWorld()") {
        auto w = defaultWorld();
        AND_GIVEN("r = Ray(point(-1.0, 0.0, 5.0), vector(0.0, 0.0, -1.0))") {
            auto r = Ray(point(-1.0, 0.0, 5.0), vector(0.0, 0.0, -1.0));
            AND_GIVEN("shape = the first object in w") {
                auto shape = w.getObjects()[0];
                AND_GIVEN("i = Intersection(6.0, shape)") {
                    auto i = Intersection(6.0, shape);
                    WHEN("comps = prepareComputatons(i, r)") {
                        auto comps = prepareComputations(i, r);
                        AND_WHEN("c = shadeHit(w, comps)") {
                            auto c = shadeHit(w, comps);
                            THEN("c == color(0.453392, 0.0, 0.0)") {
                                REQUIRE(c == color(0.453392, 0.0, 0.0));
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("Shading an intersection from the inside", "[World]") {
    GIVEN("w = defaultWorld()") {
        auto w = defaultWorld();
        AND_GIVEN("w.light = Light(point(1, 0.25, -3.0), color(1.0, 1.0, 1.0))") {
            w.addLight(Light(point(1, 0.25, -3.0), color(1.0, 1.0, 1.0)));
            AND_GIVEN("r = ray(point(1, 0, 0), vector(0.0, 0.0, -1.0)") {
                auto r = Ray(point(1.0, 0.0, -3.0), vector(0.0, 0.0, -1.0));
                AND_GIVEN("shape = the second object in w") {
                    auto shape = w.getObjects()[1];
                    AND_GIVEN("i = Intersection(0.5, shape)") {
                        auto i = Intersection(0.5, shape);
                        WHEN("comps = prepareComputatons(i, r)") {
                            auto comps = prepareComputations(i, r);
                            AND_WHEN("c = shadeHit(w, comps)") {
                                auto c = shadeHit(w, comps);
                                THEN("c == color(0.592072, 0.118414, 0.592072)") {
                                    REQUIRE(c == color(0.592072, 0.118414, 0.592072));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("The color when a ray misses", "[World]") {
    GIVEN("w = defaultWorld()") {
        auto w = defaultWorld();
        AND_GIVEN("r = Ray(point(0.0, 0.0, 5.0), vector(0.0, -1.0, 0.0))") {
            auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, -1.0, 0.0));
            WHEN("c = colorAt(w, r)") {
                auto c = colorAt(w, r);
                THEN("c == color(0.0, 0.0, 0.0)") {
                    REQUIRE(c == color(0.0, 0.0, 0.0));
                }
            }
        }
    }
}