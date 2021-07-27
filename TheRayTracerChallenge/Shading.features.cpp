#include "catch.hpp"
#include "Sphere.h"
#include "Light.h"
#include "Material.h"

SCENARIO("The normal on a sphere at a point on the x axis", "[Shading]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        WHEN("n = s.normalAt(point(1.0, 0.0, 0.0)") {
            auto n = s.normalAt(point(1.0, 0.0, 0.0));
            THEN("n == vector(1.0, 0.0, 0.0)") {
                REQUIRE(n == vector(1.0, 0.0, 0.0));
            }
        }
    }
}

SCENARIO("The normal on a sphere at a point on the y axis", "[Shading]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        WHEN("n = s.normalAt(point(0.0, 1.0, 0.0)") {
            auto n = s.normalAt(point(0.0, 1.0, 0.0));
            THEN("n == vector(1.0, 0.0, 0.0)") {
                REQUIRE(n == vector(0.0, 1.0, 0.0));
            }
        }
    }
}

SCENARIO("The normal on a sphere at a point on the z axis", "[Shading]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        WHEN("n = s.normalAt(point(0.0, 0.0, 1.0)") {
            auto n = s.normalAt(point(0.0, 0.0, 1.0));
            THEN("n == vector(0.0, 0.0, 1.0)") {
                REQUIRE(n == vector(0.0, 0.0, 1.0));
            }
        }
    }
}

SCENARIO("The normal is a normalized vector", "[Shading]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        WHEN("n = s.normalAt(point(¡Ì3/3, ¡Ì3/3, ¡Ì3/3)") {
            auto n = s.normalAt(point(std::sqrt(3.0) / 3.0, std::sqrt(3.0) / 3.0, std::sqrt(3.0) / 3.0));
            THEN("n == vector(0.0, 0.0, 1.0)") {
                REQUIRE(n == n.normalize());
            }
        }
    }
}

SCENARIO("Computing the normal on a translated sphere", "[Shading]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        AND_GIVEN("s.setTransform(translation(0.0, 1.0, 0.0)") {
            s.setTransform(translation(0.0, 1.0, 0.0));
            WHEN("n = s.normalAt(point(0, 1.70711, -0.70711)") {
                auto n = s.normalAt(point(0, 1.70711, -0.70711));
                THEN("n == vector(0, 0.70711, -0.70711)") {
                    REQUIRE(n == vector(0, 0.70711, -0.70711));
                }
            }
        }
    }
}

//SCENARIO("Computing the normal on a transformed sphere", "[Shading]") {
//    GIVEN("s = sphere()") {
//        auto s = Sphere();
//        AND_GIVEN("m = scaling(1, 0.5, 1) * rotationZ(¦Ð / 5)"
//            "s.setTransform(m)") {
//            auto m = scaling(1.0, 0.5, 1.0) * rotationZ(PI / 5);
//            s.setTransform(m);
//            WHEN("n = s.normalAt(point(0, ¡Ì2/2, -¡Ì2/2)") {
//                auto n = s.normalAt(point(0, 1.70711, -0.70711));
//                THEN("n == vector(0, 0.97014, -0.24254") {
//                    REQUIRE(n == vector(0, 0.97014, -0.24254));
//                }
//            }
//        }
//    }
//}

SCENARIO("Reflecting a vector approaching at 45¡ã", "[Shading]") {
    GIVEN("v = vector(1.0, -1.0, 0)") {
        auto v = vector(1.0, -1.0, 0.0);
        AND_GIVEN("n = vector(0.0, 1.0, 0.0)") {
            auto n = vector(0.0, 1.0, 0.0);
            WHEN("r = reflect(v, n)") {
                auto r = reflect(v, n);
                THEN("r == vector(1.0, 1.0, 0.0)") {
                    REQUIRE(r == vector(1.0, 1.0, 0.0));
                }
            }
        }
    }
}

SCENARIO("Reflecting a vector off a slanted surface", "[Shading]") {
    GIVEN("v = vector(0.0, -1.0, 0)") {
        auto v = vector(0.0, -1.0, 0.0);
        AND_GIVEN("n = vector(¡Ì2/2, ¡Ì2/2, 0.0)") {
            auto n = vector(std::sqrt(2) / 2, std::sqrt(2) / 2, 0.0);
            WHEN("r = reflect(v, n)") {
                auto r = reflect(v, n);
                THEN("r == vector(1.0, 0.0, 0.0)") {
                    REQUIRE(r == vector(1.0, 0.0, 0.0));
                }
            }
        }
    }
}

SCENARIO("A point light has a position and intensity", "[Shading]") {
    GIVEN("intensity = color(1.0, 1.0, 1.0)") {
        auto intensity = color(1.0, 1.0, 1.0);
        AND_GIVEN("position = point(0.0, 0.0, 0.0)") {
            auto position = point(0.0, 0.0, 0.0);
            WHEN("light = Light(position, intensity)") {
                auto light = Light(position, intensity);
                THEN("light.position == position") {
                    REQUIRE(light.position == position);
                    AND_WHEN("light.intensity") {
                        REQUIRE(light.intensity == intensity);
                    }
                }
            }
        }
    }
}

SCENARIO("The default material", "[Shading]") {
    GIVEN("m = Material()") {
        auto m = Material();
        THEN("m.color = color(1.0, 1.0, 1.0,)") {
            m.color = color(1.0, 1.0, 1.0);
            AND_THEN("m.ambient == 0.1")
                REQUIRE(m.ambient == 0.1);
            AND_THEN("m.diffuse == 0.9")
                REQUIRE(m.diffuse == 0.9);
            AND_THEN("m.specular == 0.9")
                REQUIRE(m.specular == 0.9);
            AND_THEN("m.shininess == 128.0")
                REQUIRE(m.shininess == 128.0);
        }
    }
}

SCENARIO("A sphere has a default material", "[Shading]") {
    GIVEN("s = Sphere()") {
        auto s = Sphere();
        WHEN("m = sphere.material") {
            auto m = s.material;
            THEN("m == Material()") {
                REQUIRE(m == Material());
            }
        }
    }
}

SCENARIO("A sphere may be assigned a material", "[Shading]") {
    GIVEN("s = Sphere()") {
        auto s = Sphere();
        AND_GIVEN("m = Matreial()") {
            auto m = Material();
            AND_GIVEN("m.ambient = 1.0") {
                m.ambient = 1.0;
                WHEN("sphere.material = m") {
                    s.material = m;
                    THEN("s.material == m") {
                        REQUIRE(s.material == m);
                    }
                }
            }
        }
    }
}

auto m = Material();
auto position = point(0.0, 0.0, 0.0);

SCENARIO("Lighting with the eye between the light and the surface", "[Shading]") {

}