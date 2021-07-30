#include "catch.hpp"
#include "Shape.h"

SCENARIO("The default transformation", "[Shape]") {
    GIVEN("s = testShape()") {
        auto s = testShape();
        THEN("s.transformation == identity_materix") {
            REQUIRE(s.transformation == Matrix4());
        }
    }
}

SCENARIO("Assigning a transformation", "[Shape]") {
    GIVEN("s = testShape()") {
        auto s = testShape();
        WHEN("s.setTransform(translation(2.0, 3.0, 4.0)") {
            s.setTransform(translation(2.0, 3.0, 4.0));
            THEN("s.transformation == translation(2.0, 3.0, 4.0)") {
                REQUIRE(s.transformation == translation(2.0, 3.0, 4.0));
            }
        }
    }
}

SCENARIO("The default material", "[Shape]") {
    GIVEN("s = testShape()") {
        auto s = testShape();
        WHEN("m = s.material") {
            auto m = s.material;
            THEN("m == Material()") {
                REQUIRE(m == Material());
            }
        }
    }
}

SCENARIO("Assigning a material", "[Shape]") {
    GIVEN("s = testShape()") {
        auto s = testShape();
        AND_GIVEN("m = Material()") {
            auto m = Material();
            AND_GIVEN("m.ambient = 1.0") {
                m.ambient = 1.0;
                WHEN("s.material = m") {
                    s.material = m;
                    THEN("s.material == m") {
                        REQUIRE(s.material == m);
                    }
                }
            }
        }
    }
}

//features / shapes.feature
//Scenario: Intersecting a scaled shape with a ray
//Given r ¡û ray(point(0, 0, -5), vector(0, 0, 1))
//And s ¡û test_shape()
//When set_transform(s, scaling(2, 2, 2))
//And xs ¡û intersect(s, r)
//Then s.saved_ray.origin = point(0, 0, -2.5)
//And s.saved_ray.direction = vector(0, 0, 0.5)
//Scenario: Intersecting a translated shape with a ray
//Given r ¡û ray(point(0, 0, -5), vector(0, 0, 1))
//And s ¡û test_shape()
//When set_transform(s, translation(5, 0, 0))
//And xs ¡û intersect(s, r)
//Then s.saved_ray.origin = point(-5, 0, -5)
//And s.saved_ray.direction = vector(0, 0, 1)

//features / shapes.feature
//Scenario : Computing the normal on a translated shape
//Given s ¡û test_shape()
//When set_transform(s, translation(0, 1, 0))
//And n ¡û normal_at(s, point(0, 1.70711, -0.70711))
//Then n = vector(0, 0.70711, -0.70711)
//Scenario : Computing the normal on a transformed shape
//Given s ¡û test_shape()
//And m ¡û scaling(1, 0.5, 1) * rotation_z(¦Ð / 5)
//When set_transform(s, m)
//And n ¡û normal_at(s, point(0, ¡Ì2 / 2, -¡Ì2 / 2))
//Then n = vector(0, 0.97014, -0.24254)