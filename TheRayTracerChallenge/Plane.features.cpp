#include "catch.hpp"
#include "Shape.h"
#include "Plane.h"
#include "Types.h"
#include "Material.h"

SCENARIO("The default transformation", "[Plane]") {
    GIVEN("s = testShape()") {
        auto s = testShape();
        THEN("s.transformation == identity_materix") {
            REQUIRE(s.transformation == Matrix4());
        }
    }
}

SCENARIO("Assigning a transformation to plane", "[Plane]") {
    GIVEN("s = testShape()") {
        auto s = testShape();
        WHEN("s.setTransformation(translate(2.0, 3.0, 4.0)") {
            s.setTransformation(translate(2.0, 3.0, 4.0));
            THEN("s.transformation == translate(2.0, 3.0, 4.0)") {
                REQUIRE(s.transformation == translate(2.0, 3.0, 4.0));
            }
        }
    }
}

SCENARIO("The default material", "[Plane]") {
    GIVEN("s = testShape()") {
        auto s = testShape();
        WHEN("m = s.material") {
            auto m = s.material;
            THEN("m == Material()") {
                REQUIRE((*m.get()) == Material());
            }
        }
    }
}

SCENARIO("Assigning a material", "[Plane]") {
    GIVEN("s = testShape()") {
        auto s = testShape();
        AND_GIVEN("m = Material()") {
            auto m = std::make_shared<Material>();
            AND_GIVEN("m.ambient = 1.0") {
                m->ambient = 1.0;
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

SCENARIO("The normal of a plane is constant everywhere", "[Plane]") {
    GIVEN("p = Plane()") {
        auto p = Plane();
        WHEN("n1 = p.normalAt(point(0.0, 0.0, 0.0))"
             "n2 = p.normalAt(point(-5.0, 0.0, 150.0))"
             "n3 = p.normalAt(point(-5.0, 0.0, 150.0))") {
            auto n1 = p.normalAt(point(0.0, 0.0, 0.0));
            auto n2 = p.normalAt(point(-5.0, 0.0, 150.0));
            auto n3 = p.normalAt(point(-5.0, 0.0, 150.0));
            THEN("n1 == vector(0.0, 1.0, 0.0)") {
                REQUIRE(n1 == vector(0.0, 1.0, 0.0));
                AND_THEN("n2 == vector(0.0, 1.0, 0.0)")
                    REQUIRE(n2 == vector(0.0, 1.0, 0.0));
                AND_THEN("n3 == vector(0.0, 1.0, 0.0)")
                    REQUIRE(n3 == vector(0.0, 1.0, 0.0));
            }
        }
    }
}

SCENARIO("Intersect with a ray parallel to the plane", "[Plane]") {
    GIVEN("p = Plane()") {
        auto p = Plane();
        AND_GIVEN("r = Ray(point(0.0, 10.0, 0.0), vector(0.0, 0.0, 1.0)") {
            auto r = Ray(point(0.0, 10.0, 0.0), vector(0.0, 0.0, 1.0));
            WHEN("xs = p.intersect(r)") {
                auto xs = p.intersect(r);
                THEN("xs is empty") {
                    REQUIRE(xs.size() == 0);
                }
            }
        }
    }
}

SCENARIO("Intersect with a coplanar ray", "[Plane]") {
    GIVEN("p = Plane()") {
        auto p = Plane();
        AND_GIVEN("r = Ray(point(0.0, 10.0, 0.0), vector(0.0, 0.0, 1.0)") {
            auto r = Ray(point(0.0, 10.0, 0.0), vector(0.0, 0.0, 1.0));
            WHEN("xs = p.intersect(r)") {
                auto xs = p.intersect(r);
                THEN("xs is empty") {
                    REQUIRE(xs.size() == 0);
                }
            }
        }
    }
}

SCENARIO("A ray intersecting a plane from above", "[Plane]") {
    GIVEN("p = Plane()") {
        auto p = std::make_shared<Plane>(point(0.0, -1.0, 0.0), vector(0.0, 1.0, 0.0));
        AND_GIVEN("r = Ray(point(0.0, 0.0, 0.0), vector(0.0, -1.0, 0.0))") {
            auto r = Ray(point(0.0, 0.0, 0.0), vector(0.0, -1.0, 0.0));
            WHEN("xs = p.intersect(r)") {
                auto xs = p->intersect(r);
                THEN("xs.size() == 1)") {
                    REQUIRE(xs.size() == 1);
                    AND_THEN("xs[0].t == 1¡£0")
                        REQUIRE(xs[0].t == 1.0);
                    AND_THEN("xs[0].object == p")
                        REQUIRE(xs[0].object == p);
                }
            }
        }
    }
}

SCENARIO("A ray intersecting a plane from below", "[Plane]") {
    GIVEN("p = Plane()") {
        auto p = std::make_shared<Plane>(point(0.0, 1.0, 0.0), vector(0.0, -1.0, 0.0));
        AND_GIVEN("r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 1.0, 0.0))") {
            auto r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 1.0, 0.0));
            WHEN("xs = p.intersect(r)") {
                auto xs = p->intersect(r);
                THEN("xs.size() == 1)") {
                    REQUIRE(xs.size() == 1);
                    AND_THEN("xs[0].t == 1")
                        REQUIRE(xs[0].t == 1.0);
                    AND_THEN("xs[0].object == p")
                        REQUIRE(xs[0].object == p);
                }
            }
        }
    }
}