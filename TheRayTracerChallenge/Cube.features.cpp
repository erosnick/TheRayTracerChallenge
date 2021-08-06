#include "catch.hpp"
#include "Cube.h"
#include "Intersection.h"

SCENARIO("A ray intersects a cube", "[Cube]") {
    GIVEN("c = Cube()") {
        auto c = std::make_shared<Cube>();
        AND_GIVEN("r = Ray(point(5.0, 0.5, 0.0), vector(-1.0, 0.0, 0.0))") {
            auto r = Ray(point(5.0, 0.5, 0.0), vector(-1.0, 0.0, 0.0));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == 4.0")
                        REQUIRE(xs[0].t == 4.0);
                    AND_THEN("xs[1].t == 6.0")
                        REQUIRE(xs[1].t == 6.0);
                }
            }
        }

        // -x
        AND_GIVEN("r = Ray(point(-5.0, 0.5, 0.0), vector(1.0, 0.0, 0.0))") {
            auto r = Ray(point(-5.0, 0.5, 0.0), vector(1.0, 0.0, 0.0));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == 4.0")
                        REQUIRE(xs[0].t == 4.0);
                    AND_THEN("xs[1].t == 6.0")
                        REQUIRE(xs[1].t == 6.0);
                }
            }
        }

        // +y
        AND_GIVEN("r = Ray(point(0.5, 5.0, 0.0), vector(0.0, -1.0, 0.0))") {
            auto r = Ray(point(0.5, 5.0, 0.0), vector(0.0, -1.0, 0.0));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == 4.0")
                        REQUIRE(xs[0].t == 4.0);
                    AND_THEN("xs[1].t == 6.0")
                        REQUIRE(xs[1].t == 6.0);
                }
            }
        }

        // -y
        AND_GIVEN("r = Ray(point(0.5, -5.0, 0.0), vector(0.0, 1.0, 0.0))") {
            auto r = Ray(point(0.5, -5.0, 0.0), vector(0.0, 1.0, 0.0));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == 4.0")
                        REQUIRE(xs[0].t == 4.0);
                    AND_THEN("xs[1].t == 6.0")
                        REQUIRE(xs[1].t == 6.0);
                }
            }
        }

        // +z
        AND_GIVEN("r = Ray(point(0.5, 0.0, 5.0), vector(0.0, 0.0, -1.0))") {
            auto r = Ray(point(0.5, 0.0, 5.0), vector(0.0, 0.0, -1.0));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == 4.0")
                        REQUIRE(xs[0].t == 4.0);
                    AND_THEN("xs[1].t == 6.0")
                        REQUIRE(xs[1].t == 6.0);
                }
            }
        }

        // -z
        AND_GIVEN("r = Ray(point(0.5, 0.0, -5.0), vector(0.0, 0.0, 1.0))") {
            auto r = Ray(point(0.5, 0.0, -5.0), vector(0.0, 0.0, 1.0));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == 4.0")
                        REQUIRE(xs[0].t == 4.0);
                    AND_THEN("xs[1].t == 6.0")
                        REQUIRE(xs[1].t == 6.0);
                }
            }
        }

        // Inside
        AND_GIVEN("r = Ray(point(0.0, 0.5, 0.0), vector(0.0, 0.0, 1.0))") {
            auto r = Ray(point(0.0, 0.5, 0.0), vector(0.0, 0.0, 1.0));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == -1.0")
                        REQUIRE(xs[0].t == -1.0);
                    AND_THEN("xs[1].t == 1.0")
                        REQUIRE(xs[1].t == 1.0);
                }
            }
        }
    }
}

SCENARIO("A ray misses a cube", "[Cube]") {
    GIVEN("c = Cube()") {
        auto c = std::make_shared<Cube>();
        AND_GIVEN("r = Ray(point(-2, 0, 0), vector(0.2673, 0.5345, 0.8018))") {
            auto r = Ray(point(-2, 0, 0), vector(0.2673, 0.5345, 0.8018));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 0") {
                    REQUIRE(xs.size() == 0);
                }
            }
        }

        AND_GIVEN("r = Ray(point(-2, 0, 0), vector(0.2673, 0.5345, 0.8018))") {
            auto r = Ray(point(0, -2, 0), vector(0.8018, 0.2673, 0.5345));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 0") {
                    REQUIRE(xs.size() == 0);
                }
            }
        }

        AND_GIVEN("r = Ray(point(0, 0, -2), vector(0.5345, 0.8018, 0.2673))") {
            auto r = Ray(point(0, 0, -2), vector(0.5345, 0.8018, 0.2673));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 0") {
                    REQUIRE(xs.size() == 0);
                }
            }
        }

        AND_GIVEN("r = Ray(point(2, 0, 2), vector(0, 0, -1))") {
            auto r = Ray(point(2, 0, 2), vector(0, 0, -1));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 0") {
                    REQUIRE(xs.size() == 0);
                }
            }
        }

        AND_GIVEN("r = Ray(point(-2, 0, 0), vector(0, -1, 0))") {
            auto r = Ray(point(0, 2, 2), vector(0, -1, 0));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 0") {
                    REQUIRE(xs.size() == 0);
                }
            }
        }

        AND_GIVEN("r = Ray(point(-2, 0, 0), vector(-1, 0, 0))") {
            auto r = Ray(point(2, 2, 0), vector(-1, 0, 0));
            WHEN("xs = c.intersect(r)") {
                auto xs = c->intersect(r);
                THEN("xs.size() == 0") {
                    REQUIRE(xs.size() == 0);
                }
            }
        }
    }
}

SCENARIO("The normal on the surface of a cube", "[Cube]") {
    GIVEN("c = Cube()") {
        auto c = std::make_shared<Cube>();
        AND_GIVEN("p = point(1, 0.5, -0.8)") {
            auto p = point(1, 0.5, -0.8);
            WHEN("normal = c.normalAt(p)") {
                auto normal = c->normalAt(p);
                THEN("normal == vector(1, 0, 0)") {
                    REQUIRE(normal == vector(1, 0, 0));
                }
            }
        }

        AND_GIVEN("p = point(-1, -0.2, 0.9)") {
            auto p = point(-1, -0.2, 0.9);
            WHEN("normal = c.normalAt(p)") {
                auto normal = c->normalAt(p);
                THEN("normal == vector(-1, 0, 0)") {
                    REQUIRE(normal == vector(-1, 0, 0));
                }
            }
        }

        AND_GIVEN("p = point(-0.4, 1, -0.1)") {
            auto p = point(-0.4, 1, -0.1);
            WHEN("normal = c.normalAt(p)") {
                auto normal = c->normalAt(p);
                THEN("normal == vector(0, 1, 0)") {
                    REQUIRE(normal == vector(0, 1, 0));
                }
            }
        }

        AND_GIVEN("p = point(0.3, -1, -0.7)") {
            auto p = point(0.3, -1, -0.7);
            WHEN("normal = c.normalAt(p)") {
                auto normal = c->normalAt(p);
                THEN("normal == vector(0, -1, 0)") {
                    REQUIRE(normal == vector(0, -1, 0));
                }
            }
        }

        AND_GIVEN("p = point(-0.6, 0.3, 1)") {
            auto p = point(-0.6, 0.3, 1);
            WHEN("normal = c.normalAt(p)") {
                auto normal = c->normalAt(p);
                THEN("normal == vector(0, 0, 1)") {
                    REQUIRE(normal == vector(0, 0, 1));
                }
            }
        }

        AND_GIVEN("p = point(0.4, 0.4, -1)") {
            auto p = point(0.4, 0.4, -1);
            WHEN("normal = c.normalAt(p)") {
                auto normal = c->normalAt(p);
                THEN("normal == vector(0, 0, -1)") {
                    REQUIRE(normal == vector(0, 0, -1));
                }
            }
        }

        AND_GIVEN("p = point(1, 1, 1)") {
            auto p = point(1, 1, 1);
            WHEN("normal = c.normalAt(p)") {
                auto normal = c->normalAt(p);
                THEN("normal == vector(1, 0, 0)") {
                    REQUIRE(normal == vector(1, 0, 0));
                }
            }
        }

        AND_GIVEN("p = point(-1, -1, -1)") {
            auto p = point(-1, -1, -1);
            WHEN("normal = c.normalAt(p)") {
                auto normal = c->normalAt(p);
                THEN("normal == vector(-1, 0, 0)") {
                    REQUIRE(normal == vector(-1, 0, 0));
                }
            }
        }
    }
}