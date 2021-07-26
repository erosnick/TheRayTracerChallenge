#include "catch.hpp"
#include "Ray.h"
#include "Intersection.h"
#include "Matrix.h"
#include <tuple>
#include <iostream>

SCENARIO("Creating and querying a ray", "[Intersection]") {
    GIVEN("origin = point(1.0, 2.0, 3.0") {
        auto origin = point(1.0, 2.0, 3.0);
        AND_GIVEN("direction = vector(4.0, 5.0, 6.0)") {
            auto direction = vector(4.0, 5.0, 6.0);
            WHEN("r = ray(origin, direction") {
                auto r = Ray(origin, direction);
                THEN("r.origin == origin") {
                    REQUIRE(r.origin == origin);
                    AND_THEN("r.direction == direction") {
                        REQUIRE(r.direction == direction);
                    }
                }
            }
        }
    }
}

SCENARIO("Computing a point from a distance", "[Intersection]") {
    GIVEN("r = ray(point(2.0, 3.0, 4.0), vector(1.0, 0.0, 0.0)") {
        auto r = Ray(point(2.0, 3.0, 4.0), vector(1.0, 0.0, 0.0));
        THEN("r.position(0) == point(2.0, 3.0, 4.0)") {
            REQUIRE(r.position(0.0) == point(2.0, 3.0, 4.0));
            AND_THEN("r.position(1.0) == point(3.0, 3.0, 4.0)") 
                REQUIRE(r.position(1.0) == point(3.0, 3.0, 4.0));
            AND_THEN("r.position(-1.0) == point(1.0, 3.0, 4.0)") 
                REQUIRE(r.position(-1.0) == point(1.0, 3.0, 4.0));
            AND_THEN("r.position(2.5) == point(4.5, 3.0, 4.0)") 
                REQUIRE(r.position(2.5) == point(4.5, 3.0, 4.0));
        }
    }
}

//SCENARIO("A ray intersects a sphere at two points", "[Intersection]") {
//    GIVEN("r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0)") {
//        auto r = Ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
//        AND_GIVEN("s = sphere()") {
//            auto s = Sphere();
//            WHEN("xs = s.intersect(r)") {
//                auto xs = s.intersect(r);
//                THEN("xs.count == 2") {
//                    REQUIRE(std::tuple_size<decltype(xs)>::value == 4);
//                    REQUIRE(std::get<1>(xs) == 2);
//                    AND_THEN("xs[1] == 4.0")
//                        REQUIRE(std::get<2>(xs) == 4.0);
//                    AND_THEN("xs[2] == 6.0")
//                        REQUIRE(std::get<3>(xs) == 6.0);
//                }
//            }
//        }
//    }
//}
//
//SCENARIO("A ray intersects a sphere at a tangent", "[Intersection]") {
//    GIVEN("r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0)") {
//        auto r = Ray(point(0.0, 1.0, -5.0), vector(0.0, 0.0, 1.0));
//        AND_GIVEN("s = sphere()") {
//            auto s = Sphere();
//            WHEN("xs = s.intersect(r)") {
//                auto xs = s.intersect(r);
//                THEN("xs.count == 2") {
//                    REQUIRE(std::tuple_size<decltype(xs)>::value == 4);
//                    REQUIRE(std::get<1>(xs) == 2);
//                    AND_THEN("xs[2] == 5.0")
//                        REQUIRE(std::get<2>(xs) == 5.0);
//                    AND_THEN("xs[3] == 5.0")
//                        REQUIRE(std::get<3>(xs) == 5.0);
//                }
//            }
//        }
//    }
//}
//
//SCENARIO("A ray misses a sphere", "[Intersection]") {
//    GIVEN("r = ray(point(0.0, 2.0, -5.0), vector(0.0, 0.0, 1.0)") {
//        auto r = Ray(point(0.0, 2.0, -5.0), vector(0.0, 0.0, 1.0));
//        AND_WHEN("s = sphere()") {
//            auto s = Sphere();
//            WHEN("xs = s.intersect(r)") {
//                auto xs = s.intersect(r);
//                THEN("xs.cout == 0") {
//                    REQUIRE(std::get<1>(xs) == 0);
//                }
//            }
//        }
//    }
//}
//
//SCENARIO("A ray originates inside a sphere", "[Intersection]") {
//    GIVEN("r = ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0)") {
//        auto r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
//        AND_WHEN("s = sphere()") {
//            auto s = Sphere();
//            WHEN("xs = s.intersect(r)") {
//                auto xs = s.intersect(r);
//                THEN("xs.cout == 2") {
//                    REQUIRE(std::get<1>(xs) == 2);
//                    AND_WHEN("xs[2] == -1.0")
//                        REQUIRE(std::get<2>(xs) == -1.0);
//                    AND_WHEN("xs[3] ==  1.0")
//                        REQUIRE(std::get<3>(xs) ==  1.0);
//                }
//            }
//        }
//    }
//}
//
//SCENARIO("A sphere is behind a ray", "[Intersection]") {
//    GIVEN("r = ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0)") {
//        auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0));
//        AND_WHEN("s = sphere()") {
//            auto s = Sphere();
//            WHEN("xs = s.intersect(r)") {
//                auto xs = s.intersect(r);
//                THEN("xs.cout == 2") {
//                    REQUIRE(std::get<1>(xs) == 2);
//                    AND_WHEN("xs[2] == -6.0")
//                        REQUIRE(std::get<2>(xs) == -6.0);
//                    AND_WHEN("xs[3] ==  -4.0")
//                        REQUIRE(std::get<3>(xs) == -4.0);
//                }
//            }
//        }
//    }
//}

SCENARIO("A ray intersects a sphere at two points", "[Intersection]") {
    GIVEN("r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0)") {
        auto r = Ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        AND_GIVEN("s = sphere()") {
            auto s = Sphere();
            WHEN("xs = s.intersect(r)") {
                auto xs = s.intersect(r);
                THEN("xs.count == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == 4.0")
                        REQUIRE(xs[0].t == 4.0);
                    AND_THEN("xs[1].t == 6.0")
                        REQUIRE(xs[1].t == 6.0);
                }
            }
        }
    }
}

SCENARIO("A ray intersects a sphere at a tangent", "[Intersection]") {
    GIVEN("r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0)") {
        auto r = Ray(point(0.0, 1.0, -5.0), vector(0.0, 0.0, 1.0));
        AND_GIVEN("s = sphere()") {
            auto s = Sphere();
            WHEN("xs = s.intersect(r)") {
                auto xs = s.intersect(r);
                THEN("xs.count == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == 5.0")
                        REQUIRE(xs[0].t == 5.0);
                    AND_THEN("xs[1].t == 5.0")
                        REQUIRE(xs[1].t == 5.0);
                }
            }
        }
    }
}

SCENARIO("A ray misses a sphere", "[Intersection]") {
    GIVEN("r = ray(point(0.0, 2.0, -5.0), vector(0.0, 0.0, 1.0)") {
        auto r = Ray(point(0.0, 2.0, -5.0), vector(0.0, 0.0, 1.0));
        AND_WHEN("s = sphere()") {
            auto s = Sphere();
            WHEN("xs = s.intersect(r)") {
                auto xs = s.intersect(r);
                THEN("xs.cout == 0") {
                    REQUIRE(xs.size() == 0);
                }
            }
        }
    }
}

SCENARIO("A ray originates inside a sphere", "[Intersection]") {
    GIVEN("r = ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0)") {
        auto r = Ray(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        AND_WHEN("s = sphere()") {
            auto s = Sphere();
            WHEN("xs = s.intersect(r)") {
                auto xs = s.intersect(r);
                THEN("xs.cout == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_WHEN("xs[0].t == -1.0")
                        REQUIRE(xs[0].t == -1.0);
                    AND_WHEN("xs[1] ==  1.0")
                        REQUIRE(xs[1].t == 1.0);
                }
            }
        }
    }
}

SCENARIO("A sphere is behind a ray", "[Intersection]") {
    GIVEN("r = ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0)") {
        auto r = Ray(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0));
        AND_WHEN("s = sphere()") {
            auto s = Sphere();
            WHEN("xs = s.intersect(r)") {
                auto xs = s.intersect(r);
                THEN("xs.cout == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_WHEN("xs[0] == -6.0")
                        REQUIRE(xs[0].t == -6.0);
                    AND_WHEN("xs[1] ==  -4.0")
                        REQUIRE(xs[1].t == -4.0);
                }
            }
        }
    }
}

SCENARIO("An intersection encapsulates t and object", "[Intersection]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        WHEN("i = Intersection(3.5, s)") {
            auto i = Intersection(3.5, s);
            THEN("i.t == 3.5") {
                REQUIRE(i.t == 3.5);
                AND_THEN("i.object == s") {
                    REQUIRE(i.object == s);
                }
            }
        }
    }
}

// 平行的AND_GIVEN也会导致AND_GIVENSCENARIO调用多次
SCENARIO("Aggregating intersections", "[Intersection]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        AND_GIVEN("i1 = Intersection(1, s)"
                  "i2 = Intersection(2, s)") {
            auto i1 = Intersection(1.0, s);
            auto i2 = Intersection(2.0, s);
            WHEN("xs = Intersections(i1, i2)") {
                auto xs = intersections({ i1, i2 });
                THEN("xs.count == 2") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].t == 1.0")
                        REQUIRE(xs[0].t == 1.0f);
                    AND_THEN("xs[1].t == 2.0")
                        REQUIRE(xs[1].t == 2.0f);
                }
            }
        }
    }
}

SCENARIO("Intersect sets the object on the intersection", "[Intersection]") {
    GIVEN("r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0)") {
        auto r = Ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        AND_GIVEN("s = sphere()") {
            auto s = Sphere();
            WHEN("xs = s.intersect(r)") {
                auto xs = s.intersect(r);
                THEN("xs.count == 2.") {
                    REQUIRE(xs.size() == 2);
                    AND_THEN("xs[0].object == s")
                        REQUIRE(xs[0].object == s);
                    AND_THEN("xs[1].object == s")
                        REQUIRE(xs[1].object == s);
                }
            }
        }
    }
}

SCENARIO("The hit, when all intersections have positive t", "[Intersection]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        AND_GIVEN("i1 = Intersection(1.0, s)"
                  "i2 = Intersection(2.0, s)"
                  "xs = intersections(i2, i1)"){
            auto i1 = Intersection(1.0, s);
            auto i2 = Intersection(2.0, s);
            auto xs = intersections({ i2, i1 });
            WHEN("i = hit(xs)") {
                auto i = hit(xs);
                THEN("i == i1") {
                    REQUIRE(i == i1);
                }
            }
        }
    }
}

SCENARIO("The hit, when some intersections have negative t", "[Intersection]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        AND_GIVEN("i1 = Intersection(-1.0, s)"
                  "i2 = Intersection(1.0, s)"
                  "xs = intersections(i2, i1)") {
            auto i1 = Intersection(-1.0, s);
            auto i2 = Intersection(1.0, s);
            auto xs = intersections({ i2, i1 });
            WHEN("i = hit(xs)") {
                auto i = hit(xs);
                THEN("i == i2") {
                    REQUIRE(i == i2);
                }
            }
        }
    }
}

SCENARIO("The hit, when all intersections have negative t", "[Intersection]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        AND_GIVEN("i1 = Intersection(-2.0, s)"
                  "i2 = Intersection(-1.0, s)"
                  "xs = intersections(i2, i1)") {
            auto i1 = Intersection(-2.0, s);
            auto i2 = Intersection(-1.0, s);
            auto xs = intersections({ i2, i1 });
            WHEN("i = hit(xs)") {
                auto i = hit(xs);
                THEN("i == i2") {
                    REQUIRE(i.bHit == false);
                }
            }
        }
    }
}

SCENARIO("The hit is always the lowest nonnegative intersection", "[Intersection]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        AND_GIVEN("i1 = Intersection(5.0, s)"
                  "i2 = Intersection(7.0, s)"
                  "i3 = Intersection(-3.0, s)"
                  "i4 = Intersection(2.0, s)"
                  "xs = intersections(i1, i2, i3, i4)") {
            auto i1 = Intersection(5.0, s);
            auto i2 = Intersection(7.0, s);
            auto i3 = Intersection(-3.0, s);
            auto i4 = Intersection(2.0, s);
            auto xs = intersections({ i1, i2, i3, i4 });
            WHEN("i = hit(xs)") {
                auto i = hit(xs);
                THEN("i == i4") {
                    REQUIRE(i == i4);
                }
            }
        }
    }
}

SCENARIO("Translating a ray", "[Intersection]") {
    GIVEN("r = ray(point(1.0, 2.0, 3.0), vector(0.0, 1.0, 0.0)") {
        auto r = Ray(point(1.0, 2.0, 3.0), vector(0.0, 1.0, 0.0));
        AND_GIVEN("m = translation(3.0, 4.0, 5.0)") {
            auto m = translation(3.0, 4.0, 5.0);
            WHEN("r2 = transform(r, m)") {
                auto r2 = transformRay(r, m);
                THEN("r2.origin == point(4.0, 6.0, 8.0)") {
                    REQUIRE(r2.origin == point(4.0, 6.0, 8.0));
                    AND_THEN("r2.direction == vector(0.0, 1.0, 0.0)") {
                        REQUIRE(r2.direction == vector(0.0, 1.0, 0.0));
                    }
                }
            }
        }
    }
}

SCENARIO("Scaling a ray", "[Intersection]") {
    GIVEN("r = ray(point(1.0, 2.0, 3.0), vector(0.0, 1.0, 0.0)") {
        auto r = Ray(point(1.0, 2.0, 3.0), vector(0.0, 1.0, 0.0));
        AND_GIVEN("m = scaling(2.0, 3.0, 4.0)") {
            auto m = scaling(2.0, 3.0, 4.0);
            WHEN("r2 = transform(r, m)") {
                auto r2 = transformRay(r, m);
                THEN("r2.origin == point(2.0, 6.0, 12.0)") {
                    REQUIRE(r2.origin == point(2.0, 6.0, 12.0));
                    AND_THEN("r2.direction == vector(0.0, 3.0, 0.0)") {
                        REQUIRE(r2.direction == vector(0.0, 3.0, 0.0));
                    }
                }
            }
        }
    }
}

SCENARIO(" A sphere's default transformation", "[Intersection]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        THEN("s.transform == I") {
            auto I = Matrix4();
            REQUIRE(s.transform == I);
        }
    }
}

SCENARIO("Changing a sphere's transformation", "[Intersection]") {
    GIVEN("s = sphere()") {
        auto s = Sphere();
        AND_GIVEN("t = translation(2.0, 3.0, 4.0)") {
            auto t = translation(2.0, 3.0, 4.0);
            WHEN("s.setTransform(t)") {
                s.setTransform(t);
                THEN("s.transform == t") {
                    REQUIRE(s.transform == t);
                }
            }
        }
    }
}

SCENARIO("Intersecting a scaled sphere with a ray", "[Intersection]") {
    GIVEN("r = ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0)") {
        auto r = Ray(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        AND_GIVEN("s = sphere()") {
            auto s = Sphere();
            WHEN("s.setTransform(scaling(2.0, 2.0, 2.0))") {
                s.setTransform(scaling(2.0, 2.0, 2.0));
                AND_WHEN("xs = s.intersect(r)") {
                    auto xs = s.intersect(r, true);
                    THEN("xs.count == 2"
                         "xs[0].t == 3.0"
                         "xs[1].t == 7.0") {
                        REQUIRE(xs.size() == 2);
                        REQUIRE(xs[0].t == 3.0);
                        REQUIRE(xs[1].t == 7.0);
                    }
                }
            }
        }
    }
}