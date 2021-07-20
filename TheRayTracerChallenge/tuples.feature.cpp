#include "catch.hpp"
#include "tuple.h"

SCENARIO("A tuple with w = 1.0 is a point", "[tuple]") {
	GIVEN("A ¡û tuple(4.3, -4.2, 3.1, 1.0") {
		auto a = tuple(4.3f, -4.2f, 3.1f, 1.0f);
		THEN("") {
			REQUIRE(a.x == 4.3f);
			REQUIRE(a.y == -4.2f);
			REQUIRE(a.z == 3.1f);
			REQUIRE(a.w == 1.0f);
		}
	}
}

SCENARIO("A tuple with w = 0.0 is a vector", "[tuple]") {
	GIVEN("A ¡û tuple(4.3, -4.2, 3.1, 1.0") {
		auto a = tuple(4.3f, -4.2f, 3.1f, 0.0f);
		THEN("") {
			REQUIRE(a.x == 4.3f);
			REQUIRE(a.y == -4.2f);
			REQUIRE(a.z == 3.1f);
			REQUIRE(a.w == 0.0f);
		}
	}
}

SCENARIO("point() creates tuples with w = 1", "[tuple]") {
	GIVEN("p ¡û point(4.0f, -4.0f, 3.0f)") {
		auto p = tuple::point(4.0f, -4.0f, 3.0f);
		THEN("p = tuple(4, -4, 3, 1)") {
			REQUIRE(p == tuple(4.0f, -4.0f, 3.0f, 1.0f));
		}
	}
}

SCENARIO("vector() creates tuples with w = 0", "[tuple]") {
	GIVEN("v ¡û point(4.0f, -4.0f, 3.0f)") {
		auto v = tuple::vector(4.0f, -4.0f, 3.0f);
		THEN("v = tuple(4, -4, 3, 1)") {
			REQUIRE(v == tuple(4.0f, -4.0f, 3.0f, 0.0f));
		}
	}
}

SCENARIO("Adding two tuples") {
	GIVEN("a1 ¡û tuple(3.0f, -2.0f, 5.0f, 1.0f") {
		auto a1 = tuple(3.0f, -2.0f, 5.0f, 1.0f);
		AND_GIVEN("a2 ¡û tuple(-2.0f, 3.0f, 1.0f, 0.0f") {
			auto a2 = tuple(-2.0f, 3.0f, 1.0f, 0.0f);
			REQUIRE(a1 + a2 == tuple(1.0f, 1.0f, 6.0f, 1.0f));
		}
	}
}

SCENARIO("Substracting two points", "[tuple]") {
	GIVEN("p1 ¡û point(3.0f, 2.0f, 1.0f") {
		auto p1 = tuple::point(3.0f, 2.0f, 1.0f);
		AND_GIVEN("p2 ¡û point(5.0f, 6.0f, 7.0f") {
			auto p2 = tuple::point(5.0f, 6.0f, 7.0f);
			REQUIRE(p1 - p2 == tuple::vector(-2.0f, -4.0f, -6.0f));
		}
	}
}

SCENARIO("Substracting a vector from a point", "[tuple]") {
	GIVEN("p1 ¡û point(3.0f, 2.0f, 1.0f") {
		auto p = tuple::point(3.0f, 2.0f, 1.0f);
		AND_GIVEN("p2 ¡û point(5.0f, 6.0f, 7.0f") {
			auto v = tuple::vector(5.0f, 6.0f, 7.0f);
			REQUIRE(p - v == tuple::point(-2.0f, -4.0f, -6.0f));
		}
	}
}

SCENARIO("Substracting two vectors", "[tuple]") {
	GIVEN("v1 ¡û point(3.0f, 2.0f, 1.0f") {
		auto v1 = tuple::point(3.0f, 2.0f, 1.0f);
		AND_GIVEN("v2 ¡û point(5.0f, 6.0f, 7.0f") {
			auto v2 = tuple::point(5.0f, 6.0f, 7.0f);
			REQUIRE(v1 - v2 == tuple::vector(-2.0f, -4.0f, -6.0f));
		}
	}
}

SCENARIO("Substracting a vector from the zero vector", "[tuple]") {
	GIVEN("zero ¡û vector(0.0f, 0.0f, 0.0f") {
		auto zero = tuple::vector(0.0f, 0.0f, 0.0f);
		AND_GIVEN("v ¡û vector(1.0f, -2.0f, 3.0f") {
			auto v = tuple::vector(1.0f, -2.0f, 3.0f);
			REQUIRE(zero - v == tuple::vector(-1.0f, 2.0f, -3.0f));
		}
	}
}

SCENARIO("Negating a tuple", "[tuple]") {
	GIVEN("a ¡û tuple(1.0f, -2.0f, 3.0f, -4.0f") {
		auto a = tuple(1.0f, -2.0f, 3.0f, -4.0);
		REQUIRE(-a == tuple(-1.0f, 2.0f, -3.0f, 4.0f));
	}
}

SCENARIO("Multiplying a tuple by a scalar", "[tuple]") {
	GIVEN("a ¡û tuple(1.0f, -2.0f, 3.0f, -4.0f") {
		auto a = tuple(1.0f, -2.0f, 3.0f, -4.0f);
		REQUIRE(a * 3.5f == tuple(3.5f, -7.0f, 10.5f, -14.0f));
	}
}

SCENARIO("Multiplying a tuple by a fraction", "[tuple]") {
	GIVEN("a ¡û tuple(1.0f, -2.0f, 3.0f, -4.0f") {
		auto a = tuple(1.0f, -2.0f, 3.0f, -4.0f);
		REQUIRE(a * 0.5f == tuple(0.5f, -1.0f, 1.5f, -2.0f));
	}
}

SCENARIO("Dividing a tuple by a scalar", "[tuple]") {
	GIVEN("a ¡û tuple(1.0f, -2.0f, 3.0f, -4.0f") {
		auto a = tuple(1.0f, -2.0f, 3.0f, -4.0f);
		REQUIRE(a / 2.0f == tuple(0.5f, -1.0f, 1.5f, -2.0f));
	}
}

SCENARIO("Computing the magnitude of vector(1.0f, 0.0f, 0.0f)", "[tuple]") {
	GIVEN("v ¡û vector(1.0f, 0.0f, 0.0f)") {
		auto v = tuple::vector(1.0f, 0.0f, 0.0f);
		REQUIRE(v.magnitude() == 1.0f);
	}
}

SCENARIO("Computing the magnitude of vector(0.0f, 1.0f, 0.0f)", "[tuple]") {
	GIVEN("v ¡û vector(0.0f, 1.0f, 0.0f)") {
		auto v = tuple::vector(0.0f, 1.0f, 0.0f);
		REQUIRE(v.magnitude() == 1.0f);
	}
}

SCENARIO("Computing the magnitude of vector(0.0f, 0.0f, 1.0f)", "[tuple]") {
	GIVEN("v ¡û vector(0.0f, 0.0f, 1.0f)") {
		auto v = tuple::vector(0.0f, 0.0f, 1.0f);
		REQUIRE(v.magnitude() == 1.0f);
	}
}

SCENARIO("Computing the magnitude of vector(1.0f, 2.0f, 3.0f)", "[tuple]") {
	GIVEN("v ¡û vector(1.0f, 2.0f, 3.0f)") {
		auto v = tuple::vector(1.0f, 2.0f, 3.0f);
		REQUIRE(v.magnitude() == std::sqrtf(14.0f));
	}
}

SCENARIO("Computing the magnitude of vector(-1.0f, -2.0f, -3.0f)", "[tuple]") {
	GIVEN("v ¡û vector(-1.0f, -2.0f, -3.0f)") {
		auto v = tuple::vector(-1.0f, -2.0f, -3.0f);
		REQUIRE(v.magnitude() == std::sqrtf(14.0f));
	}
}

SCENARIO("Normalizing vector(4.0f, 0.0f, 0.0f) gives (1.0f, 0.0f, 0.0f)", "[tuple]") {
	GIVEN("v ¡û vector(4.0f, 0.0f, 0.0f") {
		auto v = tuple::vector(4.0f, 0.0f, 0.0f);
		REQUIRE(v.normalize() == tuple::vector(1.0f, 0.0f, 0.0f));
	}
}

SCENARIO("Normalizing vector(1.0f, 2.0f, 3.0f)", "[tuple]") {
	GIVEN("v ¡û vector(1.0f, 2.0f, 3.0f") {
		auto v = tuple::vector(1.0f, 2.0f, 3.0f);
		REQUIRE(v.normalize() == tuple::vector(0.267261f, 0.534522f, 0.801784f));
	}
}

SCENARIO("The magnitude of a normalized vector", "[tuple]") {
	GIVEN("v ¡û vector(1.0f, 2.0f, 3.0f") {
		auto v = tuple::vector(4.0f, 0.0f, 0.0f);
		WHEN("") {
			auto norm = v.normalize();
			REQUIRE(norm.magnitude() == 1.0f);
		}
	}
}

SCENARIO("The dot product of two tuples 1", "[tuple]") {
	GIVEN("a ¡û vector(1.0f, 2.0f, 3.0f") {
		auto a = tuple::vector(1.0f, 2.0f, 3.0f);
		AND_GIVEN("b ¡û vector(2.0f, 3.0f, 4.0f") {
			auto b = tuple::vector(2.0f, 3.0f, 4.0f);
			REQUIRE(a * b == 20);
		}
	}
}

SCENARIO("The dot product of two tuples 2", "[tuple]") {
	GIVEN("a ¡û vector(1.0f, 2.0f, 3.0f") {
		auto a = tuple::vector(1.0f, 2.0f, 3.0f);
		AND_GIVEN("b ¡û vector(2.0f, 3.0f, 4.0f") {
			auto b = tuple::vector(2.0f, 3.0f, 4.0f);
			REQUIRE(a.dot(b) == 20);
		}
	}
}

SCENARIO("The cross product of two vectors", "[tuple]") {
	GIVEN("a ¡û vector(1.0f, 2.0f, 3.0f") {
		auto a = tuple::vector(1.0f, 2.0f, 3.0f);
		AND_GIVEN("b ¡û vector(2.0f, 3.0f, 4.0f") {
			auto b = tuple::vector(2.0f, 3.0f, 4.0f);
			REQUIRE(a.cross(b) == tuple::vector(-1.0f, 2.0f, -1.0f));
			REQUIRE(b.cross(a) == tuple::vector(1.0f, -2.0f, 1.0));
		}
	}
}