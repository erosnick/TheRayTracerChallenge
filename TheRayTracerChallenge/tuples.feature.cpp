#include "catch.hpp"
#include "tuple.h"

SCENARIO("A tuple with w = 1.0 is a point", "[Tuple]") {
	GIVEN("A = tuple(4.3, -4.2, 3.1, 1.0") {
		auto a = Tuple(4.3, -4.2, 3.1, 1.0);
		THEN("a.x == 4.3"
		     "a.y == -4.2"
			 "a.z == 3.1"
			 "a.w == 1.0") {
			REQUIRE(a.x == 4.3);
			REQUIRE(a.y == -4.2);
			REQUIRE(a.z == 3.1);
			REQUIRE(a.w == 1.0);
		}
	}
}

SCENARIO("A tuple with w = 0.0 is a vector", "[Tuple]") {
	GIVEN("A = tuple(4.3, -4.2, 3.1, 1.0") {
		auto a = Tuple(4.3, -4.2, 3.1, 0.0);
		THEN("a.x ==  4.3f" 
			 "a.y == -4.2f" 
			 "a.z ==  3.1" 
			 "a.w ==  0.0") {
			REQUIRE(a.x == 4.3);
			REQUIRE(a.y == -4.2);
			REQUIRE(a.z == 3.1);
			REQUIRE(a.w == 0.0);
		}
	}
}

SCENARIO("point() creates tuples with w = 1", "[Tuple]") {
	GIVEN("p = point(4.0, -4.0, 3.0)") {
		auto p = point(4.0, -4.0, 3.0);
		THEN("p == tuple(4, -4, 3, 1)") {
			REQUIRE(p == Tuple(4.0, -4.0, 3.0, 1.0));
		}
	}
}

SCENARIO("vector() creates tuples with w = 0", "[Tuple]") {
	GIVEN("v = point(4.0, -4.0, 3.0)") {
		auto v = vector(4.0, -4.0, 3.0);
		THEN("v == tuple(4, -4, 3, 1)") {
			REQUIRE(v == Tuple(4.0, -4.0, 3.0, 0.0));
		}
	}
}

SCENARIO("Adding two tuples") {
	GIVEN("a1 = tuple(3.0, -2.0, 5.0, 1.0") {
		auto a1 = Tuple(3.0, -2.0, 5.0, 1.0);
		AND_GIVEN("a2 = tuple(-2.0, 3.0, 1.0, 0.0") {
			auto a2 = Tuple(-2.0, 3.0, 1.0, 0.0);
			REQUIRE(a1 + a2 == Tuple(1.0, 1.0, 6.0, 1.0));
		}
	}
}

SCENARIO("Substracting two points", "[Tuple]") {
	GIVEN("p1 = point(3.0, 2.0, 1.0") {
		auto p1 = point(3.0, 2.0, 1.0);
		AND_GIVEN("p2 = point(5.0, 6.0, 7.0") {
			auto p2 = point(5.0, 6.0, 7.0);
			REQUIRE(p1 - p2 == vector(-2.0, -4.0, -6.0));
		}
	}
}

SCENARIO("Substracting a vector from a point", "[Tuple]") {
	GIVEN("p1 = point(3.0, 2.0, 1.0") {
		auto p = point(3.0, 2.0, 1.0);
		AND_GIVEN("p2 = point(5.0, 6.0, 7.0") {
			auto v = vector(5.0, 6.0, 7.0);
			REQUIRE(p - v == point(-2.0, -4.0, -6.0));
		}
	}
}

SCENARIO("Substracting two vectors", "[Tuple]") {
	GIVEN("v1 = point(3.0, 2.0, 1.0") {
		auto v1 = point(3.0, 2.0, 1.0);
		AND_GIVEN("v2 = point(5.0, 6.0, 7.0") {
			auto v2 = point(5.0, 6.0, 7.0);
			REQUIRE(v1 - v2 == vector(-2.0, -4.0, -6.0));
		}
	}
}

SCENARIO("Substracting a vector from the zero vector", "[Tuple]") {
	GIVEN("zero = vector(0.0, 0.0, 0.0") {
		auto zero = vector(0.0, 0.0, 0.0);
		AND_GIVEN("v = vector(1.0, -2.0, 3.0") {
			auto v = vector(1.0, -2.0, 3.0);
			REQUIRE(zero - v == vector(-1.0, 2.0, -3.0));
		}
	}
}

SCENARIO("Negating a tuple", "[Tuple]") {
	GIVEN("a = tuple(1.0, -2.0, 3.0, -4.0") {
		auto a = Tuple(1.0, -2.0, 3.0, -4.0);
		REQUIRE(-a == Tuple(-1.0, 2.0, -3.0, 4.0));
	}
}

SCENARIO("Multiplying a tuple by a scalar", "[Tuple]") {
	GIVEN("a = tuple(1.0, -2.0, 3.0, -4.0") {
		auto a = Tuple(1.0, -2.0, 3.0, -4.0);
		REQUIRE(a * 3.5 == Tuple(3.5, -7.0, 10.5, -14.0));
	}
}

SCENARIO("Multiplying a tuple by a fraction", "[Tuple]") {
	GIVEN("a = tuple(1.0, -2.0, 3.0, -4.0") {
		auto a = Tuple(1.0, -2.0, 3.0, -4.0);
		REQUIRE(a * 0.5 == Tuple(0.5, -1.0, 1.5, -2.0));
	}
}

SCENARIO("Dividing a tuple by a scalar", "[Tuple]") {
	GIVEN("a = tuple(1.0, -2.0, 3.0, -4.0") {
		auto a = Tuple(1.0, -2.0, 3.0, -4.0);
		REQUIRE(a / 2.0 == Tuple(0.5, -1.0, 1.5, -2.0));
	}
}

SCENARIO("Computing the magnitude of vector(1.0, 0.0, 0.0)", "[Tuple]") {
	GIVEN("v = vector(1.0, 0.0, 0.0)") {
		auto v = vector(1.0, 0.0, 0.0);
		REQUIRE(v.magnitude() == 1.0);
	}
}

SCENARIO("Computing the magnitude of vector(0.0, 1.0, 0.0)", "[Tuple]") {
	GIVEN("v = vector(0.0, 1.0, 0.0)") {
		auto v = vector(0.0, 1.0, 0.0);
		REQUIRE(v.magnitude() == 1.0);
	}
}

SCENARIO("Computing the magnitude of vector(0.0, 0.0, 1.0)", "[Tuple]") {
	GIVEN("v = vector(0.0, 0.0, 1.0)") {
		auto v = vector(0.0, 0.0, 1.0);
		REQUIRE(v.magnitude() == 1.0);
	}
}

SCENARIO("Computing the magnitude of vector(1.0, 2.0, 3.0)", "[Tuple]") {
	GIVEN("v = vector(1.0, 2.0, 3.0)") {
		auto v = vector(1.0, 2.0, 3.0);
		REQUIRE(v.magnitude() == std::sqrt(14.0));
	}
}

SCENARIO("Computing the magnitude of vector(-1.0, -2.0, -3.0)", "[Tuple]") {
	GIVEN("v = vector(-1.0, -2.0, -3.0)") {
		auto v = vector(-1.0, -2.0, -3.0);
		REQUIRE(v.magnitude() == std::sqrt(14.0));
	}
}

SCENARIO("Normalizing vector(4.0f, 0.0f, 0.0f) gives (1.0, 0.0, 0.0)", "[Tuple]") {
	GIVEN("v = vector(4.0, 0.0, 0.0") {
		auto v = vector(4.0, 0.0, 0.0);
		REQUIRE(v.normalize() == vector(1.0, 0.0, 0.0));
	}
}

SCENARIO("Normalizing vector(1.0, 2.0, 3.0)", "[Tuple]") {
	GIVEN("v = vector(1.0, 2.0, 3.0") {
		auto v = vector(1.0, 2.0, 3.0);
		REQUIRE(v.normalize() == vector(0.267261, 0.534522, 0.801784));
	}
}

SCENARIO("The magnitude of a normalized vector", "[Tuple]") {
	GIVEN("v = vector(1.0, 2.0, 3.0") {
		auto v = vector(4.0, 0.0, 0.0);
		WHEN("norm == v.normalize()") {
			auto norm = v.normalize();
			REQUIRE(norm.magnitude() == 1.0);
		}
	}
}

SCENARIO("The dot product of two tuples", "[Tuple]") {
	GIVEN("a = vector(1.0, 2.0, 3.0") {
		auto a = vector(1.0, 2.0, 3.0);
		AND_GIVEN("b = vector(2.0, 3.0, 4.0") {
			auto b = vector(2.0, 3.0, 4.0);
			REQUIRE(a.dot(b) == 20);
		}
	}
}

SCENARIO("The cross product of two vectors", "[Tuple]") {
	GIVEN("a = vector(1.0, 2.0, 3.0") {
		auto a = vector(1.0, 2.0, 3.0);
		AND_GIVEN("b = vector(2.0, 3.0, 4.0") {
			auto b = vector(2.0, 3.0, 4.0);
			REQUIRE(a.cross(b) == vector(-1.0, 2.0, -1.0));
			REQUIRE(b.cross(a) == vector(1.0, -2.0, 1.0));
		}
	}
}