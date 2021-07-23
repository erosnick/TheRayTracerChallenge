#include "Matrix.h"
#include "catch.hpp"

SCENARIO("Constructing and inspecting a 4 x 4 matrix", "[Matrix]") {
	GIVEN("the following 4x4 matrix M:"
		"| 1    | 2    | 3    | 4    |"
		"| 5.5  | 6.5  | 7.5  | 8.5  |"
		"| 9    | 10   | 11   | 12   |"
		"| 13.5 | 14.5 | 15.5 | 16.5 |") {
		auto matrix = Matrix4({
			{  1.0,  2.0,  3.0,  4.0 },
			{  5.5,  6.5,  7.5,  8.5 },
			{  9.0, 10.0, 11.0, 12.0 },
			{ 13.5, 14.5, 15.5, 16.5 }});
		REQUIRE(matrix[0][0] == 1.0);
		REQUIRE(matrix[0][3] == 4.0);
		REQUIRE(matrix[1][0] == 5.5);
		REQUIRE(matrix[1][2] == 7.5);
		REQUIRE(matrix[2][2] == 11.0);
		REQUIRE(matrix[3][0] == 13.5);
		REQUIRE(matrix[3][2] == 15.5);
	}
}

SCENARIO("A 2 x 2 matrix ought to be representable", "[Matrix]") {
	GIVEN("The following 2 x 2 matrix M:"
		  "| -3 |  5 |"
		  "|  1 | -2 |") {
		auto matrix = Matrix2({ -3.0,  5.0 }, 
							  {  1.0, -2.0 });
		REQUIRE(matrix[0][0] == -3.0);
		REQUIRE(matrix[0][1] == 5.0);
		REQUIRE(matrix[1][1] == -2.0);
	}
}

SCENARIO("A 3 x 3 matrix ought to be representable", "[Matrix]") {
	GIVEN("The following 3 x 3 matrix M:"
		"| -3 |  5 |  0 |"
		"|  1 | -2 | -7 |"
		"|  0 |  1 |  1 |") {
		auto matrix = Matrix3({ -3.0,  5.0,    0 },
							  {    1, -2.0, -7.0 },
							  {  0.0,  1.0,  1.0 });
		REQUIRE(matrix[0][0] == -3.0);
		REQUIRE(matrix[0][1] == 5.0);
		REQUIRE(matrix[1][1] == -2.0);
	}
}

SCENARIO("Matrix equality with identical matrices", "[Matrix]") {
	GIVEN("the following matrix A:") {
		auto A = Matrix4({
			{ 1.0, 2.0, 3.0, 4.0 },
			{ 5.0, 6.0, 7.0, 8.0 },
			{ 9.0, 8.0, 7.0, 6.0 },
			{ 5.0, 4.0, 3.0, 2.0 }});
		AND_GIVEN("the following matrix B:") {
			auto B = Matrix4({
			{ 1.0, 2.0, 3.0, 4.0 },
			{ 5.0, 6.0, 7.0, 8.0 },
			{ 9.0, 8.0, 7.0, 6.0 },
			{ 5.0, 4.0, 3.0, 2.0 }});
			REQUIRE(A == B);
		}
	}
}

SCENARIO("Matrix equality with different matrices", "[Matrix]") {
	GIVEN("the following matrix A:") {
		auto A = Matrix4({
			{ 1.0, 2.0, 3.0, 4.0 },
			{ 5.0, 6.0, 7.0, 8.0 },
			{ 9.0, 8.0, 7.0, 6.0 },
			{ 5.0, 4.0, 3.0, 2.0 }});
		AND_GIVEN("the following matrix B:") {
			auto B = Matrix4({
			{ 2.0, 3.0, 4.0, 5.0 },
			{ 6.0, 7.0, 8.0, 9.0 },
			{ 8.0, 7.0, 6.0, 5.0 },
			{ 4.0, 3.0, 2.0, 1.0 }});
			REQUIRE(A != B);
		}
	}
}

SCENARIO("Multiplying two matrices", "[Matrix]") {
	GIVEN("the following matrix A:") {
		auto A = Matrix4({
			{ 1.0, 2.0, 3.0, 4.0 },
			{ 5.0, 6.0, 7.0, 8.0 },
			{ 9.0, 8.0, 7.0, 6.0 },
			{ 5.0, 4.0, 3.0, 2.0 } });
		AND_GIVEN("the following matrix B:") {
			auto B = Matrix4({
			{-2.0, 1.0, 2.0,  3.0 },
			{ 3.0, 2.0, 1.0, -1.0 },
			{ 4.0, 3.0, 6.0,  5.0 },
			{ 1.0, 2.0, 7.0,  8.0 } });
			THEN("A * B is the following 4x4 matrix:" 
			"| 20 | 22 |  50 |  48 |"
			"| 44 | 54 | 114 | 108 |"
			"| 40 | 58 | 110 | 102 |"
			"| 16 | 26 |  46 |  42 |") {
				auto C = A * B;
				auto result = Matrix4({ 20.0, 22.0,  50.0,  48.0}, 
									  { 44.0, 54.0, 114.0, 108.0}, 
									  { 40.0, 58.0, 110.0, 102.0}, 
									  { 16.0, 26.0,  46.0,  42.0} );
				REQUIRE(C == result);
			}
		}
	}
}

SCENARIO("A matrix multiplied by a tuple", "[Matrix]") {
	GIVEN("the following matrix A:") {
		auto A = Matrix4({
			{ 1.0, 2.0, 3.0, 4.0 },
			{ 2.0, 4.0, 4.0, 2.0 },
			{ 8.0, 6.0, 4.0, 1.0 },
			{ 0.0, 0.0, 0.0, 1.0 } });
		AND_GIVEN("b = tuple(1, 2, 3, 1)") {
			auto b = Tuple(1.0, 2.0, 3.0, 1.0);
			THEN("A * b == tuple(18, 24, 33, 1)") {
				REQUIRE(A * b == Tuple(18.0, 24.0, 33.0, 1.0));
			}
		}
	}
}

SCENARIO("A matrix multiplied by a identity matrix", "[Matrix]") {
	GIVEN("the following matrix A:") {
		auto A = Matrix4({
			{ 0.0, 1.0, 2.0,  3.0 },
			{ 1.0, 2.0, 4.0,  8.0 },
			{ 2.0, 4.0, 8.0, 16.0 },
			{ 4.0, 8.0, 16.0, 32.0 } });
		AND_GIVEN("I = Matrix4()") {
			auto I = Matrix4();
			THEN("A * identity_matrix == A") {
				REQUIRE(A * I == A);
			}
		}
	}
}

SCENARIO("Transposing a matrix", "[Matrix]") {
	GIVEN("the following matrix A:") {
		auto A = Matrix4({
			{ 0.0, 9.0, 3.0, 0.0 },
			{ 9.0, 8.0, 0.0, 8.0 },
			{ 1.0, 8.0, 5.0, 3.0 },
			{ 0.0, 0.0, 5.0, 8.0 } });
		THEN("transpose(A) is the following matrix:") {
			auto B = Matrix4({
			{ 0.0, 9.0, 1.0,  0.0 },
			{ 9.0, 8.0, 8.0,  0.0 },
			{ 3.0, 0.0, 5.0,  5.0 },
			{ 0.0, 8.0, 3.0,  8.0 } });
			REQUIRE(A.transpose() == B);
		}
	}
}

SCENARIO("Transposing the identity matrix", "[Matrix]") {
	GIVEN("A = transpose(identity_matrix)") {
		auto A = Matrix4();
		THEN("A == identity_matrix") {
			REQUIRE(A.transpose() == A);
		}
	}
}