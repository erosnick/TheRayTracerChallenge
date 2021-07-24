#include "Matrix.h"
#include "catch.hpp"

SCENARIO("Constructing and inspecting a 4 x 4 matrix", "[Matrix]") {
	GIVEN("the following 4x4 matrix M:"
		"| 1    | 2    | 3    | 4    |"
		"| 5.5  | 6.5  | 7.5  | 8.5  |"
		"| 9    | 10   | 11   | 12   |"
		"| 13.5 | 14.5 | 15.5 | 16.5 |") {
		auto M = Matrix4({
			{  1.0,  2.0,  3.0,  4.0 },
			{  5.5,  6.5,  7.5,  8.5 },
			{  9.0, 10.0, 11.0, 12.0 },
			{ 13.5, 14.5, 15.5, 16.5 }});
		THEN("M[0,0] == 1.0") {
			REQUIRE(M[0][0] == 1.0);
			AND_THEN("M[0][3] == 4.0") REQUIRE(M[0][3] == 4.0);
			AND_THEN("M[1][0] == 5.5") REQUIRE(M[1][0] == 5.5);
			AND_THEN("M[1][2] == 7.5") REQUIRE(M[1][2] == 7.5);
			AND_THEN("M[2][2] == 11.0") REQUIRE(M[2][2] == 11.0);
			AND_THEN("M[3][0] == 13.5") REQUIRE(M[3][0] == 13.5);
			AND_THEN("M[3][2] == 15.5") REQUIRE(M[3][2] == 15.5);
		}
	}
}

SCENARIO("A 2 x 2 matrix ought to be representable", "[Matrix]") {
	GIVEN("The following 2 x 2 matrix M:"
		  "| -3 |  5 |"
		  "|  1 | -2 |") {
		auto M = Matrix2({ -3.0,  5.0 },
					     {  1.0, -2.0 });
		THEN("matrix[0][0] == -3.0") {
			REQUIRE(M[0][0] == -3.0);
			AND_THEN("M[0][1] == 5.0")  REQUIRE(M[0][1] == 5.0);
			AND_THEN("M[1][0] == 1.0")  REQUIRE(M[1][0] == 1.0);
			AND_THEN("M[1][1] == -2.0") REQUIRE(M[1][1] == -2.0);
		}
	}
}

SCENARIO("A 3 x 3 matrix ought to be representable", "[Matrix]") {
	GIVEN("The following 3 x 3 matrix M:"
		"| -3 |  5 |  0 |"
		"|  1 | -2 | -7 |"
		"|  0 |  1 |  1 |") {
		auto M = Matrix3({ -3.0,  5.0,  0.0 },
					     {  1.0, -2.0, -7.0 },
					     {  0.0,  1.0,  1.0 });
		THEN("M[0][0] == -3.0") {
			REQUIRE(M[0][0] == -3.0);
			AND_THEN("M[1][1] == -2.0") REQUIRE(M[1][1] == -2.0);
			AND_THEN("M[2][2] == 1.0") REQUIRE(M[2][2] == 1.0);
		}
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
			THEN("A == B") {
				REQUIRE(A == B);
			}
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
			THEN("A != B") {
				REQUIRE(A != B);
			}
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
		AND_GIVEN("b = tuple(1.0, 2.0, 3.0, 1.0)") {
			auto b = Tuple(1.0, 2.0, 3.0, 1.0);
			THEN("A * b == tuple(18.0, 24.0, 33.0, 1.0)") {
				REQUIRE(A * b == Tuple(18.0, 24.0, 33.0, 1.0));
			}
		}
	}
}

SCENARIO("Multiplying a matrix by the identity matrix", "[Matrix]") {
	GIVEN("the following matrix A:") {
		auto A = Matrix4({
			{ 0.0, 1.0, 2.0,  3.0 },
			{ 1.0, 2.0, 4.0,  8.0 },
			{ 2.0, 4.0, 8.0, 16.0 },
			{ 4.0, 8.0, 16.0, 32.0 } });
		AND_GIVEN("I = Matrix4()") {
			auto I = Matrix4();
			THEN("A * I == A") {
				REQUIRE(A * I == A);
			}
		}
	}
}

SCENARIO("Multiplying the identity matrix by a tuple", "[Matrix]") {
	GIVEN("a = tuple(1.0, 2.0, 3.0, 4.0") {
		auto a = Tuple(1.0, 2.0, 3.0, 4.0);
		THEN("I * a = a") {
			auto I = Matrix4();
			REQUIRE(I * a == a);
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
		THEN("A.transpose() == A") {
			REQUIRE(A.transpose() == A);
		}
	}
}

SCENARIO("Calculating the determinant of a 2 x 2 matrix", "[Matrix]") {
	GIVEN("the following 2 x 2 matrix A:"
	"| 1 | 5 |" 
	"|-3 | 2 |") {
		auto matrix = Matrix2({ 1.0, 5.0 }, 
							 { -3.0, 2.0 });
		THEN("A.determinant() == 17.0") {
			REQUIRE(matrix.determinant() == 17.0);
		}
	}
}

SCENARIO("A submatrix of a 3 x 3 matrix is a 2 x 2 matrix", "[Matrix]") {
	GIVEN("the following 3x3 matrix A:"
		"|  1 | 5 |  0 |"
		"| -3 | 2 |  7 |"
		"|  0 | 6 | -3 |") {
		auto A = Matrix3({  1.0, 5.0,  0.0 }, 
						 { -3.0, 2.0,  7.0 }, 
					     {  0.0, 6.0, -3.0 });
		THEN("A.submatrix(0, 2) is the following 2 x 2 matrix:"
			"| -3 | 2 |"
			"|  0 | 6 | ") {
			auto result = A.submatrix(0, 2);
			auto submatrix = Matrix2( {-3.0, 2.0 }, 
			                          { 0.0, 6.0 } );
			REQUIRE(result == submatrix);
		}
	}
}

SCENARIO("A submatrix of a 4 x 4 matrix is a 3 x 3 matrix", "[Matrix]") {
	GIVEN("the following 3x3 matrix A:"
		"| -6 | 1 |  1 | 6 |"
		"| -8 | 5 |  8 | 6 |"
		"|  1 | 0 |  8 | 2 |"
		"| -7 | 1 |  1 | 1 |") {
		auto A = Matrix4({ -6.0, 1.0,  1.0, 6.0 },
						 { -8.0, 5.0,  8.0, 6.0 },
						 { -1.0, 0.0,  8.0, 2.0 },
						 { -7.0, 1.0, -1.0, 1.0 });
		THEN("submatrix(A, 2, 1) is the following 3 x 3 matrix:"
			"| -6 |  1 | 6 |"
			"| -8 |  8 | 6 |"
			"| -7 | -1 | 1 |") {
			auto result = A.submatrix(2, 1);
			auto submatrix = Matrix3({ -6.0,  1.0, 6.0 },
									 { -8.0,  8.0, 6.0 },
									 { -7.0, -1.0, 1.0 });
			REQUIRE(result == submatrix);
		}
	}
}

SCENARIO("Calculating a minor of a 3 x 3 matrix", "[Matrix]") {
	GIVEN("the following 3 x 3 matrix A:"
	"| 3 |  5 |  0 |"
	"| 2 | -1 | -7 |"
	"| 6 | -1 |  5 |") {
		auto A = Matrix3( { 3.0,  5.0,  0.0 }, 
						  { 2.0, -1.0, -7.0 },
						  { 6.0, -1.0,  5.0 } );
		AND_GIVEN("B = A.submatrix(1, 0)") {
			auto B = A.submatrix(1, 0);
			THEN("B.determinant == 25.0") {
				REQUIRE(B.determinant() == 25.0);
				AND_THEN("A.minor(1, 0) == 25.0") {
					REQUIRE(A.minor(1, 0) == 25.0);
				}
			}
		}
	}
}

SCENARIO("Calculating a cofactor of a 3 x 3 matrix", "[Matrix]") {
	GIVEN("the following 3 x 3 matrix A:"
	"| 3 |  5 |  0 |"
	"| 2 | -2 | -7 |"
	"| 6 | -1 |  5 |") {
		auto A = Matrix3({ 3.0,  5.0,  0.0 }, 
		                 { 2.0, -1.0, -7.0 }, 
						 { 6.0, -1.0,  5.0 });
		THEN("A.minor(0, 0) == -12.0") {
			REQUIRE(A.minor(0, 0) == -12.0);
			AND_THEN("A.cofactor(0, 0) == -12.0") REQUIRE(A.cofactor(0, 0) == -12.0);
			AND_THEN("A.minor(1, 0) == 25.0") REQUIRE(A.minor(1, 0) == 25.0);
			AND_THEN("A.cofactor(1, 0) === -25.0") REQUIRE(A.cofactor(1, 0) == -25.0);
		}
	}
}

SCENARIO("Calculating the determinant of a 3 x 3 matrix", "[Matrix]") {
	GIVEN("the following 3 x 3 matrix A:"
	"|  1 | 2 |  6 |"
	"| -5 | 8 | -4 |"
	"|  2 | 6 |  4 |") {
		auto A = Matrix3({  1.0, 2.0,  6.0 }, 
						 { -5.0, 8.0, -4.0 }, 
						 {  2.0, 6.0,  4.0});
		THEN("A.cofactor(0, 0) == 56.0") {
			REQUIRE(A.cofactor(0, 0) == 56);
			AND_THEN("A.cofactor(0, 1) == 12.0") REQUIRE(A.cofactor(0, 1) == 12.0);
			AND_THEN("A.cofactor(0, 2) == -46.0") REQUIRE(A.cofactor(0, 2) == -46.0);
			AND_THEN("A.determinant() == -196.0") REQUIRE(A.determinant() == -196.0);
		}
	}
}

SCENARIO("Calculating the determinant of a 4 x 4 matrix", "[Matrix]") {
	GIVEN("the following 4 x 4 matrix A:"
		"| -2 | -8 |  3 |  5 |"
		"| -3 |  1 |  7 |  3 |"
		"|  1 |  2 | -9 |  6 |"
		"| -6 |  7 |  7 | -9 |") {
		auto A = Matrix4({ -2.0, -8.0,  3.0,  5.0 },
						 { -3.0,  1.0,  7.0,  3.0 },
						 {  1.0,  2.0, -9.0,  6.0 },
						 { -6.0,  7.0,  7.0, -9.0 });
		THEN("A.cofactor(0, 0) == 690.0") {
			REQUIRE(A.cofactor(0, 0) == 690.0);
			AND_THEN("A.cofactor(0, 1) == 447.0")  REQUIRE(A.cofactor(0, 1) == 447.0);
			AND_THEN("A.cofactor(0, 2) == 210.0")  REQUIRE(A.cofactor(0, 2) == 210.0);
			AND_THEN("A.cofactor(0, 3) == 51.0")   REQUIRE(A.cofactor(0, 3) == 51.0);
			AND_THEN("A.determinant() == -4071.0") REQUIRE(A.determinant() == -4071.0);
		}
	}
}

SCENARIO("Testing an invertible matrix for invertibility", "[Matrix]") {
	GIVEN("the following 4 x 4 matrix A:"
	"| 6 |  4 | 4 |  4 |"
	"| 5 |  5 | 7 |  6 |"
	"| 4 | -9 | 3 | -7 |"
	"| 9 |  1 | 7 | -6 |") {
		auto A = Matrix4({ 6.0,  4.0, 4.0,  4.0 }, 
						 { 5.0,  5.0, 7.0,  6.0 }, 
						 { 4.0, -9.0, 3.0, -7.0 }, 
						 { 9.0,  1.0, 7.0, -6.0 });
		THEN("A.determinant() == -2120.0") {
			REQUIRE(A.determinant() == -2120.0);
			AND_THEN("A is invertible") REQUIRE(A.determinant() != 0.0);
		 }
	}
}

SCENARIO("Testing an noninvertible matrix for invertibility", "[Matrix]") {
	GIVEN("the following 4 x 4 matrix A:"
		"| -4 |  2 | -2 | -3 |"
		"|  9 |  6 |  2 |  6 |"
		"|  0 | -5 |  1 | -5 |"
		"|  0 |  0 |  0 | -0 |") {
		auto A = Matrix4({ -4.0,  2.0, -2.0, -3.0 },
					     {  9.0,  6.0,  2.0,  6.0 },
					     {  0.0, -5.0,  1.0, -5.0 },
						 {  0.0,  0.0,  0.0,  0.0 });
		THEN("A.determinant() == 0.0") {
			REQUIRE(A.determinant() == 0.0);
			AND_THEN("A is not invertible") REQUIRE(A.determinant() == 0.0);
		}
	}
}

SCENARIO("Calculating the inverse of a matrix", "[Matrix]") {
	GIVEN("the following 4 x 4 matrix A:"
		"| -5 |  2 |  6 | -8 |"
		"|  1 | -5 |  1 |  8 |"
		"|  7 |  7 | -6 | -7 |"
		"|  1 | -3 |  7 |  4 |") {
		auto A = Matrix4({ -5.0,  2.0,  6.0, -8.0 }, 
						  { 1.0, -5.0,  1.0,  8.0 }, 
						  { 7.0,  7.0, -6.0, -7.0 }, 
						  { 1.0, -3.0,  7.0,  4.0});
		AND_GIVEN("B = A.inverse()") {
			auto B = A.inverse();
			THEN("A.determinant() == 532.0") {
				AND_THEN("A.cofactor(2, 3) == -160.0") REQUIRE(A.cofactor(2, 3) == -160.0);
				AND_THEN("B[3][2] == -160.0 / 532.0")  REQUIRE(B[3][2] == -160.0 / 532.0);
				AND_THEN("A.cofactor(3, 2) == 105.0")  REQUIRE(A.cofactor(3, 2) == 105.0);
				AND_THEN("B[2][3] = 105.0 / 532.0")    REQUIRE(B[2][3] == 105.0 / 532.0);
				AND_THEN("B is the following 4 x 4 matrix:"
					"|  0.21805 |  0.45113 |  0.24060 | -0.04511 |"
					"| -0.80827 | -1.45677 | -0.44361 |  0.52068 |"
					"| -0.07895 | -0.22368 | -0.05263 |  0.19737 |"
					"| -0.52256 | -0.81391 | -0.30075 |  0.30639 |") {
					auto inverse = Matrix4({  0.21805,  0.45113,  0.24060, -0.04511 },
										   { -0.80827, -1.45677, -0.44361,  0.52068 }, 
										   { -0.07895, -0.22368, -0.05263,  0.19737 }, 
										   { -0.52256, -0.81391, -0.30075,  0.30639 } );
					REQUIRE(B == inverse);
				}
			}
		}
	}
}

SCENARIO("Calculating the inverse of another matrix", "[Matrix]") {
	GIVEN("the following 4 x 4 matrix A:"
		"|  8 | -5 |  9 |  2 |"
		"|  7 |  5 |  6 |  1 |"
		"| -6 |  0 |  9 |  6 |"
		"| -3 |  0 | -9 | -4 |") {
		auto A = Matrix4({  8.0, -5.0,  9.0,  2.0 }, 
						 {  7.0,  5.0,  6.0,  1.0 }, 
						 { -6.0,  0.0,  9.0,  6.0 }, 
						 { -3.0,  0.0, -9.0, -4.0 });
		THEN("B is the following 4 x 4 matrix:"
			"| -0.15385 | -0.15385 | -0.28205 | -0.53846 |"
			"| -0.07692 |  0.12308 |  0.02564 |  0.03077 |"
			"|  0.35897 | -0.35897 |  0.43590 |  0.92308 |"
			"| -0.69231 | -0.69231 | -0.76923 | -1.92308 |") {
			auto B = A.inverse();
			auto inverse = Matrix4({ -0.15385, -0.15385, -0.28205, -0.53846 },
								   { -0.07692,  0.12308,  0.02564,  0.03077 },
								   {  0.35897, -0.35897,  0.43590,  0.92308 },
								   { -0.69231, -0.69231, -0.76923, -1.92308 });
			REQUIRE(B == inverse);
		}
	}
}

SCENARIO("Calculating the inverse of a third matrix", "[Matrix]") {
	GIVEN("the following 4 x 4 matrix A:"
		"|  9 |  3 |  0 |  9 |"
		"| -5 | -2 | -6 | -3 |"
		"| -4 |  9 |  6 |  4 |"
		"| -7 |  6 |  6 |  2 |") {
		auto A = Matrix4({  9.0,  3.0,  0.0,  9.0 }, 
						 { -5.0, -2.0, -6.0, -3.0 }, 
						 { -4.0,  9.0,  6.0,  4.0 }, 
						 { -7.0,  6.0,  6.0,  2.0 });
		THEN("B is the following 4 x 4 matrix:"
			"| -0.04074 | -0.07778 |  0.14444 | -0.22222 |"
			"| -0.07778 |  0.03333 |  0.36667 | -0.33333 |"
			"| -0.02901 | -0.14630 | -0.10926 |  0.12963 |"
			"|  0.17778 |  0.06667 | -0.26667 |  0.33333 |") {
			auto B = A.inverse();
			auto inverse = Matrix4({ -0.04074, -0.07778,  0.14444, -0.22222 },
								   { -0.07778,  0.03333,  0.36667, -0.33333 },
								   { -0.02901, -0.14630, -0.10926,  0.12963 },
								   {  0.17778,  0.06667, -0.26667,  0.33333 });
			REQUIRE(B == inverse);
		}
	}
}

SCENARIO("Multiplying a product by its inverse", "[Matrix]") {
	GIVEN("the following 4 x 4 matrix A:"
	"|  3 | -9 |  7 |  3 |"
	"|  3 | -8 |  2 | -9 |"
	"| -4 |  4 |  4 |  1 |"
	"| -6 |  5 | -1 |  1 |") {
		auto A = Matrix4({  3.0, -9.0,  7.0,  3.0 }, 
						 {  3.0, -8.0,  2.0, -9.0 }, 
						 { -4.0,  4.0,  4.0,  1.0 }, 
						 { -6.0,  5.0, -1.0,  1.0 });
		AND_GIVEN("the following 4x4 matrix B:"
		"| 8 |  2 | 2 | 2 |"
		"| 3 | -1 | 7 | 0 |"
		"| 7 |  0 | 5 | 4 |"
		"| 6 | -2 | 0 | 5 |") {
			auto B = Matrix4({ 8.0,  2.0, 2.0, 2.0 }, 
							 { 3.0, -1.0, 7.0, 0.0 }, 
							 { 7.0,  0.0, 5.0, 4.0 }, 
							 { 6.0, -2.0, 0.0, 5.0 });
			AND_GIVEN("C = A * B") {
				auto C = A * B;
				THEN("C * B.inverse() == A") {
					REQUIRE(C * B.inverse() == A);
				}
			}
		}
	}
}

SCENARIO("Multiplying by a translation matrix", "[Matrix]") {
	GIVEN("transform = translation(5.0, -3.0, 2.0)") {
		auto transform = translation(5.0, -3.0, 2.0);
		AND_GIVEN("p = point(-3.0, 4.0, 5.0)") {
			auto p = point(-3.0, 4.0, 5.0);
			THEN("transform * p = point(2.0, 1.0, 7.0)") {
				REQUIRE(transform * p == point(2.0, 1.0, 7.0));
			}
		}
	}
}

SCENARIO("Multiplying by the inverse of a translation matrix", "[Matrix]") {
	GIVEN("transform = translation(5.0, -3.0, 2.0") {
		auto transform = translation(5.0, -3.0, 2.0);
		AND_GIVEN("inverse = transform.inverse()") {
			auto inverse = transform.inverse();
			AND_GIVEN("p = point(-3.0, 4.0, 5.0") {
				auto p = point(-3.0, 4.0, 5.0);
				THEN("inverse * p == point(-8.0, 7.0, 3.0") {
					REQUIRE(inverse * p == point(-8.0, 7.0, 3.0));
				}
			}
		}
	}
}

SCENARIO("Translation does not affect vectors", "[Matrix]") {
	GIVEN("transform = translation(5.0, -3.0, 2.0)") {
		auto transform = translation(5.0, -3.0, 2.0);
		AND_GIVEN("v = vector(-3.0, 4.0, 5.0)") {
			auto v = vector(-3.0, 4.0, 5.0);
			THEN("transform * v == v") {
				REQUIRE(transform * v == v);
			}
		}
	}
}

SCENARIO("A scaling matrix applied to a point", "[Matrix]") {
	GIVEN("transform = scaling(2.0, 3.0, 4.0)") {
		auto transform = scaling(2.0, 3.0, 4.0);
		AND_GIVEN("p = point(-4.0, 6.0, 8.0") {
			auto p = point(4.0, 6.0, 8.0);
			THEN("transform * p == point(-8.0, 18.0, 32.0") {
				REQUIRE(transform * p == point(-8.0, 18.0, 32.0));
			}
		}
	}
}

SCENARIO("A scaling matrix applied to a vector", "[Matrix]") {
	GIVEN("transform = scaling(2.0, 3.0, 4.0)") {
		auto transform = scaling(2.0, 3.0, 4.0);
		AND_GIVEN("v = vector(-4.0, 6.0, 8.0") {
			auto v = vector(-4.0, 6.0, 8.0);
			THEN("transform * v == vector(-8.0, 18.0, 32.0") {
				REQUIRE(transform * v == vector(-8.0, 18.0, 32.0));
			}
		}
	}
}

SCENARIO(" Multiplying by the inverse of a scaling matrix", "[Matrix]") {
	GIVEN("transform = scaling(2.0, 3.0, 4.0") {
		auto transform = scaling(2.0, 3.0, 4.0);
		AND_GIVEN("inverse = transform.inverse()") {
			auto inverse = transform.inverse();
			AND_GIVEN("v = vector(-4.0, 6.0, 8.0") {
				auto v = vector(-4.0, 6.0, 8.0);
				THEN("inverse * v == vector(-2.0, 2.0, 2.0") {
					REQUIRE(inverse * v == vector(-2.0, 2.0, 2.0));
				}
			}
		}
	}
}

SCENARIO("Rotating a point around the x axis", "[Matrix]") {
	GIVEN("p = point(0.0, 1.0, 0.0") {
		auto p = point(0.0, 1.0, 0.0);
		AND_GIVEN("halfQuarter = rotationX(¦Ð / 4)") {
			auto halfQuarter = rotationX(PI_4);
			AND_GIVEN("fullQuarter = roatationX(¦Ð / 2)") {
				auto fullQuarter = rotationX(PI_2);
				THEN("halfQuarter * p == point(0.0, ¡Ì2/2, ¡Ì2/2)") {
					REQUIRE(halfQuarter * p == point(0.0, std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0));
					AND_THEN("fullQuarter * p == point(0.0, 0.0, 1.0)") {
						REQUIRE(fullQuarter * p == point(0.0, 0.0, 1.0));
					}
				}
			}
		}
	}
}

SCENARIO("The inverse of an x-rotation rotates in the opposite direction", "[Matrix]") {
	GIVEN("p = point(0.0, 1.0, 0.0)") {
		auto p = point(0.0, 1.0, 0.0);
		AND_GIVEN("halfQuarter = rotationX(¦Ð / 4)") {
			auto halfQuarter = rotationX(PI_4);
			AND_GIVEN("inverse = halfQuarter.inverse()") {
				auto inverse = halfQuarter.inverse();
				THEN("inverse * p == point(0.0, ¡Ì2/2, -¡Ì2/2") {
					REQUIRE(inverse * p == point(0.0, std::sqrt(2.0) / 2.0, -std::sqrt(2.0) / 2.0));
				}
			}
		}
	}
}

SCENARIO("Rotating a point around the y axis", "[Matrix]") {
	GIVEN("p = point(0.0, 0.0, 1.0") {
		auto p = point(0.0, 0.0, 1.0);
		AND_GIVEN("halfQuarter = rotationY(¦Ð / 4)") {
			auto halfQuarter = rotationY(PI_4);
			AND_GIVEN("fullQuarter = rotationY(¦Ð / 2)") {
				auto fullQuarter = rotationY(PI_2);
				THEN("halfQuarter * p == point(¡Ì2/2, 0.0, ¡Ì2/2)") {
					REQUIRE(halfQuarter * p == point(std::sqrt(2.0) / 2.0, 0.0, std::sqrt(2.0) / 2.0));
					AND_THEN("fullQuarter * p == point(1.0, 0.0, 0.0)") {
						REQUIRE(fullQuarter * p == point(1.0, 0.0, 0.0));
					}
				}
			}
		}
	}
}

SCENARIO("Rotating a point around the z axis", "[Matrix]") {
	GIVEN("p = point(0.0, 1.0, 0.0") {
		auto p = point(0.0, 1.0, 0.0);
		AND_GIVEN("halfQuarter = rotationZ(¦Ð / 4)") {
			auto halfQuarter = rotationZ(PI_4);
			AND_GIVEN("fullQuarter = rotationZ(¦Ð / 2)") {
				auto fullQuarter = rotationZ(PI_2);
				THEN("halfQuarter * p == point(-¡Ì2/2, ¡Ì2/2, 0.0)") {
					REQUIRE(halfQuarter * p == point(-std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0, 0.0));
					AND_THEN("fullQuarter * p == point(1.0, 0.0, 0.0)") {
						REQUIRE(fullQuarter * p == point(-1.0, 0.0, 0.0));
					}
				}
			}
		}
	}
}