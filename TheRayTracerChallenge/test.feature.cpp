#include "catch.hpp"

int Factorial(int number) {
    return number <= 1 ? number : Factorial(number - 1) * number; // fail
    return number <= 1 ? 1 : Factorial(number - 1) * number; // pass
}

TEST_CASE("Factorial of 0 is 1 (fail)", "[single-file]") {
    REQUIRE(Factorial(0) == 1);
}

TEST_CASE("Factorials of 1 and higher are computed (pass)", "[single-file]") {
    REQUIRE(Factorial(1) == 1);
    REQUIRE(Factorial(2) == 2);
    REQUIRE(Factorial(3) == 6);
    REQUIRE(Factorial(10) == 3628800);
}


SCENARIO("vectors can be sized and resized", "[vector]") {
    GIVEN("A vector with some items") {
        std::vector<int> v(5);

        REQUIRE(v.size() == 5);
        REQUIRE(v.capacity() >= 5);

        WHEN("The size is increased") {
            v.resize(10);

            REQUIRE(v.size() == 10);
            REQUIRE(v.capacity() >= 10);
        }

        WHEN("The size is reduced") {
            v.resize(0);

            THEN("The size changes but not capacity") {
                REQUIRE(v.size() == 0);
                REQUIRE(v.capacity() >= 5);
            }
        }

        WHEN("More capacity is reserved") {
            v.reserve(10);

            THEN("The capacity changes but not the size") {
                REQUIRE(v.size() == 5);
                REQUIRE(v.capacity() >= 10);
            }
        }

        WHEN("Less capacity is reserved") {
            v.reserve(0);

            THEN("Neither size nor capacity are changed") {
                REQUIRE(v.size() == 5);
                REQUIRE(v.capacity() >= 5);
            }
        }
    }
}

//int Fibonacci(int number) {
//    std::vector<int> numbers(number + 1, 0);
//
//    numbers[1] = 1;
//    for (auto i = 2; i <= number; i++) {
//        numbers[i] = numbers[i - 1] + numbers[i - 2];
//    }
//
//    return numbers[number];
//}

int Fibonacci(int number) {
    if (number < 2) {
        return 1;
    }

    int a = 0;
    int b = 1;
    int c = a + b;

    for (auto i = 2; i <= number; i++) {
        c = a + b;
        a = b;
        b = c;
    }

    return c;
}