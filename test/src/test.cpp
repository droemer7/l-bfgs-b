// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#include "gtest/gtest.h"

#include "test.h"
#include "lbfgsb.h"

using namespace optimize;
using LewisOvertonWeak = LewisOverton<Wolfe::weak>;
using LewisOvertonStrong = LewisOverton<Wolfe::strong>;

constexpr Scalar MAX_ERROR = 1e-5;
constexpr bool SHOW_RESULTS = false;

#define LBFGSB_TEST_CASE(line_search, function, description, x, l, u, true_min)         \
  TEST(Lbfgsb##_##line_search, function##_##description) {                              \
    function f;                                                                         \
    Lbfgsb<line_search> solver;                                                         \
    Scalar solver_min = solver.minimize(f, Vector x, Vector l, Vector u).f();           \
    EXPECT_NEAR(solver_min, true_min, MAX_ERROR);                                       \
    EXPECT_TRUE(solver.state().success());                                              \
    if (SHOW_RESULTS) { std::cout << std::endl << solver << std::endl; }                \
  }

#define LBFGSB_TEST(function, description, x, l, u, true_min)                           \
  LBFGSB_TEST_CASE(LewisOvertonWeak, function, description, x, l, u, true_min)          \
  LBFGSB_TEST_CASE(LewisOvertonStrong, function, description, x, l, u, true_min)

// ============================================================================
// L-BFGS-B: Forrester tests
// ============================================================================
// Forrester: 0 variables bounded at minimum
LBFGSB_TEST(Forrester,      // Objective function
            0Active,        // Test Description
            ({{0.5241}}),   // Initial point
            ({{0}}),        // Lower bound
            ({{1}}),        // Upper bound
            -6.020740       // Expected minimum
           )

// Forrester: 1 variable bounded at minimum
LBFGSB_TEST(Forrester,      // Objective function
            1Active,        // Test Description
            ({{0.5241}}),   // Initial point
            ({{0}}),        // Lower bound
            ({{0.7}}),      // Upper bound
            -4.605754       // Expected minimum
           )

// ============================================================================
// L-BFGS-B: Simple tests
// ============================================================================
// Simple: 0 variables bounded at minimum
LBFGSB_TEST(Simple,             // Objective function
            0Active,            // Test Description
            ({{   9,   -8}}),   // Initial point
            ({{ -10,  -10}}),   // Lower bound
            ({{  10,   10}}),   // Upper bound
            -1.250000           // Expected minimum
           )

// Simple: 1 variable bounded at minimum: x0 = -0.5
LBFGSB_TEST(Simple,             // Objective function
            1Active,            // Test Description
            ({{   9,   -8}}),   // Initial point
            ({{-0.5,  -10}}),   // Lower bound
            ({{  10,   10}}),   // Upper bound
            -1.000000           // Expected minimum
           )

// Simple: 2 variables bounded at minimum: x0 = -0.5, x1 = 2
LBFGSB_TEST(Simple,             // Objective function
            2Active,            // Test Description
            ({{   9,    6}}),   // Initial point
            ({{-0.5,    2}}),   // Lower bound
            ({{  10,   10}}),   // Upper bound
            0.000000            // Expected minimum
           )

// ============================================================================
// L-BFGS-B: Non-Smooth Tests
// ============================================================================
// Non-Smooth 2D: 0 variables bounded at minimum
LBFGSB_TEST_CASE(LewisOvertonWeak,  // Line search (only Weak for non-smooth functions)
                 NonSmooth2D,       // Objective function
                 0Active,           // Test Description
                 ({{  -9,    8}}),  // Initial point
                 ({{ -10,  -10}}),  // Lower bound
                 ({{  10,   10}}),  // Upper bound
                 0.000000           // Expected minimum
                )

// Non-Smooth 2D: 1 variable bounded at minimum: x0 = -5
LBFGSB_TEST_CASE(LewisOvertonWeak,  // Line search (only Weak for non-smooth functions)
                 NonSmooth2D,       // Objective function
                 1Active,           // Test Description
                 ({{  -9,    8}}),  // Initial point
                 ({{ -10,  -10}}),  // Lower bound
                 ({{  -5,   10}}),  // Upper bound
                 2.924018           // Expected minimum
                )

// Non-Smooth 2D: 2 variables bounded at minimum: x0 = -5, x1 = 5
LBFGSB_TEST_CASE(LewisOvertonWeak,  // Line search (only Weak for non-smooth functions)
                 NonSmooth2D,       // Objective function
                 2Active,           // Test Description
                 ({{  -9,    8}}),  // Initial point
                 ({{ -10,    5}}),  // Lower bound
                 ({{  -5,   10}}),  // Upper bound
                 14.397915          // Expected minimum
                )

// ============================================================================
// L-BFGS-B: Rosenbrock tests
// ============================================================================
// Rosenbrock: 0 variables bounded at minimum
LBFGSB_TEST(Rosenbrock,           // Objective function
            0Active,              // Test Description
            ({{  8,  -5,   3}}),  // Initial point
            ({{-10, -10, -10}}),  // Lower bound
            ({{ 10,  10,  10}}),  // Upper bound
            0.000000              // Expected minimum
           )

// Rosenbrock: 1 variable bounded at minimum: x0 = 0.5
LBFGSB_TEST(Rosenbrock,               // Objective function
            1Active,                  // Test Description
            ({{   8,   -5,    3}}),   // Initial point
            ({{ -10,  -10,  -10}}),   // Lower bound
            ({{ 0.5,   10,   10}}),   // Upper bound
            0.806931                  // Expected minimum
           )

// Rosenbrock: 2 variables bounded at minimum: x0 = 0.5, x1 = 0.5
LBFGSB_TEST(Rosenbrock,               // Objective function
            2Active,                  // Test Description
            ({{   0,    5,    5}}),   // Initial point
            ({{-0.5,  0.5,  -10}}),   // Lower bound
            ({{ 0.5,   10,   10}}),   // Upper bound
            6.750000                  // Expected minimum
           )

// Rosenbrock: 3 variables bounded at minimum: x0 = 0.5, x1 = 0.5, x2 = 0.35
LBFGSB_TEST(Rosenbrock,               // Objective function
            3Active,                  // Test Description
            ({{   0,    5,    5}}),   // Initial point
            ({{-0.5,  0.5, 0.35}}),   // Lower bound
            ({{ 0.5,   10,   10}}),   // Upper bound
            7.750000                  // Expected minimum
           )

// Rosenbrock: out of bounds initial point
LBFGSB_TEST(Rosenbrock,               // Objective function
            OutOfBoundsInitialPoint,  // Test Description
            ({{  11,  -20,   30}}),   // Initial point
            ({{ -10,  -10,  -10}}),   // Lower bound
            ({{  10,   10,   10}}),   // Upper bound
            0.000000                  // Expected minimum
           )

// ============================================================================
// L-BFGS-B: Six Hump Camel tests
// ============================================================================
// Six Hump Camel: 0 variables bounded at minimum
LBFGSB_TEST(SixHumpCamel,     // Objective function
            0Active,          // Test Description
            ({{  1,   0}}),   // Initial point
            ({{ -2,  -2}}),   // Lower bound
            ({{  2,   2}}),   // Upper bound
            -1.031628         // Expected minimum
           )

// Six Hump Camel: 1 variable bounded at minimum: x0 = 0.1
LBFGSB_TEST(SixHumpCamel,     // Objective function
            1Active,          // Test Description
            ({{  1,   0}}),   // Initial point
            ({{0.1,  -2}}),   // Lower bound
            ({{  2,   2}}),   // Upper bound
            -1.031230         // Expected minimum
           )

// Six Hump Camel: 2 variables bounded at minimum: x0 = 0.1, x1 = -0.6
LBFGSB_TEST(SixHumpCamel,     // Objective function
            2Active,          // Test Description
            ({{  1,    0}}),  // Initial point
            ({{0.1, -0.6}}),  // Lower bound
            ({{  2,    2}}),  // Upper bound
            -0.941810         // Expected minimum
           )

// ============================================================================
// L-BFGS-B: Spiral tests
// ============================================================================
// Spiral: 0 variables bounded at minimum
LBFGSB_TEST(Spiral,                   // Objective function
            0Active,                  // Test Description
            ({{   0,    0, 7.07}}),   // Initial point
            ({{-0.5, -0.5, 7.07}}),   // Lower bound
            ({{ 0.5,  0.5,  10}}),    // Upper bound
            0.313982                  // Expected minimum
           )

// Spiral: 1 variable bounded at minimum: x0 = 0.25
LBFGSB_TEST(Spiral,                     // Objective function
            1Active,                    // Test Description
            ({{   0,    0,   7.07}}),   // Initial point
            ({{-0.25, -0.25, 7.07}}),   // Lower bound
            ({{ 0.25,  0.25, 1e10}}),   // Upper bound
            6.832052                    // Expected minimum
           )

// Spiral: 2 variables bounded at minimum: x0 = 0.25, x1 = 0.2
LBFGSB_TEST(Spiral,                     // Objective function
            2Active,                    // Test Description
            ({{   0,    0,   7.07}}),   // Initial point
            ({{-0.25, -0.20, 7.07}}),   // Lower bound
            ({{ 0.25,  0.20, 1e10}}),   // Upper bound
            7.245711                    // Expected minimum
           )

// Spiral: 3 variables bounded at minimum: x0 = 0.25, x1 = 0.2, x2 = 8.0
LBFGSB_TEST(Spiral,                     // Objective function
            3Active,                    // Test Description
            ({{   0,    0,   7.07}}),   // Initial point
            ({{-0.25, -0.20, 7.07}}),   // Lower bound
            ({{ 0.25,  0.20, 8.00}}),   // Upper bound
            8.112968                    // Expected minimum
           )