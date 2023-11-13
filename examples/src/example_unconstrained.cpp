// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#include <iostream>
#include "lbfgsb.h"
#include "example.h"

using namespace optimize;

int main()
{
  Rosenbrock f;           // Objective function we wish to minimize
  Vector x {{0, 5, 5}};   // Initial guess

  Lbfgsb<Rosenbrock> solver;            // Solver using default stopping conditions
  State state = solver.minimize(f, x);  // Solve without constraints

  std::cout << "f = " << state.f() << std::endl;             // Minimum of f(x)
  std::cout << "x = " << state.x().transpose() << std::endl; // Argmin x of f(x)

  // std::cout << state << std::endl;   // Uncomment to print full state
  // std::cout << solver << std::endl;  // Uncomment to print full state + convergence info from solver

  return 0;
}