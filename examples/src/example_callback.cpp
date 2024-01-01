// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#include <iostream>
#include <iomanip>
#include "lbfgsb.h"
#include "example.h"

using namespace optimize;

// Example callback function
void exampleCallback(Solver* solver)
{
  const State& state = solver->state();

  // Print the function value every 5 iterations
  if (state.iter() % 5 == 0) {
    std::cout << "Iteration " << state.iter() << ": f = " << state.f() << std::endl;
  }
}

int main()
{
  Rosenbrock f;                   // Objective function we wish to minimize
  Vector x {{   0,    5,    5}};  // Initial guess
  Vector l {{-0.5,  0.5, 0.35}};  // Lower bounds on x
  Vector u {{ 0.5,   10,   10}};  // Upper bounds on x

  // With C++17 or greater we can omit the "<>"
  Lbfgsb<> solver(exampleCallback);           // Solver with callback function, using default stopping conditions
  State state = solver.minimize(f, x, l, u);  // Solve with constraints (solution is f = 7.75 at x = [0.5, 0.5, 0.35])

  std::cout << "f = " << state.f() << std::endl;             // Minimum of f(x)
  std::cout << "x = " << state.x().transpose() << std::endl; // Argmin x of f(x)

  // std::cout << state << std::endl;   // Uncomment to print full state
  // std::cout << solver << std::endl;  // Uncomment to print full state + convergence info from solver

  return 0;
}