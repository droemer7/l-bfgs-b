// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#include <iostream>
#include <iomanip>
#include "lbfgsb.h"

using namespace optimize;

// Rosenbrock objective function
// http://www.sfu.ca/~ssurjano/rosen.html
class Rosenbrock : public Function
{
private:
  static constexpr Scalar b = 5;

public:
  Scalar computeValue(const Vector& x) override
  {
    Scalar value = 0.0;
    Scalar t1 = 0.0;
    Scalar t2 = 0.0;
    for (Index i = 0; i < x.size() - 1; ++i) {
      t1 = x(i+1) - x(i)*x(i);
      t2 = x(i) - 1;
      value += b*t1*t1 + t2*t2;
    }
    return value;
  }

  Vector computeGradient(const Vector& x) override
  {
    Vector grad(x.size());

    for (Index i = 0; i < x.size() - 1; ++i) {
      if (i == 0) {
        grad(i) = 4*b*(x(i)*x(i)*x(i) - x(i)*x(i+1)) + 2*x(0) - 2;
      }
      if (i > 0 && i < x.size() - 1) {
        grad(i) = 4*b*(x(i)*x(i)*x(i) - x(i)*x(i+1)) + 2*b*(x(i) - x(i-1)*x(i-1)) + 2*x(i) - 2;
      }
      if (i+1 == x.size() - 1) {
        grad(i+1) = 2*b*(x(i+1) - x(i)*x(i));
      }
    }
    return grad;
  }
};

void printPoint(const State& state)
{
  for (Index i = 0; i < state.x().size(); ++i) {
    if (i == 0) {
      std::cout << "pts.append([";
    }
    std::cout << state.x()(i) << ", ";
    if (i+1 >= state.x().size()) {
      std::cout << state.f() << "])" << std::endl;
    }
  }
  if (state.x().size() == 0) {
    std::cout << std::endl;
  }
}

void displayState(Solver<Rosenbrock>* solver)
{
  printPoint(solver->state());
}

int main()
{
  Rosenbrock f;
  Vector x {{ -2, 0}};
  Vector l {{ -2, 0}};
  Vector u {{  0, 5}};

  std::cout << std::setprecision(15);
  printPoint(State(f.compute(x)));

  Lbfgsb<Rosenbrock> solver(displayState);
  State state = solver.minimize(f, x, l, u);

  std::cout << solver << std::endl;

  return 0;
}