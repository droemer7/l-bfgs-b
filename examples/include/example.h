// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "function.h"

namespace optimize
{
  // Rosenbrock objective function
  // http://www.sfu.ca/~ssurjano/rosen.html
  //
  // Create your objective function deriving from optimize::Function and implement the computeValue() and
  // computeGradient() functions. The hessian is not used in L-BFGS-B so computeHessian() should be omitted.
  class Rosenbrock : public Function
  {
  private:
    static constexpr Scalar b = 100;

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
      Vector g(x.size());

      for (Index i = 0; i < x.size() - 1; ++i) {
        if (i == 0) {
          g(i) = 4*b*(x(i)*x(i)*x(i) - x(i)*x(i+1)) + 2*x(0) - 2;
        }
        if (i > 0 && i < x.size() - 1) {
          g(i) = 4*b*(x(i)*x(i)*x(i) - x(i)*x(i+1)) + 2*b*(x(i) - x(i-1)*x(i-1)) + 2*x(i) - 2;
        }
        if (i+1 == x.size() - 1) {
          g(i+1) = 2*b*(x(i+1) - x(i)*x(i));
        }
      }
      return g;
    }
  };
}
#endif // EXAMPLE_H
