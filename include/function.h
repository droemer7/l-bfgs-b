// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#ifndef FUNCTION_H
#define FUNCTION_H

#include "util.h"

namespace optimize
{
  // Function base class
  class Function
  {
  public:
    // Constructors and destructors
    Function() = default;
    virtual ~Function() = default;

    // Evaluates and returns an iterate state
    Iterate compute(const Vector& x)
    { return Iterate(x, this->operator()(x), gradient(x), hessian(x)); }

    // Returns the objective function value
    virtual Scalar operator()(const Vector& x)
    {
      function.f_evals++;
      return computeValue(x);
    }

    // Implementation of the objective function value
    virtual Scalar computeValue(const Vector& x) = 0;

    // Returns the gradient or subgradient at the point x
    Vector gradient(const Vector& x)
    {
      function.g_evals++;
      return computeGradient(x);
    }

    // Returns the gradient or subgradient at a given state
    Vector gradient(const Iterate& state)
    {
      function.g_evals++;
      return computeGradient(state);
    }

    // Implementation of the gradient or subgradient at the point x
    virtual Vector computeGradient(const Vector& x) = 0;

    // Implementation of the gradient or subgradient at a given state
    // Override this if you want to compute the gradient using previously calculated state information.
    // This is useful when elements of the gradient contain this state, and the state is expensive to compute.
    virtual Vector computeGradient(const Iterate& state)
    { return gradient(state.x); }

    // Returns the hessian at the point x
    Matrix hessian(const Vector& x)
    {
      function.H_evals++;
      return computeHessian(x);
    }

    // Returns the hessian at a given state
    Matrix hessian(const Iterate& state)
    {
      function.H_evals++;
      return computeHessian(state);
    }

    // Implementation of the hessian at the point x
    virtual Matrix computeHessian(const Vector& /*x*/)
    { return Matrix(); }

    // Implementation of the hessian at a given state
    // Override this if you want to compute the hessian using previously calculated state information.
    // This is useful when elements of the hessian contain this state, and the state is expensive to compute.
    virtual Matrix computeHessian(const Iterate& state)
    { return hessian(state.x); }

    // Returns the function state containing the number of evaluations computed
    const FunctionState& state() const
    { return function; }

    // Returns the current number of function evaluations computed
    const Index& fEvals() const
    { return function.f_evals; }

    // Returns the current number of gradient evaluations computed
    const Index& gEvals() const
    { return function.g_evals; }

    // Returns the current number of hessian evaluations computed
    const Index& hEvals() const
    { return function.H_evals; }

    // Resets info, clearing all evaluation counts
    void reset()
    {
      function.f_evals = 0;
      function.g_evals = 0;
      function.H_evals = 0;
    }

  private:
    FunctionState function; // Number of evaluations to f(x), g(x), and H(x)
  };

} // namespace optimize

#endif // FUNCTION_H