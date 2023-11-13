// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#ifndef LINE_SEARCH_H
#define LINE_SEARCH_H

#include "function.h"
#include "util.h"

namespace optimize
{
  // Wolfe condition enumeration
  enum Wolfe
  {
    weak,
    strong
  };

  // Line Search base class
  class LineSearch
  {
  public:
    // Constructors and desctructors
    explicit LineSearch(Index iter_max) :
      iter_max(iter_max)
    {}

    virtual ~LineSearch() = default;

    // Search method
    virtual Scalar operator()(Function& f,
                              const Scalar fx,
                              const Vector& x,
                              const Vector& g,
                              const Vector& d,
                              const Scalar t_max = ScalarLimits::max()
                             ) = 0;

  protected:
    const Index iter_max;
  };

  // A modification of the Lewis-Overton line search algorithm.
  // This class can be configured to enforce the weak or strong Wolfe condition on the directional gradient, and takes
  // a parameters for the maximum step size and iterations allowed.
  //
  // Reference: A. S. Lewis and M. L. Overton. "Nonsmooth optimization via quasi-Newton methods", Mathematical
  //            Programming, Vol 141, No 1, pp. 135-163, 2013
  template <Wolfe Condition>
  class LewisOverton : public LineSearch
  {
  private:
    static constexpr Wolfe condition = Condition;

  public:
    // Constructors and desctructors
    explicit LewisOverton(Index iter_max = 25) :
      LineSearch(iter_max)
    {}

    ~LewisOverton() = default;

    Scalar operator()(Function& f,
                      const Scalar fx,
                      const Vector& x,
                      const Vector& g,
                      const Vector& d,
                      const Scalar t_max = ScalarLimits::max()
                     ) override
    {
      // Initialize
      constexpr Scalar A = 1e-4;  // Armijo constant
      constexpr Scalar W = 0.9;   // Wolfe constant
      Index iter = 0;             // Iteration counter
      bool b_set = false;         // Interval upper limit has been set at least once this search
      bool t_ok = false;          // Step length passed both Armijo and Wolfe checks
      Scalar a = 0;               // Interval lower limit
      Scalar b = t_max;           // Interval upper limit
      Scalar h = 0.0;             // h(t) = f(x + t*d) - f(x)
      Scalar hp = 0.0;            // h'(t) = ▽f(x + t*d)^T*d
      Scalar s = g.dot(d);        // s = ▽f(x)^T*d = sup|t->0 {h(t)/t, t > 0}

      // If s is not sufficiently negative, d is not a useful descent direction
      // In this case we set t = 0 to skip the search and return t = 0
      Scalar t = s < -ScalarLimits::epsilon() && t_max > 0.0 ? clip(1.0, 0.0, t_max) : 0.0;
      Scalar t_prev = t;

      // Perform search
      while (   !t_ok
             && t > 0.0
             && (t < t_max || t_prev < t_max || iter == 0)
             && iter < this->iter_max
            ) {
        iter++;

        // Compute h(t) and h'(t)
        h = f(x + t*d) - fx;
        hp = f.gradient(x + t*d).dot(d);

        // Check Armijo condition: h(t) < A*s*t  ==>  f(x + t*d) - f(x) < A*▽f(x)^T*d*t
        // This requires the function value decreases proportional to the directional gradient times the step length
        if (h >= A*s*t) {
          b = t;
          b_set = true;
        }
        // Check strong Wolfe condition: |h'(t)| < |W*s|  ==>  |▽f(x + t*d)^T*d| < |W*▽f(x)^T*d|
        // This requires a sufficient decrease in the magnitude of the directional gradient, h'(t)
        else if (   condition == Wolfe::strong
                 && std::abs(hp) >= std::abs(W*s)
                 && !(hp < 0.0 && t == t_max)
                ) {
          if (hp < 0.0) {
            a = t;
          }
          else {
            b = t;
            b_set = true;
          }
        }
        // Check weak Wolfe condition: h'(t) > W*s  ==>  ▽f(x + t*d)^T*d > W*▽f(x)^T*d
        // This requires a sufficient increase in the directional gradient, h'(t)
        else if (   condition == Wolfe::weak
                 && hp <= W*s
                 && t != t_max
                ) {
          a = t;
        }
        // Both Armijo and strong Wolfe conditions were satisified
        else {
          t_ok = true;
        }

        if (!t_ok) {
          // Save previous t
          t_prev = t;

          // Update t
          // Interval upper limit has been set: set t at the midpoint of the new interval
          //
          // In the paper, the authors check b < ∞ to determine if the Armijo check has failed at least once and the
          // interval upper limit b has been set. If t_max < ∞ though, it is possible that the Armijo check does not
          // fail until exactly t == t_max, which would then set b = t == t_max and 'incorrectly' yield
          // b < t_max == false.
          // For this reason we directly check if the interval upper limit has been set with b_set.
          if (b_set) {
            t = (a + b)/2;
          }
          // Interval upper limit has not been set: keep increasing t until max is reached
          else if (t < t_max) {
            t = 2*a;
          }

          // Clip t to be within [0, t_max]
          t = clip(t, 0.0, t_max);
        }
      }

      // Return 0 if a suitable step length could not be found and allow the solver to decide how to handle this
      return t_ok ? t : 0.0;
    }
  };

} // namespace optimize

#endif // LINE_SEARCH_H