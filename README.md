# L-BFGS-B

A lightweight, header-only C++ implementation of L-BFGS-B: the limited-memory BFGS algorithm for box-constrained problems. All code was implemented from the papers listed in the [References](README.md#references) section, namely [2](README.md#references) and [4](README.md#references).

<details><summary><b>Algorithm Overview</b></summary>
L-BFGS-B is a limited-memory, Quasi-Newton method which seeks to find a <i>local</i> solution of optimization problems of the form:
<br><br>

```
   min     f(x)
  x ∈ Rⁿ

   s.t.    l ≤ x ≤ u
```

where `f(x): Rⁿ -> R` is the objective function, and `l ∈ Rⁿ` and `u ∈ Rⁿ` are lower and upper bounds on the variables in `x`. The function `f(x)` can be nonlinear, but should be smooth and locally convex around the initial guess `x0`.

L-BFGS-B is well suited for problems with a large number of variables, and it performs similarly on unconstrained problems as L-BFGS (which can only solve unconstrained problems). It therefore can be considered to supersede L-BFGS. L-BFGS-B has a few drawbacks, however; most notably that it does not converge rapidly and does not take advantage of the structure of the problem.

The algorithm is comprised of the following steps:

1. **Convergence Test** - Evaluate the current state against all termination conditions and return this state if any condition is satisified.
2. **Generalized Cauchy Point Search** - Compute the first local minimizer of the quadratic model `m` of `f(x)` along the piecewise linear path obtained by projecting the current iterate in the steepest descent direction onto the feasible region defined by `l` and `u`.
3. **Subspace Minimization** - Compute a search direction `d` by minimizing the quadratic model of the objective function `f(x)` over the subspace of remaining free variables (after the Cauchy point search), subject to the constraints `l` and `u` on these variables.
4. **Line Search** - Perform a line search to find a suitable step length `t` from the current state along the search direction `d`, again subject to the constraints `l` and `u`on `x`.
5. **Update** - Using the step length `t` and search direction `d`, compute the new iterate `x = x + t*d`, update the limited-memory BFGS matrices, and return to step 1.

For more information on each step, refer to the papers listed in the [References](README.md#references) section.
</details>

## Requirements

The minimum requirements to use L-BFGS-B are a C++11 compiler and [Eigen >= 3.4](https://eigen.tuxfamily.org) (which is also a header-only library).

If you also want to build the [examples](https://github.com/droemer7/l-bfgs-b/tree/master/examples/src) or [unit tests](https://github.com/droemer7/l-bfgs-b/tree/master/test/src), you will need [CMake >= 3.16](https://cmake.org/). The unit tests also require a C++14 compiler and an internet connection so CMake can automatically download and link against GoogleTest to run the tests. See the section [Build Instructions](README.md#build-instructions) for more.

## Examples

Refer to the section [Running the Examples](README.md#running-the-examples) to compile and run the following example code.

### Objective Function Definition

First we need to define our objective function. This must be a class inheriting from `Function` with definitions implemented for the `computeValue()` and `computeGradient()` methods. Since the hessian is not used in L-BFGS-B, the `computeHessian()` method should be omitted.

```cpp
#include <iostream>
#include "lbfgsb.h"

using namespace optimize;

class Rosenbrock : public Function
{
public:
  Scalar computeValue(const Vector& x) override
  {
    Scalar value = 0.0;
    for (Index i = 0; i < x.size() - 1; ++i) {
      value += 100*(x(i+1) - x(i)*x(i))*(x(i+1) - x(i)*x(i)) + (x(i) - 1)*(x(i) - 1);
    }
    return value;
  }

  Vector computeGradient(const Vector& x) override
  {
    return Vector {{ 400*x(0)*x(0)*x(0) - 400*x(0)*x(1) + 2*x(0) - 2,
                     400*x(1)*x(1)*x(1) - 400*x(1)*x(2) - 200*x(0)*x(0) + 202*x(1) - 2,
                     200*(x(2) - x(1)*x(1))
                  }};
  }
};
```

### Constrained Minimization

With our objective function `Rosenbrock` defined above, we can now use this to setup our problem and perform the minimization. In the example below, given the constraints `l` and `u` the correct solution is `f(x) = 7.75` at `[0.5, 0.5, 0.35]`, not `[1, 1, 1]` as it is for the unconstrained Rosenbrock problem.

```cpp
int main()
{
  Rosenbrock f;                   // Objective function we wish to minimize
  Vector x {{   0,    5,    5}};  // Initial guess
  Vector l {{-0.5,  0.5, 0.35}};  // Lower bounds on x
  Vector u {{ 0.5,   10,   10}};  // Upper bounds on x

  Lbfgsb<Rosenbrock> solver;                  // Solver using default stopping conditions
  State state = solver.minimize(f, x, l, u);  // Solve with constraints (solution is f = 7.75 at x = [0.5, 0.5, 0.35])

  std::cout << "f = " << state.f() << std::endl;             // Minimum of f(x)
  std::cout << "x = " << state.x().transpose() << std::endl; // Argmin x of f(x)

  return 0;
}
```

Output:

```
f = 7.75
x =  0.5  0.5 0.35
```

In the above code we only displayed the minimum `f(x)` and `x`, but the `minimize()` function returns more information in a `State` object which contains:

* The solution state: `f(x)`, `x`, `g(x)`, and `H(x)` (if defined)
* The solver state: the number of iterations, total execution time, norm of `g(x)`, etc.
* The function state: number of evaluations of `f(x)`, `g(x)`, and `H(x)`

Additionally, you can print the solution `State` with `operator<<`:

```cpp
std::cout << state << std::endl;
```

Output:

```
Iterations = 16
Duration = 0.2501
Success = 1
f = 7.75
x =  0.5  0.5 0.35
g = -51  29  20
df_norm = 0.125675
dx_norm = 0.0318538
g_norm = 0
f_evals = 39
g_evals = 39
H_evals = 0
```

You can also call `operator<<` on the solver object itself, which prints the `State` plus solver status information (i.e., termination reasons, if any):

```cpp
std::cout << solver << std::endl;
```

Output:

```
Iterations = 16
Duration = 0.2501
Success = 1
f = 7.75
x =  0.5  0.5 0.35
g = -51  29  20
df_norm = 0.125675
dx_norm = 0.0318538
g_norm = 0
f_evals = 39
g_evals = 39
H_evals = 0

Gradient norm is below threshold
```

Note that all elements of the gradient at the solution are large but the gradient norm is reported to be below the convergence threshold. This is because all variables are bounded, so the gradient norm - which, for box-constrained minimization, is calculated as the infinity norm of the projected gradient - is actually zero (see `g_norm = 0` in the output).

### Unconstrained Minimization

L-BFGS-B is equally able to handle unconstrained minimization. We can modify the example above to solve the unconstrained problem by simply calling `solver.minimize(f, x)` instead. The constraint vectors `l` and `u` are ommitted from the call and need not be defined.

```cpp
int main()
{
  Rosenbrock f;           // Objective function we wish to minimize
  Vector x {{0, 5, 5}};   // Initial guess

  Lbfgsb<Rosenbrock> solver;            // Solver using default stopping conditions
  State state = solver.minimize(f, x);  // Solve without constraints

  std::cout << "f = " << state.f() << std::endl;             // Minimum of f(x)
  std::cout << "x = " << state.x().transpose() << std::endl; // Argmin x of f(x)

  return 0;
}
```

Output:

```
f = 2.51088e-18
x = 1 1 1
```

### Using a Callback Function

If you would like to run some code after each optimization step, such as to update a plot, or log the state, you can define a callback function and pass that to the solver as shown below. This function will be called after each optimization step.

```cpp
void exampleCallback(Solver<Rosenbrock>* solver)
{
  const State& state = solver->state();

  // Print the function value every 5 iterations
  if (state.iter() % 5 == 0) {
    std::cout << "Iteration " << state.iter() << ": ";
    std::cout << "f = " << state.f() << std::endl;
  }
}
```

Now create the solver by passing your callback function to the constructor. Alternatively you can call `solver.setCallback(exampleCallback)` to set/change the callback function.

```cpp
int main()
{
  Rosenbrock f;                   // Objective function we wish to minimize
  Vector x {{   0,    5,    5}};  // Initial guess
  Vector l {{-0.5,  0.5, 0.35}};  // Lower bounds on x
  Vector u {{ 0.5,   10,   10}};  // Upper bounds on x

  Lbfgsb<Rosenbrock> solver(exampleCallback); // Solver with callback function, using default stopping conditions
  State state = solver.minimize(f, x, l, u);  // Solve with constraints (solution is f = 7.75 at x = [0.5, 0.5, 0.35])

  std::cout << "f = " << state.f() << std::endl;             // Minimum of f(x)
  std::cout << "x = " << state.x().transpose() << std::endl; // Argmin x of f(x)

  return 0;
}
```

Output:

```
Iteration 5: f = 545.713
Iteration 10: f = 110.614
Iteration 15: f = 8.86398
f = 7.75
x =  0.5  0.5 0.35
```

### Changing the Line Search Method

`Lbfgsb` uses the `LewisOverton<Wolfe::strong>` class as the default line search method, which enforces the strong Wolfe condition for the curvature check. This can be changed by passing a new method to the second template parameter when creating the solver. For example, we can switch to using the `LewisOverton<Wolfe::weak>` version as shown below.

```cpp
Lbfgsb<Rosenbrock, LewisOverton<Wolfe::weak>> solver;
```

To use your own line search, define a new line search class inheriting from `LineSearch`, implement the search method as an override of `operator()`, then create an `Lbfgsb` solver with it as shown above.

## Build Instructions

This is a header-only library so just include it directly into your project. If you would like to run the examples and unit tests, continue reading below.

### Running the Unit Tests

The [unit tests](https://github.com/droemer7/l-bfgs-b/tree/master/test/src) verify the solver output on several problems with a range of initial conditions, constraints and line search methods. These can be built and run with the following commands.

```
cd test
mkdir build
cd build
cmake ..
make
./lbfgsb_example_constrained
./lbfgsb_example_unconstrained
./lbfgsb_example_callback
```

### Running the Examples

The [examples](https://github.com/droemer7/l-bfgs-b/tree/master/examples/src) described in the sections above can be built with the following commands.

```
cd examples
mkdir build
cd build
cmake ..
make
./lbfgsb_test
```

## Implementation Details

### Subspace Minimization

Following the recommendations by the authors in the latest revision of the L-BFGS-B paper, this implementation switches between using the active set or the free set - whichever is smaller - during the subspace minimization step computations. This results in fewer multiplications being performed and is helpful for large problems. Additionally, the solve operation is skipped if the active set or free set is empty.

### Line Search

The line search is the most critical component of L-BFGS-B.

The authors of L-BFGS-B revised their original paper to mention findings that a backtracking line search resulted in poor performance for some problems due to failing to satisfy the curvature condition for the BFGS update step. As a result, they currently recommend using a method that satisifies both Wolfe conditions (i.e., the Armijo sufficient decrease condition _and_ the curvature condition) while enforcing that the step size obeys the constraints `l ≤ x ≤ u`.

This code implements two versions of the Lewis-Overton line search, which both take a parameter specifying the upper bound on the step size so that the constraints on `x` are obeyed. These implementations also return a step length of 0 if they fail to satisfy the conditions, at which point `Lbfgsb` will reset all BFGS matrices and restart the search along the steepest descent direction (as recommended in [3](README.md#references)).

The two versions are:

* `LewisOverton<Wolfe::weak>`: This method follows the Lewis-Overton algorithm and checks for the 'weak' Wolfe condition; i.e., that the gradient increases sufficiently. This method has the benefit that it can, in theory, be used on non-smooth objective functions; however, using `Lbfgsb` on non-smooth functions has not been tested and likely needs modifications.
* `LewisOverton<Wolfe::strong>`: This method modifies the Lewis-Overton algorithm and checks for the 'strong' Wolfe condition; i.e., that the magnitude of the gradient decreases sufficiently. This method may perform slightly better on smooth functions than `LewisOverton<Wolfe::weak>`, but cannot be used on non-smooth objective functions.

The L-BFGS-B authors utilize the interpolation-based More-Thuente line search instead. The Lewis-Overton line search sometimes uses fewer function evaluations than the More-Thuente line search, deferring more computational work to the calculation of a new search direction. The More-Thuente line search on the other hand uses more function evaluations at times and takes bigger steps, and can generate faster convergence for some problems. Future releases of this library may implement the More-Thuente line search.

### Dynamic Eigen Matrices

This library does not use fixed-size Eigen matrices. Using fixed-size matrices allows Eigen to unroll loops and avoid dynamic memory allocation, which can be beneficial for performance if the size of the matrices are small.

I have tested a fixed-size matrix implementation of this library as a performance test, and there is a marginal speedup when the parameter vector `x` is small (this may vary depending on what SIMD instruction sets are supported on your machine). This should taper off as the size of `x` gets larger.

The main drawbacks to using fixed-size matrices are:

* Fixed-size matrices are allocated as plain arrays on the stack, so a large parameter vector (which L-BFGS-B is designed for) could cause stack overflows.
* Implementing fixed-size matrices requires additional template parameters everywhere which becomes very messy.
* Ideally, when using fixed-size matrices, sizes should be a multiple of 16 bytes for vectorization, which is an additional complexity.

## Maintenance & Contributing

If there is interest in helping make this project better, I will lay out a set of rules and specify or link to them here.

For issues and bugs, submit an [Issue](https://github.com/droemer7/l-bfgs-b/issues) or contact me at droemer7@gmail.com.

## References

1. J. Nocedal and S. Wright. _Numerical Optimization_, 2nd edition, Springer, 2006.
2. R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, _A Limited Memory Algorithm for Bound Constrained Optimization_, Tech. Report, NAM-08, EECS Department, Northwestern University, 1994.
3. C. Zhu, R.H. Byrd, P. Lu, and J. Nocedal, _L-BFGS-B: FORTRAN Subroutines for Large-Scale Bound Constrained Optimization_, EECS Department, Northwestern University, 1996.
4. A. S. Lewis and M. L. Overton. _Nonsmooth optimization via quasi-Newton methods_, Mathematical Programming, Vol 141, No 1, pp. 135-163, 2013.
