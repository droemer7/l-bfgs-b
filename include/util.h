// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#ifndef UTIL_H
#define UTIL_H

#include <iostream>    // cout, endl, ostream
#include <chrono>      // high_resolution_clock, system_clock, duration
#include <limits>      // numeric_limits
#include <vector>      // vector

#include <Eigen/Core>  // Eigen

namespace optimize
{
  using Scalar = double;
  using Eigen::all;
  using Eigen::last;
  using Index = Eigen::Index;
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using ScalarLimits = typename std::numeric_limits<Scalar>;
  using IndexLimits = typename std::numeric_limits<Index>;

  using Clock = std::chrono::high_resolution_clock;
  using Time = std::chrono::_V2::system_clock::time_point;
  using Duration = std::chrono::duration<Scalar>;

  class Function;
  struct State;

  // State of the solver
  struct SolverState
  {
    // Constructors and desctructors
    explicit SolverState(const Scalar df_norm = 0.0,  // Normalized delta in the function value, |fk+1 - fk|/max(|fk|, |fk+1|, 1)
                         const Scalar dx_norm = 0.0,  // Infinity norm of the delta in the state vector, L∞(xk+1 - xk)
                         const Scalar g_norm = 0.0,   // Infinity norm of the projected gradient, L∞(clip(xk+1 - ▽f(xk+1), l, u) - xk+1)
                         const Index iter = 0,        // Iteration count
                         const Scalar duration = 0.0, // Duration (milliseconds)
                         const bool success = false,  // The solver successfully met the convergence criteria
                         const bool stopped = true,   // The solver is stopped but may be resumed
                         const bool aborted = false,  // The solver aborted the optimization and cannot be resumed
                         const bool stalled = false   // The solver is stalled (current x == previous x)
                        ) :
      df_norm(df_norm),
      dx_norm(dx_norm),
      g_norm(g_norm),
      iter(iter),
      duration(duration),
      success(success),
      stopped(stopped),
      aborted(aborted),
      stalled(stalled)
    {}

    Scalar df_norm;   // Normalized delta in the function value, |fk+1 - fk|/max(|fk|, |fk+1|, 1)
    Scalar dx_norm;   // Infinity norm of the delta in the state vector, L∞(xk+1 - xk)
    Scalar g_norm;    // Infinity norm of the projected gradient, L∞(clip(xk+1 - ▽f(xk+1), l, u) - xk+1)
    Index iter;       // Iteration count
    Scalar duration;  // Duration (milliseconds)
    bool success;     // The solver successfully met the convergence criteria
    bool stopped;     // The solver is stopped but may be resumed
    bool aborted;     // The solver aborted the optimization and cannot be resumed
    bool stalled;     // The solver is stalled (current x == previous x)
  };

  // Iterate state
  struct Iterate
  {
    // Constructors and desctructors
    explicit Iterate(const Vector& x = Vector(), // Parameter vector
                     const Scalar f = 0.0,       // Function value f(x)
                     const Vector& g = Vector(), // Gradient ▽f(x)
                     const Matrix& H = Matrix()  // Hessian ▽^2[f(x)]
                    ) :
      x(x),
      f(f),
      g(g),
      H(H)
    {}

    Vector x;   // Parameter vector
    Scalar f;   // Function value f(x)
    Vector g;   // Gradient ▽f(x)
    Matrix H;   // Hessian ▽^2[f(x)]
  };

  // Function data
  struct FunctionState
  {
    // Constructors and desctructors
    explicit FunctionState(const Index f_evals = 0,  // Number of function evaluations
                           const Index g_evals = 0,  // Number of gradient evaluations
                           const Index H_evals = 0   // Number of hessian evaluations
                          ) :
      f_evals(f_evals),
      g_evals(g_evals),
      H_evals(H_evals)
    {}

    Index f_evals;  // Number of function evaluations
    Index g_evals;  // Number of gradient evaluations
    Index H_evals;  // Number of hessian evaluations
  };

  // Stopping state for an optimization problem
  struct StopState
  {
    // Constructors and destructors
    explicit StopState(const SolverState& solver,   // Solver state
                       const Index f_evals = 0      // Number of function evaluations (0 = unlimited)
                      ) :
      solver(solver),
      f_evals(f_evals)
    {}

    explicit StopState(const Scalar& df_norm = 0.0, // Normalized delta in the function value, |fk+1 - fk|/max(|fk|, |fk+1|, 1)
                       const Scalar& dx_norm = 0.0, // Infinity norm of the delta in the state vector, L∞(xk+1 - xk)
                       const Scalar& g_norm = 0.0,  // Infinity norm of the projected gradient, L∞(clip(xk+1 - ▽f(xk+1), l, u) - xk+1)
                       const Scalar duration = 0.0, // Duration (milliseconds) (0 = unlimited)
                       const Index f_evals = 0      // Number of function evaluations (0 = unlimited)
                      ) :
      solver(df_norm,  // df_norm
             dx_norm,  // dx_norm
             g_norm,   // g_norm
             0,        // iter
             duration  // duration
            ),
      f_evals(f_evals)
    {}

    // Solver state accessor methods
    const Scalar& dfNorm()   const { return solver.df_norm; }
    const Scalar& dxNorm()   const { return solver.dx_norm; }
    const Scalar& gNorm()    const { return solver.g_norm; }
    const Scalar& duration() const { return solver.duration; }

    Scalar& dfNorm()   { return solver.df_norm; }
    Scalar& dxNorm()   { return solver.dx_norm; }
    Scalar& gNorm()    { return solver.g_norm; }
    Scalar& duration() { return solver.duration; }

    // Function state accesor methods
    const Index& fEvals() const { return f_evals; }
    Index& fEvals() { return f_evals; }

    SolverState solver; // Solver state
    Index f_evals;      // Number of function evaluations (0 = unlimited)
  };

  // Full state of an optimization problem
  struct State
  {
    // Constructors and destructors
    explicit State(const Iterate& iterate,
                   const SolverState& solver = SolverState(),
                   const FunctionState& function = FunctionState()
                  ) :
      iterate(iterate),
      solver(solver),
      function(function)
    {}

    explicit State(const Vector& x = Vector(),  // Parameter vector
                   const Scalar f = 0.0,        // Function value f(x)
                   const Vector& g = Vector(),  // Gradient ▽f(x)
                   const Matrix& H = Matrix(),  // Hessian ▽^2[f(x)]
                   const Scalar df_norm = 0.0,  // Normalized delta in the function value, |fk+1 - fk|/max(|fk|, |fk+1|, 1)
                   const Scalar dx_norm = 0.0,  // Infinity norm of the delta in the state vector, L∞(xk+1 - xk)
                   const Scalar g_norm = 0.0,   // Infinity norm of the projected gradient, L∞(clip(xk+1 - ▽f(xk+1), l, u) - xk+1)
                   const Index iter = 0,        // Iteration count
                   const Scalar duration = 0.0, // Duration (milliseconds)
                   const bool success = false,  // The solver successfully met the convergence criteria
                   const bool stopped = true,   // The solver is stopped but may be resumed
                   const bool aborted = false,  // The solver aborted the optimization and cannot be resumed
                   const bool stalled = false,  // The solver is stalled (current x == previous x)
                   const Index f_evals = 0,     // Number of function evaluations
                   const Index g_evals = 0,     // Number of gradient evaluations
                   const Index H_evals = 0      // Number of hessian evaluations
                  ) :
      State(Iterate(x,  // x
                    f,  // f
                    g,  // g
                    H   // H
                   ),
            SolverState(df_norm,  // df_norm
                        dx_norm,  // dx_norm
                        g_norm,   // g_norm
                        duration, // duration
                        iter,     // iter
                        success,  // success
                        stopped,  // stopped
                        aborted,  // aborted
                        stalled   // stalled
                       ),
            FunctionState(f_evals,   // f_evals
                          g_evals,   // g_evals
                          H_evals    // H_evals
                         )
           )
    {}

    // Iterate state accessor functions
    const Vector& x() const { return iterate.x; }
    const Scalar& f() const { return iterate.f; }
    const Vector& g() const { return iterate.g; }
    const Matrix& H() const { return iterate.H; }

    Vector& x() { return iterate.x; }
    Scalar& f() { return iterate.f; }
    Vector& g() { return iterate.g; }
    Matrix& H() { return iterate.H; }

    // Solver state accessor methods
    const Scalar& dfNorm()   const { return solver.df_norm; }
    const Scalar& dxNorm()   const { return solver.dx_norm; }
    const Scalar& gNorm()    const { return solver.g_norm; }
    const Index& iter()      const { return solver.iter; }
    const Scalar& duration() const { return solver.duration; }
    const bool& success()    const { return solver.success; }
    const bool& stopped()    const { return solver.stopped; }
    const bool& aborted()    const { return solver.aborted; }
    const bool& stalled()    const { return solver.stalled; }

    Scalar& dfNorm()   { return solver.df_norm; }
    Scalar& dxNorm()   { return solver.dx_norm; }
    Scalar& gNorm()    { return solver.g_norm; }
    Index& iter()      { return solver.iter; }
    Scalar& duration() { return solver.duration; }
    bool& success()    { return solver.success; }
    bool& stopped()    { return solver.stopped; }
    bool& aborted()    { return solver.aborted; }
    bool& stalled()    { return solver.stalled; }

    // Function state accesor methods
    const Index& fEvals() const { return function.f_evals; }
    const Index& gEvals() const { return function.g_evals; }
    const Index& hEvals() const { return function.H_evals; }

    Index& fEvals() { return function.f_evals; }
    Index& gEvals() { return function.g_evals; }
    Index& hEvals() { return function.H_evals; }

    Iterate iterate;
    SolverState solver;
    FunctionState function;
  };

  // Prints an STL vector
  template <class T>
  inline std::ostream& operator<<(std::ostream& os, const std::vector<T> x)
  {
    for (size_t i = 0; i < x.size(); ++i) {
      if (i+1 == x.size()) {
        os << x[i];
      }
      else {
        os << x[i] << ", ";
      }
    }
    return os;
  }

  // Prints a solver state
  inline std::ostream& operator<<(std::ostream& os, const SolverState& state)
  {
    os << "Iterations = " << state.iter << std::endl;
    os << "Duration = " << state.duration << std::endl;
    os << "Success = " << (state.success ? "true" : "false") << std::endl;
    os << "df_norm = " << state.df_norm << std::endl;
    os << "dx_norm = " << state.dx_norm << std::endl;
    os << "g_norm = " << state.g_norm;
    return os;
  }

  // Prints an interate state
  inline std::ostream& operator<<(std::ostream& os, const Iterate& state)
  {
    os << "f = " << state.f << std::endl;
    os << "x = " << state.x.transpose() << std::endl;
    os << "g = " << state.g.transpose();
    for (Index i = 0; i < state.H.rows(); ++i) {
      if (i == 0) {
        os << std::endl << "H = " << state.H.row(i);
      }
      else {
        os << std::endl << "    " << state.H.row(i);
      }
    }
    return os;
  }

  // Prints a function state
  inline std::ostream& operator<<(std::ostream& os, const FunctionState& state)
  {
    os << "f_evals = " << state.f_evals << std::endl;
    os << "g_evals = " << state.g_evals << std::endl;
    os << "H_evals = " << state.H_evals;
    return os;
  }

  // Prints the full state of an optimization process
  inline std::ostream& operator<<(std::ostream& os, const State& state)
  {
    os << "Iterations = " << state.solver.iter << std::endl;
    os << "Duration = " << state.solver.duration << std::endl;
    os << "Success = " << (state.solver.success ? "true" : "false") << std::endl;
    os << state.iterate << std::endl;
    os << "df_norm = " << state.solver.df_norm << std::endl;
    os << "dx_norm = " << state.solver.dx_norm << std::endl;
    os << "g_norm = " << state.solver.g_norm << std::endl;
    os << state.function;
    return os;
  }

  // Clips a vector x to be within bounds [l, u]
  inline Vector clip(const Vector& x,
                     const Vector& l,
                     const Vector& u
                    )
  { return x.cwiseMin(u).cwiseMax(l); }

  // Clips a scalar value to be within bounds [l, u]
  inline Scalar clip(const Scalar x,
                     const Scalar l,
                     const Scalar u
                    )
  { return std::min(std::max(x, l), u); }

  // Shifts a matrix by the number of rows and columns specified, from a given starting row and column
  // Data past the end of the matrix which is shifted into the matrix is left as the old values (not reinitialized)
  template <class Derived>
  inline void shift(Eigen::MatrixBase<Derived>& x,
                    Index rows,
                    Index cols = 0,
                    Index row_start = 0,
                    Index col_start = 0
                   )
  {
    row_start = clip(row_start, 0, x.rows() - 1);
    col_start = clip(col_start, 0, x.cols() - 1);

    Index r_dir = rows > 0 ? -1 : 1;
    Index c_dir = cols > 0 ? -1 : 1;

    Index r_start = rows > 0 ? x.rows() - 1 : std::max(row_start + rows, static_cast<Index>(0));
    Index c_start = cols > 0 ? x.cols() - 1 : std::max(col_start + cols, static_cast<Index>(0));

    Index r = r_start;
    Index c = c_start;

    Index r_copy = r - rows;
    Index c_copy = c - cols;

    while (r_copy >= row_start && r_copy < x.rows()) {
      while (c_copy >= col_start && c_copy < x.cols()) {
        // Copy data to the shifted location
        x(r, c) = x(r_copy, c_copy);

        // Increment primary and runner column indexes
        c += c_dir;
        c_copy += c_dir;
      }
      // Increment primary and runner row indexes
      r += r_dir;
      r_copy += r_dir;

      // Reset column indexes for next row
      c = c_start;
      c_copy = c - cols;
    }
  }

  // Calculates the gradient projected onto the feasible region given by the bounds [l, u]
  // This represents how much x can change along the steepest descent direction when subject to the bounds.
  inline Vector projectedGradient(const Vector& x,
                                  const Vector& g,
                                  const Vector& l,
                                  const Vector& u
                                 )
  { return clip(x - g, l, u) - x; }

  // Returns the infinity norm of x
  template <class Derived>
  inline Scalar infinityNorm(const Eigen::MatrixBase<Derived>& x)
  { return x.template lpNorm<Eigen::Infinity>(); }

  inline Scalar durationMsec(const Time& start, const Time& end)
  { return std::chrono::duration_cast<Duration>(end - start).count() * 1000.0; }

  inline Scalar timeToMsec(const Time& time)
  { return std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count(); }

} // namespace optimize

#endif // UTIL_H