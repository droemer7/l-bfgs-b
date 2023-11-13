// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#ifndef SOLVER_H
#define SOLVER_H

#include <cmath>      // pow
#include <iostream>   // cout, endl
#include <functional> // function

#include "function.h" // Function
#include "util.h"     // State, StopState, Iterate, infinityNorm(), projectedGradient()

namespace optimize
{
  // Solver base class
  // Executes the high level optimization procedure, calling on the derived class to perform the optimization step
  // which computes a new state.
  template <class Function>
  class Solver
  {
  public:
    using Callback = std::function<void (Solver<Function>*)>;

  public:
    // Constructors and destructors
    explicit Solver(const Scalar accuracy = 0.7,                        // Accuracy level 0 to 1, where 1 is maximum accuracy
                    const Scalar duration_max = 0,                      // Maxmimum duration (milliseconds) (0 = unlimited)
                    const Index f_evals_max = 0,                        // Maxmimum number of function evaluations (0 = unlimited)
                    const Callback& callback = [](Solver<Function>*) {} // Callback function which is executed after each optimization step
                   ) :
      n(0),
      f(),
      l(Vector::Zero(n)),
      u(Vector::Zero(n)),
      curr_state(),
      prev_state(),
      end_time(),
      start_time(),
      stop_state(),
      callback(callback)
    {
      setStopState(accuracy,
                   duration_max,
                   f_evals_max
                  );
    }

    explicit Solver(const Callback& callback) :
      Solver(0.7,       // accuracy
             0,         // duration_max
             0,         // f_evals_max
             callback   // callback
            )
    {}

    virtual ~Solver() = default;

    // Friend operator <<
    template <class F>
    friend std::ostream& operator<<(std::ostream& os, const Solver<F>& solver);

    // Initializes the state for each new optimization
    void initialize(const Function& f,
                    const Iterate& iterate,
                    const Vector& l = Vector(),
                    const Vector& u = Vector()
                   )
    {
      // Reset the start time
      start_time = Clock::now();

      // Copy the function and constraints
      n = iterate.x.size();
      this->f = f;
      this->f.reset();
      this->l = l.size() == 0 ? Vector::Constant(n, ScalarLimits::lowest()) : l;
      this->u = u.size() == 0 ? Vector::Constant(n, ScalarLimits::max()) : u;

      // Resize l and u if necessary
      Scalar l_size = this->l.size();
      Scalar u_size = this->u.size();
      this->l.conservativeResize(n);
      this->u.conservativeResize(n);

      // If l or u increased in size, populate them with min/max bounds
      for (Index i = l_size; i < n; ++i) {
        this->l(i) = ScalarLimits::lowest();
      }
      for (Index i = u_size; i < n; ++i) {
        this->u(i) = ScalarLimits::max();
      }

      // Flip constraints if l(i) > u(i)
      for (Index i = 0; i < n; ++i) {
        Scalar li = this->l(i);
        this->l(i) = std::min<Scalar>(this->l(i), this->u(i));
        this->u(i) = std::max<Scalar>(li,         this->u(i));
      }

      // Initialize the solver state with the initial iterate, enforcing that x is within bounds [l, u]
      curr_state = State(iterate);
      curr_state.x() = clip(curr_state.x(), this->l, this->u);
      prev_state = curr_state;

      reset();            // Reset the algorithm's internal data
      updateState(true);  // Compute initial solver state with convergence data
    }

    void initialize(Function f,
                    const Vector& x,
                    const Vector& l = Vector(),
                    const Vector& u = Vector()
                   )
    { initialize(f, f.compute(x), l, u); }

    // Minimizes the function f starting from x and subject to bound constraints l and u.
    // Returns the full state when the convergence criteria is met or the solver is stopped.
    const State& minimize(Function f,
                          const Vector& x,
                          const Vector& l = Vector(),
                          const Vector& u = Vector()
                         )
    {
      initialize(f, x, l, u); // Initialize the solver for the new minimization problem
      return minimize();      // Minimize f from the initial state and return the result
    }

    // Minimizes the function f starting from the specified function state and subject to bound constraints l and u.
    // Returns the full state when the convergence criteria is met or the solver is stopped.
    const State& minimize(const Function& f,
                          const Iterate& iterate,
                          const Vector& l = Vector(),
                          const Vector& u = Vector()
                         )
    {
      initialize(f, iterate, l, u); // Initialize the solver for the new minimization problem
      return minimize();            // Minimize f from the initial state and return the result
    }

    // Minimizes the function f from the current state with previously-specified bound constraints l and u.
    // Returns the full state when the convergence criteria is met or the solver is stopped.
    const State& minimize()
    {
      curr_state.stopped() = false; // Reset the stopped flag to allow the solver to run
      start_time = Clock::now();    // Reset the start time

      // Compute a new state until the solver is done or stopped by the user
      while (!done() && !stopped()) {
        performStep();  // Perform the optimization step to compute a new iterate
        updateState();  // Update the solver state with the current iterate
        callback(this); // Execute the user callback function
      }

      return curr_state;  // Return the solution containing the minimum
    }

    // Stops the solver at the current state after the current optimization step is complete.
    // The solver may be resumed at this state by subsequently calling run().
    void stop()
    { curr_state.stopped() = true; }

    // Updates the solver state with the current iterate
    void updateState(const bool firstUpdate = false)
    {
      // Update convergence data
      // Skip calculating df and dx norms if the solver was reset because in this case, curr_state == prev_state
      if (!curr_state.reset()) {
        curr_state.dfNorm() =   std::abs(curr_state.f() - prev_state.f())
                              / std::max(std::max(std::abs(curr_state.f()),
                                                  std::abs(prev_state.f())
                                                 ),
                                         static_cast<Scalar>(1.0)
                                        );
        curr_state.dxNorm() =   infinityNorm(curr_state.x() - prev_state.x())
                              / std::max(std::max(infinityNorm(curr_state.x()),
                                                  infinityNorm(prev_state.x())
                                                 ),
                                         static_cast<Scalar>(1.0)
                                        );
      }
      curr_state.gNorm() = infinityNorm(projectedGradient(curr_state.x(), curr_state.g(), l, u));

      // Copy function data
      curr_state.function = f.state();

      // Check stopping criteria
      const bool df_min_success = (curr_state.dfNorm() <= stop_state.dfNorm() && !curr_state.reset());
      const bool dx_min_success = (curr_state.dxNorm() <= stop_state.dxNorm() && !curr_state.reset());
      const bool g_min_success = (curr_state.gNorm() <= stop_state.gNorm());
      const bool duration_exceeded = (curr_state.duration() >= stop_state.duration() && stop_state.duration() > 0.0);
      const bool f_evals_exceeded = (curr_state.fEvals() >= stop_state.fEvals() && stop_state.fEvals() > 0);

      // Update solver statuses
      curr_state.success() = (   df_min_success
                              || dx_min_success
                              || g_min_success
                             );
      curr_state.aborted() = (   curr_state.aborted()  // aborted state is controlled by the algorithm, see abort()
                              || duration_exceeded
                              || f_evals_exceeded
                             );
      curr_state.iter() = firstUpdate ? 0 : curr_state.iter() + 1;
      end_time = Clock::now();
      curr_state.duration() += durationMsec(start_time, end_time);
      start_time = end_time;

      // Update previous state
      prev_state = curr_state;
    }

    // Sets all solver stopping criteria: accuracy, duration, and number of function evaluations
    void setStopState(const Scalar accuracy,
                      const Scalar duration,
                      const Scalar f_evals
                     )
    {
      setAccuracy(accuracy);
      setDurationMax(duration);
      setFunctionEvalsMax(f_evals);
    }

    // Sets the solver convergence criteria based on the desired level of accuracy.
    // Accuracy level ranges from 0 to 1, where 1 is maximum accuracy possible based on the machine epsilon of the
    // Scalar type.
    void setAccuracy(Scalar accuracy)
    {
      // Clip accuracy to be within [0, 1]
      accuracy = clip(accuracy, 0.0, 1.0);

      // Compute converge criteria based on the machine epsilon.
      // The highest achievable accuracy in the objective value or parameter vector is the machine epsilon.
      // The target gradient norm is recommended to be at least sqrt(epsilon) by the authors of the L-BFGS-B paper.
      stop_state.dfNorm() = std::pow(10, accuracy*log10(ScalarLimits::epsilon()));
      stop_state.dxNorm() = std::pow(10, accuracy*log10(ScalarLimits::epsilon()));
      stop_state.gNorm() = std::pow(10, accuracy*log10(std::sqrt(ScalarLimits::epsilon())));
    }

    // Sets the maxmimum duration (in millseconds) the solver will run for during each computation (0 = unlimited)
    // before stopping and returning the current state
    void setDurationMax(const Scalar duration)
    { stop_state.duration() = duration; }

    // Sets the maxmimum number of function evaluations before the solver will stop and return the current state
    void setFunctionEvalsMax(const Scalar f_evals)
    { stop_state.fEvals() = f_evals; }

    // Sets the callback function
    void setCallback(const Callback& callback)
    { this->callback = callback; }

    // Returns the current state
    const State& state() const
    { return curr_state; }

    // Returns true if the solver met the convergence criteria
    bool success()
    { return curr_state.success(); }

    // Returns true if the solver is stopped
    bool stopped()
    { return curr_state.stopped(); }

    // Returns true if the solver has aborted the optimization due to a termination condition being met or, for an
    // algorithm-specific reason, further progress cannot be made. This may or may not indicate failure; in general it
    // means that the algorithm is unable to improve from the current state.
    bool aborted()
    { return curr_state.aborted(); }

    // Returns true if the solver cannot make further progress because a convergence or termination condition was met
    bool done()
    { return success() || aborted(); }

  protected:
    // Resets the algorithm's internal data and restarts the optimization at the current iterate
    virtual void reset()
    {
      // Set the reset flag to indicate this is the first optimization step since reset() was called
      this->curr_state.reset() = true;
    }

    // Performs one optimization step, updating the current and previous iterate
    virtual void performStep() = 0;

    // Aborts the solver at the current state. The solver cannot be resumed from this state.
    //
    // This function is a mechanism for the derived algorithm to indicate that, for an algorithm-specific reason,
    // further progress cannot be made. This may or may not indicate failure; in general it means that the algorithm is
    // unable to improve from the current state.
    void abort()
    { curr_state.aborted() = true; }

  protected:
    Index n;              // Size of x
    Function f;           // Function to be minimized
    Vector l;             // Lower bounds for x
    Vector u;             // Upper bounds for x
    State curr_state;     // Current state
    State prev_state;     // Previous state
    Time end_time;        // Current time point
    Time start_time;      // Previous time point
    StopState stop_state; // Stopping state
    Callback callback;    // Optional user-defined callback function
  };

  // Prints the solver state including termination messages, if any
  template <class Function>
  std::ostream& operator<<(std::ostream& os, const Solver<Function>& solver)
  {
    const State& curr_state = solver.curr_state;
    const StopState& stop_state = solver.stop_state;

    std::cout << curr_state;

    if (curr_state.success()) {
      std::cout << std::endl;
      if (curr_state.dfNorm() <= stop_state.dfNorm()) {
        std::cout << std::endl << "Change in f is below threshold";
      }
      if (curr_state.dxNorm() <= stop_state.dxNorm()) {
        std::cout << std::endl << "Change in x is below threshold";
      }
      if (curr_state.gNorm() <= stop_state.gNorm()) {
        std::cout << std::endl << "Gradient norm is below threshold";
      }
    }
    else {
      const bool duration_exceeeded = (   curr_state.duration() >= stop_state.duration()
                                       && stop_state.duration() > 0.0
                                      );
      const bool f_evals_exceeded = (   curr_state.fEvals() >= stop_state.fEvals()
                                     && stop_state.fEvals() > 0
                                    );
      if (   duration_exceeeded
          || f_evals_exceeded
          || curr_state.aborted()
          || curr_state.stopped()
         ) {
        std::cout << std::endl;
        if (duration_exceeeded) {
          std::cout << std::endl << "Maxmimum execution time exceeded";
        }
        if (f_evals_exceeded) {
          std::cout << std::endl << "Maxmimum number of function evaluations reached";
        }
        if (curr_state.aborted()) {
          std::cout << std::endl << "Optimization aborted by algorithm";
        }
        if (curr_state.stopped()) {
          std::cout << std::endl << "Optimization stopped by user";
        }
      }
    }
    return os;
  }

} // namespace optimize

#endif // SOLVER_H