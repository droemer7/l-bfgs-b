// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#ifndef LBFGSB_H
#define LBFGSB_H

#include <iostream>         // cout, endl, ostream
#include <vector>           // vector

#include <Eigen/LU>         // lu()

#include "function.h"       // Function
#include "line_search.h"    // LewisOverton
#include "solver.h"         // Solver
#include "util.h"           // Matrix, Vector, Index, clip()

namespace optimize
{
  // Breakpoint for the generalized Cauchy Point search
  struct Breakpoint
  {
    explicit Breakpoint(const Index i = -1, const double t = 0.0) :
      i(i),
      t(t)
    {}

    bool operator==(const Breakpoint& rhs)
    { return this->t == rhs.t; }

    bool operator!=(const Breakpoint& rhs)
    { return this->t != rhs.t; }

    bool operator<=(const Breakpoint& rhs)
    { return this->t <= rhs.t; }

    bool operator<(const Breakpoint& rhs)
    { return this->t < rhs.t; }

    bool operator>=(const Breakpoint& rhs)
    { return this->t >= rhs.t; }

    bool operator>(const Breakpoint& rhs)
    { return this->t > rhs.t; }

    Index i;  // Index of the variable in x
    double t; // Breakpoint for the variable i in x
  };

  std::ostream& operator<<(std::ostream& os, const std::vector<Breakpoint>& breakpoints)
  {
    os << "breakpoints = ";
    if (breakpoints.size() == 0) {
      os << std::endl;
    }
    else {
      for (size_t i = 0; i < breakpoints.size(); ++i) {
        os << std::endl << "t for x(" << breakpoints[i].i << ") = " << breakpoints[i].t;
      }
    }
    return os;
  }

  // L-BFGS-B: Limited-memory BFGS algorithm for bound constrained optimization
  //
  // Reference: R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, "A Limited Memory Algorithm for Bound Constrained
  //            Optimization", Tech. Report, NAM-08, EECS Department, Northwestern University, 1994.
  template <class Function, class LineSearch = LewisOverton<Wolfe::weak>>
  class Lbfgsb : public Solver<Function>
  {
  public:
    using IndexSet = std::vector<Index>;
    using BreakpointSet = std::vector<Breakpoint>;
    using typename Solver<Function>::Callback;

  public:
    // Constructors and destructors
    explicit Lbfgsb(const Scalar accuracy = 0.7,                                   // Accuracy level 0 to 1, where 1 is maximum accuracy
                    const Scalar duration_max = 0,                                 // Maxmimum duration (milliseconds) (0 = unlimited)
                    const Index f_evals_max = 0,                                   // Maxmimum number of function evaluations (0 = unlimited)
                    const Callback& callback = Callback([](Solver<Function>*) {})  // Callback function which is executed after each optimization step
                   ) :
      Solver<Function>(accuracy,      // accuracy
                       duration_max,  // duration_max
                       f_evals_max,   // f_evals_max
                       callback       // callback
                      ),
      m_max(5),

      line_search(),

      m(0),
      th(1.0),
      th_inv(1.0),

      I(),
      S(),
      Y(),
      SS(),
      SY(),
      YY(),

      D(),
      R_inv(),

      W(),
      Wb(),

      M(),
      Mb(),

      c(),

      free_set(),
      active_set()
    {}

  explicit Lbfgsb(const Callback& callback) :
    Lbfgsb(0.7,       // accuracy
           0,         // duration_max
           0,         // f_evals_max
           callback   // callback
          )
  {}

  private:
    // Resets the algorithm's internal data
    void reset() override
    {
      // Set the reset flag to indicate this is the first optimization step since reset() was called
      this->curr_state.reset() = true;

      // Initialize BFGS information and matrices
      m = 0;
      th = 1.0;
      th_inv = 1.0;

      I = Matrix::Identity(this->n, this->n);
      S = Matrix::Zero(this->n, 1);
      Y = Matrix::Zero(this->n, 1);
      SS = Matrix::Zero(1, 1);
      SY = Matrix::Zero(1, 1);
      YY = Matrix::Zero(1, 1);

      D = Matrix::Zero(1, 1);
      R_inv = Matrix::Zero(1, 1);

      W = Matrix::Zero(this->n, 2);
      Wb = Matrix::Zero(this->n, 2);

      M = Matrix::Zero(2, 2);
      Mb = Matrix::Zero(2, 2);

      c = Vector::Zero(2);

      free_set.reserve(this->n);
      free_set.clear();
      active_set.reserve(this->n);
      active_set.clear();
    }

    // Performs one optimization step of the algorithm, updating the current and previous iterate
    void performStep() override
    {
      // Create aliases for readability
      Function& f = this->f;
      const Vector& l = this->l;
      const Vector& u = this->u;
      Iterate& prev = this->prev_state.iterate;
      Iterate& curr = this->curr_state.iterate;

      Vector xc = cauchyPoint(curr.x, curr.g, l, u);                // Compute the generalized Cauchy point xc
      Vector d = searchDir(xc, curr.x, curr.g, l, u);               // Compute the search direction d
      Scalar t_max = maxStep(curr.x, l, u, d);                      // Compute the max step possible along d
      Scalar t = line_search(f, curr.f, curr.x, curr.g, d, t_max);  // Compute the step to take along d

      // A suitable step was found: compute the next iterate and update the limited memory matrices
      if (t > 0.0) {
        prev = curr;                      // Save the previous iterate
        curr.x += t*d;                    // Compute the next iterate x = x + t*d
        curr.f = f(curr.x);               // Compute the function value f(x)
        curr.g = f.gradient(curr.x);      // Compute the gradient ▽f(x)
        updateMatrices(curr.x - prev.x,   // Update the limited memory matrices
                       curr.g - prev.g
                      );
        this->curr_state.reset() = false; // Clear the reset flag now a new iterate has been successfully computed
      }
      // A suitable step was not found: discard all correction pairs and restart along the steepest descent direction
      else if (!this->curr_state.reset()) {
        reset();
      }
      // Restarting along the steepest descent direction failed, abort
      else {
        this->abort();
      }
    }

    // Determines the first local minimizer of the univariate, piecewise quadratic q(t):
    //
    //   q(t) = m(P(xk - t*g, l u))
    //
    // where
    //   m(x) = f(xk) + ▽f(xk)^T(x - xk) + 1/2(x - xk)^T*Bk*(x - xk)
    //   P(x, l, u)(i) = { l(i)  if x(i) < l(i)
    //                   { x(i)  if x(i) is in [l(i), u(i)]
    //                   { u(i)  if x(i) > u(i)
    //   Bk = th*I - W*M*W^T
    Vector cauchyPoint(const Vector& x,
                       const Vector& g,
                       const Vector& l,
                       const Vector& u
                      )
    {
      // Initialize the search
      // Set xc = x to start - we will modify xc if/when appropriate during the search
      Vector xc = x;
      BreakpointSet breakpoints;
      Vector t = Vector::Zero(this->n);
      Vector d = Vector::Zero(this->n);
      Vector p = Vector::Zero(W.cols());
      Vector w = Vector::Zero(W.cols());
      c = Vector::Zero(W.cols());
      free_set.clear();
      active_set.clear();

      size_t q = 0;
      Index b = -1;
      Scalar fp = 0;
      Scalar fpp = 0;
      Scalar dt_min = 0;
      Scalar t_start = 0;
      Scalar t_end = 0;
      Scalar dt = 0;
      Scalar t_min = 0;
      Scalar zb = 0;

      // Calculate breakpoints t(i) and descent directions d(i)
      for (Index i = 0; i < this->n; ++i) {
        // Calculate breakpoints t(i)
        if (g(i) < 0) {
          t(i) = (x(i) - u(i))/g(i);
        }
        else if (g(i) > 0) {
          t(i) = (x(i) - l(i))/g(i);
        }
        else {
          t(i) = ScalarLimits::max();
        }
        // Calculate descent directions d(i) for variables with breakpoints t(i) > 0
        // Otherwise, t(i) == 0 so we leave d(i) = 0 and xc(i) = x(i) from initialization above
        if (t(i) > 0) {
          d(i) = -g(i);
          breakpoints.push_back(Breakpoint(i, t(i)));
        }
      }

      if (breakpoints.size() > 0) {
        // Sort indexes i by ascending breakpoint value t
        std::sort(breakpoints.begin(), breakpoints.end());

        // Initialize the storage vector p used to update f' and f''
        p = W.transpose()*d;

        // Calculate the initial dt_min
        fp = -d.dot(d);
        fpp = -th*fp - p.transpose()*M*p;
        dt_min = (fpp == 0) ? ScalarLimits::max() : -fp/fpp;

        // Define the first interval
        b = breakpoints[q].i;
        t_start = t_end;
        t_end = t(b);
        dt = t_end - t_start;

        // Search for the step to the first minimum along the steepest descent direction d(i) = -g(i)
        // Inspect each interval in t until dt_min is less than or equal to the interval
        while (dt_min > dt) {
          assert (d(b) != 0);

          // Since dt_min > dt, the current component's minimum is not in the interval and must be bounded
          // Accordingly, we also zero out the current component's search direction
          xc(b) = d(b) > 0 ? u(b) : l(b);
          zb = xc(b) - x(b);
          d(b) = 0;

          // Update vector c which will be used to initialize the subspace minimization problem
          c += dt*p;

          // Calculate f' and f'' for determining the location of the minimum
          w = W(b, Eigen::all).transpose();
          fp += dt*fpp + g(b)*g(b) + th*g(b)*zb - g(b)*w.transpose()*M*c;
          fpp += -th*g(b)*g(b) - 2*g(b)*w.transpose()*M*p - g(b)*g(b)*w.transpose()*M*w;
          dt_min = (fpp == 0) ? ScalarLimits::max() : -fp/fpp;

          // Update p for the next calculation of f' and f''
          p += g(b)*w;

          // Update to the next interval
          t_start = t_end;

          if (++q < breakpoints.size()) {
            b = breakpoints[q].i;
            t_end = t(b);
            dt = t_end - t_start;
          }
          // If there are no more intervals, the previous iteration determined that the final component of x(b=n) is
          // bounded. In this case dt_min will be past the final breakpoint t(b=n) and there will be no free variables.
          else {
            t_end = t_start;
            dt = 0;
            break;
          }
        }
      }
      // Calculate t_min
      dt_min = std::max<Scalar>(dt_min, 0);
      t_min = t_start + dt_min;

      // Construct the sets of free and active variable indexes i and calculate the remaining components xc(i) of the
      // Cauchy point from t_min.
      //
      // Note: There is an error in the paper here: t_min should be used instead of t_end. The final steps specify
      //       updating xc(i) for t(i) >= t_end and removing i from the free set if t(i) == t_end. This would mean
      //       removing indexes for xc(i) that are NOT at their bound because usually t_min < t_end, and
      //       xc(i) = x(i) + t_min*d(i). Since the minimum is found at t_min, we should compute all xc(i) for
      //       t(i) >= t_min and then remove i from F if t(i) == t_min, because that means xc(i) will be at its bound.
      for (Index i = 0; i < this->n; ++i) {
        // If q is past the end of the breakpoint array, there are no free variables.
        // Either: 1) There were no breakpoints t(i) > 0, so all xc(i) are already bounded (and xc == x from
        //            initialization)
        //         2) The Cauchy point lies beyond the largest breakpoint t(i) in x, in which case all xc(i) were set to
        //            their bound u(i) or l(i) during the search.
        if (q < breakpoints.size()) {
          // Compute xc(i) using the step t_min to the first minimum found earlier
          if (t(i) >= t_min) {
            xc(i) = x(i) + t_min*d(i);
          }

          // Add i to the free set if x(i) is unbounded (its breakpoint t(i) lies beyond the minimum t_min)
          // Otherwise, add i to the active set
          if (t(i) > t_min) {
            free_set.push_back(i);
          }
          else {
            active_set.push_back(i);
          }
        }
        else {
          active_set.push_back(i);
        }
      }
      c += dt_min*p;  // Update vector c for the subspace minimization problem

      return xc;
    }

    // Approximately solves for the search direction d of the subspace minimization problem m(d):
    //
    //   m(d) = d^T*rc + 1/2(d^T*B*d) + y
    //   subject to l(i) - xc(i) <= d(i) <= u(i) - xc(i)
    //
    // where
    //   rc is the reduced gradient of m at xc
    //   B is the reduced hessian of m
    Vector searchDir(const Vector& xc,
                     const Vector& x,
                     const Vector& g,
                     const Vector& l,
                     const Vector& u
                    )
    {
      Matrix A = I(all, active_set);  // Matrix of unit vectors spanning the active set at the Cauchy point xc (n x ta)
      Matrix Z = I(all, free_set);    // Matrix of unit vectors spanning the free set at the Cauchy point xc (n x tf)
      Vector v;                       // Vector v (2m x 1)
      Matrix N;                       // Matrix N (2m x 2m)

      Vector rc = Z.transpose()*(g + th*(xc - x) - W*M*c);  // Reduced gradient of the quadratic model mk at xc (tf x 1)

      // Compute v and N using the active set or free set, whichever is smaller
      if (A.cols() < Z.cols()) {
        v = Mb*W.transpose()*Z*rc;

        // If A is empty, N remains empty (mathematically N = I)
        if (A.cols() > 0) {
          N = Matrix::Identity(Mb.rows(), Mb.cols()) + th*Mb*Wb.transpose()*A*A.transpose()*Wb;
        }
      }
      else {
        v = M*W.transpose()*Z*rc;

        // If Z is empty, N remains empty (mathematically N = I)
        if (Z.cols() > 0) {
          N = Matrix::Identity(M.rows(), M.cols()) - th_inv*M*W.transpose()*Z*Z.transpose()*W;
        }
      }

      // Compute v = N^-1*v (2m x 1)
      // If N is empty here, mathematically N = I therefore N^-1*v = v and we can skip this calculation
      if (N.size() > 0) {
        v = N.lu().solve(v);
      }

      // Compute du = -B^-1*rc = -(1/th)*rc - (1/th^2)*Z^T*W*v (tf x 1)
      //
      // Note: There is an error in the paper here. Equation 5.7 correctly specifies the solution to m(d) as
      //       d = -B^-1*rc, then Equation 5.11 mistakenly drops the (-) sign and states d = B^-1*rc. The calculation
      //       below follows the correct solution to m(d), d = -B^-1*rc, which can be verified by setting the derivative
      //       m'(d) = 0 and solving for d.
      Vector du = -th_inv*rc - th_inv*th_inv*Z.transpose()*W*v;

      // Find a_star = max{a : a <= 1, l(i) - xc(i) <= a*du(i) <= u(i) - xc(i), i ∈ F}
      Scalar a_star = std::min<Scalar>(1.0, maxStep(xc(free_set), l(free_set), u(free_set), du));

      // Compute d (n x 1)
      // = xc(i) - x                    if i ∉ F
      // = xc(i) - x + (Z*a_star*du)(i) if i ∈ F
      Vector d = xc - x;
      Vector Zd_star = Z*a_star*du;

      for (Index i : free_set) {
        d(i) += Zd_star(i);
      }

      return d;
    }

    // Performs the limited-memory BFGS update of th, S, Y, W, Wb, M, Mb and related matrices.
    template <class Derived>
    void updateMatrices(const Eigen::MatrixBase<Derived>& s,
                        const Eigen::MatrixBase<Derived>& y
                       )
    {
      // Discard {s, y} if the curvature condition s^T * y > 0 is not satisified
      if (s.dot(y) > ScalarLimits::epsilon()*y.squaredNorm()) {
        // Increase the size m of the matrices with each correction pair we keep up to m_max
        m = std::min(m+1, m_max);

        // Once we've reached the max memory size, begin discarding old {s, y} pairs by shifting matrix data left & up
        if (Y.cols() == m_max) {
          shift(S, 0, -1);
          shift(Y, 0, -1);
          shift(SS, -1, -1);
          shift(SY, -1, -1);
          shift(YY, -1, -1);
        }
        else {
          S.conservativeResize(Eigen::NoChange, m);
          Y.conservativeResize(Eigen::NoChange, m);
          SS.conservativeResize(m, m);
          SY.conservativeResize(m, m);
          YY.conservativeResize(m, m);

          W.conservativeResize(Eigen::NoChange, 2*m);
          Wb.conservativeResize(Eigen::NoChange, 2*m);

          M.conservativeResize(2*m, 2*m);
          Mb.conservativeResize(2*m, 2*m);
        }

        // Update th
        th = y.dot(y)/y.dot(s);
        th_inv = 1/th;

        // Add new s, y to S and Y
        S.col(m-1) = s;
        Y.col(m-1) = y;

        // Update SS
        SS.row(m-1) = S.col(m-1).transpose()*S;
        SS.col(m-1) = SS.row(m-1).transpose();

        // Update SY
        SY.row(m-1) = S.col(m-1).transpose()*Y;
        SY.col(m-1) = S.transpose()*Y.col(m-1);

        // Update YY
        YY.row(m-1) = Y.col(m-1).transpose()*Y;
        YY.col(m-1) = YY.row(m-1).transpose();

        // Form D (diagonal of S^T*Y) and R^-1 (inverse of the upper triangular matrix of S^T*Y)
        D = SY.diagonal().asDiagonal();
        R_inv = SY.template triangularView<Eigen::Upper>();
        R_inv = R_inv.inverse().eval();

        // Update W
        W.leftCols(m) = Y;
        W.rightCols(m) = th*S;

        // Update Wb
        Wb.leftCols(m) = th_inv*Y;
        Wb.rightCols(m) = S;

        // Update M
        M.topLeftCorner(m, m) = -D;
        M.topRightCorner(m, m) = SY.template triangularView<Eigen::StrictlyLower>().transpose();
        M.bottomLeftCorner(m, m) = SY.template triangularView<Eigen::StrictlyLower>();
        M.bottomRightCorner(m, m) = th*SS;
        M = M.inverse().eval();

        // Update Mb
        Mb.topLeftCorner(m, m) = Matrix::Zero(m, m);
        Mb.topRightCorner(m, m) = -R_inv;
        Mb.bottomLeftCorner(m, m) = (-R_inv).transpose();
        Mb.bottomRightCorner(m, m) = R_inv.transpose()*(D + th_inv*YY)*R_inv;
      }
    }

    // Finds max{ t: l(i) <= x(i) + t*d(i) <= u(i) ∀ d(i) }
    template <class Fixed, class Dynamic>
    Scalar maxStep(const Eigen::MatrixBase<Fixed>& x,
                   const Eigen::MatrixBase<Fixed>& l,
                   const Eigen::MatrixBase<Fixed>& u,
                   const Eigen::MatrixBase<Dynamic>& d
                  )
    {
      // Scaler to prevent precision issues from generating a max step that violates the bounds by a small amount
      constexpr Scalar s = 1 - ScalarLimits::epsilon();

      Scalar t = 0.0;
      Scalar t_max = ScalarLimits::max();

      for (Index i = 0; i < d.size(); ++i) {
        if (d(i) != 0.0) {
          t = s * std::max<Scalar>((u(i) - x(i))/d(i),
                                   (l(i) - x(i))/d(i)
                                  );
          t = clip(t, 0.0, ScalarLimits::max());
        }
        else {
          t = ScalarLimits::max();
        }
        t_max = std::min<Scalar>(t, t_max);
      }

      return t_max;
    }

  private:
    const Index m_max;      // Maximum number of correction pairs {s, y} to store in the limited memory matrices

    LineSearch line_search; // Line search method

    Index m;         // Current number of correction pairs {s, y} stored in the limited memory matrices
    Scalar th;       // Theta scaling parameter
    Scalar th_inv;   // 1/Theta scaling parameter

    Matrix I;        // Identity Matrix (n x n)
    Matrix S;        // Matrix S where each element is the correction pair xk+1 - xk (n x m)
    Matrix Y;        // Matrix Y where each element is the correction pair gk+1 - gk (n x m)
    Matrix SS;       // Matrix S^T*S (m x m)
    Matrix SY;       // Matrix S^T*Y (m x m)
    Matrix YY;       // Matrix Y^T*Y (m x m)

    Matrix D;        // Diagonal matrix of S^T*Y (m x m)
    Matrix R_inv;    // Inverse of the upper triangular matrix of S^T*Y (m x m)

    Matrix W;        // Matrix W from the L-BFGS-B paper (n x 2m)
    Matrix Wb;       // Matrix Wb from the L-BFGS-B paper (n x 2m)

    Matrix M;        // Matrix M from the L-BFGS-B paper (2m x 2m)
    Matrix Mb;       // Matrix Mb from the L-BFGS-B paper (2m x 2m)

    Vector c;        // Vector which represents a component needed to compute the reduced gradient of free
                            // variables at the Cauchy point (2m x 1)

    IndexSet free_set;      // Set of free (unconstrained) indexes (tf x 1)
    IndexSet active_set;    // Set of active (constrained) indexes (ta x 1)
  };

} // namespace optimize

#endif // LBFGSB_H