// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#ifndef TEST_H
#define TEST_H

#include <cmath>

#include "function.h"

namespace optimize
{
  // Forrester objective function
  // http://www.sfu.ca/~ssurjano/forretal08.html
  class Forrester : public Function
  {
  public:
    Scalar computeValue(const Vector& x) override
    { return (6*x(0) - 2)*(6*x(0) - 2)*std::sin(12*x(0) - 4); }

    Vector computeGradient(const Vector& x) override
    {
      return Vector {{  12*(6*x(0) - 2)*(  std::sin(12*x(0) - 4)
                                         + (6*x(0) - 2)*std::cos(12*x(0) - 4))
                    }};
    }
  };

  // Simple objective function
  // https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods#Numeric_Example
  class Simple : public Function
  {
  public:
    Scalar computeValue(const Vector& x) override
    { return x(0) - x(1) + 2*x(0)*x(1) + 2*x(0)*x(0) + x(1)*x(1); }

    Vector computeGradient(const Vector& x) override
    {
      return Vector {{ 1 + 2*x(1) + 4*x(0),
                      -1 + 2*x(0) + 2*x(1)
                    }};
    }
  };

  // Rosenbrock objective function
  // http://www.sfu.ca/~ssurjano/rosen.html
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

  // Six-Hump Camel objective function
  // http://www.sfu.ca/~ssurjano/camel6.html
  class SixHumpCamel : public Function
  {
  public:
    Scalar computeValue(const Vector& x) override
    { return x(0)*x(0)*(4 - 2.1*x(0)*x(0) + x(0)*x(0)*x(0)*x(0)/3) + x(0)*x(1) + 4*x(1)*x(1)*(-1 + x(1)*x(1)); }

    Vector computeGradient(const Vector& x) override
    {
      return Vector {{ 8*x(0) - 8.4*x(0)*x(0)*x(0) + 2*x(0)*x(0)*x(0)*x(0)*x(0) + x(1),
                       x(0) - 8*x(1) + 16*x(1)*x(1)*x(1)
                    }};
    }
  };

  // 2D point (x, y) and angle of orientation
  struct Pose {
    explicit Pose(const double x = 0.0,
                  const double y = 0.0,
                  const double th = 0.0
                 ):
      x(x),
      y(y),
      th(th)
    {}

    double x;  // X position
    double y;  // Y position
    double th; // Angle
  };

  // Spiral objective function
  class Spiral : public Function
  {
  public:
    // Define a non-zero-initialized default constructor for the purposes of testing
    // This wouldn't be defined this way in normal use
    explicit Spiral(const Pose p0 = Pose(0, 0, 0),
                    const Pose pf = Pose(5.0, 5.0, M_PI/3),
                    const Pose weights = Pose(25, 25, 30)
                   ) :
      p0(p0),
      pf(pf),
      weights(weights)
    {}

    Scalar computeValue(const Vector& x) override
    {
      Scalar p1 = x(0);
      Scalar p2 = x(1);
      Scalar p4 = x(2);

      // Bending energy
      Scalar be = 27*p4*(4*p1*p1 - p1*p2 + 4*p2*p2)/280;

      // X error
      Scalar xe_t1 = (p4*(    cos(p0.th)
                          + 4*cos(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th)
                          + 2*cos(    0.17724609375*p1*p4 -     0.06005859375*p2*p4 + p0.th)
                          + 4*cos(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th)
                          +   cos(            0.375*p1*p4 +             0.375*p2*p4 + p0.th)
                          + 2*cos(        0.3984375*p1*p4 -         0.0234375*p2*p4 + p0.th)
                          + 4*cos(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th)
                          + 2*cos(    0.43505859375*p1*p4 +     0.19775390625*p2*p4 + p0.th)
                          + 4*cos(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)
                         )
                      - 24*pf.x
                      + 24*p0.x
                     );
      Scalar xe = xe_t1*xe_t1/576;

      // Y error
      Scalar ye_t1 = (p4*(    sin(p0.th)
                          + 4*sin(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th)
                          + 2*sin(    0.17724609375*p1*p4 -     0.06005859375*p2*p4 + p0.th)
                          + 4*sin(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th)
                          +   sin(            0.375*p1*p4 +             0.375*p2*p4 + p0.th)
                          + 2*sin(        0.3984375*p1*p4 -         0.0234375*p2*p4 + p0.th)
                          + 4*sin(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th)
                          + 2*sin(    0.43505859375*p1*p4 +     0.19775390625*p2*p4 + p0.th)
                          + 4*sin(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)
                         )
                      - 24*pf.y
                      + 24*p0.y
                     );
      Scalar ye = ye_t1*ye_t1/576;

      // Angular error
      Scalar the_t1 = 0.375*p1*p4 + 0.375*p2*p4 - pf.th + p0.th;
      Scalar the = the_t1*the_t1;

      return (  be
              + weights.x*xe
              + weights.y*ye
              + weights.th*the
             );
    }

    Vector computeGradient(const Vector& x) override
    {
      Scalar p1 = x(0);
      Scalar p2 = x(1);
      Scalar p4 = x(2);

      Vector grad_be(Vector::Zero(x.rows()));
      Vector grad_xe(Vector::Zero(x.rows()));
      Vector grad_ye(Vector::Zero(x.rows()));
      Vector grad_the(Vector::Zero(x.rows()));

      // Bending energy computeGradient
      grad_be << 27*p4*(8*p1 - p2)/280,
                 27*p4*(-p1 + 8*p2)/280,
                 27*p1*p1/70 - 27*p1*p2/280 + 27*p2*p2/70;

      // X error computeGradient
      grad_xe << weights.x*p4*(p4*(cos(p0.th) + 4*cos(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 2*cos(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 4*cos(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + cos(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 2*cos(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 4*cos(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 2*cos(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 4*cos(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)) - 24*pf.x + 24*p0.x)*(-0.2259521484375*p4*sin(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) - 0.3544921875*p4*sin(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) - 1.2161865234375*p4*sin(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) - 0.375*p4*sin(0.375*p1*p4 + 0.375*p2*p4 + p0.th) - 0.796875*p4*sin(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) - 1.5970458984375*p4*sin(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) - 0.8701171875*p4*sin(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) - 1.7669677734375*p4*sin(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th))/288,
                 weights.x*p4*(p4*(cos(p0.th) + 4*cos(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 2*cos(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 4*cos(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + cos(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 2*cos(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 4*cos(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 2*cos(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 4*cos(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)) - 24*pf.x + 24*p0.x)*( 0.0970458984375*p4*sin(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 0.1201171875*p4*sin(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 0.2669677734375*p4*sin(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) - 0.375*p4*sin(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 0.046875*p4*sin(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) - 1.2740478515625*p4*sin(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) - 0.3955078125*p4*sin(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) - 0.2838134765625*p4*sin(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th))/288,
                 weights.x*   (p4*(cos(p0.th) + 4*cos(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 2*cos(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 4*cos(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + cos(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 2*cos(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 4*cos(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 2*cos(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 4*cos(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)) - 24*pf.x + 24*p0.x)*(2*p4*(-4*(0.056488037109375*p1 - 0.024261474609375*p2)*sin(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) - 2*(0.17724609375*p1 - 0.06005859375*p2)*sin(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) - 4*(0.304046630859375*p1 - 0.066741943359375*p2)*sin(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) - (0.375*p1 + 0.375*p2)*sin(0.375*p1*p4 + 0.375*p2*p4 + p0.th) - 2*(0.3984375*p1 - 0.0234375*p2)*sin(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) - 4*(0.399261474609375*p1 + 0.318511962890625*p2)*sin(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) - 2*(0.43505859375*p1 + 0.19775390625*p2)*sin(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) - 4*(0.441741943359375*p1 + 0.070953369140625*p2)*sin(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)) + 2*cos(p0.th) + 8*cos(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 4*cos(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 8*cos(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + 2*cos(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 4*cos(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 8*cos(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 4*cos(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 8*cos(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th))/576;

      // Y error computeGradient
      grad_ye << weights.y*p4*(p4*(sin(p0.th) + 4*sin(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 2*sin(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 4*sin(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + sin(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 2*sin(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 4*sin(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 2*sin(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 4*sin(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)) - 24*pf.y + 24*p0.y)*( 0.2259521484375*p4*cos(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 0.3544921875*p4*cos(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 1.2161865234375*p4*cos(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + 0.375*p4*cos(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 0.796875*p4*cos(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 1.5970458984375*p4*cos(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 0.8701171875*p4*cos(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 1.7669677734375*p4*cos(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th))/288,
                 weights.y*p4*(p4*(sin(p0.th) + 4*sin(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 2*sin(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 4*sin(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + sin(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 2*sin(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 4*sin(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 2*sin(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 4*sin(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)) - 24*pf.y + 24*p0.y)*(-0.0970458984375*p4*cos(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) - 0.1201171875*p4*cos(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) - 0.2669677734375*p4*cos(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + 0.375*p4*cos(0.375*p1*p4 + 0.375*p2*p4 + p0.th) - 0.046875*p4*cos(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 1.2740478515625*p4*cos(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 0.3955078125*p4*cos(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 0.2838134765625*p4*cos(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th))/288,
                 weights.y*   (p4*(sin(p0.th) + 4*sin(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 2*sin(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 4*sin(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + sin(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 2*sin(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 4*sin(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 2*sin(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 4*sin(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)) - 24*pf.y + 24*p0.y)*(2*p4*(4*(0.056488037109375*p1 - 0.024261474609375*p2)*cos(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 2*(0.17724609375*p1 - 0.06005859375*p2)*cos(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 4*(0.304046630859375*p1 - 0.066741943359375*p2)*cos(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + (0.375*p1 + 0.375*p2)*cos(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 2*(0.3984375*p1 - 0.0234375*p2)*cos(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 4*(0.399261474609375*p1 + 0.318511962890625*p2)*cos(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th)  + 2*(0.43505859375*p1 + 0.19775390625*p2)*cos(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 4*(0.441741943359375*p1 + 0.070953369140625*p2)*cos(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th)) + 2*sin(p0.th) + 8*sin(0.056488037109375*p1*p4 - 0.024261474609375*p2*p4 + p0.th) + 4*sin(0.17724609375*p1*p4 - 0.06005859375*p2*p4 + p0.th) + 8*sin(0.304046630859375*p1*p4 - 0.066741943359375*p2*p4 + p0.th) + 2*sin(0.375*p1*p4 + 0.375*p2*p4 + p0.th) + 4*sin(0.3984375*p1*p4 - 0.0234375*p2*p4 + p0.th) + 8*sin(0.399261474609375*p1*p4 + 0.318511962890625*p2*p4 + p0.th) + 4*sin(0.43505859375*p1*p4 + 0.19775390625*p2*p4 + p0.th) + 8*sin(0.441741943359375*p1*p4 + 0.070953369140625*p2*p4 + p0.th))/576;

      // Angular error computeGradient
      grad_the << 0.75*weights.th*p4*(0.375*p1*p4 + 0.375*p2*p4 - pf.th + p0.th),
                  0.75*weights.th*p4*(0.375*p1*p4 + 0.375*p2*p4 - pf.th + p0.th),
                       weights.th*   (0.75*p1 + 0.75*p2)*(0.375*p1*p4 + 0.375*p2*p4 - pf.th + p0.th);

      // Combine into final computeGradient
      return grad_be + grad_xe + grad_ye + grad_the;
    }

  private:
    Pose p0;
    Pose pf;
    Pose weights;
  };

} // namespace optimize

#endif // TEST_H