#include "triangle.h"

#include "Eigen/Geometry"

namespace Rain {
namespace Triangle {
bool RayIntersect(const Vec3& v0, const Vec3& v1, const Vec3& v2,
                  const Vec3& origin, const Vec3& dir, real& dist) {
  real eps = 1e-7;
  Vec3 e1, e2, h, s, q;
  real a, f, u, v;
  e1 = v1 - v0;
  e2 = v2 - v0;
  h = dir.cross(e2);
  a = e1.dot(h);
  if (a > -eps && a < eps) return false;
  f = 1.0 / a;
  s = origin - v0;
  u = f * s.dot(h);
  if (u < 0 || u > 1) return false;
  q = s.cross(e1);
  v = f * dir.dot(q);
  if (v < 0 || u + v > 1) return false;
  dist = f * e2.dot(q);
  if (dist > eps)
    return true;
  else
    return false;
}

real PointEdgeDistSqr(const Vec3& p, const Vec3& v0, const Vec3& v1,
                      real& weight) {
  Vec3 v01 = v1 - v0;
  Vec3 dp = p - v0;
  real pdotv = dp.dot(v01);
  real vdotv = v01.dot(v01);
  if (pdotv < 0)
    weight = 0;
  else if (pdotv > vdotv)
    weight = 1;
  else
    weight = pdotv / vdotv;
  v01 = weight * v0 + (1 - weight) * v1;
  return (p - v01).squaredNorm();
}

void BaryCentricWeight(const Vec3& p, const Vec3& v1, const Vec3& v2,
                       const Vec3& v3, real& w1, real& w2) {
  real S = (v1 - v3).cross(v2 - v3).norm();
  real S1 = (p - v3).cross(v2 - v3).norm();
  real S2 = (v1 - v3).cross(p - v3).norm();
  w1 = S1 / S;
  if (w1 > 1) w1 = 1;
  w2 = S2 / S;
  if (w2 > 1) w2 = 1;
}

void EdgeEdgeCrossWeight(const Vec3& v1, const Vec3& v2, const Vec3& v3,
                         const Vec3& v4, real& w1, real& w2) {
  Vec3 v21 = v1 - v2;
  Vec3 v43 = v3 - v4;
  Vec3 v24 = v4 - v2;
  real a = v21.dot(v21);
  real b = -v21.dot(v43);
  real d = v43.dot(v43);
  // check parallel
  real deno = a * d - b * b;
  if (std::abs(deno) < 1e-6) {
    w1 = w2 = 0.5;
    return;
  }
  real r1 = v21.dot(v24);
  real r2 = -v43.dot(v24);
  w1 = (d * r1 - b * r2) / deno;
  w2 = (-b * r1 + a * r2) / deno;
  if (w1 < 0)
    w1 = 0;
  else if (w1 > 1)
    w1 = 1;
  if (w2 < 0)
    w2 = 0;
  else if (w2 > 1)
    w2 = 1;
}
};  // namespace Triangle
};  // namespace Rain