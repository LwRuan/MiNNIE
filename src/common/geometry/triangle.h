#pragma once

#include "mathtype.h"

namespace Rain {
namespace Triangle {
bool RayIntersect(const Vec3& v0, const Vec3& v1, const Vec3& v2,
                  const Vec3& origin, const Vec3& dir, real& dist);
real PointEdgeDistSqr(const Vec3& p, const Vec3& v0, const Vec3& v1,
                      real& weight);
void BaryCentricWeight(const Vec3& p, const Vec3& v1, const Vec3& v2,
                       const Vec3& v3, real& w1, real& w2);
void EdgeEdgeCrossWeight(const Vec3& v1, const Vec3& v2, const Vec3& v3,
                       const Vec3& v4, real& w1, real& w2);
};  // namespace Triangle
};  // namespace Rain