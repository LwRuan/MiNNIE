#include "tetmesh.h"

#include "math.h"

namespace Rain {
namespace TetMesh {
void GetSurface(uint32_t* indices, Vec3* vertices, uint64_t n_tet,
                uint32_t** surface_indices, uint64_t* n_tri) {
  struct ArrayHasher {
    std::size_t operator()(const std::array<uint32_t, 3>& a) const {
      std::size_t h = 0;
      for (auto e : a) {
        h ^= std::hash<uint32_t>{}(e) + 0x9e3779b9 + (h << 6) + (h >> 2);
      }
      return h;
    }
  };
  std::unordered_map<std::array<uint32_t, 3>, uint32_t, ArrayHasher>
      surface_map;
  for (uint64_t t = 0; t < n_tet; ++t) {
    std::array<uint32_t, 4> sorted_tet =
        SortTetIndices(indices[4 * t], indices[4 * t + 1], indices[4 * t + 2],
                       indices[4 * t + 3]);
    std::array<uint32_t, 3> face{sorted_tet[0], sorted_tet[1], sorted_tet[2]};
    // 0 1 2
    if (surface_map.find(face) != surface_map.end()) {
      surface_map.erase(face);
    } else {
      surface_map.insert(std::make_pair(face, sorted_tet[3]));
    }
    face[2] = sorted_tet[3];  // 0 1 3
    if (surface_map.find(face) != surface_map.end()) {
      surface_map.erase(face);
    } else {
      surface_map.insert(std::make_pair(face, sorted_tet[2]));
    }
    face[1] = sorted_tet[2];  // 0 2 3
    if (surface_map.find(face) != surface_map.end()) {
      surface_map.erase(face);
    } else {
      surface_map.insert(std::make_pair(face, sorted_tet[1]));
    }
    face[0] = sorted_tet[1];  // 1 2 3
    if (surface_map.find(face) != surface_map.end()) {
      surface_map.erase(face);
    } else {
      surface_map.insert(std::make_pair(face, sorted_tet[0]));
    }
  }
  *n_tri = surface_map.size();
  *surface_indices = new uint32_t[3 * surface_map.size()];
  uint64_t idx = 0;
  for (auto iter = surface_map.begin(); iter != surface_map.end(); ++iter) {
    const std::array<uint32_t, 3>& tri = iter->first;
    (*surface_indices)[idx] = tri[0];
    // check normal direction
    Vec3 dx1 = vertices[tri[1]] - vertices[tri[0]];
    Vec3 dx2 = vertices[tri[2]] - vertices[tri[0]];
    Vec3 dx3 = vertices[iter->second] - vertices[tri[0]];
    if (dx3.dot(dx1.cross(dx2)) > 0) {
      (*surface_indices)[idx + 1] = tri[2];
      (*surface_indices)[idx + 2] = tri[1];
    } else {
      (*surface_indices)[idx + 1] = tri[1];
      (*surface_indices)[idx + 2] = tri[2];
    }
    idx += 3;
  }
}

std::array<uint32_t, 4> SortTetIndices(uint32_t v1, uint32_t v2, uint32_t v3,
                                       uint32_t v4) {
  std::array<uint32_t, 4> ret;
  if (v1 < v2)
    ret[0] = v1, ret[1] = v2;
  else
    ret[0] = v2, ret[1] = v1;
  if (v3 < v4)
    ret[2] = v3, ret[3] = v4;
  else
    ret[2] = v4, ret[3] = v3;
  if (ret[0] > ret[2]) std::swap(ret[0], ret[2]);
  if (ret[1] > ret[3]) std::swap(ret[1], ret[3]);
  if (ret[1] > ret[2]) std::swap(ret[1], ret[2]);
  return ret;
}
};  // namespace TetMesh

namespace Tet {
static inline bool SameSide(const Vec3& v1, const Vec3& v2, const Vec3& v3,
                            const Vec3& p1, const Vec3& p2) {
  Vec3 n = (v2 - v1).cross(v3 - v1);
  real d1 = n.dot(p1 - v1);
  real d2 = n.dot(p2 - v1);
  return (std::signbit(d1) == std::signbit(d2));
}

bool VertexInside(const Vec3& p, const Vec3& v1, const Vec3& v2, const Vec3& v3,
                  const Vec3& v4) {
  return SameSide(v1, v2, v3, v4, p) && SameSide(v2, v3, v4, v1, p) &&
         SameSide(v3, v4, v1, v2, p) && SameSide(v4, v1, v2, v3, p);
}
};  // namespace Tet
};  // namespace Rain