#pragma once

#include <array>
#include <unordered_map>

#include "mathtype.h"

namespace Rain {
namespace TetMesh {
void GetSurface(uint32_t* indices, Vec3* vertices, uint64_t n_tet,
                uint32_t** surface_indices, uint64_t* n_tri);
std::array<uint32_t, 4> SortTetIndices(uint32_t v1, uint32_t v2, uint32_t v3,
                                       uint32_t v4);
};  // namespace TetMesh

namespace Tet {
bool VertexInside(const Vec3& p, const Vec3& v1, const Vec3& v2, const Vec3& v3,
                  const Vec3& v4);
};
};  // namespace Rain