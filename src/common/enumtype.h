#pragma once

namespace Rain {
enum class ElasticModelType { StVK, Corotation, NeoHookean, PD, NeoHookeanLog };

// Multigird operation type
enum class MGOpType { Jacobi = 0, GS, RVanka, IUzawa, CG, PCG, Direct, DS, US };

enum class ReductionType { Same, Derivative };

enum class ObjType { Kinematic, Dynamic };

enum class ImplicitGeoType { None, Sphere, Plane, Cylinder, Torus };
}  // namespace Rain
