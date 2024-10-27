#pragma once

#include <spdlog/spdlog.h>

#include <optional>
#include <string>
#include <vector>

#include "enumtype.h"
#include "geometry/tetmesh.h"
#include "helper/yaml.h"
#include "mathtype.h"

namespace Rain {
struct Material {
  Vec3f Ka_ = Vec3f(0.2f, 0.2f, 0.2f);  // ambient color
  Vec3f Kd_ = Vec3f(0.8f, 0.8f, 0.8f);  // diffuse color
  Vec3f Ks_ = Vec3f(1.0f, 1.0f, 1.0f);  // specular color
  float d_ = 1.0f;                      // non-transparency
  float Ns_ = 0.0f;                     // shininess
};

class RenderModel;
class Object {
 public:
  std::string name_;
  uint64_t n_vert_ = 0;
  Vec3* vertices_ = nullptr;
  Vec3* normals_ = nullptr;
  Vec3* colors_ = nullptr;
  uint64_t n_texc_ = 0;
  Vec2f* texcoords_ = nullptr;
  uint64_t n_ele_ = 0;
  uint32_t n_elevert_ = 3;  // 3 for triangle, 4 for tet
  uint32_t* indices_ = nullptr;
  uint64_t n_surfidx_;
  uint64_t n_face_;
  uint32_t* surface_indices_ = nullptr;
  bool* is_fixed_ = nullptr;
  Vec3 bbox_min_;
  Vec3 bbox_max_;
  Material material_;
  Mat4f transformation_;

  RenderModel* render_model_;
  ObjType obj_type_ = ObjType::Dynamic;
  ImplicitGeoType implicit_geo_type_ = ImplicitGeoType::None;
  std::vector<real> implicit_geo_params_;
  int32_t implicit_geo_sign_ = 1;

  std::optional<uint32_t> selected_idx_;
  Vec3 control_pos_;

  bool Init(const std::string& obj_file, const Mat3f& rot, const Vec3f& trans,
            float scale);
  void UpdateNormal();
  void WriteObj(const std::string& obj_file);
  bool InitTet(const std::string& ele_node, const Mat3f& rot,
               const Vec3f& trans, float scale);
  bool InitImplicit(ImplicitGeoType type, const std::vector<real>& params);
  bool SelectVert(const Vec3& origin, const Vec3& dir);
  void Destroy();
};

class Scene {
 public:
  std::vector<Object> objects_;
  std::optional<uint32_t> selected_obj_idx_;
  void Init(const YAML::Node& objs);
  bool SelectObjVert(const Vec3& origin, const Vec3& dir);
  void ClearSelectObjVert();
  void Destroy();
};
};  // namespace Rain