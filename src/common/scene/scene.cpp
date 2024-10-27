#include "scene.h"

#include "geometry/triangle.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <Eigen/Geometry>
#include <fstream>

#include "tiny_obj_loader.h"

namespace Rain {
bool Object::Init(const std::string& obj_file, const Mat3f& rot,
                  const Vec3f& trans, float scale) {
  transformation_ = Mat4f::Identity();
  transformation_.block<3, 3>(0, 0) = scale * rot;
  transformation_.block<3, 1>(0, 3) = trans;

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              obj_file.c_str());
  if (!warn.empty()) spdlog::warn(warn);
  if (!err.empty()) spdlog::error(err);
  if (!ret) return ret;
  if (shapes.size() != 1) {
    spdlog::error("multiple shapes in one model");
    return false;
  }
  tinyobj::mesh_t& mesh = shapes[0].mesh;
  n_vert_ = attrib.vertices.size() / 3;
  vertices_ = new Vec3[n_vert_];
  for (size_t i = 0; i < n_vert_; ++i) {
    vertices_[i] = Vec3(attrib.vertices[3 * i], attrib.vertices[3 * i + 1],
                        attrib.vertices[3 * i + 2]);
    vertices_[i] = static_cast<real>(scale) * rot.cast<real>() * vertices_[i] +
                   trans.cast<real>();
  }
  if (attrib.normals.size() > 0) {
    assert(attrib.normals.size() == n_vert_);
    normals_ = new Vec3[n_vert_];
    for (size_t i = 0; i < n_vert_; ++i) {
      normals_[i] = Vec3(attrib.normals[3 * i], attrib.normals[3 * i + 1],
                         attrib.normals[3 * i + 2]);
      normals_[i] = rot.cast<real>() * normals_[i];
    }
  }
  if (attrib.texcoords.size() > 0) {
    n_texc_ = attrib.texcoords.size() / 2;
    texcoords_ = new Vec2f[n_texc_];
    for (size_t i = 0; i < n_texc_; ++i) {
      texcoords_[i] =
          Vec2f(attrib.texcoords[2 * i], attrib.texcoords[2 * i + 1]);
    }
  }
  n_elevert_ = mesh.num_face_vertices[0];
  n_ele_ = mesh.indices.size() / n_elevert_;
  indices_ = new uint32_t[mesh.indices.size()];
  size_t index_offset = 0;
  for (size_t f = 0; f < mesh.num_face_vertices.size(); ++f) {
    assert(mesh.num_face_vertices[f] == n_elevert_);
    for (size_t v = 0; v < n_elevert_; ++v) {
      tinyobj::index_t& idx = mesh.indices[index_offset + v];
      indices_[index_offset + v] = idx.vertex_index;
    }
    index_offset += n_elevert_;
  }
  if (n_elevert_ == 3) {
    n_surfidx_ = n_ele_ * 3;
    n_face_ = n_ele_;
    surface_indices_ = indices_;
  } else if (n_elevert_ == 4) {
    TetMesh::GetSurface(indices_, vertices_, n_ele_, &surface_indices_,
                        &n_face_);
    n_surfidx_ = n_face_ * 3;
  }

  if (attrib.normals.size() == 0) {
    // compute surface normal
    normals_ = new Vec3[n_vert_];
    memset(normals_, 0, n_vert_ * sizeof(Vec3));
    UpdateNormal();
  }
  is_fixed_ = new bool[n_vert_];
  memset(is_fixed_, 0, sizeof(bool) * n_vert_);
  return true;
}

bool Object::InitTet(const std::string& ele_node, const Mat3f& rot,
                     const Vec3f& trans, float scale) {
  transformation_ = Mat4f::Identity();
  transformation_.block<3, 3>(0, 0) = scale * rot;
  transformation_.block<3, 1>(0, 3) = trans;

  std::ifstream ele_file(ele_node + ".ele");
  int n_attr;
  real tmp;
  ele_file >> n_ele_ >> n_elevert_ >> n_attr;
  assert(n_elevert_ == 4);
  uint64_t idx;
  indices_ = new uint32_t[n_ele_ * n_elevert_];
  spdlog::info("#elements: {}", n_ele_);
  for (uint64_t i = 0; i < n_ele_; ++i) {
    ele_file >> idx >> indices_[i * 4] >> indices_[i * 4 + 1] >>
        indices_[i * 4 + 2] >> indices_[i * 4 + 3];
    for (int j = 0; j < n_attr; ++j) ele_file >> tmp;
    assert(idx == i + 1);
  }
  for (uint64_t i = 0; i < n_ele_ * 4; ++i) {
    indices_[i] -= 1;
  }
  ele_file.close();

  std::ifstream node_file(ele_node + ".node");
  int dim, marker;
  node_file >> n_vert_ >> dim >> n_attr >> marker;
  assert(dim == 3);
  // assert(n_attr == 0);
  // assert(marker == 0);
  spdlog::info("#vertices: {}", n_vert_);
  vertices_ = new Vec3[n_vert_];
  bbox_min_ = Vec3::Ones() * 1e16, bbox_max_ = -Vec3::Ones() * 1e16;
  for (uint64_t i = 0; i < n_vert_; ++i) {
    node_file >> idx >> vertices_[i][0] >> vertices_[i][1] >> vertices_[i][2];
    vertices_[i] = static_cast<real>(scale) * rot.cast<real>() * vertices_[i] +
                   trans.cast<real>();
    for (int j = 0; j < n_attr + marker; ++j) node_file >> tmp;
    for (int d = 0; d < 3; ++d) {
      bbox_min_[d] = std::min(bbox_min_[d], vertices_[i][d]);
      bbox_max_[d] = std::max(bbox_max_[d], vertices_[i][d]);
    }
    assert(idx == i + 1);
  }
  spdlog::info("bbox_min: {} {} {}", bbox_min_[0], bbox_min_[1], bbox_min_[2]);
  spdlog::info("bbox_max: {} {} {}", bbox_max_[0], bbox_max_[1], bbox_max_[2]);
  node_file.close();

  // fix inverse tet

  for (uint64_t i = 0; i < n_ele_; ++i) {
    uint32_t* tet = &indices_[i * 4];
    Mat3 Dm;
    Dm << vertices_[tet[0]] - vertices_[tet[3]],
        vertices_[tet[1]] - vertices_[tet[3]],
        vertices_[tet[2]] - vertices_[tet[0]];
    real vol = Dm.determinant() / 6;
    if (vol < 0.) {
      std::swap(tet[0], tet[1]);
    }
  }

  TetMesh::GetSurface(indices_, vertices_, n_ele_, &surface_indices_, &n_face_);
  n_surfidx_ = n_face_ * 3;

  // compute surface normal
  normals_ = new Vec3[n_vert_];
  memset(normals_, 0, n_vert_ * sizeof(Vec3));
  UpdateNormal();
  is_fixed_ = new bool[n_vert_];
  memset(is_fixed_, 0, sizeof(bool) * n_vert_);
  // WriteObj("armadillo.obj");
  return true;
}

bool Object::InitImplicit(ImplicitGeoType type,
                          const std::vector<real>& params) {
  implicit_geo_type_ = type;
  implicit_geo_params_ = params;
  // gen mesh
  if (type == ImplicitGeoType::Sphere) {
    if (params.size() != 6) {
      spdlog::error("incorrect parameters");
      exit(1);
    }
    // params: center.x center.y center.z radius u_divide v_divide
    Vec3 center{params[0], params[1], params[2]};
    real radius = params[3];
    int u_divide = int(params[4]);
    int v_divide = int(params[5]);

    n_vert_ = (v_divide - 1) * u_divide + 2;
    vertices_ = new Vec3[n_vert_];
    normals_ = new Vec3[n_vert_];
    n_ele_ = (2 * v_divide - 2) * u_divide;
    n_elevert_ = 3;
    indices_ = new uint32_t[3 * n_ele_];
    n_surfidx_ = 3 * n_ele_;
    n_face_ = n_ele_;
    surface_indices_ = indices_;

    vertices_[0] = center + radius * Vec3::Unit(1);
    real du = 2 * PI_ / u_divide;
    real dv = PI_ / v_divide;
    int idx = 1;
    for (int i = 1; i < v_divide; ++i) {
      for (int j = 0; j < u_divide; ++j) {
        vertices_[idx] =
            center + Vec3(std::sin(j * du) * std::sin(i * dv), std::cos(i * dv),
                          std::cos(j * du) * std::sin(i * dv)) *
                         radius;
        ++idx;
      }
    }
    vertices_[idx] = center - radius * Vec3::Unit(1);
    idx = 0;
    for (int i = 1; i <= u_divide; ++i) {
      indices_[3 * idx + 0] = 0;
      indices_[3 * idx + 1] = i;
      indices_[3 * idx + 2] = i % u_divide + 1;
      ++idx;
    }
    for (int i = 0; i < v_divide - 2; ++i) {
      for (int j = 1; j <= u_divide; ++j) {
        indices_[3 * idx + 0] = i * u_divide + j;
        indices_[3 * idx + 1] = i * u_divide + j + u_divide;
        indices_[3 * idx + 2] = i * u_divide + j % u_divide + u_divide + 1;
        ++idx;
        indices_[3 * idx + 0] = i * u_divide + j;
        indices_[3 * idx + 1] = i * u_divide + j % u_divide + u_divide + 1;
        indices_[3 * idx + 2] = i * u_divide + j % u_divide + 1;
        ++idx;
      }
    }
    for (int i = 1; i <= u_divide; ++i) {
      indices_[3 * idx + 0] = (v_divide - 2) * u_divide + i;
      indices_[3 * idx + 1] = (v_divide - 1) * u_divide + 1;
      indices_[3 * idx + 2] = (v_divide - 2) * u_divide + i % u_divide + 1;
      ++idx;
    }
  } else if (type == ImplicitGeoType::Plane) {
    if (params.size() != 10) {
      spdlog::error("incorrect parameters");
      exit(1);
    }
    Vec3 c{params[0], params[1], params[2]};
    Vec3 n{params[3], params[4], params[5]};
    real w = params[6];
    real h = params[7];
    int w_divide = int(params[8]);
    int h_divide = int(params[9]);

    n_vert_ = (w_divide + 1) * (h_divide + 1);
    vertices_ = new Vec3[n_vert_];
    normals_ = new Vec3[n_vert_];
    n_ele_ = w_divide * h_divide * 2;
    n_elevert_ = 3;
    indices_ = new uint32_t[3 * n_ele_];
    n_surfidx_ = 3 * n_ele_;
    n_face_ = n_ele_;
    surface_indices_ = indices_;

    Vec3 a = (n.cross(Vec3::Unit(1))).normalized();
    if (a == Vec3::Zero()) a = Vec3::Unit(0);
    Vec3 b = (n.cross(a)).normalized();
    for (int i = 0; i < w_divide + 1; ++i) {
      for (int j = 0; j < h_divide + 1; ++j) {
        real dx = (i - w_divide * 0.5) * w / w_divide;
        real dy = (j - h_divide * 0.5) * h / h_divide;
        vertices_[j * (w_divide + 1) + i] = c + dx * a + dy * b;
      }
    }
    int idx = 0;
    for (int i = 0; i < w_divide; ++i) {
      for (int j = 0; j < h_divide; ++j) {
        indices_[3 * idx + 0] = j * (w_divide + 1) + i;
        indices_[3 * idx + 1] = j * (w_divide + 1) + i + 1;
        indices_[3 * idx + 2] = (j + 1) * (w_divide + 1) + i;
        ++idx;
        indices_[3 * idx + 0] = (j + 1) * (w_divide + 1) + i;
        indices_[3 * idx + 1] = j * (w_divide + 1) + i + 1;
        indices_[3 * idx + 2] = (j + 1) * (w_divide + 1) + i + 1;
        ++idx;
      }
    }
  } else if (type == ImplicitGeoType::Cylinder) {
    Vec3 c{params[0], params[1], params[2]};
    Vec3 n{params[3], params[4], params[5]};
    real r = params[6];
    real h = params[7];
    int r_divide = int(params[8]);
    int h_divide = int(params[9]);

    n_vert_ = (h_divide + 1) * r_divide + 2;
    vertices_ = new Vec3[n_vert_];
    normals_ = new Vec3[n_vert_];
    n_ele_ = r_divide * (2 * h_divide + 2);
    n_elevert_ = 3;
    indices_ = new uint32_t[3 * n_ele_];
    n_surfidx_ = 3 * n_ele_;
    n_face_ = n_ele_;
    surface_indices_ = indices_;

    vertices_[0] = c + h / 2 * n;
    int idx = 1;
    Vec3 a = (n.cross(Vec3::Unit(1))).normalized();
    if (a == Vec3::Zero()) a = Vec3::Unit(0);
    Vec3 b = (n.cross(a)).normalized();
    real dtheta = 2 * PI_ / r_divide;
    for (int i = 0; i < h_divide + 1; ++i) {
      for (int j = 0; j < r_divide; ++j) {
        vertices_[idx] = c + (- i + h_divide * 0.5) * h / h_divide * n;
        vertices_[idx] += (std::cos(j * dtheta) * a + std::sin(j * dtheta) * b) * r;
        ++idx;
      }
    }
    vertices_[idx] = c - h / 2 * n;
    idx = 0;
    for (int i = 1; i <= r_divide; ++i) {
      indices_[3 * idx + 0] = 0;
      indices_[3 * idx + 1] = i;
      indices_[3 * idx + 2] = i % r_divide + 1;
      ++idx;
    }
    for (int i = 0; i < h_divide; ++i) {
      for (int j = 1; j <= r_divide; ++j) {
        indices_[idx * 3 + 0] = i * r_divide + j;
        indices_[idx * 3 + 1] = (i + 1) * r_divide + j;
        indices_[idx * 3 + 2] = (i + 1) * r_divide + j % r_divide + 1;
        ++idx;
        indices_[idx * 3 + 0] = i * r_divide + j;
        indices_[idx * 3 + 1] = (i + 1) * r_divide + j % r_divide + 1;
        indices_[idx * 3 + 2] = i * r_divide + j % r_divide + 1;
        ++idx;
      }
    }
    for (int i = 1; i <= r_divide; ++i) {
      indices_[3 * idx + 0] = h_divide * r_divide + i;
      indices_[3 * idx + 1] = (h_divide + 1) * r_divide + 1;
      indices_[3 * idx + 2] = h_divide * r_divide + i % r_divide + 1;
      ++idx;
    }
  } else if (type == ImplicitGeoType::Torus) {
    Vec3 center {params[0], params[1], params[2]};
    Vec3 normal {params[3], params[4], params[5]};
    real a = params[6];
    real r = params[7];
    int a_divide = int(params[8]);
    int r_divide = int(params[9]);
    
    n_vert_ = a_divide * r_divide;
    vertices_ = new Vec3[n_vert_];
    normals_ = new Vec3[n_vert_];
    n_ele_ = a_divide * r_divide * 2;
    n_elevert_ = 3;
    indices_ = new uint32_t[3 * n_ele_];
    n_surfidx_ = 3 * n_ele_;
    n_face_ = n_ele_;
    surface_indices_ = indices_;

    Vec3 x = normal.cross(Vec3::Unit(1)).normalized();
    if (x == Vec3::Zero()) x = Vec3::Unit(0);
    Vec3 y = normal.cross(x);

    real dphi = 2 * PI_ / a_divide;
    real dtheta = 2 * PI_ / r_divide;
    
    for (int i = 0; i < a_divide; ++i) {
      Vec3 cx = std::cos(i * dphi) * x + std::sin(i * dphi) * y;
      Vec3 ca = center + a * cx;
      for (int j = 0; j < r_divide; ++j) {
        vertices_[i * r_divide + j] = ca + r * std::cos(j * dtheta) * cx + r * std::sin(j * dtheta) * normal;
      }
    }
    int idx = 0;
    for (int i = 0; i < a_divide; ++i) {
      for (int j = 0; j < r_divide; ++j) {
        indices_[3 * idx + 0] = i * r_divide + j;
        indices_[3 * idx + 1] = ((i + 1) % a_divide) * r_divide + (j + 1) % r_divide;
        indices_[3 * idx + 2] = i * r_divide + (j + 1) % r_divide;
        ++idx;
        indices_[3 * idx + 0] = i * r_divide + j;
        indices_[3 * idx + 1] = ((i + 1) % a_divide) * r_divide + j;
        indices_[3 * idx + 2] = ((i + 1) % a_divide) * r_divide + (j + 1) % r_divide;
        ++idx;
      }
    }
  } else {
    spdlog::error("unknown implicit geometry type");
    exit(1);
  }
  UpdateNormal();
  return true;
}

void Object::UpdateNormal() {
  for (size_t f = 0; f < n_face_; ++f) {
    uint32_t v1 = surface_indices_[3 * f];
    uint32_t v2 = surface_indices_[3 * f + 1];
    uint32_t v3 = surface_indices_[3 * f + 2];
    Vec3 dx1 = vertices_[v2] - vertices_[v1];
    Vec3 dx2 = vertices_[v3] - vertices_[v1];
    Vec3 normal = (dx1.cross(dx2)).normalized();
    normals_[v1] += normal;
    normals_[v2] += normal;
    normals_[v3] += normal;
  }
  for (size_t i = 0; i < n_vert_; ++i) {
    normals_[i] = normals_[i].normalized();
  }
}

void Object::WriteObj(const std::string& obj_file) {
  Eigen::Map<const Eigen::Matrix<real, -1, 3, Eigen::RowMajor>> _vertices(
      reinterpret_cast<const real*>(vertices_), n_vert_, 3);
  Eigen::Map<const Eigen::Matrix<uint32_t, -1, 3, Eigen::RowMajor>> _faces(
      surface_indices_, n_face_, 3);
  std::ofstream s(obj_file);
  if (!s.is_open()) {
    spdlog::error("can't open file {}", obj_file);
    return;
  }
  if (colors_) {
    for (uint32_t i = 0; i < n_vert_; ++i) {
      s << "v " << vertices_[i](0) << " " << vertices_[i](1) << " "
        << vertices_[i](2) << " " << colors_[i](0) << " " << colors_[i](1)
        << " " << colors_[i](2) << std::endl;
    }
  } else {
    s << _vertices.format(Eigen::IOFormat(Eigen::FullPrecision,
                                          Eigen::DontAlignCols, " ", "\n", "v ",
                                          "", "", "\n"));
  }

  s << (_faces.array() + 1)
           .format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols,
                                   " ", "\n", "f ", "", "", "\n"));
  s.close();
}

bool Object::SelectVert(const Vec3& origin, const Vec3& dir) {
  selected_idx_.reset();
  real dist = std::numeric_limits<real>::max();
  std::optional<uint32_t> selected_face;
  for (uint32_t i = 0; i < n_face_; ++i) {
    real tmp = std::numeric_limits<real>::max();
    uint32_t* tri = &surface_indices_[3 * i];
    bool inter = Triangle::RayIntersect(vertices_[tri[0]], vertices_[tri[1]],
                                        vertices_[tri[2]], origin, dir, tmp);
    if (inter && tmp < dist) {
      selected_face = i;
      dist = tmp;
    }
  }
  if (selected_face.has_value()) {
    uint32_t* tri = &surface_indices_[3 * selected_face.value()];
    Vec3 q = origin + 1.2 * dist * dir;
    real w;
    real d0 = Triangle::PointEdgeDistSqr(vertices_[tri[0]], q, origin, w);
    real d1 = Triangle::PointEdgeDistSqr(vertices_[tri[1]], q, origin, w);
    real d2 = Triangle::PointEdgeDistSqr(vertices_[tri[2]], q, origin, w);
    if (d0 < d1 && d0 < d2) {
      selected_idx_ = tri[0];
      control_pos_ = origin + (vertices_[tri[0]] - origin).dot(dir) * dir;
    } else if (d1 < d2) {
      selected_idx_ = tri[1];
      control_pos_ = origin + (vertices_[tri[1]] - origin).dot(dir) * dir;
    } else {
      selected_idx_ = tri[2];
      control_pos_ = origin + (vertices_[tri[2]] - origin).dot(dir) * dir;
    }
    return true;
  } else
    return false;
}

void Object::Destroy() {
  if (vertices_) delete[] vertices_;
  if (normals_) delete[] normals_;
  if (colors_) delete[] colors_;
  if (texcoords_) delete[] texcoords_;
  if (indices_) delete[] indices_;
  if ((n_elevert_ == 4) && surface_indices_) delete[] surface_indices_;
  if (is_fixed_) delete[] is_fixed_;
}

void Scene::Init(const YAML::Node& obj_nodes) {
  for (const auto& obj_node : obj_nodes) {
    std::string name = obj_node["name"].as<std::string>();
    spdlog::info("name: {}", name);
    std::string mesh_type = obj_node["mesh_type"].as<std::string>();
    Mat3f rotation = Mat3f::Identity();
    if (obj_node["rotation_angle"]) {
      float angle = obj_node["rotation_angle"].as<float>();
      Vec3f axis = obj_node["rotation_axis"].as<Vec3f>();
      rotation = Eigen::AngleAxisf(angle / 180 * 3.1415926F, axis).toRotationMatrix();
    }
    Vec3f translation = Vec3f::Zero();
    if (obj_node["translation"])
      translation = obj_node["translation"].as<Vec3f>();
    float scale = 1.0f;
    if (obj_node["scale"]) scale = obj_node["scale"].as<float>();
    Object tmp;
    tmp.name_ = name;
    if (mesh_type == "tet_ele_node") {
      std::string mesh_file = obj_node["mesh"].as<std::string>();
      tmp.InitTet(mesh_file, rotation, translation, scale);
    } else if (mesh_type == "obj") {
      std::string mesh_file = obj_node["mesh"].as<std::string>();
      tmp.Init(mesh_file, rotation, translation, scale);
    } else if (mesh_type == "implicit") {
      std::string type_name = obj_node["implicit_geo_type"].as<std::string>();
      if (obj_node["implicit_geo_sign"]) {
        tmp.implicit_geo_sign_ = obj_node["implicit_geo_sign"].as<int32_t>();
      }
      std::vector<real> params =
          obj_node["implicit_geo_params"].as<std::vector<real>>();
      if (type_name == "Sphere") {
        tmp.InitImplicit(ImplicitGeoType::Sphere, params);
      } else if (type_name == "Plane") {
        tmp.InitImplicit(ImplicitGeoType::Plane, params);
      } else if (type_name == "Cylinder") {
        tmp.InitImplicit(ImplicitGeoType::Cylinder, params);
      } else if (type_name == "Torus") {
        tmp.InitImplicit(ImplicitGeoType::Torus, params);
      } else {
        spdlog::error("unknown implicit geometry type");
        return;
      }
    } else {
      spdlog::error("unknown mesh type");
      return;
    }
    if (obj_node["color_diffuse"])
      tmp.material_.Ka_ = obj_node["color_diffuse"].as<Vec3f>();
    if (obj_node["fixed_vertices"]) {
      for (size_t i = 0; i < obj_node["fixed_vertices"].size(); ++i) {
        uint32_t idx = obj_node["fixed_vertices"][i].as<uint32_t>();
        tmp.is_fixed_[idx] = true;
      }
    }
    if (obj_node["obj_type"]) {
      std::string obj_type_name = obj_node["obj_type"].as<std::string>();
      if (obj_type_name == "dynamic") {
        tmp.obj_type_ = ObjType::Dynamic;
        spdlog::info("dynamic object");
      } else if (obj_type_name == "kinematic") {
        tmp.obj_type_ = ObjType::Kinematic;
        spdlog::info("kinematic object");
      } else {
        spdlog::error("unsupport obj type: {}", obj_type_name);
        exit(1);
      }
    }
    objects_.push_back(tmp);
  }
}

bool Scene::SelectObjVert(const Vec3& origin, const Vec3& dir) {
  selected_obj_idx_.reset();
  bool ret = false;
  real dist = std::numeric_limits<real>::max();
  for (uint32_t i = 0; i < objects_.size(); ++i) {
    if (objects_[i].obj_type_ != ObjType::Dynamic) continue;
    bool tmp = objects_[i].SelectVert(origin, dir);
    if (tmp) {
      real tdist =
          (objects_[i].vertices_[objects_[i].selected_idx_.value()] - origin)
              .norm();
      if (tdist < dist) {
        dist = tdist;
        selected_obj_idx_ = i;
      }
    }
  }
  return selected_obj_idx_.has_value();
}

void Scene::ClearSelectObjVert() {
  if (selected_obj_idx_.has_value()) {
    auto& obj = objects_[selected_obj_idx_.value()];
    obj.selected_idx_.reset();
    selected_obj_idx_.reset();
  }
}

void Scene::Destroy() {
  for (Object obj : objects_) {
    obj.Destroy();
  }
}
};  // namespace Rain