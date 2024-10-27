#include "kinematicobject.h"
#include <Eigen/Geometry>

namespace Rain {
void KinematicObject::Init(Object* obj, const Vec3& pivot, const Vec3& vel,
                           const Vec3& angular_vel, int st_frame, int ed_frame) {
  obj_ = obj;
  pivot_ = pivot;
  velocity_ = vel;
  angular_velocity_ = angular_vel;
  rel_pos_ = new Vec3[obj_->n_vert_];
  for (uint32_t i = 0; i < obj_->n_vert_; ++i) {
    rel_pos_[i] = obj_->vertices_[i] - pivot_;
  }
  translation_ = pivot_;
  start_frame_ = st_frame;
  end_frame_ = ed_frame;
}

void KinematicObject::Reset() {
  rotation_ = Mat3::Identity();
  translation_ = pivot_;
  for (uint32_t i = 0; i < obj_->n_vert_; ++i) {
    obj_->vertices_[i] = pivot_ + rel_pos_[i];
  }
  obj_->UpdateNormal();
}

void KinematicObject::Update(float dt, uint32_t n_frame) {
  if (n_frame < start_frame_) return;
  if ((end_frame_ != -1) && n_frame >= end_frame_) return;
  real theta = angular_velocity_.norm() * dt;
  Vec3 n = angular_velocity_.normalized();
  Mat3 cross;
  cross << 0, -n.z(), n.y(), n.z(), 0, -n.x(), -n.y(), n.x(), 0;
  Mat3 r = std::cos(theta) * Mat3::Identity() + std::sin(theta) * cross +
           (1 - std::cos(theta)) * n * n.transpose();
  rotation_ = r * rotation_;
  translation_ = translation_ + velocity_ * dt;
  for (uint32_t i = 0; i < obj_->n_vert_; ++i) {
    obj_->vertices_[i] = translation_ + rotation_ * rel_pos_[i];
  }
}

real KinematicObject::GetSignedDistance(const Vec3& pos) {
  if (obj_->implicit_geo_type_ == ImplicitGeoType::None) {
    // !TODO: explicit mesh sdf
    return 0;
  }
  Vec3 p = rotation_.transpose() * (pos - translation_) + pivot_;
  if (obj_->implicit_geo_type_ == ImplicitGeoType::Sphere) {
    Vec3 center{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    real radius = obj_->implicit_geo_params_[3];
    return (p - center).norm() - radius;
  } else if (obj_->implicit_geo_type_ == ImplicitGeoType::Plane) {
    Vec3 center{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    Vec3 normal{obj_->implicit_geo_params_[3], obj_->implicit_geo_params_[4],
                obj_->implicit_geo_params_[5]};
    return (p - center).dot(normal);
  } else if (obj_->implicit_geo_type_ == ImplicitGeoType::Cylinder) {
    Vec3 c{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    Vec3 n{obj_->implicit_geo_params_[3], obj_->implicit_geo_params_[4],
                obj_->implicit_geo_params_[5]};
    real r = obj_->implicit_geo_params_[6];
    real h = obj_->implicit_geo_params_[7];
    Vec3 pz = (p - c).dot(n) * n;
    Vec3 pr = p - c - pz;
    // !TODO: ignore h
    return pr.norm() - r;
  } else if (obj_->implicit_geo_type_ == ImplicitGeoType::Torus) {
    Vec3 c{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    Vec3 n{obj_->implicit_geo_params_[3], obj_->implicit_geo_params_[4],
                obj_->implicit_geo_params_[5]};
    real a = obj_->implicit_geo_params_[6];
    real r = obj_->implicit_geo_params_[7];
    real y = (p - c).dot(n);
    Vec3 dp = (p - c) - y * n;
    real x = dp.norm();
    return std::sqrt((x - a) * (x - a) + y * y) - r;
  }
}

Vec3 KinematicObject::GetClosestNormal(const Vec3& pos) {
  if (obj_->implicit_geo_type_ == ImplicitGeoType::None) {
    // !TODO: explicit mesh sdf
    return Vec3::Zero();
  }
  Vec3 p = rotation_.transpose() * (pos - translation_) + pivot_;
  if (obj_->implicit_geo_type_ == ImplicitGeoType::Sphere) {
    Vec3 center{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    return (rotation_ * (p - center)).normalized();
  } else if (obj_->implicit_geo_type_ == ImplicitGeoType::Plane) {
    Vec3 normal{obj_->implicit_geo_params_[3], obj_->implicit_geo_params_[4],
                obj_->implicit_geo_params_[5]};
    return (rotation_ * normal).normalized();
  } else if (obj_->implicit_geo_type_ == ImplicitGeoType::Cylinder) {
    Vec3 c{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    Vec3 n{obj_->implicit_geo_params_[3], obj_->implicit_geo_params_[4],
                obj_->implicit_geo_params_[5]};
    real r = obj_->implicit_geo_params_[6];
    real h = obj_->implicit_geo_params_[7];
    Vec3 pz = (p - c).dot(n) * n;
    Vec3 pr = p - c - pz;
    return (rotation_ * pr).normalized();
  } else if (obj_->implicit_geo_type_ == ImplicitGeoType::Torus) {
    Vec3 c{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    Vec3 n{obj_->implicit_geo_params_[3], obj_->implicit_geo_params_[4],
                obj_->implicit_geo_params_[5]};
    real a = obj_->implicit_geo_params_[6];
    real r = obj_->implicit_geo_params_[7];
    real y = (p - c).dot(n);
    Vec3 dp = (p - c) - y * n;
    real x = dp.norm();
    Vec3 normal = (x - a) * dp.normalized() + y * n;
    return (rotation_ * normal).normalized();
  }
}

Vec3 KinematicObject::GetClosestSurfacePosition(const Vec3& pos) {
  if (obj_->implicit_geo_type_ == ImplicitGeoType::None) {
    // !TODO: explicit mesh sdf
    return Vec3::Zero();
  }
  Vec3 p = rotation_.transpose() * (pos - translation_) + pivot_;
  if (obj_->implicit_geo_type_ == ImplicitGeoType::Sphere) {
    Vec3 center{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    real radius = obj_->implicit_geo_params_[3];
    Vec3 q = (p - center).normalized() * radius + center;
    return rotation_ * (q - pivot_) + translation_;
  } else if (obj_->implicit_geo_type_ == ImplicitGeoType::Plane) {
    Vec3 center{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    Vec3 normal{obj_->implicit_geo_params_[3], obj_->implicit_geo_params_[4],
                obj_->implicit_geo_params_[5]};
    Vec3 q = p - (p - center).dot(normal) * normal;
    return rotation_ * (q - pivot_) + translation_;
  } else if (obj_->implicit_geo_type_ == ImplicitGeoType::Cylinder) {
    Vec3 c{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    Vec3 n{obj_->implicit_geo_params_[3], obj_->implicit_geo_params_[4],
                obj_->implicit_geo_params_[5]};
    real r = obj_->implicit_geo_params_[6];
    real h = obj_->implicit_geo_params_[7];
    Vec3 pz = (p - c).dot(n) * n;
    Vec3 pr = p - c - pz;
    Vec3 q = c + pz + pr.normalized() * r;
    return rotation_ * (q - pivot_) + translation_;
  } else if (obj_->implicit_geo_type_ == ImplicitGeoType::Torus) {
    Vec3 c{obj_->implicit_geo_params_[0], obj_->implicit_geo_params_[1],
                obj_->implicit_geo_params_[2]};
    Vec3 n{obj_->implicit_geo_params_[3], obj_->implicit_geo_params_[4],
                obj_->implicit_geo_params_[5]};
    real a = obj_->implicit_geo_params_[6];
    real r = obj_->implicit_geo_params_[7];
    real y = (p - c).dot(n);
    Vec3 dp = (p - c) - y * n;
    real x = dp.norm();
    real s = std::sqrt((x - a) * (x - a) + y * y);
    Vec3 q = ((x - a) / s * r + a) * dp.normalized() + y / s * r * n + c;
    return rotation_ * (q - pivot_) + translation_;
  }
}

Vec3 KinematicObject::GetVelocity(const Vec3& pos, int n_frame) {
  if (n_frame < start_frame_) return Vec3::Zero();
  if ((end_frame_ != -1) && n_frame >= end_frame_) return Vec3::Zero();
  return angular_velocity_.cross(pos - translation_) + velocity_;
}

void KinematicObject::Destroy() { delete[] rel_pos_; }

void KinematicObject::ShowUI() { ImGui::Text("Kinematic Object"); }
}  // namespace Rain