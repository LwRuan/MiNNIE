#pragma once

#include "imgui.h"
#include "scene/scene.h"

namespace Rain {
class KinematicObject {
 public:
  Object* obj_;
  Vec3* rel_pos_;
  Mat3 rotation_ = Mat3::Identity();
  Vec3 translation_ = Vec3::Zero();
  Vec3 pivot_;
  Vec3 velocity_;
  Vec3 angular_velocity_;
  int start_frame_;
  int end_frame_;

  void Init(Object* obj, const Vec3& pivot, const Vec3& vel,
            const Vec3& angular_vel, int st_frame, int ed_frame);
  void Reset();
  real GetSignedDistance(const Vec3& pos);
  Vec3 GetClosestNormal(const Vec3& pos);
  Vec3 GetClosestSurfacePosition(const Vec3& pos);
  Vec3 GetVelocity(const Vec3& pos, int n_frame);
  void Update(float dt, uint32_t n_frame);
  void ShowUI();
  void Destroy();
};
};  // namespace Rain