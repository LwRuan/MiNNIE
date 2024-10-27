#include "camera.h"

#include <Eigen/Dense>
#include <cmath>

namespace Rain {
// VkResult Camera::CreateBuffer(Device* device, SwapChain* swap_chain) {
//   aspect_ = float(swap_chain->extent_.width) / swap_chain->extent_.height;
//   VkResult result;
//   VkDeviceSize size = sizeof(CameraData);

//   buffers_.resize(swap_chain->images_.size());
//   for (uint32_t i = 0; i < swap_chain->images_.size(); ++i) {
//     result = buffers_[i].Allocate(device, nullptr, size,
//                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
//     if (result != VK_SUCCESS) return result;
//   }

//   return VK_SUCCESS;
// }

void Camera::InitData(const float aspect, const float fovy, const float z_near,
                      const float z_far, const float radius, const float phi,
                      const float theta, const Vec3f& target) {
  aspect_ = aspect;
  saved_radius_ = radius;
  saved_phi_ = phi;
  saved_theta_ = theta;
  saved_target_ = target;
  SetPerspective(fovy, z_near, z_far);
  SetSpherical(radius, phi, theta, target);
  UpdateData();
}

void Camera::Rotate(const float dx, const float dy) {
  static constexpr float rotate_ratio = 0.25f * PI_ / 180.0f;
  phi_ -= rotate_ratio * dx;
  theta_ -= rotate_ratio * dy;
  phi_ = std::fmod(phi_, 2.0f * PI_);
  if (phi_ < 0.0f) phi_ += 2.0f * PI_;
  theta_ = std::clamp(theta_, 0.1f, PI_ - 0.1f);
  LookAt(SphericalToCartesian(radius_, phi_, theta_), target_);
}

void Camera::Translate(const float dx, const float dy) {
  static constexpr float trans_ratio = 8e-4f;
  target_ += trans_ratio * radius_ * (dy * up_ - dx * right_);
  LookAt(SphericalToCartesian(radius_, phi_, theta_), target_);
}

void Camera::Scale(const float dy) {
  static constexpr float scale_ratio = 0.05f;
  radius_ /= std::exp(scale_ratio * dy);
  radius_ = std::clamp(radius_, 0.1f, 150.0f);
  LookAt(SphericalToCartesian(radius_, phi_, theta_), target_);
}

void Camera::ResetAspect(const float aspect) {
  aspect_ = aspect;
  proj_dirty_ = true;
}

void Camera::UpdateData() {
  if (proj_dirty_) {
    float y_scale = 1.0f / std::tan(fovy_ / 2);
    float x_scale = y_scale / aspect_;
    // opengl style: z \in [-1, 1]
    // proj_ << x_scale, 0, 0, 0, 0, y_scale, 0, 0, 0, 0,
    //     -(z_far_ + z_near_) / (z_far_ - z_near_),
    //     -2 * z_near_ * z_far_ / (z_far_ - z_near_), 0, 0, -1, 0;

    // Vulkan style: z \in [0, 1]
    proj_ << x_scale, 0, 0, 0, 0, y_scale, 0, 0, 0, 0,
        -z_far_ / (z_far_ - z_near_), -z_near_ * z_far_ / (z_far_ - z_near_), 0,
        0, -1, 0;
  }
  if (view_dirty_) {
    view_ << right_.x(), right_.y(), right_.z(), -right_.dot(pos_), up_.x(),
        up_.y(), up_.z(), -up_.dot(pos_), -lookat_.x(), -lookat_.y(),
        -lookat_.z(), lookat_.dot(pos_), 0, 0, 0, 1;
  }
  if (proj_dirty_ || view_dirty_) {
    proj_view_ = proj_ * view_;
  }
  proj_dirty_ = false;
  view_dirty_ = false;
}

void Camera::LookAt(const Vec3f& pos, const Vec3f& target) {
  pos_ = pos;
  target_ = target;
  lookat_ = (target - pos).normalized();
  right_ = lookat_.cross(Vec3f::Unit(1)).normalized();
  up_ = right_.cross(lookat_).normalized();

  view_dirty_ = true;
}

void Camera::SetPerspective(const float fovy, const float z_near,
                            const float z_far) {
  fovy_ = fovy;
  z_near_ = z_near;
  z_far_ = z_far;
  proj_dirty_ = true;
}

void Camera::SetSpherical(const float radius, const float phi,
                          const float theta, const Vec3f& target) {
  radius_ = radius;
  phi_ = phi;
  theta_ = theta;
  LookAt(SphericalToCartesian(radius, phi, theta), target);
}

void Camera::GetSelectRay(const Vec2f& mouse_pos, Vec3f& origin, Vec3f& dir) {
  UpdateData();
  origin = pos_;
  Vec4f q(mouse_pos.x(), mouse_pos.y(), 0, 1);
  Vec4f p = proj_view_.inverse() * q;
  dir = (p.segment<3>(0) / p(3) - pos_).normalized();
}

// void Camera::Destroy(VkDevice device) {
//   for (auto buffer : buffers_) {
//     buffer.Destroy(device);
//   }
// }
};  // namespace Rain