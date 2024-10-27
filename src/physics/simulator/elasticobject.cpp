#include "elasticobject.h"

#include <Eigen/Dense>

namespace Rain {
void ElasticObject::Init(Object* obj, real density, ElasticModelType type,
                         real young, real poisson) {
  obj_ = obj;
  n_tet_ = (uint32_t)obj_->n_ele_;
  tets_ = obj_->indices_;
  n_vert_ = (uint32_t)obj_->n_vert_;
  verts_ = obj_->vertices_;
  initial_vertices_ = new Vec3[obj_->n_vert_];
  memcpy(initial_vertices_, obj_->vertices_, sizeof(Vec3) * obj_->n_vert_);
  Dm_inv_ = new Mat3[obj_->n_ele_];
  volumes_ = new real[obj_->n_ele_];
  for (size_t e = 0; e < obj_->n_ele_; ++e) {
    const uint32_t& v1 = obj_->indices_[4 * e];
    const uint32_t& v2 = obj_->indices_[4 * e + 1];
    const uint32_t& v3 = obj_->indices_[4 * e + 2];
    const uint32_t& v4 = obj_->indices_[4 * e + 3];
    Mat3 Dm;
    Dm.col(0) = initial_vertices_[v1] - initial_vertices_[v4];
    Dm.col(1) = initial_vertices_[v2] - initial_vertices_[v4];
    Dm.col(2) = initial_vertices_[v3] - initial_vertices_[v4];
    volumes_[e] = std::abs(Dm.determinant()) / 6.0;
    Dm_inv_[e] = Dm.inverse();
  }
  masses_ = new real[obj_->n_vert_];
  memset(masses_, 0, sizeof(real) * obj_->n_vert_);
  for (size_t e = 0; e < obj_->n_ele_; ++e) {
    for (size_t i = 0; i < 4; ++i) {
      const auto& idx = obj_->indices_[4 * e + i];
      masses_[idx] += density_ * volumes_[e] / 4.0;
    }
  }

  velocities_ = new Vec3[obj_->n_vert_];
  memset(velocities_, 0, sizeof(Vec3) * obj_->n_vert_);
  forces_ = new Vec3[obj_->n_vert_];
  memset(forces_, 0, sizeof(Vec3) * obj_->n_vert_);
  young_ = young;
  poisson_ = poisson;
  switch (type) {
    case ElasticModelType::StVK:
      elastic_model_ = new StVK;
      elastic_model_->SetLame(young, poisson);
      break;
    case ElasticModelType::PD:
      elastic_model_ = new PD;
      elastic_model_->SetLame(young, poisson);
      break;
    case ElasticModelType::NeoHookean:
      elastic_model_ = new NeoHookean;
      elastic_model_->SetLame(young, poisson);
      break;
    default:
      break;
  }
}

void ElasticObject::Reset() {
  memset(velocities_, 0, sizeof(Vec3) * obj_->n_vert_);
  memset(forces_, 0, sizeof(Vec3) * obj_->n_vert_);
  memcpy(obj_->vertices_, initial_vertices_, sizeof(Vec3) * obj_->n_vert_);
  // for (uint32_t v = 0; v < n_vert_; ++v) {
  //   // obj_->vertices_[v](1) = 0.6 * initial_vertices_[v](1);
  //   obj_->vertices_[v] = 10 * Vec3::Unit(1) + 8 * Vec3::Random();
  // }
  // for (uint32_t v = 0; v < n_vert_; ++v) {
  //   velocities_[v] = -15 * Vec3::Unit(1);
  // }
  obj_->UpdateNormal();
}

void ElasticObject::Update(float Dt, const Vec3& grav, real damping,
                           uint32_t n_substep, real substep_size,
                           uint32_t n_frame,
                           const std::vector<KinematicObject*>& kobjs) {
  for (int step = 0; step < n_substep; ++step) {
    real dt = std::min((real)Dt / n_substep, substep_size);
    // compute force
    memset(forces_, 0, sizeof(Vec3) * obj_->n_vert_);
    for (size_t e = 0; e < obj_->n_ele_; ++e) {
      const uint32_t& v1 = obj_->indices_[4 * e];
      const uint32_t& v2 = obj_->indices_[4 * e + 1];
      const uint32_t& v3 = obj_->indices_[4 * e + 2];
      const uint32_t& v4 = obj_->indices_[4 * e + 3];
      Mat3 Ds;
      Ds.col(0) = obj_->vertices_[v1] - obj_->vertices_[v4];
      Ds.col(1) = obj_->vertices_[v2] - obj_->vertices_[v4];
      Ds.col(2) = obj_->vertices_[v3] - obj_->vertices_[v4];
      Mat3 Piola;
      elastic_model_->GetPiola(Ds * Dm_inv_[e], Piola);
      Mat3 H = -volumes_[e] * Piola * Dm_inv_[e].transpose();
      forces_[v1] += H.col(0);
      forces_[v2] += H.col(1);
      forces_[v3] += H.col(2);
      forces_[v4] += -H.col(0) - H.col(1) - H.col(2);
    }
    // update velocity and position
    for (size_t i = 0; i < obj_->n_vert_; ++i) {
      if (obj_->is_fixed_[i]) continue;
      velocities_[i] += (forces_[i] / masses_[i] + grav) * dt;
      velocities_[i] -= damping * velocities_[i];
      obj_->vertices_[i] += velocities_[i] * dt;
    }
  }
  obj_->UpdateNormal();
}

void ElasticObject::ShowUI() {
  ImGui::Text("density: %.2e", density_);
  ImGui::Text("elastic model: %s", elastic_model_->GetName().c_str());
  ImGui::Text("young's modulus: %.2e", young_);
  ImGui::Text("poisson's ratio: %.2f", poisson_);
}

void ElasticObject::Destroy() {
  delete[] initial_vertices_;
  delete[] volumes_;
  delete[] Dm_inv_;
  delete[] masses_;
  delete[] velocities_;
  delete[] forces_;
  delete elastic_model_;
}
};  // namespace Rain