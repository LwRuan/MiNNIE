#include "cudaelasticobject.h"

#include <Eigen/Dense>
#include <fstream>

#include "imgui.h"
#include "renderscene/renderscene.h"

namespace Rain {
void CudaElasticObject::Init(VkDevice device, Object* obj, real density,
                             ElasticModelType type, real young, real poisson,
                             Joint* skeleton, int32_t n_joint) {
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
    volumes_[e] = Dm.determinant() / 6.0;
    if (volumes_[e] < 0) spdlog::error("negative volume: {}", volumes_[e]);
    Dm_inv_[e] = Dm.inverse();
  }

  density_ = density;
  masses_ = new real[obj_->n_vert_];
  memset(masses_, 0, sizeof(real) * obj_->n_vert_);
  for (size_t e = 0; e < obj_->n_ele_; ++e) {
    for (size_t i = 0; i < 4; ++i) {
      const auto& idx = obj_->indices_[4 * e + i];
      masses_[idx] += density_ * volumes_[e] / 4.0;
    }
  }

  // young_ = young;
  // poisson_ = poisson;
  switch (type) {
    case ElasticModelType::StVK:
      elastic_model_ = new StVK;
      break;
    case ElasticModelType::PD:
      elastic_model_ = new PD;
      break;
    case ElasticModelType::NeoHookean:
      elastic_model_ = new NeoHookean;
      break;
    case ElasticModelType::Corotation:
      elastic_model_ = new Corotation;
      break;
    case ElasticModelType::NeoHookeanLog:
      elastic_model_ = new NeoHookeanLog;
      break;
    default:
      break;
  }
  elastic_model_->SetLame(young, poisson);

  if (n_joint > 0) {
    n_joint_ = n_joint;
    skeleton_ = new Joint[n_joint];
    memcpy(skeleton_, skeleton, sizeof(Joint) * n_joint);

    for (int i = 0; i < n_joint; ++i) {
      skeleton_[i].name = std::string(skeleton[i].name);
      spdlog::info("joint {}: name {}, parent {}", i, skeleton_[i].name,
                   skeleton_[i].parent);
    }
    joint_trans_ = new JointTransform[n_joint];
    CheckCuda(cudaMalloc(&djoint_trans_, sizeof(JointTransform) * n_joint_));
    UpdateSkeleton();

    bone_ids_ = new int32_t[n_vert_];
    memset(bone_ids_, -1, sizeof(int32_t) * n_vert_);
    local_pos_ = new Vec3[n_vert_];
    for (int i = 0; i < n_vert_; ++i) {
      for (int b = 0; b < n_joint_; ++b) {
        Vec3 spos = GetGlobalPos(b, skeleton_[b].bone_start);
        Vec3 epos = GetGlobalPos(b, skeleton_[b].bone_end);
        Vec3 dir = (epos - spos).normalized();
        real proj = (verts_[i] - spos).dot(dir);
        real dist = 0.;
        if (proj < 0) {
          dist = (verts_[i] - spos).norm();
        } else if (proj > (epos - spos).norm()) {
          dist = (verts_[i] - epos).norm();
        } else {
          dist = (verts_[i] - spos - dir * proj).norm();
        }
        if (dist < skeleton_[b].bone_radius) {
          bone_ids_[i] = b;
          local_pos_[i] = joint_trans_[b].global_rot.transpose() *
                          (verts_[i] - joint_trans_[b].global_pos);
          break;
        }
      }
    }
    CheckCuda(cudaMalloc(&dbone_ids_, sizeof(int32_t) * n_vert_));
    CheckCuda(cudaMemcpy(dbone_ids_, bone_ids_, sizeof(int32_t) * n_vert_,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&dlocal_pos_, sizeof(Vec3) * n_vert_));
    CheckCuda(cudaMemcpy(dlocal_pos_, local_pos_, sizeof(Vec3) * n_vert_,
                         cudaMemcpyHostToDevice));
  }

  VkDeviceSize size = (VkDeviceSize)(sizeof(Vec3)) * obj_->n_vert_;
  ImportCudaExternalMemory((void**)&dvertices_, &vert_mem_, device,
                           obj_->render_model_->vertex_buffers_[0].memory_,
                           size, GetDefaultMemHandleType());
  ImportCudaExternalMemory((void**)&dnormals_, &norm_mem_, device,
                           obj_->render_model_->vertex_buffers_[1].memory_,
                           size, GetDefaultMemHandleType());
  size = (VkDeviceSize)(sizeof(uint32_t)) * obj_->n_surfidx_;
  ImportCudaExternalMemory((void**)&dfaces_, &face_mem_, device,
                           obj_->render_model_->index_buffer_.memory_, size,
                           GetDefaultMemHandleType());
  vert_blocks_ =
      (obj_->n_vert_ + vert_threads_per_block_ - 1) / vert_threads_per_block_;
  tet_blocks_ =
      (obj_->n_ele_ + tet_threads_per_block_ - 1) / tet_threads_per_block_;
  face_blocks_ =
      (obj_->n_face_ + face_threads_per_block_ - 1) / face_threads_per_block_;

  CheckCuda(
      cudaMalloc((void**)&dindices_, sizeof(uint32_t) * obj_->n_ele_ * 4));
  CheckCuda(cudaMalloc((void**)&dforces_, sizeof(Vec3) * obj_->n_vert_));
  CheckCuda(cudaMalloc((void**)&dvelocities_, sizeof(Vec3) * obj_->n_vert_));
  CheckCuda(cudaMalloc((void**)&dmasses_, sizeof(real) * obj_->n_vert_));
  CheckCuda(cudaMalloc((void**)&dfixed_, sizeof(bool) * obj_->n_vert_));
  CheckCuda(cudaMalloc((void**)&dvolumes_, sizeof(real) * obj_->n_ele_));
  CheckCuda(cudaMalloc((void**)&dDm_inv_, sizeof(Mat3) * obj_->n_ele_));

  CheckCuda(cudaMemcpy(dindices_, obj_->indices_,
                       sizeof(uint32_t) * obj_->n_ele_ * 4,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemset(dvelocities_, 0, sizeof(Vec3) * obj_->n_vert_));
  CheckCuda(cudaMemcpy(dmasses_, masses_, sizeof(real) * obj_->n_vert_,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dfixed_, obj_->is_fixed_, sizeof(bool) * obj_->n_vert_,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dvolumes_, volumes_, sizeof(real) * obj_->n_ele_,
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dDm_inv_, Dm_inv_, sizeof(Mat3) * obj_->n_ele_,
                       cudaMemcpyHostToDevice));
}

void CudaElasticObject::Reset() {
  CheckCuda(cudaMemset(dvelocities_, 0, sizeof(Vec3) * obj_->n_vert_));
  CheckCuda(cudaMemset(dforces_, 0, sizeof(Vec3) * obj_->n_vert_));
  memcpy(obj_->vertices_, initial_vertices_, sizeof(Vec3) * obj_->n_vert_);
  // for (int i = 0; i < n_vert_; ++i) {
  //   real y = obj_->vertices_[i][1];
  //   if (2. - y < 1e-1 || y + 2. < 1e-1) {
  //     std::cout << i << ", ";
  //   }
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < n_vert_; ++i) {
  //   if (obj_->is_fixed_[i]) continue;
  //   // obj_->vertices_[i] = 2. * Vec3::Random();
  //   obj_->vertices_[i] = obj_->vertices_[i] + 0.1 * Vec3::Random();
  // }
  // for (int i = 0; i < n_vert_; ++i) {
  //   real theta = 0.5 * PI_ * obj_->vertices_[i][1] / 2.;
  //   obj_->vertices_[i][1] = 0.7 * obj_->vertices_[i][1];
  //   real x = obj_->vertices_[i][0];
  //   real z = obj_->vertices_[i][2];
  //   obj_->vertices_[i][0] = cos(theta) * x - sin(theta) * z;
  //   obj_->vertices_[i][2] = sin(theta) * x + cos(theta) * z;
  // }
  // for (int i = 0; i < n_vert_; ++i) {
  //   if (!obj_->is_fixed_[i]) continue;
  //   obj_->vertices_[i] = 1.3 * obj_->vertices_[i];
  // }
  CheckCuda(cudaMemcpy(dfixed_verts_, obj_->vertices_,
                       sizeof(Vec3) * obj_->n_vert_, cudaMemcpyHostToDevice));
  obj_->UpdateNormal();
  CheckCuda(cudaMemcpy(dvertices_, obj_->vertices_,
                       sizeof(Vec3) * obj_->n_vert_, cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dnormals_, obj_->normals_, sizeof(Vec3) * obj_->n_vert_,
                       cudaMemcpyHostToDevice));
}

void CudaElasticObject::ShowUI() {
  if (ImGui::Button("Output Frame Data")) {
    output_frame_data_ = true;
  }
  ImGui::Text("density: %.2e", density_);
  ImGui::Text("elastic model: %s", elastic_model_->GetName().c_str());
  ImGui::Text("young's modulus: %.2e", elastic_model_->young_);
  ImGui::Text("poisson's ratio: %.4f", elastic_model_->poisson_);
  ImGui::Text("Skeleton Control:");
  for (int i = 0; i < n_joint_; ++i) {
    real delta = skeleton_[i].rot_limit[1] - skeleton_[i].rot_limit[0];
    if (delta > 1e-3) {
      float rot = (float)skeleton_[i].rot;
      ImGui::SliderAngle(skeleton_[i].name.c_str(), &rot,
                         skeleton_[i].rot_limit[0], skeleton_[i].rot_limit[1]);
      skeleton_[i].rot = (real)rot;
    }
  }
}

void CudaElasticObject::Destroy() {
  if (initial_vertices_) delete[] initial_vertices_;
  if (volumes_) delete[] volumes_;
  if (Dm_inv_) delete[] Dm_inv_;
  if (masses_) delete[] masses_;
  if (elastic_model_) delete elastic_model_;
  if (skeleton_) delete[] skeleton_;
  if (joint_trans_) delete[] joint_trans_;
  if (bone_ids_) delete[] bone_ids_;
  if (local_pos_) delete[] local_pos_;
  if (djoint_trans_) CheckCuda(cudaFree(djoint_trans_));
  if (dbone_ids_) CheckCuda(cudaFree(dbone_ids_));
  if (dlocal_pos_) CheckCuda(cudaFree(dlocal_pos_));
  if (dvertices_) CheckCuda(cudaDestroyExternalMemory(vert_mem_));
  if (dnormals_) CheckCuda(cudaDestroyExternalMemory(norm_mem_));
  if (dfaces_) CheckCuda(cudaDestroyExternalMemory(face_mem_));
  if (dindices_) CheckCuda(cudaFree(dindices_));
  if (dforces_) CheckCuda(cudaFree(dforces_));
  if (dvelocities_) CheckCuda(cudaFree(dvelocities_));
  if (dmasses_) CheckCuda(cudaFree(dmasses_));
  if (dfixed_) CheckCuda(cudaFree(dfixed_));
  if (dvolumes_) CheckCuda(cudaFree(dvolumes_));
  if (dDm_inv_) CheckCuda(cudaFree(dDm_inv_));
}

void CudaElasticObject::UpdateSkeleton() {
  for (int i = 0; i < n_joint_; ++i) {
    if (skeleton_[i].parent == -1) {
      real angle = skeleton_[i].rot + skeleton_[i].rot_init;
      Eigen::AngleAxis<real> rot(angle, skeleton_[i].axis);
      joint_trans_[i].global_rot = rot.toRotationMatrix();
      joint_trans_[i].global_pos = rot * skeleton_[i].offset;
    } else {
      real angle = skeleton_[i].rot + skeleton_[i].rot_init;
      const JointTransform& pj = joint_trans_[skeleton_[i].parent];
      Eigen::AngleAxis<real> rot(angle, skeleton_[i].axis);
      joint_trans_[i].global_rot = pj.global_rot * rot.toRotationMatrix();
      joint_trans_[i].global_pos =
          pj.global_rot * skeleton_[i].offset + pj.global_pos;
    }
  }
  CheckCuda(cudaMemcpy(djoint_trans_, joint_trans_,
                       sizeof(JointTransform) * n_joint_,
                       cudaMemcpyHostToDevice));
}

Vec3 CudaElasticObject::GetGlobalPos(const int i, const Vec3& pos) const {
  return joint_trans_[i].global_rot * pos + joint_trans_[i].global_pos;
}
};  // namespace Rain