#include "cudakinematicobject.h"

#include "imgui.h"
#include "renderscene/renderscene.h"

namespace Rain {
void CudaKinematicObject::Init(VkDevice device, Object* obj, const Vec3& pivot,
                               const Vec3& vel, const Vec3& angular_vel,
                               int st_frame, int ed_frame) {
  obj_ = obj;
  pivot_ = pivot;
  velocity_ = vel;
  angular_velocity_ = angular_vel;
  translation_ = pivot_;
  start_frame_ = st_frame;
  end_frame_ = ed_frame;

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

  rel_pos_ = new Vec3[obj_->n_vert_];
  for (uint32_t i = 0; i < obj_->n_vert_; ++i) {
    rel_pos_[i] = obj_->vertices_[i] - pivot_;
  }
  CheckCuda(cudaMalloc((void**)&drel_pos_, sizeof(Vec3) * obj_->n_vert_));
  CheckCuda(cudaMemcpy(drel_pos_, rel_pos_, sizeof(Vec3) * obj_->n_vert_,
                       cudaMemcpyHostToDevice));

  vert_blocks_ =
      (obj_->n_vert_ + vert_threads_per_block_ - 1) / vert_threads_per_block_;
}

void CudaKinematicObject::Reset() {
  rotation_ = Mat3::Identity();
  translation_ = pivot_;
  for (uint32_t i = 0; i < obj_->n_vert_; ++i) {
    obj_->vertices_[i] = pivot_ + rel_pos_[i];
  }
  obj_->UpdateNormal();
  CheckCuda(cudaMemcpy(dvertices_, obj_->vertices_,
                       sizeof(Vec3) * obj_->n_vert_, cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dnormals_, obj_->normals_, sizeof(Vec3) * obj_->n_vert_,
                       cudaMemcpyHostToDevice));
}

void CudaKinematicObject::ShowUI() { ImGui::Text("Kinematic Object"); }

void CudaKinematicObject::Destroy() {
  if (dvertices_) CheckCuda(cudaDestroyExternalMemory(vert_mem_));
  if (dnormals_) CheckCuda(cudaDestroyExternalMemory(norm_mem_));
  if (dfaces_) CheckCuda(cudaDestroyExternalMemory(face_mem_));
  if (drel_pos_) CheckCuda(cudaFree(drel_pos_));
  if (rel_pos_) delete[] rel_pos_;
}
};  // namespace Rain