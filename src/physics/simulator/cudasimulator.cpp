#include "cudasimulator.h"

#include "configtype.h"
#include "cudahelper.h"

namespace Rain {
void CudaSimulator::Init(Scene* scene, const YAML::Node& config,
                         VkDevice device, uint8_t* vkDeviceUUID,
                         size_t UUID_SIZE) {
  InitCuda(vkDeviceUUID, UUID_SIZE);

  grav_ = config["gravity"].as<Vec3>();
  damping_ = config["damping"].as<real>();
  n_substep_ = config["n_substep"].as<uint32_t>();
  substep_size_ = config["substep_size"].as<real>();
  if (config["kinematic_penalty"])
    kinematic_penalty_ = config["kinematic_penalty"].as<real>();
  if (config["self_penalty"]) self_penalty_ = config["self_penalty"].as<real>();
  // elastic_objs_.resize(scene->objects_.size());
  for (size_t i = 0; i < scene->objects_.size(); ++i) {
    if (scene->objects_[i].obj_type_ != ObjType::Dynamic) continue;
    const YAML::Node& obj_node = config["objects"][i];
    std::string elastic_model = obj_node["elastic_model"].as<std::string>();
    ElasticModelType model_type = ElasticModelType::StVK;
    if (elastic_model == "PD") {
      model_type = ElasticModelType::PD;
    } else if (elastic_model == "NeoHookean") {
      model_type = ElasticModelType::NeoHookean;
    } else if (elastic_model == "StVK") {
      model_type = ElasticModelType::StVK;
    } else if (elastic_model == "Corotation") {
      model_type = ElasticModelType::Corotation;
    } else if (elastic_model == "NeoHookeanLog") {
      model_type = ElasticModelType::NeoHookeanLog;
    } else {
      spdlog::error("elastic model not supported: {}", elastic_model);
      exit(1);
    }
    real young = obj_node["young"].as<real>();
    real poisson = obj_node["poisson"].as<real>();
    real density = obj_node["density"].as<real>();
    std::string solver = obj_node["solver"].as<std::string>();
    if (solver == "Explicit") {
      CudaElasticObject* tobj = new CudaElasticObject;
      elastic_objs_.push_back(tobj);
      tobj->Init(device, &scene->objects_[i], density, model_type, young,
                 poisson);
    } else if (solver == "MG19") {
      CudaElasobjMG19* tobj = new CudaElasobjMG19;
      elastic_objs_.push_back(tobj);
      CudaElasobjMG19InitInfo info;
      info.device = device;
      info.obj = &scene->objects_[i];
      info.density = density;
      info.type = model_type;
      info.young = young;
      info.poisson = poisson;
      info.control_mag = obj_node["control_mag"].as<real>();
      info.relaxation = obj_node["relaxation"].as<real>();
      info.n_layer = obj_node["n_layer"].as<uint32_t>();
      std::vector<uint32_t> n_handles;
      for (auto i = 0; i < obj_node["n_handles"].size(); ++i) {
        n_handles.push_back(obj_node["n_handles"][i].as<uint32_t>());
      }
      info.n_handles = n_handles.data();
      std::vector<unsigned char> A_as_LDU;
      std::vector<unsigned char> A_as_dense;
      for (auto i = 0; i < obj_node["A_as_LDU"].size(); ++i) {
        A_as_LDU.push_back(obj_node["A_as_LDU"][i].as<unsigned char>());
      }
      for (auto i = 0; i < obj_node["A_as_dense"].size(); ++i) {
        A_as_dense.push_back(obj_node["A_as_dense"][i].as<unsigned char>());
      }
      info.A_as_LDU = (bool*)A_as_LDU.data();
      info.A_as_dense = (bool*)A_as_dense.data();
      info.dt = substep_size_;
      if (obj_node["quasi_static"])
        info.quasi_static = obj_node["quasi_static"].as<bool>();
      else
        info.quasi_static = false;
      info.n_iter = obj_node["n_iter"].as<uint32_t>();
      std::vector<MGOpConfig> ops;
      for (auto i = 0; i < obj_node["operations"].size(); ++i) {
        MGOpConfig op;
        const YAML::Node& op_node = obj_node["operations"][i];
        std::string type_str = op_node["type"].as<std::string>();
        if (type_str == "Jacobi") {
          op.type = MGOpType::Jacobi;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "GS") {
          op.type = MGOpType::GS;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "CG") {
          op.type = MGOpType::CG;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "PCG") {
          op.type = MGOpType::PCG;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "Direct") {
          op.type = MGOpType::Direct;
        } else if (type_str == "DS") {
          op.type = MGOpType::DS;
        } else if (type_str == "US") {
          op.type = MGOpType::US;
        }
        ops.push_back(op);
      }
      info.n_op = ops.size();
      info.operations = ops.data();
      std::vector<Joint> joints;
      if (obj_node["skeleton"]) {  
        for (int i = 0; i < obj_node["skeleton"].size(); ++i) {
          const YAML::Node& jnode = obj_node["skeleton"][i];
          Joint jt;
          jt.name = jnode["joint"].as<std::string>();
          jt.parent = jnode["parent"].as<int>();
          jt.offset = jnode["offset"].as<Vec3>();
          jt.axis = jnode["axis"].as<Vec3>();
          jt.rot_init = jnode["rot_init"].as<real>() * PI_ / 180.;
          jt.rot_limit = jnode["rot_limit"].as<Vec2>();
          jt.bone_start = jnode["bone_start"].as<Vec3>();
          jt.bone_end = jnode["bone_end"].as<Vec3>();
          jt.bone_radius = jnode["bone_radius"].as<real>();
          joints.push_back(jt);
        }
        info.skeleton = joints.data();
        info.n_joint = joints.size();
      } else {
        info.skeleton = nullptr;
        info.n_joint = 0;
      }
      if (obj_node["self_collision"]) {
        tobj->self_collision_ = obj_node["self_collision"].as<bool>();
      }
      tobj->Init(&info);
    } else if (solver == "MixedMG19") {
      CudaElasobjMixedMG19* tobj = new CudaElasobjMixedMG19;
      elastic_objs_.push_back(tobj);
      CudaElasobjMixedMG19InitInfo info;
      info.device = device;
      info.obj = &scene->objects_[i];
      info.density = density;
      info.type = model_type;
      info.young = young;
      info.poisson = poisson;
      info.control_mag = obj_node["control_mag"].as<real>();
      info.relaxation = obj_node["relaxation"].as<real>();
      info.n_layer = obj_node["n_layer"].as<uint32_t>();
      std::vector<uint32_t> n_handles;
      for (auto i = 0; i < obj_node["n_handles"].size(); ++i) {
        n_handles.push_back(obj_node["n_handles"][i].as<uint32_t>());
      }
      info.n_handles = n_handles.data();
      std::vector<unsigned char> A_as_LDU;
      std::vector<unsigned char> A_as_dense;
      for (auto i = 0; i < obj_node["A_as_LDU"].size(); ++i) {
        A_as_LDU.push_back(obj_node["A_as_LDU"][i].as<unsigned char>());
      }
      for (auto i = 0; i < obj_node["A_as_dense"].size(); ++i) {
        A_as_dense.push_back(obj_node["A_as_dense"][i].as<unsigned char>());
      }
      info.A_as_LDU = (bool*)A_as_LDU.data();
      info.A_as_dense = (bool*)A_as_dense.data();
      info.dt = substep_size_;
      if (obj_node["quasi_static"])
        info.quasi_static = obj_node["quasi_static"].as<bool>();
      else
        info.quasi_static = false;
      if (obj_node["minres_iter"])
        info.minres_iter = obj_node["minres_iter"].as<uint32_t>();
      else
        info.minres_iter = 0;
      info.n_iter = obj_node["n_iter"].as<uint32_t>();
      if (obj_node["p_smooth"]) 
        info.p_smooth = obj_node["p_smooth"].as<real>();
      else 
        info.p_smooth = 5.;
      
      if (obj_node["p_scale"]) 
        info.p_scale = obj_node["p_scale"].as<real>();
      else 
        info.p_scale = 1.e8;
      std::vector<MGOpConfig> ops;
      for (auto i = 0; i < obj_node["operations"].size(); ++i) {
        MGOpConfig op;
        const YAML::Node& op_node = obj_node["operations"][i];
        std::string type_str = op_node["type"].as<std::string>();
        if (type_str == "Jacobi") {
          op.type = MGOpType::Jacobi;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "GS") {
          op.type = MGOpType::GS;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "IUzawa") {
          op.type = MGOpType::IUzawa;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "RVanka") {
          op.type = MGOpType::RVanka;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "CG") {
          op.type = MGOpType::CG;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "PCG") {
          op.type = MGOpType::PCG;
          op.max_iter = op_node["max_iter"].as<uint32_t>();
        } else if (type_str == "Direct") {
          op.type = MGOpType::Direct;
        } else if (type_str == "DS") {
          op.type = MGOpType::DS;
        } else if (type_str == "US") {
          op.type = MGOpType::US;
        }
        ops.push_back(op);
      }
      std::vector<Joint> joints;
      if (obj_node["skeleton"]) {  
        for (int i = 0; i < obj_node["skeleton"].size(); ++i) {
          const YAML::Node& jnode = obj_node["skeleton"][i];
          Joint jt;
          jt.name = jnode["joint"].as<std::string>();
          jt.parent = jnode["parent"].as<int>();
          jt.offset = jnode["offset"].as<Vec3>();
          jt.axis = jnode["axis"].as<Vec3>();
          jt.rot_init = jnode["rot_init"].as<real>() * PI_ / 180.;
          jt.rot_limit = jnode["rot_limit"].as<Vec2>();
          jt.bone_start = jnode["bone_start"].as<Vec3>();
          jt.bone_end = jnode["bone_end"].as<Vec3>();
          jt.bone_radius = jnode["bone_radius"].as<real>();
          joints.push_back(jt);
        }
        info.skeleton = joints.data();
        info.n_joint = joints.size();
      } else {
        info.skeleton = nullptr;
        info.n_joint = 0;
      }
      info.n_op = ops.size();
      info.operations = ops.data();
      if (obj_node["self_collision"]) {
        tobj->self_collision_ = obj_node["self_collision"].as<bool>();
      }
      tobj->Init(&info);
    } 
    else {
      spdlog::error("unkown solver type");
      exit(1);
    }
  }

  for (size_t i = 0; i < scene->objects_.size(); ++i) {
    if (scene->objects_[i].obj_type_ != ObjType::Kinematic) continue;
    const YAML::Node& obj_node = config["objects"][i];
    CudaKinematicObject* obj = new CudaKinematicObject;
    kinematic_objs_.push_back(obj);
    Vec3 pivot = Vec3::Zero();
    if (obj_node["pivot"]) pivot = obj_node["pivot"].as<Vec3>();
    Vec3 velocity = Vec3::Zero();
    if (obj_node["velocity"]) velocity = obj_node["velocity"].as<Vec3>();
    Vec3 angular_velocity = Vec3::Zero();
    if (obj_node["angular_velocity"])
      angular_velocity = obj_node["angular_velocity"].as<Vec3>();
    int st_frame = 0;
    int ed_frame = -1;
    if (obj_node["start_frame"]) st_frame = obj_node["start_frame"].as<int>();
    if (obj_node["end_frame"]) ed_frame = obj_node["end_frame"].as<int>();
    obj->Init(device, &scene->objects_[i], pivot, velocity, angular_velocity,
              st_frame, ed_frame);
  }
}

void CudaSimulator::InitCuda(uint8_t* vkDeviceUUID, size_t UUID_SIZE) {
  int cuda_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;
  cudaDeviceProp device_prop;
  CheckCuda(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    spdlog::error("no devices supporting CUDA");
    exit(1);
  }
  for (int i = 0; i < device_count; ++i) {
    CheckCuda(cudaGetDeviceProperties(&device_prop, i));
    if (device_prop.computeMode != cudaComputeModeProhibited) {
      int ret = memcmp((void*)&device_prop.uuid, vkDeviceUUID, UUID_SIZE);
      if (ret == 0) {  // same device as Vulkan
        CheckCuda(cudaSetDevice(i));
        CheckCuda(cudaGetDeviceProperties(&device_prop, i));
        spdlog::info("GPU {} picked for CUDA: {}", i, device_prop.name);
        spdlog::info("compute capability: {}.{}", device_prop.major,
                     device_prop.minor);
        cuda_device = i;
      }
    } else {
      ++devices_prohibited;
    }
  }
  if (devices_prohibited == device_count) {
    spdlog::error("no Vulkan-CUDA interop capable GPU found");
    exit(1);
  }

  CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamDefault));
  return;
}

void CudaSimulator::Update(float dt) {
  for (auto obj : kinematic_objs_) {
    obj->Update(stream_, dt, n_substep_, substep_size_, n_frame_);
  }
  for (auto obj : elastic_objs_) {
    obj->Update(stream_, dt, grav_, damping_, kinematic_penalty_, self_penalty_,
                kinematic_objs_, n_substep_, substep_size_, n_frame_);
  }
  ++n_frame_;
}

void CudaSimulator::Reset() {
  n_frame_ = 0;
  for (auto obj : elastic_objs_) {
    obj->Reset();
  }
  for (auto obj : kinematic_objs_) {
    obj->Reset();
  }
}

void CudaSimulator::ShowUI() {
  ImGui::Text("gravity: %.1f %.1f %.1f", grav_[0], grav_[1], grav_[2]);
  ImGui::Text("damping: %.2e", damping_);
  ImGui::Text("n_substep: %d", n_substep_);
  ImGui::Text("substep_size: %.2e", substep_size_);
  ImGui::Text("n_frame: %d", n_frame_);
  ImGui::Separator();
  for (size_t i = 0; i < elastic_objs_.size(); ++i) {
    if (i == 0) ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    CudaElasticObject& obj = *elastic_objs_[i];
    if (ImGui::TreeNode((void*)(intptr_t)i, obj.obj_->name_.c_str())) {
      obj.ShowUI();
      ImGui::TreePop();
    }
  }
}

void CudaSimulator::Destroy() {
  for (auto obj : elastic_objs_) {
    obj->Destroy();
    delete obj;
  }
  for (auto obj : kinematic_objs_) {
    obj->Destroy();
    delete obj;
  }
}
};  // namespace Rain