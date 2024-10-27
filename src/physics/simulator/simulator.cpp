#include "simulator.h"

#include "elasobjmg19.h"

namespace Rain {
void Simulator::Init(Scene* scene, const YAML::Node& config) {
  grav_ = config["gravity"].as<Vec3>();
  damping_ = config["damping"].as<real>();
  n_substep_ = config["n_substep"].as<uint32_t>();
  substep_size_ = config["substep_size"].as<real>();
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
    } else {
      spdlog::error("elastic model not supported: {}", elastic_model);
      exit(1);
    }
    real density = obj_node["density"].as<real>();
    real young = obj_node["young"].as<real>();
    real poisson = obj_node["poisson"].as<real>();
    std::string solver = obj_node["solver"].as<std::string>();
    if (solver == "Explicit") {
      ElasticObject* tobj = new ElasticObject;
      elastic_objs_.push_back(tobj);
      tobj->Init(&scene->objects_[i], density, model_type, young, poisson);
    } else if (solver == "MG19") {
      ElasobjMG19* tobj = new ElasobjMG19;
      elastic_objs_.push_back(tobj);
      ElasobjMG19InitInfo info;
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
      tobj->Init(&info);
      if (obj_node["self_collision"]) {
        tobj->self_collision_ = obj_node["self_collision"].as<bool>();
      }
    } else {
      spdlog::error("unkown solver type");
      exit(1);
    }
  }

  for (size_t i = 0; i < scene->objects_.size(); ++i) {
    if (scene->objects_[i].obj_type_ != ObjType::Kinematic) continue;
    const YAML::Node& obj_node = config["objects"][i];
    KinematicObject* obj = new KinematicObject;
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
    obj->Init(&scene->objects_[i], pivot, velocity, angular_velocity, st_frame, ed_frame);
  }
}

void Simulator::Update(float dt) {
  for (auto obj : kinematic_objs_) {
    obj->Update(dt, n_frame_);
  }
  for (auto obj : elastic_objs_) {
    obj->Update(dt, grav_, damping_, n_substep_, substep_size_, n_frame_,
                kinematic_objs_);
  }
  ++n_frame_;
}

void Simulator::Reset() {
  n_frame_ = 0;
  for (auto obj : elastic_objs_) {
    obj->Reset();
  }
  for (auto obj : kinematic_objs_) {
    obj->Reset();
  }
}

void Simulator::ShowUI() {
  // ImGui::Text("gravity: %.1f %.1f %.1f", grav_[0], grav_[1], grav_[2]);
  // ImGui::SliderFloat("gravity", &grav_[1], 0.0f, -10.0f, "%.1f");
  ImGui::Text("damping: %.2e", damping_);
  ImGui::Text("n_substep: %d", n_substep_);
  ImGui::Text("substep_size: %.2e", substep_size_);
  ImGui::Text("n_frame: %d", n_frame_);
  ImGui::Separator();
  for (size_t i = 0; i < elastic_objs_.size(); ++i) {
    if (i == 0) ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    ElasticObject* obj = elastic_objs_[i];
    if (ImGui::TreeNode((void*)(intptr_t)i, obj->obj_->name_.c_str())) {
      obj->ShowUI();
      ImGui::TreePop();
    }
  }
}

void Simulator::Destroy() {
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