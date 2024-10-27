#include "engine.h"

#include <spdlog/spdlog.h>

#include <cstring>
#include <iostream>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "helper/yaml.h"

namespace Rain {
Engine::Engine() {
#ifdef NDEBUG
  enable_validation_layers_ = false;
#else
  enable_validation_layers_ = true;
#endif
}

void Engine::Init(const std::string& config_file) {
  YAML::Node config = YAML::LoadFile(config_file);
  if (config["GPU_sim"]) GPU_SIM = config["GPU_sim"].as<bool>();
  if (config["timestep"]) timestep_ = config["timestep"].as<float>();
  {  // init window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window_ =
        glfwCreateWindow(width_, height_, "MultiGridSim", nullptr, nullptr);
    glfwGetWindowContentScale(window_, &width_scale_, &height_scale_);
    width_ *= width_scale_;
    height_ *= height_scale_;
    glfwSetWindowSize(window_, width_, height_);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    glfwShowWindow(window_);

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, WindowResizeCallback);
    glfwSetKeyCallback(window_, KeyCallback);
    glfwSetMouseButtonCallback(window_, MouseButtonCallback);
    glfwSetCursorPosCallback(window_, CursorPosCallback);
    glfwSetScrollCallback(window_, ScrollCallback);
  }

  if (enable_validation_layers_) {
    if (!CheckValidationLayerSupport()) {
      spdlog::error("validation layers requested, but not available");
      exit(1);
    }
    debug_utils_ext_ = new DebugUtilsEXT;
  }

  {  // create instance
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "MultiGridSim";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    if (enable_validation_layers_) {
      create_info.enabledLayerCount =
          static_cast<uint32_t>(validation_layers_.size());
      create_info.ppEnabledLayerNames = validation_layers_.data();
      VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
      DebugUtilsEXT::GetCreateInfo(debug_create_info);
      // create_info.pNext =
      //     (VkDebugUtilsMessengerCreateInfoEXT*)&debug_create_info;
    } else {
      create_info.enabledLayerCount = 0;
      create_info.pNext = nullptr;
    }
    auto extensions = GetRequiredExtensions();
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
      spdlog::error("instance creation failed");
      CleanUp();
      exit(1);
    } else {
      spdlog::debug("instance created");
    }
  }

  if (enable_validation_layers_) {  // create debug messenger
    if (debug_utils_ext_->Init(instance_) != VK_SUCCESS) {
      spdlog::error("debug messenger creation failed");
      CleanUp();
      exit(1);
    } else
      spdlog::debug("debug messenger created");
  }

  {  // create surface
    VkResult result;
    if ((result = glfwCreateWindowSurface(instance_, window_, nullptr,
                                          &surface_)) != VK_SUCCESS) {
      spdlog::error("{} window surface creation failed", result);
      CleanUp();
      exit(1);
    } else
      spdlog::debug("window surface created");
  }

  {  // create physical device
    physical_device_ = new PhysicalDevice;
    if (physical_device_->Init(instance_, surface_) != VK_SUCCESS) {
      CleanUp();
      exit(1);
    } else
      spdlog::debug("physical device created");
  }

  {  // create logical device
    device_ = new Device;
    VkResult result = VK_SUCCESS;
    uint32_t graphics_queue_family, present_queue_family;
    physical_device_->GetGraphicsPresentQueueFamily(graphics_queue_family,
                                                    present_queue_family);
    if (enable_validation_layers_) {
      result = device_->Init(physical_device_->device_, graphics_queue_family,
                             present_queue_family, &validation_layers_,
                             &physical_device_->device_extensions_);
    } else
      result = device_->Init(physical_device_->device_, graphics_queue_family,
                             present_queue_family, nullptr,
                             &physical_device_->device_extensions_);
    if (result != VK_SUCCESS) {
      CleanUp();
      exit(1);
    } else
      spdlog::debug("logical device created");
  }

  {  // create swap chain
    swap_chain_ = new SwapChain;
    if (swap_chain_->Init(device_, physical_device_, window_, surface_) !=
        VK_SUCCESS) {
      spdlog::error("swap chain creation failed");
      CleanUp();
      exit(1);
    } else {
      spdlog::debug("swap chain created");
    }
  }

  {  // scene
    scene_.Init(config["objects"]);
    if (render_scene_.Init(device_, swap_chain_, &scene_, GPU_SIM) !=
        VK_SUCCESS) {
      CleanUp();
      exit(1);
    }
  }

  {  // create render pass
    render_pass_ = new RenderPass;
    if (render_pass_->Init(device_, swap_chain_->image_format_) != VK_SUCCESS) {
      CleanUp();
      exit(1);
    } else {
      spdlog::debug("render pass created");
    }
    imgui_render_pass_ = new ImGuiRenderPass;
    if (imgui_render_pass_->Init(device_, swap_chain_->image_format_) !=
        VK_SUCCESS) {
      CleanUp();
      exit(1);
    }
  }

  {  // create pipeline
    pipeline_ = new Pipeline;
    if (pipeline_->Init(device_->device_, swap_chain_->extent_,
                        render_pass_->render_pass_,
                        &render_scene_) != VK_SUCCESS) {
      spdlog::error("pipeline creation failed");
      CleanUp();
      exit(1);
    } else {
      spdlog::debug("pipeline created");
    }
  }

  {  // create framebuffers
    framebuffers_.resize(swap_chain_->image_views_.size());
    imgui_framebuffers_.resize(swap_chain_->image_views_.size());
    for (size_t i = 0; i < swap_chain_->image_views_.size(); ++i) {
      if (framebuffers_[i].Init(device_, swap_chain_->extent_,
                                swap_chain_->image_views_[i],
                                render_pass_->render_pass_) != VK_SUCCESS) {
        CleanUp();
        exit(1);
      }
      if (imgui_framebuffers_[i].Init(
              device_, swap_chain_->extent_, swap_chain_->image_views_[i],
              imgui_render_pass_->render_pass_) != VK_SUCCESS) {
        CleanUp();
        exit(1);
      }
    }
    spdlog::debug("framebuffers created");
  }

  // allocate and record command buffers
  device_->AllocateCommandBuffers(swap_chain_);
  InitImGui();

  if (GPU_SIM) {
    cuda_simulator_ = new CudaSimulator;
    cuda_simulator_->Init(&scene_, config, device_->device_,
                          physical_device_->device_UUID_, VK_UUID_SIZE);
    ImportCudaExternalSemaphore(cuda_wait_, device_->device_,
                                swap_chain_->cuda_sim_vk_signaled_,
                                GetDefaultSemaphoreHandleType());
    ImportCudaExternalSemaphore(cuda_signaled_, device_->device_,
                                swap_chain_->cuda_sim_vk_wait_,
                                GetDefaultSemaphoreHandleType());
  } else {
    simulator_ = new Simulator;
    simulator_->Init(&scene_, config);
  }
  spdlog::info("initialization finished");
}

VkResult Engine::InitImGui() {
  VkResult result;
  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000;
  pool_info.poolSizeCount = std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;

  result = vkCreateDescriptorPool(device_->device_, &pool_info, nullptr,
                                  &imgui_pool_);
  if (result != VK_SUCCESS) {
    spdlog::error("imgui descriptor pool creation failed");
    return result;
  }
  uint32_t graphics_queue_family, present_queue_family;
  physical_device_->GetGraphicsPresentQueueFamily(graphics_queue_family,
                                                  present_queue_family);
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->AddFontFromFileTTF("../ext/imgui/misc/fonts/ProggyClean.ttf",
                               int(13 * width_scale_));
  ImGui::GetStyle().ScaleAllSizes(width_scale_);
  ImGui_ImplGlfw_InitForVulkan(window_, true);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = instance_;
  init_info.PhysicalDevice = physical_device_->device_;
  init_info.Device = device_->device_;
  init_info.QueueFamily = graphics_queue_family;
  init_info.Queue = device_->graphics_queue_;
  init_info.DescriptorPool = imgui_pool_;
  init_info.MinImageCount = swap_chain_->images_.size();
  init_info.ImageCount = swap_chain_->images_.size();
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&init_info, imgui_render_pass_->render_pass_);
  VkCommandBuffer command_buffer = device_->BeginSingleTimeCommands();
  ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
  device_->EndSingleTimeCommands(command_buffer);
  ImGui_ImplVulkan_DestroyFontUploadObjects();
  return VK_SUCCESS;
}

void Engine::DrawFrame() {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  // ImGui::ShowDemoWindow();
  ImGui::Begin("Configurations", nullptr, ImGuiWindowFlags_None);
  ImGui::Text("time/frame: %.1f ms, fps: %.1f", 1000.0f / fps_, fps_);
  if (ImGui::CollapsingHeader("Renderer")) {
    ImGui::Text("Directional Light");
    ImGui::SliderFloat("x angle", &render_scene_.light_x_angle_, 0.0f, 180.0f,
                       "%.0f degree");
    ImGui::SliderFloat("y angle", &render_scene_.light_y_angle_, 0.0f, 90.0f,
                       "%.0f degree");

    ImGui::PushID(0);
    ImGui::SliderFloat("intensity", &render_scene_.directional_light_intensity_,
                       0.0f, 5.0f);
    ImGui::ColorEdit3("color", render_scene_.directional_light_.data());
    ImGui::PopID();

    ImGui::Text("Ambient Light");
    ImGui::PushID(1);
    ImGui::SliderFloat("intensity", &render_scene_.ambient_light_intensity_,
                       0.0f, 5.0f);
    ImGui::ColorEdit3("color", render_scene_.ambient_light_.data());
    ImGui::PopID();
    for (uint32_t i = 0; i < render_scene_.models_.size(); ++i) {
      ImGui::Checkbox(render_scene_.models_[i].obj_->name_.c_str(),
                      &render_scene_.models_[i].visible_);
    }
  }
  if (ImGui::CollapsingHeader("Simulator", nullptr,
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    if (GPU_SIM) {
      if (ImGui::Button("Play")) {
        enable_sim_ = !enable_sim_;
      };
      ImGui::SameLine();
      if (ImGui::Button("Reset")) {
        vkWaitForFences(device_->device_, 1, &swap_chain_->cpu_sim_fence_, true,
                        UINT64_MAX);
        cuda_simulator_->Reset();
      }
      ImGui::SameLine();
      if (ImGui::Button("Step")) {
        step_sim_ = true;
      }
      cuda_simulator_->ShowUI();
    } else {
      if (ImGui::Button("Play")) {
        enable_sim_ = !enable_sim_;
      };
      ImGui::SameLine();
      if (ImGui::Button("Reset")) {
        vkWaitForFences(device_->device_, 1, &swap_chain_->cpu_sim_fence_, true,
                        UINT64_MAX);
        simulator_->Reset();
        render_scene_.UpdateVertex(device_->device_);
      }
      ImGui::SameLine();
      if (ImGui::Button("Step")) {
        step_sim_ = true;
      }
      simulator_->ShowUI();
    }
  }
  ImGui::End();
  ImGui::Render();
  uint32_t image_index = swap_chain_->BeginFrame(window_resized_);
  if (window_resized_) {
    window_resized_ = false;
    RecreateSwapChain();
    image_index = swap_chain_->BeginFrame(window_resized_);
  }
  render_scene_.UpdateUniform(device_->device_, image_index);

  VkCommandBuffer command_buffer = device_->command_buffers_[image_index];
  vkResetCommandBuffer(command_buffer,
                       VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VkResult result = vkBeginCommandBuffer(command_buffer, &begin_info);
  if (result != VK_SUCCESS) {
    spdlog::error("command buffer recording start failed");
    CleanUp();
    exit(1);
  }
  VkRenderPassBeginInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_info.renderPass = render_pass_->render_pass_;
  render_pass_info.framebuffer = framebuffers_[image_index].framebuffer_;
  render_pass_info.renderArea.offset = {0, 0};
  render_pass_info.renderArea.extent = swap_chain_->extent_;
  std::array<VkClearValue, 2> clear_values{};
  // clear_values[0].color = {{0.6f, 0.6f, 0.6f, 1.0f}};
  clear_values[0].color = {{1.f, 1.f, 1.f, 1.0f}};
  clear_values[1].depthStencil = {1.0f, 0};
  render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
  render_pass_info.pClearValues = clear_values.data();
  vkCmdBeginRenderPass(command_buffer, &render_pass_info,
                       VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline_->pipeline_);
  for (size_t i = 0; i < render_scene_.models_.size(); ++i) {
    if (!render_scene_.models_[i].visible_) continue;
    render_scene_.BindAndDraw(command_buffer, pipeline_->layout_, image_index,
                              i);
  }
  vkCmdEndRenderPass(command_buffer);
  VkRenderPassBeginInfo imgui_pass_info = {};
  imgui_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  imgui_pass_info.renderPass = imgui_render_pass_->render_pass_;
  imgui_pass_info.framebuffer = imgui_framebuffers_[image_index].framebuffer_;
  imgui_pass_info.renderArea.extent = swap_chain_->extent_;
  imgui_pass_info.clearValueCount = 1;
  imgui_pass_info.pClearValues = &clear_values[0];
  vkCmdBeginRenderPass(command_buffer, &imgui_pass_info,
                       VK_SUBPASS_CONTENTS_INLINE);
  ImDrawData* imgui_data = ImGui::GetDrawData();
  ImGui_ImplVulkan_RenderDrawData(imgui_data, command_buffer);
  vkCmdEndRenderPass(command_buffer);
  if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
    spdlog::error("command buffer recording failed");
    CleanUp();
    exit(1);
  }
  swap_chain_->EndFrame(image_index);
}

void Engine::MainLoop() {
  timer_.Reset();
  while (!glfwWindowShouldClose(window_)) {
    timer_.Tick();
    dt_ = timer_.delta_time_.count();
    static uint32_t fps_cnt = 0;
    static double last_tot_time = timer_.TotalTime().count();
    double curr_tot_time = timer_.TotalTime().count();
    ++fps_cnt;
    if (curr_tot_time - last_tot_time >= 0.5) {
      fps_ = fps_cnt / (curr_tot_time - last_tot_time);
      fps_cnt = 0;
      last_tot_time = curr_tot_time;
    }
    glfwPollEvents();
    DrawFrame();
    if ((step_sim_ || enable_sim_) && GPU_SIM) {  // GPU simulation
      cudaExternalSemaphoreWaitParams wait_params = {};
      wait_params.flags = 0;
      wait_params.params.fence.value = 0;
      cudaExternalSemaphoreSignalParams signal_params = {};
      signal_params.flags = 0;
      signal_params.params.fence.value = 0;
      CheckCuda(cudaWaitExternalSemaphoresAsync(&cuda_wait_, &wait_params, 1,
                                                cuda_simulator_->stream_));
      if (timestep_) cuda_simulator_->Update(timestep_);
      else cuda_simulator_->Update(dt_);
      CheckCuda(cudaSignalExternalSemaphoresAsync(
          &cuda_signaled_, &signal_params, 1, cuda_simulator_->stream_));
      if (step_sim_) step_sim_ = false;
    } else if (step_sim_ || enable_sim_) {  // CPU simulation
      if (timestep_) simulator_->Update(timestep_);
      else simulator_->Update(dt_);
      vkWaitForFences(device_->device_, 1, &swap_chain_->cpu_sim_fence_, true,
                      UINT64_MAX);
      render_scene_.UpdateVertex(device_->device_);
      if (step_sim_) step_sim_ = false;
    }
  }
  vkDeviceWaitIdle(device_->device_);
}

void Engine::CleanUp() {
  ImGui_ImplVulkan_Shutdown();
  if (imgui_pool_) {
    vkDestroyDescriptorPool(device_->device_, imgui_pool_, nullptr);
  }
  if (GPU_SIM) {
    CheckCuda(cudaDestroyExternalSemaphore(cuda_signaled_));
    CheckCuda(cudaDestroyExternalSemaphore(cuda_wait_));
  }
  if (simulator_) {
    simulator_->Destroy();
    delete simulator_;
  }
  if (cuda_simulator_) {
    cuda_simulator_->Destroy();
    delete cuda_simulator_;
  }
  render_scene_.Destroy(device_->device_);
  scene_.Destroy();
  if (instance_) {
    if (device_) {
      if (swap_chain_) {
        swap_chain_->Destroy();
        spdlog::debug("swap chain destroyed");
        delete swap_chain_;
      }
      if (pipeline_) {
        pipeline_->Destroy(device_->device_);
        spdlog::debug("pipeline destroyed");
        delete pipeline_;
      }
      if (render_pass_) {
        render_pass_->Destroy(device_->device_);
        spdlog::debug("render pass destroyed");
        delete render_pass_;
      }
      if (imgui_render_pass_) {
        imgui_render_pass_->Destroy(device_->device_);
        delete imgui_render_pass_;
      }
      for (auto framebuffer : framebuffers_) {
        framebuffer.Destroy(device_->device_);
      }
      for (auto framebuffer : imgui_framebuffers_) {
        framebuffer.Destroy(device_->device_);
      }
      spdlog::debug("framebuffers destroyed");
      device_->Destroy();
      delete device_;
    }
    if (debug_utils_ext_) {
      debug_utils_ext_->Destroy(instance_);
      delete debug_utils_ext_;
    }
    if (surface_) {
      vkDestroySurfaceKHR(instance_, surface_, nullptr);
      spdlog::debug("window surface destroyed");
    }
    vkDestroyInstance(instance_, nullptr);
    spdlog::debug("instance destroyed");
  }
  if (physical_device_) delete physical_device_;
  glfwDestroyWindow(window_);
  glfwTerminate();
}

bool Engine::CheckValidationLayerSupport() {
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
  for (const char* layer_name : validation_layers_) {
    bool layer_found = false;
    for (const auto& layer_property : available_layers) {
      if (strcmp(layer_name, layer_property.layerName) == 0) {
        layer_found = true;
        break;
      }
    }
    if (!layer_found)
      return false;
    else
      spdlog::debug("{} found", layer_name);
  }
  return true;
}

std::vector<const char*> Engine::GetRequiredExtensions() {
  // functional extensions
  std::vector<const char*> extensions(instance_extensions_);
  // debug extensions
  if (enable_validation_layers_) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  // glfw extensions
  uint32_t glfw_extension_count = 0;
  const char** glfw_extensions =
      glfwGetRequiredInstanceExtensions(&glfw_extension_count);
  extensions.insert(extensions.end(), glfw_extensions,
                    glfw_extensions + glfw_extension_count);
  return extensions;
}

void Engine::CleanUpSwapChain() {
  render_scene_.DestroyUniform(device_->device_);
  for (auto framebuffer : framebuffers_) {
    framebuffer.Destroy(device_->device_);
  }
  for (auto framebuffer : imgui_framebuffers_) {
    framebuffer.Destroy(device_->device_);
  }
  vkFreeCommandBuffers(device_->device_, device_->command_pool_,
                       static_cast<uint32_t>(device_->command_buffers_.size()),
                       device_->command_buffers_.data());
  if (pipeline_) {
    pipeline_->Destroy(device_->device_);
  }
  if (render_pass_) {
    render_pass_->Destroy(device_->device_);
  }
  if (imgui_render_pass_) {
    imgui_render_pass_->Destroy(device_->device_);
  }
  if (GPU_SIM) {
    CheckCuda(cudaDestroyExternalSemaphore(cuda_signaled_));
    CheckCuda(cudaDestroyExternalSemaphore(cuda_wait_));
  }
  if (swap_chain_) {
    swap_chain_->Destroy();
  }
}

void Engine::RecreateSwapChain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window_, &width, &height);
    glfwWaitEvents();
  }
  vkDeviceWaitIdle(device_->device_);
  CleanUpSwapChain();
  width_ = width;
  height_ = height;
  physical_device_->swap_chain_support_details_ =
      physical_device_->QuerySwapChainSupport(physical_device_->device_,
                                              surface_, false);
  {  // create swap chain
    if (swap_chain_->Init(device_, physical_device_, window_, surface_) !=
        VK_SUCCESS) {
      spdlog::error("swap chain creation failed");
      CleanUp();
      exit(1);
    }
  }

  if (GPU_SIM) {
    ImportCudaExternalSemaphore(cuda_wait_, device_->device_,
                                swap_chain_->cuda_sim_vk_signaled_,
                                GetDefaultSemaphoreHandleType());
    ImportCudaExternalSemaphore(cuda_signaled_, device_->device_,
                                swap_chain_->cuda_sim_vk_wait_,
                                GetDefaultSemaphoreHandleType());
  }

  {
    render_scene_.InitUniform(device_, swap_chain_);
    float aspect = 1.0;
    if (swap_chain_->extent_.height)
      aspect = float(swap_chain_->extent_.width) / swap_chain_->extent_.height;
    render_scene_.camera_->ResetAspect(aspect);
    render_scene_.camera_->UpdateData();
  }

  {  // create render pass
    if (render_pass_->Init(device_, swap_chain_->image_format_) != VK_SUCCESS) {
      CleanUp();
      exit(1);
    }
    if (imgui_render_pass_->Init(device_, swap_chain_->image_format_) !=
        VK_SUCCESS) {
      CleanUp();
      exit(1);
    }
  }

  {  // create pipeline
    if (pipeline_->Init(device_->device_, swap_chain_->extent_,
                        render_pass_->render_pass_,
                        &render_scene_) != VK_SUCCESS) {
      spdlog::error("pipeline creation failed");
      CleanUp();
      exit(1);
    }
  }

  {  // create framebuffers
    framebuffers_.resize(swap_chain_->image_views_.size());
    imgui_framebuffers_.resize(swap_chain_->image_views_.size());
    for (size_t i = 0; i < swap_chain_->image_views_.size(); ++i) {
      if (framebuffers_[i].Init(device_, swap_chain_->extent_,
                                swap_chain_->image_views_[i],
                                render_pass_->render_pass_) != VK_SUCCESS) {
        CleanUp();
        exit(1);
      }
      if (imgui_framebuffers_[i].Init(
              device_, swap_chain_->extent_, swap_chain_->image_views_[i],
              imgui_render_pass_->render_pass_) != VK_SUCCESS) {
        CleanUp();
        exit(1);
      }
    }
  }

  // allocate and record command buffers
  device_->AllocateCommandBuffers(swap_chain_);

  // imgui
  ImGui_ImplVulkan_SetMinImageCount(swap_chain_->images_.size());
}

void Engine::WindowResizeCallback(GLFWwindow* window, int width, int height) {
  auto engine = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
  engine->window_resized_ = true;
}

void Engine::KeyCallback(GLFWwindow* window, int key, int scancode, int action,
                         int mods) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  auto engine = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, true);
        break;
      case GLFW_KEY_R:
        vkWaitForFences(engine->device_->device_, 1,
                        &engine->swap_chain_->cpu_sim_fence_, true, UINT64_MAX);
        if (engine->GPU_SIM) {
          engine->cuda_simulator_->Reset();
        } else {
          engine->simulator_->Reset();
          engine->render_scene_.UpdateVertex(engine->device_->device_);
          break;
        }

      case GLFW_KEY_P:
        engine->enable_sim_ = !engine->enable_sim_;
      default:
        break;
    }
  }
  return;
}

void Engine::MouseButtonCallback(GLFWwindow* window, int button, int action,
                                 int mods) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  auto engine = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
  if (action == GLFW_PRESS) {
    glfwGetCursorPos(window, &engine->last_mouse_pos_.x(),
                     &engine->last_mouse_pos_.y());
    // select vert
    Vec2f p;
    p.x() = engine->last_mouse_pos_.x() / real(engine->width_) * 2 - 1;
    p.y() = 1 - engine->last_mouse_pos_.y() / real(engine->height_) * 2;
    Vec3f origin, dir;
    engine->render_scene_.camera_->GetSelectRay(p, origin, dir);
    engine->select_mode_ =
        engine->scene_.SelectObjVert(origin.cast<real>(), dir.cast<real>());
  } else if (action == GLFW_RELEASE) {
    engine->select_mode_ = false;
    engine->scene_.ClearSelectObjVert();
  }
}

void Engine::CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  auto engine = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
    if (engine->select_mode_) {
      Vec2f p;
      p.x() = xpos / real(engine->width_) * 2 - 1;
      p.y() = 1 - ypos / real(engine->height_) * 2;
      Vec3f origin, dir;
      engine->render_scene_.camera_->GetSelectRay(p, origin, dir);
      auto& obj =
          engine->scene_.objects_[engine->scene_.selected_obj_idx_.value()];
      Vec3& select_pos = obj.vertices_[obj.selected_idx_.value()];
      obj.control_pos_ =
          origin.cast<real>() +
          (select_pos - origin.cast<real>()).dot(dir.cast<real>()) *
              dir.cast<real>();
    } else {
      engine->render_scene_.camera_->Rotate(
          float(xpos - engine->last_mouse_pos_.x()),
          float(ypos - engine->last_mouse_pos_.y()));
    }
  } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) ==
             GLFW_PRESS) {
    engine->render_scene_.camera_->Translate(
        float(xpos - engine->last_mouse_pos_.x()),
        float(ypos - engine->last_mouse_pos_.y()));
  }
  engine->last_mouse_pos_ << xpos, ypos;
}

void Engine::ScrollCallback(GLFWwindow* window, double xoffset,
                            double yoffset) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  auto engine = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
  engine->render_scene_.camera_->Scale(float(yoffset));
}
};  // namespace Rain