#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <cmath>
#include <vector>

#include "cudahelper.h"
#include "camera/camera.h"
#include "device/device.h"
#include "device/physicaldevice.h"
#include "framebuffer/framebuffer.h"
#include "imgui.h"
#include "mathtype.h"
#include "pipeline/pipeline.h"
#include "renderpass/renderpass.h"
#include "renderscene/renderscene.h"
#include "scene/scene.h"
#include "simulator/cudasimulator.h"
#include "simulator/simulator.h"
#include "steptimer.h"
#include "surface/swapchain.h"
#include "vkext/debugutils.h"

namespace Rain {
class Engine {
 public:
  bool GPU_SIM = true;
  bool enable_sim_ = false;
  bool step_sim_ = false;
  GLFWwindow* window_ = nullptr;
  VkInstance instance_ = VK_NULL_HANDLE;
  PhysicalDevice* physical_device_ = nullptr;
  Device* device_ = nullptr;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  SwapChain* swap_chain_ = nullptr;
  RenderPass* render_pass_ = nullptr;
  Pipeline* pipeline_ = nullptr;
  std::vector<Framebuffer> framebuffers_;
  StepTimer timer_;
  float timestep_= 0;
  float fps_ = 1000.0f;
  float dt_ = 1e-8f;

  VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;
  ImGuiRenderPass* imgui_render_pass_ = nullptr;
  std::vector<ImGuiFramebuffer> imgui_framebuffers_;

  DebugUtilsEXT* debug_utils_ext_ = nullptr;
  bool enable_validation_layers_;
  const std::vector<const char*> validation_layers_ = {
      "VK_LAYER_KHRONOS_validation"};
  const std::vector<const char*> instance_extensions_ = {
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME
  };

  uint32_t width_ = 1080;
  uint32_t height_ = 720;
  float width_scale_;
  float height_scale_;
  bool window_resized_ = false;

  Scene scene_;
  RenderScene render_scene_;
  Simulator* simulator_ = nullptr;

  CudaSimulator* cuda_simulator_ = nullptr;
  cudaExternalSemaphore_t cuda_wait_;
  cudaExternalSemaphore_t cuda_signaled_;

  Vec2d last_mouse_pos_;
  bool select_mode_ = false;

  Engine();
  void Init(const std::string& config_file);
  VkResult InitImGui();
  void DrawFrame();
  void UpdateGlobalUniformBuffer(uint32_t image_index);
  void MainLoop();
  void CleanUp();
  bool CheckValidationLayerSupport();
  std::vector<const char*> GetRequiredExtensions();
  void CleanUpSwapChain();
  void RecreateSwapChain();

  static void WindowResizeCallback(GLFWwindow* window, int width, int height);
  static void KeyCallback(GLFWwindow* window, int key, int scancode, int action,
                          int mods);
  static void MouseButtonCallback(GLFWwindow* window, int button, int action,
                                  int mods);
  static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
  static void ScrollCallback(GLFWwindow* window, double xoffset,
                             double yoffset);
};
};  // namespace Rain