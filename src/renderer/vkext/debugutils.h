#pragma once

#include <vulkan/vulkan.h>

namespace Rain {

class DebugUtilsEXT {
 public:
  VkDebugUtilsMessengerEXT debug_messenger_ = nullptr;

  VkResult Init(VkInstance instance);
  void Destroy(VkInstance instance);

  static void GetCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  DebugCallBack(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                  VkDebugUtilsMessageTypeFlagsEXT messageType,
                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                  void* pUserData);
};

};  // namespace Rain