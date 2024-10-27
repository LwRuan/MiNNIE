#include "debugutils.h"

#include "spdlog/spdlog.h"
#include <iostream>

namespace Rain {

VkResult DebugUtilsEXT::Init(VkInstance instance) {
  VkDebugUtilsMessengerCreateInfoEXT create_info{};
  GetCreateInfo(create_info);
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, &create_info, nullptr, &debug_messenger_);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DebugUtilsEXT::GetCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = DebugCallBack;
}

void DebugUtilsEXT::Destroy(VkInstance instance) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func && debug_messenger_) {
    spdlog::debug("debug messenger destroyed");
    func(instance, debug_messenger_, nullptr);
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsEXT::DebugCallBack(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
  if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    spdlog::error("validation layer: {}", pCallbackData->pMessage);
  } else if (messageSeverity ==
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    spdlog::warn("validation layer: {}", pCallbackData->pMessage);
  }
  return VK_FALSE;
}
};  // namespace Rain