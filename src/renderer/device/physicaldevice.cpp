#include "physicaldevice.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "mathtype.h"

namespace Rain {
VkBool32 PhysicalDevice::Init(VkInstance instance, VkSurfaceKHR surface) {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  if (device_count == 0) {
    spdlog::error("no avaliable GPU with Vulkan support");
    return VK_FALSE;
  } else
    spdlog::info("{} GPU(s) found", device_count);
  std::vector<VkPhysicalDevice> physical_devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());
  std::vector<int> device_scores(device_count);
  for (auto i = 0; i < device_count; ++i) {
    const auto& device = physical_devices[i];
    VkPhysicalDeviceProperties device_properties;
    VkPhysicalDeviceFeatures device_features;
    vkGetPhysicalDeviceProperties(device, &device_properties);
    vkGetPhysicalDeviceFeatures(device, &device_features);
    spdlog::info("GPU{}: {}", i, device_properties.deviceName);
    if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      device_scores[i] += 1000;
    }
    if (std::is_same<real, double>() && !device_features.shaderFloat64) {
      device_scores[i] = -1;
    }
    QueueFamilyIndices indices = FindQueueFamilies(device, surface, true);
    bool ext_supported = CheckDeviceExtensionSupport(device, true);
    bool swap_chain_adequate = false;
    if (ext_supported) {
      SwapChainSupportDetails details =
          QuerySwapChainSupport(device, surface, true);
      swap_chain_adequate =
          !details.formats_.empty() && !details.present_modes_.empty();
    }
    if ((!indices.IsComplete()) || (!ext_supported) || (!swap_chain_adequate))
      device_scores[i] = -1;
  }
  int idx_device = std::distance(
      device_scores.begin(),
      std::max_element(device_scores.begin(), device_scores.end()));
  if (device_scores[idx_device] < 0) {
    // no queue family+extension supported
    spdlog::error("no GPU with sufficient Vulkan support");
    return VK_FALSE;
  }
  device_ = physical_devices[idx_device];
  queue_family_indices_ = FindQueueFamilies(device_, surface, false);
  swap_chain_support_details_ = QuerySwapChainSupport(device_, surface, false);

  VkPhysicalDeviceIDProperties ID_props{};
  ID_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  ID_props.pNext = nullptr;
  VkPhysicalDeviceProperties2 props_2{};
  props_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props_2.pNext = &ID_props;
  PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;
  fpGetPhysicalDeviceProperties2 =
      (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
          instance, "vkGetPhysicalDeviceProperties2");
  if (!fpGetPhysicalDeviceProperties2) {
    spdlog::error("vkGetPhysicalDeviceProperties2 not found");
    return VK_ERROR_UNKNOWN;
  }
  fpGetPhysicalDeviceProperties2(device_, &props_2);
  memcpy(device_UUID_, ID_props.deviceUUID, VK_UUID_SIZE);
  spdlog::info("GPU{} picked for Vulkan", idx_device);
  return VK_SUCCESS;
}

void PhysicalDevice::GetGraphicsPresentQueueFamily(
    uint32_t& graphics_queue_family, uint32_t& present_queue_family) {
  std::vector<uint32_t> intersect_family{};
  std::set_intersection(queue_family_indices_.graphics_family_.begin(),
                        queue_family_indices_.graphics_family_.end(),
                        queue_family_indices_.present_family_.begin(),
                        queue_family_indices_.present_family_.end(),
                        std::back_inserter(intersect_family));
  if (intersect_family.size()) {
    graphics_queue_family = intersect_family[0];
    present_queue_family = graphics_queue_family;
  } else {
    graphics_queue_family = *queue_family_indices_.graphics_family_.begin();
    present_queue_family = *queue_family_indices_.present_family_.begin();
  }
}

PhysicalDevice::QueueFamilyIndices PhysicalDevice::FindQueueFamilies(
    VkPhysicalDevice device, VkSurfaceKHR surface, bool verbose) {
  QueueFamilyIndices indices;
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           queue_families.data());
  for (auto i = 0; i < queue_family_count; ++i) {
    const auto& queue_family = queue_families[i];
    std::string flags = "";
    if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      flags += " graphics";
      indices.graphics_family_.insert(i);
    }
    if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      indices.compute_family_.insert(i);
      flags += " compute";
    }
    if (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT) {
      flags += " transfer";
    }
    VkBool32 present_support = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
    if (present_support) {
      indices.present_family_.insert(i);
      flags += " present";
    }
    if (verbose)
      spdlog::debug(" queue family {} : count {},{}", i,
                    queue_family.queueCount, flags);
  }
  return indices;
}

bool PhysicalDevice::CheckDeviceExtensionSupport(VkPhysicalDevice device,
                                                 bool verbose) {
  uint32_t extension_count = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       nullptr);
  std::vector<VkExtensionProperties> available_extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       available_extensions.data());
  std::set<std::string> required_extensions(device_extensions_.begin(),
                                            device_extensions_.end());
  for (const auto& extension : available_extensions) {
    required_extensions.erase(extension.extensionName);
  }
  return required_extensions.empty();
}

PhysicalDevice::SwapChainSupportDetails PhysicalDevice::QuerySwapChainSupport(
    VkPhysicalDevice device, VkSurfaceKHR surface, bool verbose) {
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                            &details.capabilities_);

  uint32_t format_count = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
  if (format_count != 0) {
    details.formats_.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count,
                                         details.formats_.data());
  }

  uint32_t present_mode_count = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                            &present_mode_count, nullptr);
  if (present_mode_count != 0) {
    details.present_modes_.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface, &present_mode_count, details.present_modes_.data());
  }
  return details;
}

VkSurfaceFormatKHR PhysicalDevice::ChooseSurfaceFormat() {
  for (const auto& format : swap_chain_support_details_.formats_) {
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      // spdlog::debug("surface format picked: sRGB32 nonlinear");
      return format;
    }
  }
  const auto& format = swap_chain_support_details_.formats_[0];
  // spdlog::debug("default surface format picked: format {}, color space {}",
  //               format.format, format.colorSpace);
  return swap_chain_support_details_.formats_[0];
}

VkPresentModeKHR PhysicalDevice::ChoosePresentMode(
    const VkPresentModeKHR& recommand) {
  std::optional<VkPresentModeKHR> ret;
  for (const auto& mode : swap_chain_support_details_.present_modes_) {
    if (mode == recommand) {
      ret = mode;
      break;
    }
  }
  if (!ret.has_value()) ret = swap_chain_support_details_.present_modes_[0];
  // if (ret.value() == VK_PRESENT_MODE_FIFO_KHR)
  //   spdlog::debug("present mode picked: FIFO");
  // else if (ret.value() == VK_PRESENT_MODE_MAILBOX_KHR)
  //   spdlog::debug("present mode picked: MAILBOX");
  // else
  //   spdlog::debug("present mode picked: {}", ret.value());
  return ret.value();
}

VkExtent2D PhysicalDevice::ChooseSwapExtent(GLFWwindow* window) {
  VkExtent2D extent = swap_chain_support_details_.capabilities_.currentExtent;
  if (extent.width != UINT32_MAX) {
    spdlog::debug("swap extent picked: {}*{}", extent.width, extent.height);
    return extent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    extent.width = static_cast<uint32_t>(width);
    extent.height = static_cast<uint32_t>(height);
    extent.width = std::clamp(
        extent.width,
        swap_chain_support_details_.capabilities_.minImageExtent.width,
        swap_chain_support_details_.capabilities_.maxImageExtent.width);
    extent.height = std::clamp(
        extent.height,
        swap_chain_support_details_.capabilities_.minImageExtent.height,
        swap_chain_support_details_.capabilities_.maxImageExtent.height);
    spdlog::debug("swap extent picked: {}*{}", extent.width, extent.height);
    return extent;
  }
}

};  // namespace Rain