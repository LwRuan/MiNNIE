#include "device.h"

#include <spdlog/spdlog.h>

#include <set>
#include <type_traits>

#include "mathtype.h"

namespace Rain {
VkResult Device::Init(VkPhysicalDevice physical_device,
                      uint32_t graphics_queue_family_index,
                      uint32_t present_queue_family_index,
                      const std::vector<const char*>* layers,
                      const std::vector<const char*>* extensions) {
  physical_device_ = physical_device;
  vkGetPhysicalDeviceMemoryProperties(physical_device,
                                      &physicalmem_properties_);
  spdlog::debug("queue family {} picked for graphics",
                graphics_queue_family_index);
  spdlog::debug("queue family {} picked for present",
                present_queue_family_index);
  std::set<uint32_t> unique_queue_families = {graphics_queue_family_index,
                                              present_queue_family_index};
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  float queue_priority = 1.0f;
  for (uint32_t queue_family : unique_queue_families) {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = graphics_queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }
  VkPhysicalDeviceFeatures device_features{};
  if constexpr (std::is_same<real, double>())
    device_features.shaderFloat64 = true;

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.queueCreateInfoCount =
      static_cast<uint32_t>(queue_create_infos.size());
  create_info.pEnabledFeatures = &device_features;

  if (extensions) {
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(extensions->size());
    create_info.ppEnabledExtensionNames = extensions->data();
  } else
    create_info.enabledExtensionCount = 0;
  if (layers) {
    create_info.enabledLayerCount = static_cast<uint32_t>(layers->size());
    create_info.ppEnabledLayerNames = layers->data();
  } else
    create_info.enabledLayerCount = 0;
  VkResult result =
      vkCreateDevice(physical_device, &create_info, nullptr, &device_);
  if (result == VK_SUCCESS) {
    vkGetDeviceQueue(device_, graphics_queue_family_index, 0, &graphics_queue_);
    vkGetDeviceQueue(device_, present_queue_family_index, 0, &present_queue_);
  } else {
    spdlog::error("logical device creation failed");
    return result;
  }

  {  // create command pool
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = graphics_queue_family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkResult result =
        vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);
    if (result != VK_SUCCESS) {
      spdlog::error("command pool creation failed");
      return result;
    } else {
      spdlog::debug("command pool created");
    }
  }
  return VK_SUCCESS;
}

VkResult Device::AllocateCommandBuffers(SwapChain* swap_chain) {
  swap_chain_ = swap_chain;
  command_buffers_.resize(swap_chain_->images_.size());
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool_;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = (uint32_t)command_buffers_.size();
  VkResult result =
      vkAllocateCommandBuffers(device_, &alloc_info, command_buffers_.data());
  if (result != VK_SUCCESS) {
    spdlog::error("command buffers allocation failed");
    return result;
  } else {
    // spdlog::debug("command buffer allocated");
  }
  return VK_SUCCESS;
}

VkCommandBuffer Device::BeginSingleTimeCommands() {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool_;
  alloc_info.commandBufferCount = 1;
  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(command_buffer, &beginInfo);
  return command_buffer;
}

void Device::EndSingleTimeCommands(VkCommandBuffer command_buffer) {
  vkEndCommandBuffer(command_buffer);
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue_);
  vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
}

uint32_t Device::FindMemoryTypeIndex(uint32_t type_filter,
                                     VkMemoryPropertyFlags properties) {
  for (uint32_t i = 0; i < physicalmem_properties_.memoryTypeCount; ++i) {
    if ((type_filter & (1 << i)) &&
        (physicalmem_properties_.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }
  spdlog::error("no suitable memory type found");
  exit(1);
  return 0;
}

VkFormat Device::FindDepthFormat() {
  return FindSupportFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                            VK_FORMAT_D24_UNORM_S8_UINT},
                           VK_IMAGE_TILING_OPTIMAL,
                           VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VkFormat Device::FindSupportFormat(const std::vector<VkFormat>& candidates,
                                   VkImageTiling tiling,
                                   VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physical_device_, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }
  return VK_FORMAT_UNDEFINED;
}

uint32_t Device::GetAlignedUniformByteOffset(const uint32_t offset) {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physical_device_, &props);
  static const uint32_t min_alignment =
      props.limits.minUniformBufferOffsetAlignment;
  uint32_t aligned_offset = offset + min_alignment - 1;
  aligned_offset = aligned_offset - (aligned_offset % min_alignment);
  return aligned_offset;
}

void Device::Destroy() {
  if (command_pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, command_pool_, nullptr);
    spdlog::debug("command pool destroyed");
  }
  if (device_) {
    vkDestroyDevice(device_, nullptr);
    spdlog::debug("logical device destroyed");
  }
}
};  // namespace Rain