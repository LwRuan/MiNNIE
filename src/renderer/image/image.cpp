#include "image.h"

namespace Rain {
VkResult Image::InitDepthImage(Device* device, uint32_t width,
                               uint32_t height) {
  VkResult result;
  format_ = device->FindDepthFormat();
  result = CreateImage(device, width, height, format_, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (result != VK_SUCCESS) {
    return result;
  }
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format_;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;
  result = vkCreateImageView(device->device_, &view_info, nullptr, &view_);
  if (result != VK_SUCCESS) {
    spdlog::error("image view creation failed");
    return result;
  }
  // TransitionLayout(device, VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL);

  return VK_SUCCESS;
}

VkResult Image::InitColorImage(Device* device, VkFormat format, uint32_t width,
                               uint32_t height) {
  VkResult result;
  result = CreateImage(device, width, height, format, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (result != VK_SUCCESS) {
    return result;
  }
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format_;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;
  result = vkCreateImageView(device->device_, &view_info, nullptr, &view_);
  if (result != VK_SUCCESS) {
    spdlog::error("image view creation failed");
    return result;
  }

  return VK_SUCCESS;
}

VkResult Image::CreateImage(Device* device, uint32_t width, uint32_t height,
                            VkFormat format, VkImageTiling tiling,
                            VkImageUsageFlags usages,
                            VkMemoryPropertyFlags properties) {
  VkResult result;
  width_ = width;
  height_ = height;
  format_ = format;
  usages_ = usages;
  layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.extent.width = width_;
  image_info.extent.height = height_;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = tiling;
  image_info.format = format_;
  image_info.usage = usages_;
  image_info.initialLayout = layout_;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  result = vkCreateImage(device->device_, &image_info, nullptr, &image_);
  if (result != VK_SUCCESS) {
    spdlog::error("image creation failed");
    return result;
  }

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  VkMemoryRequirements mem_reqs;
  vkGetImageMemoryRequirements(device->device_, image_, &mem_reqs);
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex =
      device->FindMemoryTypeIndex(mem_reqs.memoryTypeBits, properties);
  result = vkAllocateMemory(device->device_, &alloc_info, nullptr, &memory_);
  if (result != VK_SUCCESS) {
    spdlog::error("memory allocation failed");
    return result;
  }

  result = vkBindImageMemory(device->device_, image_, memory_, 0);
  if (result != VK_SUCCESS) {
    spdlog::error("image memory binding failed");
    return result;
  }

  return VK_SUCCESS;
}

void Image::TransitionLayout(Device* device, VkImageLayout new_layout) {
  VkCommandBuffer command_buffer = device->BeginSingleTimeCommands();
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = layout_;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image_;
  if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (HasStencilComponent(format_)) {
      barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  } else {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  }
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  VkPipelineStageFlags source_stage;
  VkPipelineStageFlags destination_stage;
  if (layout_ == VK_IMAGE_LAYOUT_UNDEFINED &&
      new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (layout_ == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (layout_ == VK_IMAGE_LAYOUT_UNDEFINED &&
             new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  } else {
    spdlog::error("layout transition unsupported");
  }
  vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);
  device->EndSingleTimeCommands(command_buffer);
  layout_ = new_layout;
}

void Image::Destroy(VkDevice device) {
  if (view_ != VK_NULL_HANDLE) vkDestroyImageView(device, view_, nullptr);
  if (image_ != VK_NULL_HANDLE) vkDestroyImage(device, image_, nullptr);
  if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device, memory_, nullptr);
}
};  // namespace Rain