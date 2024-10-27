#include "framebuffer.h"

#include <array>

namespace Rain {
VkResult Framebuffer::Init(Device* device, const VkExtent2D& extent,
                           VkImageView swap_image_view,
                           VkRenderPass render_pass) {
  VkResult result;
  result = depth_image_.InitDepthImage(device, extent.width, extent.height);
  if (result != VK_SUCCESS) return result;
  swap_image_view_ = swap_image_view;

  std::array<VkImageView, 2> attachments = {swap_image_view_,
                                            depth_image_.view_};

  VkFramebufferCreateInfo framebuffer_info{};
  framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_info.renderPass = render_pass;
  framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
  framebuffer_info.pAttachments = attachments.data();
  framebuffer_info.width = extent.width;
  framebuffer_info.height = extent.height;
  framebuffer_info.layers = 1;

  result = vkCreateFramebuffer(device->device_, &framebuffer_info, nullptr,
                               &framebuffer_);
  if (result != VK_SUCCESS) {
    spdlog::error("framebuffer creation failed");
  }
  return result;
};

void Framebuffer::Destroy(VkDevice device) {
  depth_image_.Destroy(device);
  if (framebuffer_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(device, framebuffer_, nullptr);
  }
}

VkResult ImGuiFramebuffer::Init(Device* device, const VkExtent2D& extent,
                           VkImageView swap_image_view,
                           VkRenderPass render_pass) {
  VkResult result;
  swap_image_view_ = swap_image_view;

  VkFramebufferCreateInfo framebuffer_info{};
  framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_info.renderPass = render_pass;
  framebuffer_info.attachmentCount = 1;
  framebuffer_info.pAttachments = &swap_image_view_;
  framebuffer_info.width = extent.width;
  framebuffer_info.height = extent.height;
  framebuffer_info.layers = 1;

  result = vkCreateFramebuffer(device->device_, &framebuffer_info, nullptr,
                               &framebuffer_);
  if (result != VK_SUCCESS) {
    spdlog::error("framebuffer creation failed");
  }
  return result;
};

void ImGuiFramebuffer::Destroy(VkDevice device) {
  if (framebuffer_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(device, framebuffer_, nullptr);
  }
}
};  // namespace Rain