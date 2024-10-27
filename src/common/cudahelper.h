#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusolverRf.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.h>
#if defined(_WIN64)
#define NOMINMAX
#include <AccCtrl.h>
#include <AclAPI.h>
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#include <vulkan/vulkan_win32.h>
#include <windows.h>
#endif

namespace Rain {
#if defined(_WIN64)
class WindowsSecurityAttributes {
 protected:
  SECURITY_ATTRIBUTES attributes_;
  PSECURITY_DESCRIPTOR descriptor_;

 public:
  WindowsSecurityAttributes();
  SECURITY_ATTRIBUTES *operator&();
  ~WindowsSecurityAttributes();
};
#endif

void *GetSemaphoreHandle(VkDevice device, VkSemaphore semaphore,
                         VkExternalSemaphoreHandleTypeFlagBits handle_type);

void ImportCudaExternalSemaphore(
    cudaExternalSemaphore_t &cuda_sem, VkDevice device, VkSemaphore &vk_sem,
    VkExternalSemaphoreHandleTypeFlagBits handle_type);

void *GetMemoryHandle(VkDevice device, VkDeviceMemory memory,
                      VkExternalMemoryHandleTypeFlagBits handle_type);

void ImportCudaExternalMemory(void **cuda_ptr, cudaExternalMemory_t *cuda_mem,
                              VkDevice device, VkDeviceMemory vk_mem,
                              VkDeviceSize size,
                              VkExternalMemoryHandleTypeFlagBits handle_type);

VkExternalSemaphoreHandleTypeFlagBits GetDefaultSemaphoreHandleType();
VkExternalMemoryHandleTypeFlagBits GetDefaultMemHandleType();

const char *_cudaGetErrorEnum(cudaError_t error);
// cuSPARSE API errors
const char *_cudaGetErrorEnum(cusparseStatus_t error);
// cuBLAS API errors
const char *_cudaGetErrorEnum(cublasStatus_t error);
// cuSOLVER API errors
const char *_cudaGetErrorEnum(cusolverStatus_t error);

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    spdlog::error("CUDA error at {}:{} code={}({}) \"{}\" \n", file, line,
                  static_cast<unsigned int>(result), _cudaGetErrorEnum(result),
                  func);
    exit(1);
  }

#define CheckCuda(val) check((val), #val, __FILE__, __LINE__)
}
};  // namespace Rain
