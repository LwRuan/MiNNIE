#include "cudahelper.h"

namespace Rain {
#if defined(_WIN64)
WindowsSecurityAttributes::WindowsSecurityAttributes() {
  descriptor_ = (PSECURITY_DESCRIPTOR)calloc(
      1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
  if (!descriptor_) {
    spdlog::error("Failed to allocate memory for security descriptor");
  }

  PSID *ppSID = (PSID *)((PBYTE)descriptor_ + SECURITY_DESCRIPTOR_MIN_LENGTH);
  PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

  InitializeSecurityDescriptor(descriptor_, SECURITY_DESCRIPTOR_REVISION);

  SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
      SECURITY_WORLD_SID_AUTHORITY;
  AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0,
                           0, 0, 0, 0, 0, ppSID);

  EXPLICIT_ACCESS explicitAccess;
  ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
  explicitAccess.grfAccessPermissions =
      STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
  explicitAccess.grfAccessMode = SET_ACCESS;
  explicitAccess.grfInheritance = INHERIT_ONLY;
  explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
  explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
  explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

  SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

  SetSecurityDescriptorDacl(descriptor_, TRUE, *ppACL, FALSE);

  attributes_.nLength = sizeof(attributes_);
  attributes_.lpSecurityDescriptor = descriptor_;
  attributes_.bInheritHandle = TRUE;
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() {
  return &attributes_;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
  PSID *ppSID = (PSID *)((PBYTE)descriptor_ + SECURITY_DESCRIPTOR_MIN_LENGTH);
  PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

  if (*ppSID) {
    FreeSid(*ppSID);
  }
  if (*ppACL) {
    LocalFree(*ppACL);
  }
  free(descriptor_);
}
#endif

void *GetSemaphoreHandle(VkDevice device, VkSemaphore semaphore,
                         VkExternalSemaphoreHandleTypeFlagBits handle_type) {
  VkResult result;
#if defined(_WIN64)
  HANDLE handle;
  VkSemaphoreGetWin32HandleInfoKHR win32_info{};
  win32_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  win32_info.pNext = nullptr;
  win32_info.semaphore = semaphore;
  win32_info.handleType = handle_type;
  PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
  fpGetSemaphoreWin32HandleKHR =
      (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
          device, "vkGetSemaphoreWin32HandleKHR");
  if (!fpGetSemaphoreWin32HandleKHR) {
    spdlog::error("vkGetSemaphoreWin32HandleKHR not found");
    return nullptr;
  }
  result = fpGetSemaphoreWin32HandleKHR(device, &win32_info, &handle);
  if (result != VK_SUCCESS) spdlog::error("semaphore handle not get");
  return handle;
#else
  int fd;
  VkSemaphoreGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.semaphore = semaphore;
  fd_info.handleType = handle_type;
  PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
  fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
      device, "vkGetSemaphoreFdKHR");
  if (!fpGetSemaphoreFdKHR) {
    spdlog::error("semaphore handle not get");
    return nullptr;
  }
  result = fpGetSemaphoreFdKHR(device, &fd_info, &fd);
  if (result != VK_SUCCESS) spdlog::error("semaphore handle not get");
  return (void *)(uintptr_t)fd;
#endif
}

void ImportCudaExternalSemaphore(
    cudaExternalSemaphore_t &cuda_sem, VkDevice device, VkSemaphore &vk_sem,
    VkExternalSemaphoreHandleTypeFlagBits handle_type) {
  cudaExternalSemaphoreHandleDesc desc{};
  if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  } else if (handle_type &
             VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
  } else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  } else {
    spdlog::error("unkown semaphore handle type");
    return;
  }
#if defined(_WIN64)
  desc.handle.win32.handle =
      (HANDLE)GetSemaphoreHandle(device, vk_sem, handle_type);
#else
  desc.handle.fd =
      (int)(uintptr_t)GetSemaphoreHandle(device, vk_sem, handle_type);
#endif
  desc.flags = 0;
  CheckCuda(cudaImportExternalSemaphore(&cuda_sem, &desc));
}

void *GetMemoryHandle(VkDevice device, VkDeviceMemory memory,
                      VkExternalMemoryHandleTypeFlagBits handle_type) {
  VkResult result;
#if defined(_WIN64)
  HANDLE handle = 0;
  VkMemoryGetWin32HandleInfoKHR win32_info = {};
  win32_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  win32_info.pNext = nullptr;
  win32_info.memory = memory;
  win32_info.handleType = handle_type;

  PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
  fpGetMemoryWin32HandleKHR =
      (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
          device, "vkGetMemoryWin32HandleKHR");
  if (!fpGetMemoryWin32HandleKHR) {
    spdlog::error("vkGetMemoryWin32HandleKHR not found");
    return nullptr;
  }
  result = fpGetMemoryWin32HandleKHR(device, &win32_info, &handle);
  if (result != VK_SUCCESS) spdlog::error("memory handle not get");
  return (void *)handle;
#else
  int fd = -1;
  VkMemoryGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.memory = memory;
  fd_info.handleType = handle_type;

  PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
  fpGetMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
  if (!fpGetMemoryFdKHR) {
    spdlog::error("vkGetMemoryFdKHR not found");
    return nullptr;
  }
  result = fpGetMemoryFdKHR(device, &fd_info, &fd);
  if (result != VK_SUCCESS) spdlog::error("memory handle not get");
  return (void *)(uintptr_t)fd;
#endif
}

void ImportCudaExternalMemory(void **cuda_ptr, cudaExternalMemory_t *cuda_mem,
                              VkDevice device, VkDeviceMemory vk_mem,
                              VkDeviceSize size,
                              VkExternalMemoryHandleTypeFlagBits handle_type) {
  cudaExternalMemoryHandleDesc handle_desc{};
  if (handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
    handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  } else if (handle_type &
             VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
    handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
  } else if (handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
    handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  } else {
    spdlog::error("unknown memory handle type");
    exit(1);
  }

  handle_desc.size = size;
#if defined(_WIN64)
  handle_desc.handle.win32.handle =
      (HANDLE)GetMemoryHandle(device, vk_mem, handle_type);
  handle_desc.handle.win32.name = nullptr;
#else
  handle_desc.handle.fd =
      (int)(uintptr_t)GetMemoryHandle(device, vk_mem, handle_type);
#endif
  CheckCuda(cudaImportExternalMemory(cuda_mem, &handle_desc));
  cudaExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = 0;
  buffer_desc.size = size;
  buffer_desc.flags = 0;
  CheckCuda(
      cudaExternalMemoryGetMappedBuffer(cuda_ptr, *cuda_mem, &buffer_desc));
}

VkExternalSemaphoreHandleTypeFlagBits GetDefaultSemaphoreHandleType() {
#if defined(_WIN64)
  return IsWindows8OrGreater()
             ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
             : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

VkExternalMemoryHandleTypeFlagBits GetDefaultMemHandleType() {
#if defined(_WIN64)
  return IsWindows8Point1OrGreater()
             ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
             : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
// cuSPARSE API errors
const char *_cudaGetErrorEnum(cusparseStatus_t error) {
  switch (error) {
    case CUSPARSE_STATUS_SUCCESS:
      return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
      return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }
  return "<unknown>";
}
// cuBLAS API errors
const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

// cuSOLVER API errors
const char *_cudaGetErrorEnum(cusolverStatus_t error) {
  switch (error) {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "CUSOLVER_STATUS_INVALID_LICENSE";
  }

  return "<unknown>";
}
};  // namespace Rain