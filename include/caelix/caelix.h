#ifndef CAELIX_H
#define CAELIX_H
/*
 * Caelix renderer C ABI. C99, no Vulkan headers.
 *
 * Vulkan objects cross this ABI as plain integers and pointers so that a host
 * does not need the Vulkan headers to bind the library:
 *   - Every Vulkan handle, dispatchable (VkInstance, VkDevice, VkQueue,
 *     VkCommandBuffer, ...) and non-dispatchable (VkImage, ...), is a uint64_t
 *     that holds the raw handle value.
 *   - Vulkan function pointers are void*.
 *   - Vulkan enums (VkFormat, VkImageLayout, ...) are uint32_t; the doc comment
 *     of each field names the Vulkan type.
 *
 * Ownership model: the HOST creates and owns the Vulkan instance, device,
 * queues, swapchain and command buffers. The library adopts the host's device
 * (caelix_device_adopt) and records its work into command buffers that the
 * host provides. The library never submits or presents.
 */
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32)
  #if defined(CAELIX_BUILD_SHARED)
    #define CAELIX_API __declspec(dllexport)
  #else
    #define CAELIX_API __declspec(dllimport)
  #endif
#else
  #define CAELIX_API __attribute__((visibility("default")))
#endif

/* Bumped on every breaking change of this header. */
#define CAELIX_ABI_VERSION 1u

typedef struct CaelixVersion {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
  uint32_t abi;   /* == CAELIX_ABI_VERSION of the built library */
} CaelixVersion;

/* Fills *out with the library's version. Never fails. */
CAELIX_API void caelix_get_version(CaelixVersion* out);
/* Static string like "0.1.0", owned by the library. */
CAELIX_API const char* caelix_version_string(void);

/* Result of every fallible ABI call. */
typedef enum CaelixResult {
  CAELIX_OK = 0,
  CAELIX_ERROR_INVALID_ARGUMENT = 1,
  CAELIX_ERROR_UNSUPPORTED = 2,      /* the adopted device lacks a requirement; see log */
  CAELIX_ERROR_VULKAN = 3,           /* a Vulkan call failed; see log */
  CAELIX_ERROR_OUT_OF_MEMORY = 4
} CaelixResult;

/* ---------------------------------------------------------------------------
 * Logging
 * ------------------------------------------------------------------------- */

typedef enum CaelixLogLevel {
  CAELIX_LOG_DEBUG = 0,
  CAELIX_LOG_INFO = 1,
  CAELIX_LOG_WARNING = 2,
  CAELIX_LOG_ERROR = 3
} CaelixLogLevel;

/* `message` is a NUL-terminated UTF-8 string, valid only during the call. */
typedef void (*CaelixLogCallback)(CaelixLogLevel level, const char* message, void* user_data);
/* Thread-safe to set; messages may come from any thread the library is called on. NULL disables logging. */
CAELIX_API void caelix_set_log_callback(CaelixLogCallback callback, void* user_data);

/* ---------------------------------------------------------------------------
 * Device requirements and adoption
 * ------------------------------------------------------------------------- */

/* Device features the library needs ENABLED on the adopted device. Vulkan 1.3 core features are
   listed too because the host must still enable them at vkCreateDevice. */
typedef enum CaelixDeviceFeatureBits {
  CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS   = 1u << 0, /* VkPhysicalDeviceVulkan12Features::bufferDeviceAddress */
  CAELIX_FEATURE_DESCRIPTOR_INDEXING     = 1u << 1, /* runtimeDescriptorArray, shaderStorageBufferArrayNonUniformIndexing, descriptorBindingPartiallyBound, descriptorBindingVariableDescriptorCount */
  CAELIX_FEATURE_SYNCHRONIZATION2        = 1u << 2, /* VkPhysicalDeviceVulkan13Features::synchronization2 */
  CAELIX_FEATURE_DYNAMIC_RENDERING       = 1u << 3, /* VkPhysicalDeviceVulkan13Features::dynamicRendering */
  CAELIX_FEATURE_TIMELINE_SEMAPHORE      = 1u << 4, /* VkPhysicalDeviceVulkan12Features::timelineSemaphore */
  CAELIX_FEATURE_ACCELERATION_STRUCTURE  = 1u << 5, /* VkPhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure */
  CAELIX_FEATURE_RAY_QUERY               = 1u << 6, /* VkPhysicalDeviceRayQueryFeaturesKHR::rayQuery */
  CAELIX_FEATURE_SHADER_INT64            = 1u << 7, /* VkPhysicalDeviceFeatures::shaderInt64 */
  CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT     = 1u << 8  /* VkPhysicalDeviceVulkan12Features::scalarBlockLayout */
} CaelixDeviceFeatureBits;

typedef struct CaelixDeviceRequirements {
  uint32_t api_version;                 /* VK_MAKE_API_VERSION(0,1,3,0) */
  uint32_t device_extension_count;
  const char* const* device_extensions; /* static strings owned by the library */
  uint32_t feature_flags;               /* CaelixDeviceFeatureBits */
} CaelixDeviceRequirements;

/* Fills *out with what the host must request at vkCreateInstance / vkCreateDevice.
   Call before creating the device. Never fails; a NULL `out` is ignored. */
CAELIX_API void caelix_get_device_requirements(CaelixDeviceRequirements* out);

typedef struct CaelixDeviceAdoptDesc {
  void*    get_instance_proc_addr;  /* PFN_vkGetInstanceProcAddr */
  uint64_t instance;                /* VkInstance */
  uint64_t physical_device;         /* VkPhysicalDevice */
  uint64_t device;                  /* VkDevice */
  uint64_t graphics_queue;          /* VkQueue, graphics + compute capable */
  uint32_t graphics_queue_family;
} CaelixDeviceAdoptDesc;

typedef struct CaelixDevice CaelixDevice;  /* opaque */

/* Adopts the host's device. Checks that the physical device supports every requirement of
   caelix_get_device_requirements; a missing item is logged and returns CAELIX_ERROR_UNSUPPORTED.
   The library loads Vulkan through `get_instance_proc_addr`, so it uses the host's loader and
   layers. One adopted device per process at a time.
   The host keeps ownership of every Vulkan object in the desc and must keep them alive until caelix_device_destroy. */
CAELIX_API CaelixResult caelix_device_adopt(const CaelixDeviceAdoptDesc* desc, CaelixDevice** out_device);
/* Releases the library's objects on the device (never the host's objects). NULL is ignored.
   The host must ensure the device is idle (vkDeviceWaitIdle) before calling this. */
CAELIX_API void caelix_device_destroy(CaelixDevice* device);

/* ---------------------------------------------------------------------------
 * Render targets
 * ------------------------------------------------------------------------- */

typedef struct CaelixTargetImage {
  uint64_t image;   /* VkImage */
  uint32_t format;  /* VkFormat */
  uint32_t width;
  uint32_t height;
  uint32_t layout;  /* VkImageLayout of the image when the library starts recording */
} CaelixTargetImage;

/* TEMPORARY, removed when the renderer lands: records into `command_buffer` (a VkCommandBuffer in the
   recording state, owned by the host) a clear of `target` to `rgba`. On return the image is in
   VK_IMAGE_LAYOUT_GENERAL. This is the layout contract every later render call keeps: the library
   leaves the target in GENERAL; the host transitions it for present.
   The target image needs VK_IMAGE_USAGE_TRANSFER_DST_BIT. */
CAELIX_API CaelixResult caelix_clear_target(CaelixDevice* device, uint64_t command_buffer,
                                            const CaelixTargetImage* target, const float rgba[4]);

#ifdef __cplusplus
}
#endif
#endif
