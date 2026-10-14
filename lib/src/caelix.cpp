#include "caelix/caelix.h"

// Included only to prove the include paths of the dependencies.
// No Vulkan function is called yet.
#include <volk.h>
#include <vk_mem_alloc.h>
#include <VkBootstrap.h>

extern "C" {

CAELIX_API void caelix_get_version(CaelixVersion* out) {
  if (out == nullptr) {
    return;
  }
  out->major = CAELIX_VERSION_MAJOR;
  out->minor = CAELIX_VERSION_MINOR;
  out->patch = CAELIX_VERSION_PATCH;
  out->abi = CAELIX_ABI_VERSION;
}

CAELIX_API const char* caelix_version_string(void) {
  return CAELIX_VERSION_STRING;
}

}  // extern "C"
