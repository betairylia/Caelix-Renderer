// The C ABI. Every function here is extern "C" and exception-free: internal
// code may throw, and each entry point converts exceptions to a CaelixResult.
#include "caelix/caelix.h"

#include "device.h"
#include "gpu_buffer.h"
#include "log.h"
#include "scene.h"
#include "vk_common.h"

#include <memory>
#include <new>

// The opaque handle behind CaelixDevice*.
struct CaelixDevice {
  std::unique_ptr<caelix::Device> impl;
};

// The opaque handle behind CaelixScene*.
struct CaelixScene {
  std::unique_ptr<caelix::Scene> impl;
};

namespace {

// Converts the exception in flight to a CaelixResult and logs it. Call only
// inside a catch block.
CaelixResult TranslateException(const char* function) {
  try {
    throw;
  } catch (const caelix::VulkanError& error) {
    const VkResult result = error.result();
    CAELIX_LOGE("%s: %s (VkResult %d)", function, error.what(), static_cast<int>(result));
    return result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY
               ? CAELIX_ERROR_OUT_OF_MEMORY
               : CAELIX_ERROR_VULKAN;
  } catch (const std::bad_alloc&) {
    CAELIX_LOGE("%s: out of memory", function);
    return CAELIX_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    CAELIX_LOGE("%s: unexpected exception", function);
    return CAELIX_ERROR_VULKAN;
  }
}

// The live scene behind a handle, or null (logged) when the handle or the entity is not valid.
caelix::Scene* SceneFor(CaelixScene* scene, const char* function) {
  if (scene == nullptr || scene->impl == nullptr) {
    CAELIX_LOGE("%s: scene is null", function);
    return nullptr;
  }
  return scene->impl.get();
}

bool CheckEntity(const caelix::Scene& scene, CaelixEntityId entity, const char* function) {
  if (!scene.EntityAlive(entity)) {
    CAELIX_LOGE("%s: entity %u does not exist", function, entity);
    return false;
  }
  return true;
}

CaelixResult ClearTarget(CaelixDevice* device, uint64_t command_buffer,
                         const CaelixTargetImage* target, const float rgba[4]) {
  if (device == nullptr || device->impl == nullptr || command_buffer == 0 ||
      target == nullptr || rgba == nullptr || target->image == 0) {
    CAELIX_LOGE("caelix_clear_target: null argument");
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  const VkCommandBuffer cmd = caelix::HandleFromU64<VkCommandBuffer>(command_buffer);
  const VkImage image = caelix::HandleFromU64<VkImage>(target->image);
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                                      VK_REMAINING_ARRAY_LAYERS};

  // Any layout -> TRANSFER_DST_OPTIMAL. The source scope is ALL_COMMANDS, so it
  // chains with whatever the host did before (a semaphore wait, an earlier pass).
  VkImageMemoryBarrier2 to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  to_transfer.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  to_transfer.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  to_transfer.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
  to_transfer.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  to_transfer.oldLayout = static_cast<VkImageLayout>(target->layout);
  to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_transfer.image = image;
  to_transfer.subresourceRange = range;

  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &to_transfer;
  vkCmdPipelineBarrier2(cmd, &dependency);

  VkClearColorValue color{};
  color.float32[0] = rgba[0];
  color.float32[1] = rgba[1];
  color.float32[2] = rgba[2];
  color.float32[3] = rgba[3];
  vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);

  // TRANSFER_DST_OPTIMAL -> GENERAL, the layout contract of every render call.
  // The destination scope is ALL_COMMANDS; the host adds its own barrier for present.
  VkImageMemoryBarrier2 to_general = to_transfer;
  to_general.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
  to_general.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  to_general.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  to_general.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  dependency.pImageMemoryBarriers = &to_general;
  vkCmdPipelineBarrier2(cmd, &dependency);
  return CAELIX_OK;
}

}  // namespace

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

CAELIX_API void caelix_set_log_callback(CaelixLogCallback callback, void* user_data) {
  caelix::SetLogCallback(callback, user_data);
}

CAELIX_API void caelix_get_device_requirements(CaelixDeviceRequirements* out) {
  if (out == nullptr) {
    return;
  }
  *out = caelix::GetDeviceRequirements();
}

CAELIX_API CaelixResult caelix_device_adopt(const CaelixDeviceAdoptDesc* desc,
                                            CaelixDevice** out_device) {
  if (out_device == nullptr) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  *out_device = nullptr;
  if (desc == nullptr) {
    CAELIX_LOGE("caelix_device_adopt: desc is null");
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    std::unique_ptr<caelix::Device> impl;
    const CaelixResult result = caelix::Device::Adopt(*desc, impl);
    if (result != CAELIX_OK) {
      return result;
    }
    *out_device = new CaelixDevice{std::move(impl)};
    return CAELIX_OK;
  } catch (const std::bad_alloc&) {
    CAELIX_LOGE("caelix_device_adopt: out of memory");
    return CAELIX_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    CAELIX_LOGE("caelix_device_adopt: unexpected exception");
    return CAELIX_ERROR_VULKAN;
  }
}

CAELIX_API void caelix_device_destroy(CaelixDevice* device) {
  try {
    delete device;
  } catch (...) {
    CAELIX_LOGE("caelix_device_destroy: unexpected exception");
  }
}

CAELIX_API CaelixResult caelix_clear_target(CaelixDevice* device, uint64_t command_buffer,
                                            const CaelixTargetImage* target,
                                            const float rgba[4]) {
  try {
    return ClearTarget(device, command_buffer, target, rgba);
  } catch (const std::bad_alloc&) {
    CAELIX_LOGE("caelix_clear_target: out of memory");
    return CAELIX_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    CAELIX_LOGE("caelix_clear_target: unexpected exception");
    return CAELIX_ERROR_VULKAN;
  }
}

CAELIX_API CaelixResult caelix_scene_create(CaelixDevice* device, const CaelixSceneDesc* desc,
                                            CaelixScene** out_scene) {
  if (out_scene == nullptr) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  *out_scene = nullptr;
  if (device == nullptr || device->impl == nullptr) {
    CAELIX_LOGE("caelix_scene_create: device is null");
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    std::unique_ptr<caelix::Scene> impl;
    const CaelixResult result = caelix::Scene::Create(*device->impl, desc, impl);
    if (result != CAELIX_OK) {
      return result;
    }
    *out_scene = new CaelixScene{std::move(impl)};
    return CAELIX_OK;
  } catch (...) {
    return TranslateException("caelix_scene_create");
  }
}

CAELIX_API void caelix_scene_destroy(CaelixScene* scene) {
  try {
    delete scene;
  } catch (...) {
    CAELIX_LOGE("caelix_scene_destroy: unexpected exception");
  }
}

CAELIX_API CaelixResult caelix_scene_add_entity(CaelixScene* scene, const CaelixEntityDesc* desc,
                                                CaelixEntityId* out_id) {
  constexpr const char* kFunction = "caelix_scene_add_entity";
  if (out_id != nullptr) {
    *out_id = CAELIX_INVALID_ENTITY;
  }
  caelix::Scene* impl = SceneFor(scene, kFunction);
  if (impl == nullptr || desc == nullptr || out_id == nullptr) {
    if (impl != nullptr) {
      CAELIX_LOGE("%s: null argument", kFunction);
    }
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    return impl->AddEntity(*desc, *out_id);
  } catch (...) {
    return TranslateException(kFunction);
  }
}

CAELIX_API CaelixResult caelix_scene_remove_entity(CaelixScene* scene, CaelixEntityId entity) {
  constexpr const char* kFunction = "caelix_scene_remove_entity";
  caelix::Scene* impl = SceneFor(scene, kFunction);
  if (impl == nullptr || !CheckEntity(*impl, entity, kFunction)) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    return impl->RemoveEntity(entity);
  } catch (...) {
    return TranslateException(kFunction);
  }
}

CAELIX_API CaelixResult caelix_scene_set_entity_transform(CaelixScene* scene, CaelixEntityId entity,
                                                          const float position[3],
                                                          const float rotation[4]) {
  constexpr const char* kFunction = "caelix_scene_set_entity_transform";
  caelix::Scene* impl = SceneFor(scene, kFunction);
  if (impl == nullptr || !CheckEntity(*impl, entity, kFunction)) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (position == nullptr || rotation == nullptr) {
    CAELIX_LOGE("%s: null argument", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    return impl->SetEntityTransform(entity, position, rotation);
  } catch (...) {
    return TranslateException(kFunction);
  }
}

CAELIX_API CaelixResult caelix_scene_upload_bricks(CaelixScene* scene, CaelixEntityId entity,
                                                   uint32_t count, const CaelixBrickKey* keys,
                                                   const void* blocks, uint32_t* consumed) {
  constexpr const char* kFunction = "caelix_scene_upload_bricks";
  if (consumed != nullptr) {
    *consumed = 0;
  }
  caelix::Scene* impl = SceneFor(scene, kFunction);
  if (impl == nullptr || !CheckEntity(*impl, entity, kFunction)) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (consumed == nullptr || (count > 0 && (keys == nullptr || blocks == nullptr))) {
    CAELIX_LOGE("%s: null argument", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    return impl->UploadBricks(entity, count, keys, blocks, *consumed);
  } catch (...) {
    return TranslateException(kFunction);
  }
}

CAELIX_API CaelixResult caelix_scene_remove_bricks(CaelixScene* scene, CaelixEntityId entity,
                                                   uint32_t count, const CaelixBrickKey* keys) {
  constexpr const char* kFunction = "caelix_scene_remove_bricks";
  caelix::Scene* impl = SceneFor(scene, kFunction);
  if (impl == nullptr || !CheckEntity(*impl, entity, kFunction)) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (count > 0 && keys == nullptr) {
    CAELIX_LOGE("%s: null argument", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    return impl->RemoveBricks(entity, count, keys);
  } catch (...) {
    return TranslateException(kFunction);
  }
}

CAELIX_API CaelixResult caelix_scene_flush(CaelixScene* scene, const CaelixFrameDesc* desc) {
  constexpr const char* kFunction = "caelix_scene_flush";
  caelix::Scene* impl = SceneFor(scene, kFunction);
  if (impl == nullptr) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (desc == nullptr || desc->command_buffer == 0) {
    CAELIX_LOGE("%s: desc or its command buffer is null", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    return impl->Flush(*desc);
  } catch (...) {
    return TranslateException(kFunction);
  }
}

CAELIX_API CaelixResult caelix_scene_recycle(CaelixScene* scene, uint64_t safe_frame_number) {
  constexpr const char* kFunction = "caelix_scene_recycle";
  caelix::Scene* impl = SceneFor(scene, kFunction);
  if (impl == nullptr) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    impl->Recycle(safe_frame_number);
    return CAELIX_OK;
  } catch (...) {
    return TranslateException(kFunction);
  }
}

CAELIX_API void caelix_scene_get_stats(const CaelixScene* scene, CaelixSceneStats* out) {
  if (scene == nullptr || scene->impl == nullptr || out == nullptr) {
    return;
  }
  scene->impl->GetStats(*out);
}

CAELIX_API CaelixResult caelix_scene_query_brick(const CaelixScene* scene, CaelixEntityId entity,
                                                 CaelixBrickKey key, CaelixBrickInfo* out) {
  constexpr const char* kFunction = "caelix_scene_query_brick";
  if (out != nullptr) {
    *out = CaelixBrickInfo{};
  }
  if (scene == nullptr || scene->impl == nullptr) {
    CAELIX_LOGE("%s: scene is null", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (!CheckEntity(*scene->impl, entity, kFunction)) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (out == nullptr) {
    CAELIX_LOGE("%s: out is null", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    return scene->impl->QueryBrick(entity, key, *out);
  } catch (...) {
    return TranslateException(kFunction);
  }
}

}  // extern "C"
