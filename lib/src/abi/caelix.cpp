// The C ABI. Every function here is extern "C" and exception-free: internal
// code may throw, and each entry point converts exceptions to a CaelixResult.
#include "caelix/caelix.h"

#include "core/log.h"
#include "core/vk_common.h"
#include "render/renderer.h"
#include "rhi/device.h"
#include "rhi/gpu_buffer.h"
#include "scene/scene.h"

#include <exception>
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

// The opaque handle behind CaelixRenderer*.
struct CaelixRenderer {
  std::unique_ptr<caelix::Renderer> impl;
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
  } catch (const std::exception& error) {
    CAELIX_LOGE("%s: %s", function, error.what());
    return CAELIX_ERROR_VULKAN;
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

CAELIX_API CaelixResult caelix_scene_get_tlas(const CaelixScene* scene, uint64_t* out_handle,
                                              uint64_t* out_address) {
  constexpr const char* kFunction = "caelix_scene_get_tlas";
  if (scene == nullptr || scene->impl == nullptr) {
    CAELIX_LOGE("%s: scene is null", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (out_handle == nullptr || out_address == nullptr) {
    CAELIX_LOGE("%s: null argument", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  scene->impl->GetTlas(*out_handle, *out_address);
  return CAELIX_OK;
}

CAELIX_API CaelixResult caelix_renderer_create(CaelixDevice* device, CaelixScene* scene,
                                               CaelixRenderer** out_renderer) {
  constexpr const char* kFunction = "caelix_renderer_create";
  if (out_renderer == nullptr) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  *out_renderer = nullptr;
  if (device == nullptr || device->impl == nullptr || scene == nullptr ||
      scene->impl == nullptr) {
    CAELIX_LOGE("%s: device or scene is null", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (&scene->impl->device() != device->impl.get()) {
    CAELIX_LOGE("%s: the scene belongs to another device", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    std::unique_ptr<caelix::Renderer> impl;
    const CaelixResult result = caelix::Renderer::Create(*device->impl, *scene->impl, impl);
    if (result != CAELIX_OK) {
      return result;
    }
    *out_renderer = new CaelixRenderer{std::move(impl)};
    return CAELIX_OK;
  } catch (...) {
    return TranslateException(kFunction);
  }
}

CAELIX_API void caelix_renderer_destroy(CaelixRenderer* renderer) {
  try {
    delete renderer;
  } catch (...) {
    CAELIX_LOGE("caelix_renderer_destroy: unexpected exception");
  }
}

CAELIX_API CaelixResult caelix_renderer_render(CaelixRenderer* renderer, const CaelixFrameDesc* frame,
                                               const CaelixRenderTargets* targets,
                                               const CaelixCamera* camera,
                                               const CaelixRenderOptions* options) {
  constexpr const char* kFunction = "caelix_renderer_render";
  if (renderer == nullptr || renderer->impl == nullptr) {
    CAELIX_LOGE("%s: renderer is null", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (frame == nullptr || targets == nullptr || camera == nullptr || options == nullptr) {
    CAELIX_LOGE("%s: null argument", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  try {
    return renderer->impl->Render(*frame, *targets, *camera, *options);
  } catch (...) {
    return TranslateException(kFunction);
  }
}

CAELIX_API void caelix_renderer_release_target_views(CaelixRenderer* renderer) {
  if (renderer == nullptr || renderer->impl == nullptr) {
    return;
  }
  renderer->impl->ReleaseTargetViews();
}

CAELIX_API void caelix_renderer_get_gpu_timings(const CaelixRenderer* renderer,
                                                CaelixGpuTimings* out) {
  if (renderer == nullptr || renderer->impl == nullptr || out == nullptr) {
    return;
  }
  renderer->impl->GetTimings(*out);
}

CAELIX_API CaelixResult caelix_renderer_get_nrd_info(const CaelixRenderer* renderer,
                                                     CaelixNrdInfo* out) {
  if (renderer == nullptr || renderer->impl == nullptr || out == nullptr) {
    CAELIX_LOGE("caelix_renderer_get_nrd_info: null argument");
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  renderer->impl->GetNrdInfo(*out);
  return CAELIX_OK;
}

}  // extern "C"
