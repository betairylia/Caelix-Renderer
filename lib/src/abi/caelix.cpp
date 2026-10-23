// The C ABI. Every function here is extern "C" and exception-free: internal
// code may throw, and each entry point converts exceptions to a CaelixResult.
#include "caelix/caelix.h"

#include "core/log.h"
#include "core/vk_common.h"
#include "render/dlss_upscaler.h"
#include "render/render_kernels.h"
#include "render/renderer.h"
#include "rhi/device.h"
#include "rhi/gpu_buffer.h"
#include "scene/scene.h"

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <new>

// Layout of the structs that hosts embed by value (the Unity plugin's render packet holds a
// CaelixRenderOptions): appended fields must keep these offsets.
static_assert(sizeof(CaelixSkyCubemap) == 24);
static_assert(offsetof(CaelixRenderOptions, bounce_count_transparent) == 124);
static_assert(offsetof(CaelixRenderOptions, emission_scale) == 128);
static_assert(offsetof(CaelixRenderOptions, fog_density) == 132);
static_assert(offsetof(CaelixRenderOptions, fog_color) == 136);
static_assert(offsetof(CaelixRenderOptions, pad3) == 148);
static_assert(offsetof(CaelixRenderOptions, sky) == 152);
static_assert(offsetof(CaelixRenderOptions, upscaler) == 176);
static_assert(offsetof(CaelixRenderOptions, upscale_quality) == 180);
static_assert(offsetof(CaelixRenderOptions, upscale_preset) == 184);
static_assert(offsetof(CaelixRenderOptions, upscale_flags) == 188);
static_assert(sizeof(CaelixRenderOptions) == 192);
static_assert(sizeof(CaelixMaterialEntry) == 52);
static_assert(offsetof(CaelixGpuTimings, upscale_ms) == 24);
static_assert(sizeof(CaelixGpuTimings) == 32);
static_assert(sizeof(CaelixUpscalerQueryDesc) == 16);
static_assert(sizeof(CaelixUpscalerQueryResult) == 40);
static_assert(sizeof(CaelixUpscalerInfo) == 36);
static_assert(sizeof(CaelixLightOptions) == 16);
static_assert(sizeof(CaelixLightStats) == 104);
static_assert(offsetof(CaelixLightStats, tree_leaves) == 72);
static_assert(offsetof(CaelixLightStats, passes) == 48);
static_assert(offsetof(CaelixLightStats, gpu_ms) == 64);

// The opaque handle behind CaelixDevice*. Members are destroyed in reverse order: the NGX
// runtime goes first (every renderer, and so every DLSS feature, is already destroyed), then
// the kernels, then the device.
struct CaelixDevice
{
    std::unique_ptr<caelix::Device> impl;
    std::unique_ptr<caelix::RenderKernels> kernels;  // created by the first caelix_renderer_create
#if CAELIX_DLSS
    std::unique_ptr<caelix::NgxRuntime> ngx;  // created by the first upscaler query or upscaled render
    bool ngx_failed = false;                  // NGX did not start (logged once): never tried again
#endif
};

// The opaque handle behind CaelixScene*.
struct CaelixScene
{
    std::unique_ptr<caelix::Scene> impl;
};

// The opaque handle behind CaelixRenderer*.
struct CaelixRenderer
{
    std::unique_ptr<caelix::Renderer> impl;
};

namespace {

// Converts the exception in flight to a CaelixResult and logs it. Call only
// inside a catch block.
CaelixResult TranslateException(const char* function)
{
    try
    {
        throw;
    }
    catch (const caelix::VulkanError& error)
    {
        const VkResult result = error.result();
        CAELIX_LOGE("%s: %s (VkResult %d)", function, error.what(), static_cast<int>(result));
        return result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY
                   ? CAELIX_ERROR_OUT_OF_MEMORY
                   : CAELIX_ERROR_VULKAN;
    }
    catch (const std::bad_alloc&)
    {
        CAELIX_LOGE("%s: out of memory", function);
        return CAELIX_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::exception& error)
    {
        CAELIX_LOGE("%s: %s", function, error.what());
        return CAELIX_ERROR_VULKAN;
    }
    catch (...)
    {
        CAELIX_LOGE("%s: unexpected exception", function);
        return CAELIX_ERROR_VULKAN;
    }
}

// The live scene behind a handle, or null (logged) when the handle or the entity is not valid.
caelix::Scene* SceneFor(CaelixScene* scene, const char* function)
{
    if (scene == nullptr || scene->impl == nullptr)
    {
        CAELIX_LOGE("%s: scene is null", function);
        return nullptr;
    }
    return scene->impl.get();
}

bool CheckEntity(const caelix::Scene& scene, CaelixEntityId entity, const char* function)
{
    if (!scene.EntityAlive(entity))
    {
        CAELIX_LOGE("%s: entity %u does not exist", function, entity);
        return false;
    }
    return true;
}

#if CAELIX_DLSS
// The device's NGX runtime, created on the first call; null when NGX does not start (logged once
// by NgxRuntime, not tried again).
caelix::NgxRuntime* NgxFor(CaelixDevice& device)
{
    if (device.ngx == nullptr && !device.ngx_failed)
    {
        try
        {
            device.ngx = std::make_unique<caelix::NgxRuntime>(*device.impl);
        }
        catch (const caelix::NgxError&)
        {
            device.ngx_failed = true;
        }
    }
    return device.ngx.get();
}
#endif

// The radical inverse of `index` in `base` (Halton sequence).
float RadicalInverse(uint32_t index, uint32_t base)
{
    double result = 0.0;
    double fraction = 1.0 / static_cast<double>(base);
    while (index > 0)
    {
        result += static_cast<double>(index % base) * fraction;
        index /= base;
        fraction /= static_cast<double>(base);
    }
    return static_cast<float>(result);
}

}  // namespace

extern "C" {

CAELIX_API void caelix_get_version(CaelixVersion* out)
{
    if (out == nullptr)
    {
        return;
    }
    out->major = CAELIX_VERSION_MAJOR;
    out->minor = CAELIX_VERSION_MINOR;
    out->patch = CAELIX_VERSION_PATCH;
    out->abi = CAELIX_ABI_VERSION;
}

CAELIX_API const char* caelix_version_string(void)
{
    return CAELIX_VERSION_STRING;
}

CAELIX_API void caelix_set_log_callback(CaelixLogCallback callback, void* user_data)
{
    caelix::SetLogCallback(callback, user_data);
}

CAELIX_API void caelix_get_device_requirements(CaelixDeviceRequirements* out)
{
    if (out == nullptr)
    {
        return;
    }
    *out = caelix::GetDeviceRequirements();
}

CAELIX_API CaelixResult caelix_device_adopt(const CaelixDeviceAdoptDesc* desc,
                                            CaelixDevice** out_device)
{
    if (out_device == nullptr)
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    *out_device = nullptr;
    if (desc == nullptr)
    {
        CAELIX_LOGE("caelix_device_adopt: desc is null");
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        std::unique_ptr<caelix::Device> impl;
        const CaelixResult result = caelix::Device::Adopt(*desc, impl);
        if (result != CAELIX_OK)
        {
            return result;
        }
        *out_device = new CaelixDevice{std::move(impl), nullptr};
        return CAELIX_OK;
    }
    catch (const std::bad_alloc&)
    {
        CAELIX_LOGE("caelix_device_adopt: out of memory");
        return CAELIX_ERROR_OUT_OF_MEMORY;
    }
    catch (...)
    {
        CAELIX_LOGE("caelix_device_adopt: unexpected exception");
        return CAELIX_ERROR_VULKAN;
    }
}

CAELIX_API void caelix_device_destroy(CaelixDevice* device)
{
    try
    {
        delete device;
    }
    catch (...)
    {
        CAELIX_LOGE("caelix_device_destroy: unexpected exception");
    }
}

CAELIX_API CaelixResult caelix_scene_create(CaelixDevice* device, const CaelixSceneDesc* desc,
                                            CaelixScene** out_scene)
{
    if (out_scene == nullptr)
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    *out_scene = nullptr;
    if (device == nullptr || device->impl == nullptr)
    {
        CAELIX_LOGE("caelix_scene_create: device is null");
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        std::unique_ptr<caelix::Scene> impl;
        const CaelixResult result = caelix::Scene::Create(*device->impl, desc, impl);
        if (result != CAELIX_OK)
        {
            return result;
        }
        *out_scene = new CaelixScene{std::move(impl)};
        return CAELIX_OK;
    }
    catch (...)
    {
        return TranslateException("caelix_scene_create");
    }
}

CAELIX_API void caelix_scene_destroy(CaelixScene* scene)
{
    try
    {
        delete scene;
    }
    catch (...)
    {
        CAELIX_LOGE("caelix_scene_destroy: unexpected exception");
    }
}

CAELIX_API CaelixResult caelix_scene_add_entity(CaelixScene* scene, const CaelixEntityDesc* desc,
                                                CaelixEntityId* out_id)
{
    constexpr const char* kFunction = "caelix_scene_add_entity";
    if (out_id != nullptr)
    {
        *out_id = CAELIX_INVALID_ENTITY;
    }
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr || desc == nullptr || out_id == nullptr)
    {
        if (impl != nullptr)
        {
            CAELIX_LOGE("%s: null argument", kFunction);
        }
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->AddEntity(*desc, *out_id);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_remove_entity(CaelixScene* scene, CaelixEntityId entity)
{
    constexpr const char* kFunction = "caelix_scene_remove_entity";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr || !CheckEntity(*impl, entity, kFunction))
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->RemoveEntity(entity);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_set_entity_transform(CaelixScene* scene, CaelixEntityId entity,
                                                          const float position[3],
                                                          const float rotation[4])
{
    constexpr const char* kFunction = "caelix_scene_set_entity_transform";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr || !CheckEntity(*impl, entity, kFunction))
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (position == nullptr || rotation == nullptr)
    {
        CAELIX_LOGE("%s: null argument", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->SetEntityTransform(entity, position, rotation);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_upload_bricks(CaelixScene* scene, CaelixEntityId entity,
                                                   uint32_t count, const CaelixBrickKey* keys,
                                                   const void* blocks, uint32_t* consumed)
{
    constexpr const char* kFunction = "caelix_scene_upload_bricks";
    if (consumed != nullptr)
    {
        *consumed = 0;
    }
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr || !CheckEntity(*impl, entity, kFunction))
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (consumed == nullptr || (count > 0 && (keys == nullptr || blocks == nullptr)))
    {
        CAELIX_LOGE("%s: null argument", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->UploadBricks(entity, count, keys, blocks, *consumed);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_remove_bricks(CaelixScene* scene, CaelixEntityId entity,
                                                   uint32_t count, const CaelixBrickKey* keys)
{
    constexpr const char* kFunction = "caelix_scene_remove_bricks";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr || !CheckEntity(*impl, entity, kFunction))
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (count > 0 && keys == nullptr)
    {
        CAELIX_LOGE("%s: null argument", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->RemoveBricks(entity, count, keys);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_flush(CaelixScene* scene, const CaelixFrameDesc* desc)
{
    constexpr const char* kFunction = "caelix_scene_flush";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr)
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (desc == nullptr || desc->command_buffer == 0)
    {
        CAELIX_LOGE("%s: desc or its command buffer is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->Flush(*desc);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_recycle(CaelixScene* scene, uint64_t safe_frame_number)
{
    constexpr const char* kFunction = "caelix_scene_recycle";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr)
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        impl->Recycle(safe_frame_number);
        return CAELIX_OK;
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API void caelix_scene_get_stats(const CaelixScene* scene, CaelixSceneStats* out)
{
    if (scene == nullptr || scene->impl == nullptr || out == nullptr)
    {
        return;
    }
    scene->impl->GetStats(*out);
}

CAELIX_API CaelixResult caelix_scene_query_brick(const CaelixScene* scene, CaelixEntityId entity,
                                                 CaelixBrickKey key, CaelixBrickInfo* out)
{
    constexpr const char* kFunction = "caelix_scene_query_brick";
    if (out != nullptr)
    {
        *out = CaelixBrickInfo{};
    }
    if (scene == nullptr || scene->impl == nullptr)
    {
        CAELIX_LOGE("%s: scene is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (!CheckEntity(*scene->impl, entity, kFunction))
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (out == nullptr)
    {
        CAELIX_LOGE("%s: out is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return scene->impl->QueryBrick(entity, key, *out);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_get_tlas(const CaelixScene* scene, uint64_t* out_handle,
                                              uint64_t* out_address)
{
    constexpr const char* kFunction = "caelix_scene_get_tlas";
    if (scene == nullptr || scene->impl == nullptr)
    {
        CAELIX_LOGE("%s: scene is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (out_handle == nullptr || out_address == nullptr)
    {
        CAELIX_LOGE("%s: null argument", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    scene->impl->GetTlas(*out_handle, *out_address);
    return CAELIX_OK;
}

CAELIX_API CaelixResult caelix_scene_set_materials(CaelixScene* scene, const CaelixMaterial* entries,
                                                   uint32_t count)
{
    constexpr const char* kFunction = "caelix_scene_set_materials";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr)
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->SetMaterials(entries, count);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_set_glass_palette(CaelixScene* scene,
                                                       const CaelixMaterial* entries, uint32_t count)
{
    constexpr const char* kFunction = "caelix_scene_set_glass_palette";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr)
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->SetGlassPalette(entries, count);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_set_material_entries(CaelixScene* scene,
                                                          const CaelixMaterialEntry* entries,
                                                          uint32_t count)
{
    constexpr const char* kFunction = "caelix_scene_set_material_entries";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr)
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->SetMaterialEntries(entries, count);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_get_material(const CaelixScene* scene, uint32_t code,
                                                  CaelixMaterial* out)
{
    constexpr const char* kFunction = "caelix_scene_get_material";
    if (scene == nullptr || scene->impl == nullptr)
    {
        CAELIX_LOGE("%s: scene is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (out == nullptr)
    {
        CAELIX_LOGE("%s: out is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    return scene->impl->GetMaterial(code, *out);
}

CAELIX_API CaelixResult caelix_scene_set_entity_motion(CaelixScene* scene, CaelixEntityId entity,
                                                       uint32_t motion)
{
    constexpr const char* kFunction = "caelix_scene_set_entity_motion";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr || !CheckEntity(*impl, entity, kFunction))
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->SetEntityMotion(entity, motion);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API CaelixResult caelix_scene_set_light_options(CaelixScene* scene, const CaelixLightOptions* options)
{
    constexpr const char* kFunction = "caelix_scene_set_light_options";
    caelix::Scene* impl = SceneFor(scene, kFunction);
    if (impl == nullptr)
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options == nullptr)
    {
        CAELIX_LOGE("%s: options is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return impl->SetLightOptions(*options);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API void caelix_scene_get_light_stats(const CaelixScene* scene, CaelixLightStats* out)
{
    if (scene == nullptr || scene->impl == nullptr || out == nullptr)
    {
        return;
    }
    scene->impl->GetLightStats(*out);
}

CAELIX_API uint64_t caelix_scene_get_light_tree(const CaelixScene* scene)
{
    if (scene == nullptr || scene->impl == nullptr)
    {
        return 0;
    }
    return scene->impl->light_tree_address();
}

CAELIX_API CaelixResult caelix_renderer_create(CaelixDevice* device, CaelixScene* scene,
                                               CaelixRenderer** out_renderer)
{
    constexpr const char* kFunction = "caelix_renderer_create";
    if (out_renderer == nullptr)
    {
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    *out_renderer = nullptr;
    if (device == nullptr || device->impl == nullptr || scene == nullptr ||
        scene->impl == nullptr)
    {
        CAELIX_LOGE("%s: device or scene is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (&scene->impl->device() != device->impl.get())
    {
        CAELIX_LOGE("%s: the scene belongs to another device", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        // The device's shared kernels: created by the first renderer (a throw leaves them null,
        // and the next call tries again).
        if (device->kernels == nullptr)
        {
            device->kernels = std::make_unique<caelix::RenderKernels>(*device->impl);
        }
        // The renderer reaches the device's NGX runtime through this accessor (lazy creation).
        std::function<caelix::NgxRuntime*()> ngx;
#if CAELIX_DLSS
        ngx = [device]() { return NgxFor(*device); };
#endif
        std::unique_ptr<caelix::Renderer> impl;
        const CaelixResult result =
            caelix::Renderer::Create(*device->impl, *device->kernels, *scene->impl, std::move(ngx), impl);
        if (result != CAELIX_OK)
        {
            return result;
        }
        *out_renderer = new CaelixRenderer{std::move(impl)};
        return CAELIX_OK;
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API void caelix_renderer_destroy(CaelixRenderer* renderer)
{
    try
    {
        delete renderer;
    }
    catch (...)
    {
        CAELIX_LOGE("caelix_renderer_destroy: unexpected exception");
    }
}

CAELIX_API CaelixResult caelix_renderer_render(CaelixRenderer* renderer, const CaelixFrameDesc* frame,
                                               const CaelixRenderTargets* targets,
                                               const CaelixCamera* camera,
                                               const CaelixRenderOptions* options)
{
    constexpr const char* kFunction = "caelix_renderer_render";
    if (renderer == nullptr || renderer->impl == nullptr)
    {
        CAELIX_LOGE("%s: renderer is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (frame == nullptr || targets == nullptr || camera == nullptr || options == nullptr)
    {
        CAELIX_LOGE("%s: null argument", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    try
    {
        return renderer->impl->Render(*frame, *targets, *camera, *options);
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
}

CAELIX_API void caelix_renderer_release_target_views(CaelixRenderer* renderer)
{
    if (renderer == nullptr || renderer->impl == nullptr)
    {
        return;
    }
    renderer->impl->ReleaseTargetViews();
}

CAELIX_API void caelix_renderer_get_gpu_timings(const CaelixRenderer* renderer,
                                                CaelixGpuTimings* out)
{
    if (renderer == nullptr || renderer->impl == nullptr || out == nullptr)
    {
        return;
    }
    renderer->impl->GetTimings(*out);
}

CAELIX_API CaelixResult caelix_renderer_get_nrd_info(const CaelixRenderer* renderer,
                                                     CaelixNrdInfo* out)
{
    if (renderer == nullptr || renderer->impl == nullptr || out == nullptr)
    {
        CAELIX_LOGE("caelix_renderer_get_nrd_info: null argument");
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    renderer->impl->GetNrdInfo(*out);
    return CAELIX_OK;
}

CAELIX_API CaelixResult caelix_upscaler_query(CaelixDevice* device, const CaelixUpscalerQueryDesc* desc,
                                              CaelixUpscalerQueryResult* out)
{
    constexpr const char* kFunction = "caelix_upscaler_query";
    if (out != nullptr)
    {
        *out = CaelixUpscalerQueryResult{};
    }
    if (device == nullptr || device->impl == nullptr || desc == nullptr || out == nullptr)
    {
        CAELIX_LOGE("%s: null argument", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (desc->upscaler > CAELIX_UPSCALER_DLSS_RR || desc->quality > CAELIX_UPSCALE_ULTRA_PERFORMANCE)
    {
        CAELIX_LOGE("%s: unknown upscaler %u or quality %u", kFunction, desc->upscaler, desc->quality);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (desc->upscaler == CAELIX_UPSCALER_NONE)
    {
        out->supported = 1;
        out->render_width = out->render_min_width = out->render_max_width = desc->output_width;
        out->render_height = out->render_min_height = out->render_max_height = desc->output_height;
        return CAELIX_OK;
    }
#if CAELIX_DLSS
    if (desc->output_width < 64 || desc->output_height < 32)
    {
        out->reason = CAELIX_UPSCALER_BAD_ARGUMENT;
        return CAELIX_OK;
    }
    try
    {
        caelix::NgxRuntime* ngx = NgxFor(*device);
        if (ngx == nullptr)
        {
            out->reason = CAELIX_UPSCALER_NO_NGX;
            return CAELIX_OK;
        }
        ngx->Query(static_cast<caelix::UpscalerKind>(desc->upscaler), desc->quality, desc->output_width,
                   desc->output_height, *out);
        return CAELIX_OK;
    }
    catch (...)
    {
        return TranslateException(kFunction);
    }
#else
    out->reason = CAELIX_UPSCALER_NOT_BUILT;
    return CAELIX_OK;
#endif
}

CAELIX_API void caelix_upscaler_instance_extensions(uint32_t* out_count, const char* const** out_names)
{
    uint32_t count = 0;
    const char* const* names = nullptr;
#if CAELIX_DLSS
    try
    {
        caelix::UpscalerInstanceExtensions(count, names);
    }
    catch (...)
    {
        CAELIX_LOGE("caelix_upscaler_instance_extensions: unexpected exception");
        count = 0;
        names = nullptr;
    }
#endif
    if (out_count != nullptr)
    {
        *out_count = count;
    }
    if (out_names != nullptr)
    {
        *out_names = names;
    }
}

CAELIX_API CaelixResult caelix_upscaler_device_extensions(uint64_t instance, uint64_t physical_device,
                                                          uint32_t* out_count, const char* const** out_names)
{
    constexpr const char* kFunction = "caelix_upscaler_device_extensions";
    if (out_count != nullptr)
    {
        *out_count = 0;
    }
    if (out_names != nullptr)
    {
        *out_names = nullptr;
    }
    if (out_count == nullptr || out_names == nullptr || instance == 0 || physical_device == 0)
    {
        CAELIX_LOGE("%s: null argument or zero handle", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
#if CAELIX_DLSS
    try
    {
        return caelix::UpscalerDeviceExtensions(caelix::HandleFromU64<VkInstance>(instance),
                                                caelix::HandleFromU64<VkPhysicalDevice>(physical_device),
                                                *out_count, *out_names);
    }
    catch (...)
    {
        *out_count = 0;
        *out_names = nullptr;
        return TranslateException(kFunction);
    }
#else
    return CAELIX_OK;
#endif
}

CAELIX_API uint32_t caelix_upscaler_jitter_phases(uint32_t upscaler, uint32_t render_width, uint32_t output_width)
{
    // 8 * ceil((output / render)^2), in integers: ceil(o^2 / r^2) = (o^2 + r^2 - 1) / r^2.
    uint64_t phases = 8;
    if (render_width != 0 && output_width != 0)
    {
        const uint64_t o2 = static_cast<uint64_t>(output_width) * output_width;
        const uint64_t r2 = static_cast<uint64_t>(render_width) * render_width;
        phases = 8 * ((o2 + r2 - 1) / r2);
    }
    if (upscaler == CAELIX_UPSCALER_DLSS_RR && phases < 32)
    {
        phases = 32;
    }
    return phases > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(phases);
}

CAELIX_API void caelix_jitter_halton(uint32_t index, uint32_t phases, float out_jitter[2])
{
    if (out_jitter == nullptr)
    {
        return;
    }
    const uint32_t wrapped = phases != 0 ? index % phases : index;
    // Point number wrapped + 1: Halton point 0 is (0, 0), a corner of the pixel, not a jitter.
    const uint32_t point = wrapped == UINT32_MAX ? wrapped : wrapped + 1;
    out_jitter[0] = RadicalInverse(point, 2) - 0.5f;
    out_jitter[1] = RadicalInverse(point, 3) - 0.5f;
}

CAELIX_API CaelixResult caelix_renderer_get_upscaler_info(const CaelixRenderer* renderer, CaelixUpscalerInfo* out)
{
    if (renderer == nullptr || renderer->impl == nullptr || out == nullptr)
    {
        CAELIX_LOGE("caelix_renderer_get_upscaler_info: null argument");
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    renderer->impl->GetUpscalerInfo(*out);
    return CAELIX_OK;
}

}  // extern "C"
