// The plugin's C API for C# and the render-thread work of the plugin events.
//
// Packet version 6: three plugin events, each with its own packet kind (caelix_unity.h).
//
// Frame event (once per Unity frame, before any camera renders): applies the frame packet's scene
// records (entities, brick removes, brick uploads) and flushes the scene. A packet with
// CAELIX_UNITY_FRAME_RESET starts a session: it creates the scene with the packet's encoding and
// staging size and applies its material entries first. The plugin holds no scene between
// sessions: the release event with CAELIX_UNITY_RELEASE_SCENE destroys it.
//
// Frame numbers: Unity's own, no conversion. The flush and every render of a Unity frame use
// frame_number = currentFrameNumber and safe_frame_number = safeFrameNumber of the recording
// state. The scene is flushed at most once per Unity frame; a second frame event in the same Unity
// frame applies its records and leaves them for the next frame's flush.
//
// Uploads: the plugin has no queue of its own. Bricks the staging ring refuses are dropped and
// counted (CaelixUnityStatus::bricks_dropped); C# sizes its uploads from upload_bricks_free and
// upload_bricks_cap so that this does not happen.
//
// Render event (once per view per Unity frame): renders the view's renderer (created on the
// view's first render event) over the shared scene into the packet's targets, with the sky
// cubemap target when the packet has one. A path trace packet with an upscaler is upscaled when
// the upscaler query (cached per view and key, published in CaelixUnityStatus::upscalers) says
// supported and the geometry targets have the query's render size; otherwise the frame renders at
// the colour size without the upscaler.
//
// Release event (graphics queue access, queue idle): releases cached target views, one view's
// renderer, or every renderer and the scene.
#include "plugin_state.h"

#include <caelix_unity.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace caelix_unity {
namespace {

constexpr uint32_t kBlocksPerBrick = 512;
constexpr uint32_t kNoScene = 0xFFFFFFFFu;

// The sizes caelix_unity.h states, which C# mirrors.
static_assert(sizeof(CaelixUnityPacketHeader) == 24, "CaelixUnityPacketHeader is 24 bytes");
static_assert(sizeof(CaelixUnityFramePacket) == 80, "CaelixUnityFramePacket is 80 bytes");
// 504 since the DLSS step (CaelixRenderOptions grew 176 -> 192 bytes); the C# mirror must follow.
static_assert(sizeof(CaelixUnityViewPacket) == 504, "CaelixUnityViewPacket is 504 bytes");
static_assert(sizeof(CaelixUnityReleasePacket) == 32, "CaelixUnityReleasePacket is 32 bytes");
static_assert(sizeof(CaelixUnityStatus) == 536, "CaelixUnityStatus is 536 bytes");
static_assert(sizeof(CaelixUnityUpscalerStatus) == 32, "CaelixUnityUpscalerStatus is 32 bytes");
static_assert(offsetof(CaelixUnityStatus, upscaler_extensions) == 164 &&
                  offsetof(CaelixUnityStatus, upscale_ms) == 276 && offsetof(CaelixUnityStatus, upscalers) == 280,
              "CaelixUnityStatus offsets of the upscaler fields (C# mirrors them)");
// CaelixMaterialEntry is 116 bytes on the subsurface experiment branch sss-expr (52 on main): the C#
// mirror has NOT followed, the plugin is not used on that branch.
static_assert(sizeof(CaelixUnityEntity) == 48 && sizeof(CaelixUnityBrickRemove) == 16 &&
                  sizeof(CaelixUnityUpload) == 16 && sizeof(CaelixMaterialEntry) == 116,
              "record sizes of caelix_unity.h");

struct Guid
{
    uint64_t lo = 0;
    uint64_t hi = 0;
    bool operator==(const Guid& other) const
    {
        return lo == other.lo && hi == other.hi;
    }
};

struct GuidHash
{
    size_t operator()(const Guid& guid) const
    {
        return static_cast<size_t>(guid.lo * 0x9E3779B97F4A7C15ull ^ guid.hi);
    }
};

// The upscaler query of one view, cached by its key (see "Upscalers" in caelix_unity.h).
struct UpscalerCache
{
    bool valid = false;
    CaelixUpscalerQueryDesc key{};
    CaelixUpscalerQueryResult result{};
    bool logged_fallback = false;  // the frame of this key rendered without the upscaler (logged)
    bool logged_active = false;    // the first upscaled frame of this key (logged)
};

// Render thread only.
struct RenderThreadState
{
    const char* event_name = "plugin";  // the event that runs now, for the error log
    uint32_t last_generation = 0;
    bool released_since_generation = true;
    bool logged_first_render = false;
    bool logged_access_fallback = false;
    bool logged_level = false;
    bool warned_render_without_scene = false;
    uint32_t dropped_target_logs = 0;
    uint32_t sky_logs = 0;
    bool scene_busy = false;        // the scene had records, staged bricks or pending builds
    uint64_t scene_busy_since = 0;  // the newest flushed Unity frame when the current burst began
    uint32_t scene_encoding = kNoScene;
    uint64_t last_flushed_frame = 0;  // the newest Unity frame the scene was flushed in
    uint64_t renders_without_scene = 0;
    uint64_t error_logs = 0;
    uint32_t access_logs = 0;
    uint64_t upscaler_logs = 0;
    UpscalerCache upscalers[CAELIX_UNITY_MAX_VIEWS];

    std::unordered_map<Guid, CaelixEntityId, GuidHash> entities;
    std::vector<CaelixEntityId> packet_ids;      // scratch: entity id per packet entity record
    std::vector<CaelixBrickKey> remove_scratch;  // scratch: keys of one remove run
};
RenderThreadState g_rt;

// Logs the first errors of a kind, then one in 600, so a broken setup does not flood the log.
bool ShouldLogError()
{
    ++g_rt.error_logs;
    return g_rt.error_logs <= 10 || g_rt.error_logs % 600 == 0;
}

void Fail(CaelixResult result, const char* what)
{
    State& state = GetState();
    state.failed_events.fetch_add(1);
    if (result != CAELIX_OK)
    {
        state.last_error.store(static_cast<int32_t>(result));
    }
    if (ShouldLogError())
    {
        Log(LogLevel::Error, "%s event: %s (CaelixResult %d)", g_rt.event_name, what,
            static_cast<int>(result));
    }
}

// The upscaler lines (once per view and key): the first 10, then one in 600, like the errors.
bool ShouldLogUpscaler()
{
    ++g_rt.upscaler_logs;
    return g_rt.upscaler_logs <= 10 || g_rt.upscaler_logs % 600 == 0;
}

// A scene call failed but the event goes on.
void SceneError(CaelixResult result, const char* what)
{
    GetState().last_error.store(static_cast<int32_t>(result));
    if (ShouldLogError())
    {
        Log(LogLevel::Error, "scene feed: %s (CaelixResult %d)", what, static_cast<int>(result));
    }
}

// Publishes a packet's sequence as the consumed sequence of its kind: C# may reuse its memory from
// now on.
void PublishConsumed(std::atomic<uint64_t>& last_consumed, uint64_t sequence)
{
    uint64_t consumed = last_consumed.load();
    while (sequence > consumed && !last_consumed.compare_exchange_weak(consumed, sequence))
    {
    }
}

// Reads the fixed struct of a packet of `kind` (the memory may not be aligned for it). False
// (logged, rate limited) when the header's size, version or kind differs from what the plugin was
// compiled with; the packet's sequence cannot be trusted then and is not published.
template <typename Packet>
bool ReadPacket(const void* data, uint32_t kind, Packet& packet)
{
    CaelixUnityPacketHeader header;
    std::memcpy(&header, data, sizeof(header));
    if (header.size != sizeof(Packet) || header.version != CAELIX_UNITY_PACKET_VERSION ||
        header.kind != kind)
    {
        if (ShouldLogError())
        {
            Log(LogLevel::Error, "%s event: packet size %u version %u kind %u, the plugin expects size "
                                 "%u version %u kind %u (C# and plugin differ)",
                g_rt.event_name, header.size, header.version, header.kind,
                static_cast<unsigned>(sizeof(Packet)), CAELIX_UNITY_PACKET_VERSION, kind);
        }
        return false;
    }
    std::memcpy(&packet, data, sizeof(packet));
    return true;
}

// Unity's recording state for the current event. Logs the command buffer once.
bool GetRecordingState(UnityVulkanRecordingState& recording)
{
    State& state = GetState();
    if (!state.vulkan->CommandRecordingState(&recording, kUnityVulkanGraphicsQueueAccess_DontCare))
    {
        return false;
    }
    if (!g_rt.logged_level)
    {
        g_rt.logged_level = true;
        Log(LogLevel::Info, "%s event: command buffer %p level %s, render pass %p, framebuffer %p, "
                            "subpass %d, Unity frame %llu, safe %llu",
            g_rt.event_name, static_cast<void*>(recording.commandBuffer),
            recording.commandBufferLevel == VK_COMMAND_BUFFER_LEVEL_PRIMARY ? "primary" : "secondary",
            static_cast<void*>(recording.renderPass), static_cast<void*>(recording.framebuffer),
            recording.subPassIndex, recording.currentFrameNumber, recording.safeFrameNumber);
    }
    return true;
}

void UpdateLiveViews()
{
    State& state = GetState();
    uint32_t mask = 0;
    for (uint32_t view = 0; view < CAELIX_UNITY_MAX_VIEWS; ++view)
    {
        if (state.renderers[view] != nullptr) mask |= 1u << view;
    }
    state.live_views.store(mask);
}

bool HasRenderer()
{
    const State& state = GetState();
    for (const CaelixRenderer* renderer : state.renderers)
    {
        if (renderer != nullptr) return true;
    }
    return false;
}

// Creates the renderer of `view` over the shared scene when there is none. The scene must exist
// and `view` must be < CAELIX_UNITY_MAX_VIEWS.
bool EnsureRenderer(uint32_t view)
{
    State& state = GetState();
    if (state.renderers[view] == nullptr)
    {
        const CaelixResult result = caelix_renderer_create(state.device, state.scene, &state.renderers[view]);
        if (result != CAELIX_OK)
        {
            state.renderers[view] = nullptr;
            Fail(result, "caelix_renderer_create failed");
            return false;
        }
        UpdateLiveViews();
        Log(LogLevel::Info, "renderer created for view %u", view);
    }
    return true;
}

// Destroys every renderer and the scene and forgets the scene's contents. The device must be idle
// (or the GPU done with every frame that used them).
void DestroySceneObjects()
{
    State& state = GetState();
    for (CaelixRenderer*& renderer : state.renderers)
    {
        caelix_renderer_destroy(renderer);
        renderer = nullptr;
    }
    caelix_scene_destroy(state.scene);
    state.scene = nullptr;
    g_rt.entities.clear();
    g_rt.scene_encoding = kNoScene;
    g_rt.scene_busy = false;
    state.scene_encoding.store(kNoScene);
    state.upload_bricks_free.store(0);
    state.upload_bricks_cap.store(0);
    state.scene_entities.store(0);
    state.scene_groups.store(0);
    state.bricks_resident.store(0);
    state.as_builds_pending.store(0);
    state.live_views.store(0);
}

bool AccessTarget(const CaelixUnityTarget& target, UnityVulkanImage& image)
{
    State& state = GetState();
    const VkImageSubresource subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    // PipelineBarrier: Unity records a barrier from its tracked layout and access to GENERAL /
    // compute shader write and then tracks the image as GENERAL with that access. The library
    // leaves the image in GENERAL with its writes available to all later commands, so Unity's
    // tracked state stays true and its next use of the texture transitions from it correctly.
    if (target.native_texture != nullptr &&
        state.vulkan->AccessTexture(target.native_texture, &subresource, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                    kUnityVulkanResourceAccess_PipelineBarrier, &image))
    {
        return true;
    }
    if (target.native_render_buffer != nullptr &&
        state.vulkan->AccessRenderBufferTexture(
            static_cast<UnityRenderBuffer>(target.native_render_buffer), &subresource,
            VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            kUnityVulkanResourceAccess_PipelineBarrier, &image))
    {
        if (!g_rt.logged_access_fallback)
        {
            g_rt.logged_access_fallback = true;
            Log(LogLevel::Warning, "AccessTexture refused the native texture; AccessRenderBufferTexture "
                                   "accepted the render buffer");
        }
        return true;
    }
    return false;
}

// An optional geometry target of the packet: accessed and checked like the colour target, but a
// failure only drops the target (logged a few times), the frame still renders. `width` x `height`
// is the expected size: the colour target's, or the upscaler's render size.
bool AccessOptionalTarget(const CaelixUnityTarget& target, const char* name, uint32_t width,
                          uint32_t height, CaelixTargetImage& out)
{
    out = CaelixTargetImage{};
    if (target.native_texture == nullptr && target.native_render_buffer == nullptr) return false;
    UnityVulkanImage image{};
    char problem[128] = {};
    if (!AccessTarget(target, image))
    {
        std::snprintf(problem, sizeof(problem), "AccessTexture failed");
    }
    else if ((image.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0)
    {
        std::snprintf(problem, sizeof(problem), "no STORAGE usage (RenderTexture.enableRandomWrite is off)");
    }
    else if (image.extent.width != width || image.extent.height != height)
    {
        std::snprintf(problem, sizeof(problem), "it is %ux%u, the frame expects %ux%u", image.extent.width,
                      image.extent.height, width, height);
    }
    if (problem[0] != '\0')
    {
        if (g_rt.dropped_target_logs < 8)
        {
            ++g_rt.dropped_target_logs;
            Log(LogLevel::Warning, "the %s target is dropped: %s", name, problem);
        }
        return false;
    }
    out.image = reinterpret_cast<uint64_t>(image.image);
    out.format = static_cast<uint32_t>(image.format);
    out.width = image.extent.width;
    out.height = image.extent.height;
    out.layout = static_cast<uint32_t>(image.layout);
    return true;
}

const char* UpscalerName(uint32_t upscaler)
{
    switch (upscaler)
    {
        case CAELIX_UPSCALER_NONE: return "none";
        case CAELIX_UPSCALER_DLSS_SR: return "DLSS SR";
        case CAELIX_UPSCALER_DLSS_RR: return "DLSS RR";
        default: return "unknown upscaler";
    }
}

const char* UpscalerReasonName(uint32_t reason)
{
    switch (reason)
    {
        case CAELIX_UPSCALER_OK: return "ok";
        case CAELIX_UPSCALER_NOT_BUILT: return "the library was built without the upscaler";
        case CAELIX_UPSCALER_NO_NGX: return "NGX did not start (driver, DLL or device extension missing)";
        case CAELIX_UPSCALER_FEATURE_UNAVAILABLE: return "the GPU or the NGX runtime lacks the feature";
        case CAELIX_UPSCALER_DRIVER_TOO_OLD: return "the driver is too old";
        case CAELIX_UPSCALER_QUALITY_UNAVAILABLE: return "no render size for this quality";
        case CAELIX_UPSCALER_BAD_ARGUMENT: return "bad argument (output below 64 x 32, or an unknown value)";
        default: return "unknown reason";
    }
}

// Publishes the query of `view` for caelix_unity_status (key and result together).
void PublishUpscalerStatus(uint32_t view, const UpscalerCache& cache)
{
    CaelixUnityUpscalerStatus entry{};
    entry.upscaler = cache.key.upscaler;
    entry.quality = cache.key.quality;
    entry.output_width = cache.key.output_width;
    entry.output_height = cache.key.output_height;
    entry.supported = cache.result.supported;
    entry.reason = cache.result.reason;
    entry.render_width = cache.result.render_width;
    entry.render_height = cache.result.render_height;
    State& state = GetState();
    std::lock_guard<std::mutex> lock(state.upscalers_mutex);
    state.upscalers[view] = entry;
}

// True when the packet target is absent or declares `width` x `height`.
bool DeclaresSize(const CaelixUnityTarget& target, uint32_t width, uint32_t height)
{
    const bool present = target.native_texture != nullptr || target.native_render_buffer != nullptr;
    return !present || (target.width == width && target.height == height);
}

// Decides whether this frame of `view` is upscaled (see "Upscalers" in caelix_unity.h). With
// options.upscaler set in the path trace mode it queries the upscaler for the key (upscaler,
// quality, colour size), cached per view, and publishes the result. The frame is upscaled when the
// query says supported and every present geometry target of the packet declares the query's render
// size; otherwise options.upscaler is cleared for this frame (logged once per key). Writes the size
// the geometry targets must have (render size or colour size) to geometry_width / geometry_height.
void DecideUpscaler(uint32_t view, const CaelixUnityViewPacket& packet, uint32_t color_width,
                    uint32_t color_height, CaelixRenderOptions& options, uint32_t& geometry_width,
                    uint32_t& geometry_height)
{
    geometry_width = color_width;
    geometry_height = color_height;
    if (options.upscaler == CAELIX_UPSCALER_NONE) return;
    if (options.mode != CAELIX_MODE_PATH_TRACE)
    {
        // The library ignores the upscaler in the other modes; C# sends none there.
        options.upscaler = CAELIX_UPSCALER_NONE;
        return;
    }

    State& state = GetState();
    UpscalerCache& cache = g_rt.upscalers[view];
    const CaelixUpscalerQueryDesc key{options.upscaler, options.upscale_quality, color_width, color_height};
    if (!cache.valid || std::memcmp(&cache.key, &key, sizeof(key)) != 0)
    {
        cache = UpscalerCache{};
        cache.valid = true;
        cache.key = key;
        // The first query of the device creates the NGX runtime (it costs once).
        const CaelixResult result = caelix_upscaler_query(state.device, &key, &cache.result);
        if (result != CAELIX_OK)
        {
            cache.result = CaelixUpscalerQueryResult{};
            cache.result.reason = CAELIX_UPSCALER_BAD_ARGUMENT;
            state.last_error.store(static_cast<int32_t>(result));
        }
        PublishUpscalerStatus(view, cache);
    }

    if (cache.result.supported == 0)
    {
        options.upscaler = CAELIX_UPSCALER_NONE;
        if (!cache.logged_fallback)
        {
            cache.logged_fallback = true;
            if (ShouldLogUpscaler())
            {
                Log(LogLevel::Warning, "view %u: %s quality %u at %ux%u is not available (reason %u: %s); the view "
                                       "renders without the upscaler",
                    view, UpscalerName(key.upscaler), key.quality, key.output_width, key.output_height,
                    cache.result.reason, UpscalerReasonName(cache.result.reason));
            }
        }
        return;
    }

    const uint32_t render_width = cache.result.render_width;
    const uint32_t render_height = cache.result.render_height;
    if (!DeclaresSize(packet.depth, render_width, render_height) ||
        !DeclaresSize(packet.motion, render_width, render_height) ||
        !DeclaresSize(packet.albedo, render_width, render_height))
    {
        // The normal first frame of a key: C# reads the render size from the status and allocates
        // its geometry targets at that size for the next frame.
        options.upscaler = CAELIX_UPSCALER_NONE;
        if (!cache.logged_fallback)
        {
            cache.logged_fallback = true;
            if (ShouldLogUpscaler())
            {
                Log(LogLevel::Info, "view %u: %s quality %u needs geometry targets of %ux%u for the output %ux%u; "
                                    "this frame renders without the upscaler",
                    view, UpscalerName(key.upscaler), key.quality, render_width, render_height,
                    key.output_width, key.output_height);
            }
        }
        return;
    }
    geometry_width = render_width;
    geometry_height = render_height;
    if (!cache.logged_active)
    {
        cache.logged_active = true;
        if (ShouldLogUpscaler())
        {
            Log(LogLevel::Info, "view %u: %s quality %u preset %u upscales %ux%u -> %ux%u", view,
                UpscalerName(key.upscaler), key.quality, options.upscale_preset, render_width, render_height,
                key.output_width, key.output_height);
        }
    }
}

// The sky cubemap target of a view packet: observed (ObserveOnly records nothing and ignores the
// layout, stage and access arguments, so the image stays in the layout Unity keeps it in) and
// described in `sky`. An image that is not a 6-layer square cube, or whose layout the library cannot
// sample in, is dropped for the frame (logged a few times): the renderer then uses the procedural
// sky.
void AccessSky(const CaelixUnityTarget& target, CaelixSkyCubemap& sky)
{
    sky = CaelixSkyCubemap{};
    if (target.native_texture == nullptr) return;
    State& state = GetState();
    const VkImageSubresource subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    UnityVulkanImage image{};
    char problem[96] = {};
    if (!state.vulkan->AccessTexture(target.native_texture, &subresource, VK_IMAGE_LAYOUT_UNDEFINED, 0, 0,
                                     kUnityVulkanResourceAccess_ObserveOnly, &image))
    {
        std::snprintf(problem, sizeof(problem), "AccessTexture refused it");
    }
    else if (image.layers != 6 || image.extent.width != image.extent.height || image.extent.width == 0 ||
             image.mipCount < 1)
    {
        std::snprintf(problem, sizeof(problem), "not a cube (%d layers, %ux%u, %d mips)", image.layers,
                      image.extent.width, image.extent.height, image.mipCount);
    }
    else if (image.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && image.layout != VK_IMAGE_LAYOUT_GENERAL)
    {
        std::snprintf(problem, sizeof(problem), "layout %d (neither SHADER_READ_ONLY_OPTIMAL nor GENERAL)",
                      static_cast<int>(image.layout));
    }
    if (problem[0] != '\0')
    {
        if (g_rt.sky_logs < 8)
        {
            ++g_rt.sky_logs;
            Log(LogLevel::Warning, "the sky cubemap is dropped for this frame: %s; the procedural sky "
                                   "is used",
                problem);
        }
        return;
    }
    sky.image = reinterpret_cast<uint64_t>(image.image);
    sky.format = static_cast<uint32_t>(image.format);
    sky.layout = static_cast<uint32_t>(image.layout);
    sky.size = image.extent.width;
    sky.mip_levels = static_cast<uint32_t>(image.mipCount);
}

// ---------------------------------------------------------------------------------------------
// Scene feed

// True when [offset, offset + count * stride) lies inside the frame packet's memory block, after
// the fixed struct, and offset is aligned.
bool InPacket(const CaelixUnityFramePacket& packet, uint32_t offset, uint32_t count, uint32_t stride,
              uint32_t alignment)
{
    if (count == 0) return true;
    if (offset % alignment != 0 || offset < sizeof(CaelixUnityFramePacket)) return false;
    const uint64_t end = static_cast<uint64_t>(offset) + static_cast<uint64_t>(count) * stride;
    return end <= packet.total_bytes;
}

void RemoveEntity(const Guid& guid)
{
    State& state = GetState();
    auto it = g_rt.entities.find(guid);
    if (it == g_rt.entities.end()) return;
    const CaelixResult result = caelix_scene_remove_entity(state.scene, it->second);
    if (result != CAELIX_OK) SceneError(result, "caelix_scene_remove_entity failed");
    g_rt.entities.erase(it);
}

// Applies the session config of a frame packet with CAELIX_UNITY_FRAME_RESET: creates the scene
// (or, when the release event that should have destroyed the previous one did not run, empties the
// existing scene), then applies the material entries. False when there is no scene afterwards.
bool ResetSession(const CaelixUnityFramePacket& packet, const uint8_t* base)
{
    State& state = GetState();
    if (state.scene == nullptr)
    {
        CaelixSceneDesc desc{};
        desc.encoding = packet.encoding;
        desc.staging_bytes = static_cast<uint64_t>(packet.staging_mib) << 20;  // 0 = the library default
        const CaelixResult result = caelix_scene_create(state.device, &desc, &state.scene);
        if (result != CAELIX_OK)
        {
            state.scene = nullptr;
            Fail(result, "caelix_scene_create failed");
            return false;
        }
        g_rt.scene_encoding = packet.encoding;
        state.scene_encoding.store(packet.encoding);
        state.scene_resets.fetch_add(1);
        // The ring is empty now: a quarter of it is the most one frame packet should carry, because
        // the frames in flight keep their staging space until they complete.
        CaelixSceneStats stats{};
        caelix_scene_get_stats(state.scene, &stats);
        state.upload_bricks_free.store(stats.upload_bricks_free);
        state.upload_bricks_cap.store(stats.upload_bricks_free / 4);
        state.staging_bytes.store(stats.staging_bytes);
    }
    else
    {
        if (ShouldLogError())
        {
            Log(LogLevel::Error, "session reset while the plugin still has a scene (no release event with "
                                 "CAELIX_UNITY_RELEASE_SCENE before it): %zu entities removed, built-in "
                                 "materials restored; the scene keeps encoding %u (packet: %u) and its "
                                 "staging ring",
                g_rt.entities.size(), g_rt.scene_encoding, packet.encoding);
        }
        for (const auto& entry : g_rt.entities)
        {
            const CaelixResult result = caelix_scene_remove_entity(state.scene, entry.second);
            if (result != CAELIX_OK) SceneError(result, "caelix_scene_remove_entity failed (reset)");
        }
        g_rt.entities.clear();
        const CaelixResult result = caelix_scene_set_materials(state.scene, nullptr, 0);
        if (result != CAELIX_OK) SceneError(result, "caelix_scene_set_materials(default) failed (reset)");
    }

    if (!InPacket(packet, packet.materials_offset, packet.material_count, sizeof(CaelixMaterialEntry), 4))
    {
        SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "material entries outside the packet: built-in table kept");
    }
    else if (packet.material_count != 0)
    {
        const auto* entries = reinterpret_cast<const CaelixMaterialEntry*>(base + packet.materials_offset);
        const CaelixResult result = caelix_scene_set_material_entries(state.scene, entries, packet.material_count);
        if (result != CAELIX_OK) SceneError(result, "caelix_scene_set_material_entries failed");
    }
    Log(LogLevel::Info, "session reset: encoding %u, staging %u MiB (0 = library default), %u material "
                        "entries, upload cap %llu bricks per frame packet",
        g_rt.scene_encoding, packet.staging_mib, packet.material_count,
        static_cast<unsigned long long>(state.upload_bricks_cap.load()));
    return true;
}

// Uploads one batch. Bricks the staging ring (or an error) refuses are dropped and counted.
void Upload(uint32_t entity_index, CaelixEntityId entity, uint32_t count, const CaelixBrickKey* keys,
            const uint16_t* blocks)
{
    State& state = GetState();
    uint32_t consumed = 0;
    const CaelixResult result = caelix_scene_upload_bricks(state.scene, entity, count, keys, blocks, &consumed);
    if (result != CAELIX_OK) SceneError(result, "caelix_scene_upload_bricks failed");
    if (consumed > count) consumed = count;
    state.uploaded_total.fetch_add(consumed);
    if (consumed < count)
    {
        const uint64_t dropped = state.bricks_dropped.fetch_add(count - consumed) + (count - consumed);
        if (ShouldLogError())
        {
            Log(LogLevel::Error, "scene feed: the staging ring took %u of %u bricks of entity record %u "
                                 "(CaelixResult %d); %u dropped, %llu dropped so far (C# sends more than "
                                 "upload_bricks_free)",
                consumed, count, entity_index, static_cast<int>(result), count - consumed,
                static_cast<unsigned long long>(dropped));
        }
    }
}

// Applies the frame packet's records: the entities in list order, then the brick removes in runs
// of one entity, then the uploads. `base` is the packet memory (read in place). Returns true when
// the packet had records.
bool ApplyRecords(const CaelixUnityFramePacket& packet, const uint8_t* base)
{
    State& state = GetState();
    const bool has_records = packet.entity_count != 0 || packet.remove_count != 0 || packet.upload_count != 0;
    if (!has_records) return false;

    if (!InPacket(packet, packet.entities_offset, packet.entity_count, sizeof(CaelixUnityEntity), 4) ||
        !InPacket(packet, packet.removes_offset, packet.remove_count, sizeof(CaelixUnityBrickRemove), 4) ||
        !InPacket(packet, packet.uploads_offset, packet.upload_count, sizeof(CaelixUnityUpload), 4))
    {
        SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "scene records outside the packet: records ignored");
        return false;
    }

    // Entities, in list order.
    std::vector<CaelixEntityId>& ids = g_rt.packet_ids;
    ids.assign(packet.entity_count, CAELIX_INVALID_ENTITY);
    const auto* entities = reinterpret_cast<const CaelixUnityEntity*>(base + packet.entities_offset);
    for (uint32_t i = 0; i < packet.entity_count; ++i)
    {
        CaelixUnityEntity record;
        std::memcpy(&record, entities + i, sizeof(record));
        Guid guid;
        std::memcpy(&guid, record.guid, sizeof(guid));
        if ((record.flags & CAELIX_UNITY_ENTITY_REMOVE) != 0)
        {
            RemoveEntity(guid);
            continue;
        }
        if ((record.flags & CAELIX_UNITY_ENTITY_ADD) != 0)
        {
            RemoveEntity(guid);
            CaelixEntityDesc desc{};
            std::memcpy(desc.guid, record.guid, sizeof(desc.guid));
            std::memcpy(desc.position, record.position, sizeof(desc.position));
            std::memcpy(desc.rotation, record.rotation, sizeof(desc.rotation));
            CaelixEntityId id = CAELIX_INVALID_ENTITY;
            const CaelixResult result = caelix_scene_add_entity(state.scene, &desc, &id);
            if (result != CAELIX_OK)
            {
                SceneError(result, "caelix_scene_add_entity failed");
                continue;
            }
            g_rt.entities[guid] = id;
            ids[i] = id;
            continue;
        }
        auto it = g_rt.entities.find(guid);
        if (it == g_rt.entities.end())
        {
            SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "a record names an entity the scene does not have");
            continue;
        }
        ids[i] = it->second;
        if ((record.flags & CAELIX_UNITY_ENTITY_TRANSFORM) != 0)
        {
            const CaelixResult result =
                caelix_scene_set_entity_transform(state.scene, it->second, record.position, record.rotation);
            if (result != CAELIX_OK) SceneError(result, "caelix_scene_set_entity_transform failed");
        }
    }

    // Brick removes, in runs of one entity.
    const auto* removes = reinterpret_cast<const CaelixUnityBrickRemove*>(base + packet.removes_offset);
    for (uint32_t i = 0; i < packet.remove_count;)
    {
        CaelixUnityBrickRemove first;
        std::memcpy(&first, removes + i, sizeof(first));
        std::vector<CaelixBrickKey>& keys = g_rt.remove_scratch;
        keys.clear();
        uint32_t j = i;
        // The run ends where the entity record changes.
        for (; j < packet.remove_count; ++j)
        {
            CaelixUnityBrickRemove record;
            std::memcpy(&record, removes + j, sizeof(record));
            if (record.entity_index != first.entity_index) break;
            keys.push_back(record.key);
        }
        i = j;
        if (first.entity_index >= ids.size() || ids[first.entity_index] == CAELIX_INVALID_ENTITY)
        {
            SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "brick removes name an unknown entity record");
            continue;
        }
        const CaelixResult result = caelix_scene_remove_bricks(state.scene, ids[first.entity_index],
                                                               static_cast<uint32_t>(keys.size()), keys.data());
        if (result != CAELIX_OK)
            SceneError(result, "caelix_scene_remove_bricks failed");
        else
            state.removed_total.fetch_add(keys.size());
    }

    // Uploads.
    const auto* uploads = reinterpret_cast<const CaelixUnityUpload*>(base + packet.uploads_offset);
    for (uint32_t i = 0; i < packet.upload_count; ++i)
    {
        CaelixUnityUpload upload;
        std::memcpy(&upload, uploads + i, sizeof(upload));
        if (upload.brick_count == 0) continue;
        if (upload.entity_index >= ids.size() || ids[upload.entity_index] == CAELIX_INVALID_ENTITY)
        {
            SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "an upload names an unknown entity record");
            continue;
        }
        if (!InPacket(packet, upload.keys_offset, upload.brick_count, sizeof(CaelixBrickKey), 4) ||
            !InPacket(packet, upload.blocks_offset, upload.brick_count, kBlocksPerBrick * 2, 16))
        {
            SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "upload data outside the packet");
            continue;
        }
        Upload(upload.entity_index, ids[upload.entity_index], upload.brick_count,
               reinterpret_cast<const CaelixBrickKey*>(base + upload.keys_offset),
               reinterpret_cast<const uint16_t*>(base + upload.blocks_offset));
    }
    return true;
}

// Publishes the scene statistics for C# after a flush. Logs once when the scene becomes idle (no
// staged bricks, no pending builds) after a burst of work that lasted a while.
void PublishSceneStats()
{
    State& state = GetState();
    CaelixSceneStats stats{};
    caelix_scene_get_stats(state.scene, &stats);
    state.scene_entities.store(stats.entities);
    state.scene_groups.store(stats.groups);
    state.bricks_resident.store(stats.bricks_resident);
    state.as_builds_pending.store(stats.as_builds_pending);
    state.record_bytes.store(stats.record_bytes);
    state.blas_bytes.store(stats.blas_bytes);
    state.staging_bytes.store(stats.staging_bytes);
    state.upload_bricks_free.store(stats.upload_bricks_free);
    UpdateLiveViews();
    const bool busy = stats.as_builds_pending != 0 || stats.pending_bricks != 0;
    if (g_rt.scene_busy && !busy)
    {
        g_rt.scene_busy = false;
        // Only a burst that lasted a while is worth a line: a world that changes every frame goes
        // busy and idle once per frame and would write one line per frame to the Editor log.
        constexpr uint64_t kIdleLogMinFrames = 30;
        if (g_rt.last_flushed_frame - g_rt.scene_busy_since < kIdleLogMinFrames) return;
        Log(LogLevel::Info, "scene idle after Unity frame %llu: %u entities, %u groups, %llu bricks "
                            "resident, %llu uploaded so far, %llu dropped, records %.1f MiB, BLAS %.1f MiB",
            static_cast<unsigned long long>(g_rt.last_flushed_frame), stats.entities, stats.groups,
            static_cast<unsigned long long>(stats.bricks_resident),
            static_cast<unsigned long long>(state.uploaded_total.load()),
            static_cast<unsigned long long>(state.bricks_dropped.load()),
            static_cast<double>(stats.record_bytes) / (1 << 20),
            static_cast<double>(stats.blas_bytes) / (1 << 20));
    }
}

// ---------------------------------------------------------------------------------------------
// Events

void FrameEvent(void* data)
{
    State& state = GetState();
    if (data == nullptr)
    {
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "no packet");
        return;
    }
    CaelixUnityFramePacket packet;
    if (!ReadPacket(data, CAELIX_UNITY_PACKET_FRAME, packet))
    {
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "packet size, version or kind mismatch");
        return;
    }
    const auto* base = static_cast<const uint8_t*>(data);
    const uint64_t sequence = packet.header.sequence;

    if (state.device == nullptr)
    {
        PublishConsumed(state.last_consumed_frame_sequence, sequence);
        Fail(CAELIX_OK, "no adopted device (see the caelix_device_adopt log)");
        return;
    }
    const bool reset = (packet.flags & CAELIX_UNITY_FRAME_RESET) != 0;
    if (reset)
    {
        if (!ResetSession(packet, base))
        {
            PublishConsumed(state.last_consumed_frame_sequence, sequence);
            return;
        }
    }
    else if (state.scene == nullptr)
    {
        PublishConsumed(state.last_consumed_frame_sequence, sequence);
        Fail(CAELIX_OK, "frame packet without a scene (no session reset yet)");
        return;
    }

    const bool has_records = ApplyRecords(packet, base);
    if ((reset || has_records) && !g_rt.scene_busy)
    {
        g_rt.scene_busy = true;
        g_rt.scene_busy_since = g_rt.last_flushed_frame;
    }
    // The records are in the library now (the bricks in its staging ring): C# may reuse the memory.
    // The free staging space goes out first: C# subtracts the bricks of the frame packets it has
    // issued since the status it reads, so the two must describe the same moment.
    {
        CaelixSceneStats stats{};
        caelix_scene_get_stats(state.scene, &stats);
        state.upload_bricks_free.store(stats.upload_bricks_free);
    }
    PublishConsumed(state.last_consumed_frame_sequence, sequence);

    UnityVulkanRecordingState recording{};
    if (!GetRecordingState(recording))
    {
        // The records were applied; the flush waits for the next frame event.
        Fail(CAELIX_OK, "CommandRecordingState failed");
        return;
    }
    const uint64_t frame = recording.currentFrameNumber;
    const uint64_t safe = recording.safeFrameNumber;
    state.last_unity_frame.store(frame);
    state.last_safe_frame.store(safe);
    if (frame <= safe)
    {
        char what[128];
        std::snprintf(what, sizeof(what), "Unity frame %llu is not above the safe frame %llu: no flush",
                      static_cast<unsigned long long>(frame), static_cast<unsigned long long>(safe));
        Fail(CAELIX_OK, what);
        return;
    }
    if (frame <= g_rt.last_flushed_frame)
    {
        // A second frame event in this Unity frame: its records reach the GPU with the next flush.
        if (ShouldLogError())
        {
            Log(LogLevel::Warning, "frame event: Unity frame %llu was already flushed (a second frame "
                                   "event in one Unity frame); the records wait for the next flush",
                static_cast<unsigned long long>(frame));
        }
        return;
    }

    CaelixResult result = caelix_scene_recycle(state.scene, safe);
    if (result != CAELIX_OK) SceneError(result, "caelix_scene_recycle failed");

    CaelixFrameDesc frame_desc{};
    frame_desc.command_buffer = reinterpret_cast<uint64_t>(recording.commandBuffer);
    frame_desc.frame_number = frame;
    frame_desc.safe_frame_number = safe;
    result = caelix_scene_flush(state.scene, &frame_desc);
    if (result != CAELIX_OK)
    {
        // OUT_OF_MEMORY leaves acceleration structure builds pending but records everything else,
        // and the library counts the frame as flushed.
        Fail(result, "caelix_scene_flush failed");
        if (result != CAELIX_ERROR_OUT_OF_MEMORY) return;
    }
    g_rt.last_flushed_frame = frame;
    state.last_flushed_frame.store(frame);
    state.frames.fetch_add(1);
    PublishSceneStats();
}

void RenderEvent(void* data)
{
    State& state = GetState();
    if (data == nullptr)
    {
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "no packet");
        return;
    }
    CaelixUnityViewPacket packet;
    if (!ReadPacket(data, CAELIX_UNITY_PACKET_VIEW, packet))
    {
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "packet size, version or kind mismatch");
        return;
    }
    const uint64_t sequence = packet.header.sequence;

    if (state.device == nullptr)
    {
        PublishConsumed(state.last_consumed_view_sequence, sequence);
        Fail(CAELIX_OK, "no adopted device (see the caelix_device_adopt log)");
        return;
    }
    if (state.scene == nullptr)
    {
        // A camera can render before the first frame packet of a session (or after its end).
        PublishConsumed(state.last_consumed_view_sequence, sequence);
        ++g_rt.renders_without_scene;
        if (!g_rt.warned_render_without_scene)
        {
            g_rt.warned_render_without_scene = true;
            Log(LogLevel::Warning, "render event without a scene (no session reset yet): the view is not "
                                   "rendered (logged once)");
        }
        return;
    }
    if (packet.view >= CAELIX_UNITY_MAX_VIEWS)
    {
        PublishConsumed(state.last_consumed_view_sequence, sequence);
        char what[96];
        std::snprintf(what, sizeof(what), "packet view %u out of range (max %u)", packet.view,
                      CAELIX_UNITY_MAX_VIEWS - 1);
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, what);
        return;
    }
    if (packet.target_generation != g_rt.last_generation)
    {
        if (!g_rt.released_since_generation && ShouldLogError())
        {
            Log(LogLevel::Warning, "target generation changed (%u -> %u) without a release event: a "
                                   "cached image view may refer to a destroyed image",
                g_rt.last_generation,
                packet.target_generation);
        }
        g_rt.last_generation = packet.target_generation;
    }
    g_rt.released_since_generation = false;

    // Access calls may record commands and invalidate the recording state: all of them first.
    UnityVulkanImage color{};
    if (!AccessTarget(packet.color, color))
    {
        PublishConsumed(state.last_consumed_view_sequence, sequence);
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "AccessTexture failed for the colour target");
        return;
    }
    if ((color.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0)
    {
        PublishConsumed(state.last_consumed_view_sequence, sequence);
        Fail(CAELIX_ERROR_INVALID_ARGUMENT,
             "the colour target has no STORAGE usage (RenderTexture.enableRandomWrite is off)");
        return;
    }
    // The upscaler decision sets the size the geometry targets must have (no command recorded).
    CaelixRenderOptions options = packet.options;
    uint32_t geometry_width = 0;
    uint32_t geometry_height = 0;
    DecideUpscaler(packet.view, packet, color.extent.width, color.extent.height, options, geometry_width,
                   geometry_height);
    CaelixRenderTargets targets{};
    AccessOptionalTarget(packet.depth, "depth", geometry_width, geometry_height, targets.depth);
    AccessOptionalTarget(packet.motion, "motion", geometry_width, geometry_height, targets.motion);
    AccessOptionalTarget(packet.albedo, "albedo", geometry_width, geometry_height, targets.albedo);
    AccessSky(packet.sky, options.sky);

    UnityVulkanRecordingState recording{};
    if (!GetRecordingState(recording))
    {
        PublishConsumed(state.last_consumed_view_sequence, sequence);
        Fail(CAELIX_OK, "CommandRecordingState failed");
        return;
    }
    const uint64_t frame = recording.currentFrameNumber;
    const uint64_t safe = recording.safeFrameNumber;
    state.last_unity_frame.store(frame);
    state.last_safe_frame.store(safe);
    if (frame <= safe)
    {
        PublishConsumed(state.last_consumed_view_sequence, sequence);
        char what[128];
        std::snprintf(what, sizeof(what), "Unity frame %llu is not above the safe frame %llu: no render",
                      static_cast<unsigned long long>(frame), static_cast<unsigned long long>(safe));
        Fail(CAELIX_OK, what);
        return;
    }

    targets.color.image = reinterpret_cast<uint64_t>(color.image);
    targets.color.format = static_cast<uint32_t>(color.format);
    targets.color.width = color.extent.width;
    targets.color.height = color.extent.height;
    targets.color.layout = static_cast<uint32_t>(color.layout);
    if (color.extent.width != packet.color.width || color.extent.height != packet.color.height)
    {
        if (ShouldLogError())
        {
            Log(LogLevel::Warning, "colour target is %ux%u, the packet says %ux%u; using the image size",
                color.extent.width, color.extent.height, packet.color.width, packet.color.height);
        }
    }
    if (!EnsureRenderer(packet.view))
    {
        PublishConsumed(state.last_consumed_view_sequence, sequence);
        return;
    }

    CaelixFrameDesc frame_desc{};
    frame_desc.command_buffer = reinterpret_cast<uint64_t>(recording.commandBuffer);
    frame_desc.frame_number = frame;
    frame_desc.safe_frame_number = safe;
    const CaelixResult result = caelix_renderer_render(state.renderers[packet.view], &frame_desc, &targets,
                                                       &packet.camera, &options);
    PublishConsumed(state.last_consumed_view_sequence, sequence);
    if (result != CAELIX_OK)
    {
        // The library has logged the reason (for example a second render of this view in one Unity
        // frame).
        Fail(result, "caelix_renderer_render failed");
        return;
    }
    state.renders.fetch_add(1);
    if ((packet.flags & CAELIX_UNITY_VIEW_REPORT_TIMINGS) != 0)
    {
        CaelixGpuTimings timings{};
        caelix_renderer_get_gpu_timings(state.renderers[packet.view], &timings);
        state.flush_ms.store(timings.flush_ms);
        state.trace_ms.store(timings.trace_ms);
        state.composite_ms.store(timings.composite_ms);
        state.denoise_ms.store(timings.denoise_ms);
        state.upscale_ms.store(timings.upscale_ms);
        state.timings_frame.store(timings.frame_number);
        state.timings_view.store(packet.view);
    }
    if (g_rt.access_logs < 4)
    {
        ++g_rt.access_logs;
        Log(LogLevel::Info, "render, Unity frame %llu, view %u: AccessTexture reported layout %d (VkImage %p)",
            static_cast<unsigned long long>(frame), packet.view, static_cast<int>(color.layout),
            static_cast<void*>(color.image));
    }
    if (!g_rt.logged_first_render)
    {
        g_rt.logged_first_render = true;
        Log(LogLevel::Info, "first render: view %u, %ux%u, VkFormat %d, layout at access %d, Unity frame "
                            "%llu (safe %llu), mode %u, sky cubemap %s, upscaler %s",
            packet.view, color.extent.width, color.extent.height, static_cast<int>(color.format),
            static_cast<int>(color.layout), static_cast<unsigned long long>(frame),
            static_cast<unsigned long long>(safe), options.mode, options.sky.image != 0 ? "yes" : "no",
            UpscalerName(options.upscaler));
    }
}

// Runs with graphics queue access (ConfigureEvents below): Unity has submitted its command buffers
// and keeps its worker threads off the queue, so waiting for the queue to be idle is legal here.
void ReleaseEvent(void* data)
{
    State& state = GetState();
    if (data == nullptr)
    {
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "no packet");
        return;
    }
    CaelixUnityReleasePacket packet;
    if (!ReadPacket(data, CAELIX_UNITY_PACKET_RELEASE, packet))
    {
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "packet size, version or kind mismatch");
        return;
    }
    const bool release_scene = (packet.flags & CAELIX_UNITY_RELEASE_SCENE) != 0;
    const bool release_view = (packet.flags & CAELIX_UNITY_RELEASE_VIEW) != 0;
    const bool release_target_views = (packet.flags & CAELIX_UNITY_RELEASE_TARGET_VIEWS) != 0;
    if (release_view && packet.view >= CAELIX_UNITY_MAX_VIEWS)
    {
        char what[96];
        std::snprintf(what, sizeof(what), "release of view %u out of range (max %u)", packet.view,
                      CAELIX_UNITY_MAX_VIEWS - 1);
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, what);
    }
    const bool view_in_range = packet.view < CAELIX_UNITY_MAX_VIEWS;

    const bool work = (release_scene && (state.scene != nullptr || HasRenderer())) ||
                      (release_view && view_in_range && state.renderers[packet.view] != nullptr) ||
                      (release_target_views && HasRenderer());
    if (work && state.queue_wait_idle != nullptr)
    {
        state.queue_wait_idle(state.vk_queue);
    }
    if (release_scene)
    {
        // Every renderer goes with the scene, so there are no target views left to release.
        if (state.scene != nullptr || HasRenderer())
        {
            const size_t entities = g_rt.entities.size();
            DestroySceneObjects();
            Log(LogLevel::Info, "session end: renderers and scene destroyed (%zu entities)", entities);
        }
    }
    else
    {
        if (release_view && view_in_range && state.renderers[packet.view] != nullptr)
        {
            caelix_renderer_destroy(state.renderers[packet.view]);
            state.renderers[packet.view] = nullptr;
            UpdateLiveViews();
            Log(LogLevel::Info, "renderer destroyed for view %u", packet.view);
        }
        if (release_target_views)
        {
            for (CaelixRenderer* renderer : state.renderers)
            {
                if (renderer != nullptr) caelix_renderer_release_target_views(renderer);
            }
        }
    }
    if (release_scene || release_target_views)
    {
        g_rt.released_since_generation = true;
    }
    state.releases.fetch_add(1);
    PublishConsumed(state.last_consumed_release_sequence, packet.header.sequence);
}

}  // namespace

void UNITY_INTERFACE_API OnPluginEvent(int event_id, void* data)
{
    const State& state = GetState();
    if (event_id == state.frame_event_id)
    {
        g_rt.event_name = "frame";
        FrameEvent(data);
    }
    else if (event_id == state.render_event_id)
    {
        g_rt.event_name = "render";
        RenderEvent(data);
    }
    else if (event_id == state.release_event_id)
    {
        g_rt.event_name = "release";
        ReleaseEvent(data);
    }
}

void ConfigureEvents()
{
    State& state = GetState();
    // Frame and render events: we bind compute pipelines and descriptor sets, so Unity must not
    // assume its bindings survive.
    UnityVulkanPluginEventConfig render{};
    render.renderPassPrecondition = kUnityVulkanRenderPass_EnsureOutside;
    render.graphicsQueueAccess = kUnityVulkanGraphicsQueueAccess_DontCare;
    render.flags = kUnityVulkanEventConfigFlag_EnsurePreviousFrameSubmission |
                   kUnityVulkanEventConfigFlag_ModifiesCommandBuffersState;
    state.vulkan->ConfigureEvent(state.frame_event_id, &render);
    state.vulkan->ConfigureEvent(state.render_event_id, &render);

    UnityVulkanPluginEventConfig release{};
    release.renderPassPrecondition = kUnityVulkanRenderPass_EnsureOutside;
    release.graphicsQueueAccess = kUnityVulkanGraphicsQueueAccess_Allow;
    release.flags = kUnityVulkanEventConfigFlag_EnsurePreviousFrameSubmission |
                    kUnityVulkanEventConfigFlag_FlushCommandBuffers |
                    kUnityVulkanEventConfigFlag_SyncWorkerThreads;
    state.vulkan->ConfigureEvent(state.release_event_id, &release);
    Log(LogLevel::Info, "plugin events configured: frame %d and render %d (outside render pass, no queue "
                        "access), release %d (queue access, flush)",
        state.frame_event_id, state.render_event_id, state.release_event_id);
}

void DestroyRenderObjects()
{
    DestroySceneObjects();
    g_rt = RenderThreadState{};
    // The upscaler queries belonged to the device that goes away.
    State& state = GetState();
    std::lock_guard<std::mutex> lock(state.upscalers_mutex);
    for (CaelixUnityUpscalerStatus& entry : state.upscalers)
    {
        entry = CaelixUnityUpscalerStatus{};
    }
}

}  // namespace caelix_unity

extern "C" CAELIX_UNITY_API void* caelix_unity_get_render_event_func(void)
{
    return reinterpret_cast<void*>(&caelix_unity::OnPluginEvent);
}

extern "C" CAELIX_UNITY_API void caelix_unity_status(CaelixUnityStatus* out)
{
    if (out == nullptr)
    {
        return;
    }
    caelix_unity::State& state = caelix_unity::GetState();
    CaelixUnityStatus status{};
    status.size = sizeof(CaelixUnityStatus);
    status.packet_version = CAELIX_UNITY_PACKET_VERSION;
    status.frame_packet_size = sizeof(CaelixUnityFramePacket);
    status.view_packet_size = sizeof(CaelixUnityViewPacket);
    status.release_packet_size = sizeof(CaelixUnityReleasePacket);
    status.unity_renderer = state.unity_renderer.load();
    status.intercept_registered = state.intercept_registered.load();
    status.intercept_called = state.intercept_called.load();
    status.instance_patched = state.instance_patched.load();
    status.device_patched = state.device_patched.load();
    status.device_adopted = state.device_adopted.load();
    status.last_error = state.last_error.load();
    status.frame_event_id = state.frame_event_id;
    status.render_event_id = state.render_event_id;
    status.release_event_id = state.release_event_id;
    status.last_consumed_frame_sequence = state.last_consumed_frame_sequence.load();
    status.last_consumed_view_sequence = state.last_consumed_view_sequence.load();
    status.last_consumed_release_sequence = state.last_consumed_release_sequence.load();
    status.last_unity_frame = state.last_unity_frame.load();
    status.last_safe_frame = state.last_safe_frame.load();
    status.last_flushed_frame = state.last_flushed_frame.load();
    status.frames = state.frames.load();
    status.renders = state.renders.load();
    status.releases = state.releases.load();
    status.failed_events = state.failed_events.load();
    status.scene_encoding = state.scene_encoding.load();
    status.entities = state.scene_entities.load();
    status.groups = state.scene_groups.load();
    status.scene_resets = state.scene_resets.load();
    status.live_views = state.live_views.load();
    status.upscaler_extensions = state.upscaler_extensions.load();
    status.bricks_resident = state.bricks_resident.load();
    status.as_builds_pending = state.as_builds_pending.load();
    status.uploaded_total = state.uploaded_total.load();
    status.removed_total = state.removed_total.load();
    status.bricks_dropped = state.bricks_dropped.load();
    status.upload_bricks_free = state.upload_bricks_free.load();
    status.upload_bricks_cap = state.upload_bricks_cap.load();
    status.record_bytes = state.record_bytes.load();
    status.blas_bytes = state.blas_bytes.load();
    status.staging_bytes = state.staging_bytes.load();
    status.flush_ms = state.flush_ms.load();
    status.trace_ms = state.trace_ms.load();
    status.composite_ms = state.composite_ms.load();
    status.denoise_ms = state.denoise_ms.load();
    status.timings_frame = state.timings_frame.load();
    status.timings_view = state.timings_view.load();
    status.upscale_ms = state.upscale_ms.load();
    {
        std::lock_guard<std::mutex> lock(state.upscalers_mutex);
        std::memcpy(status.upscalers, state.upscalers, sizeof(status.upscalers));
    }
    *out = status;
}
