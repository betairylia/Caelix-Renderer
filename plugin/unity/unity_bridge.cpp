// The plugin's C API for C# and the render-thread work of the plugin events.
//
// Frame numbers: the library needs one strictly increasing frame number per flush, and a safe
// frame number below it. Unity's currentFrameNumber is not strictly increasing per event (two
// cameras render in one Unity frame), so the plugin counts its own frames: every render event is
// plugin frame n + 1, and the plugin remembers the Unity frame of each one. The safe plugin frame
// is the newest plugin frame whose Unity frame is <= Unity's safeFrameNumber.
//
// Scene feed (packet version 2): each render event applies the packet's scene records (entities,
// brick removes, brick uploads, materials) before the flush. Bricks the staging ring cannot take
// are copied into a plugin-owned pending queue and uploaded first at the next event; while the
// queue is not empty, new uploads are appended to it, so the bricks of one entity reach the
// library in the order C# sent them.
//
// Views (packet version 4): one scene, one renderer per view (CaelixUnityPacket::view). A packet
// applies its scene records and flushes the scene whatever its view; only the render call uses
// the view's renderer, which is created on the view's first render event.
#include "plugin_state.h"

#include <caelix_unity.h>

#include <cstdio>
#include <cstring>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace caelix_unity {
namespace {

constexpr uint32_t kBlocksPerBrick = 512;
constexpr uint32_t kMaterialEntries = 65536;
constexpr uint32_t kNoScene = 0xFFFFFFFFu;

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

struct KeyHash
{
    size_t operator()(const CaelixBrickKey& key) const
    {
        return static_cast<size_t>(static_cast<uint32_t>(key.x) * 73856093u ^
                                   static_cast<uint32_t>(key.y) * 19349663u ^
                                   static_cast<uint32_t>(key.z) * 83492791u);
    }
};

struct KeyEqual
{
    bool operator()(const CaelixBrickKey& a, const CaelixBrickKey& b) const
    {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
};

// Bricks the staging ring could not take, copied out of a packet. `offset` is the first brick not
// uploaded yet.
struct PendingBatch
{
    CaelixEntityId entity = CAELIX_INVALID_ENTITY;
    std::vector<CaelixBrickKey> keys;
    std::vector<uint16_t> blocks;
    size_t offset = 0;
};

// Render thread only.
struct RenderThreadState
{
    std::deque<std::pair<uint64_t, uint64_t>> frames;  // (plugin frame, Unity frame), oldest first
    uint64_t plugin_frame = 0;
    uint64_t safe_plugin_frame = 0;
    uint32_t last_generation = 0;
    bool released_since_generation = true;
    bool logged_first_render = false;
    bool logged_access_fallback = false;
    bool logged_level = false;
    bool warned_encoding = false;
    uint32_t dropped_target_logs = 0;
    bool scene_busy = false;        // the scene had uploads, pending bricks or pending builds
    uint64_t scene_busy_since = 0;  // plugin frame of the first busy packet of the current burst
    uint32_t scene_encoding = kNoScene;
    uint64_t error_logs = 0;
    uint32_t access_logs = 0;

    std::unordered_map<Guid, CaelixEntityId, GuidHash> entities;
    std::deque<PendingBatch> pending;
    uint64_t pending_bricks = 0;
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
        Log(LogLevel::Error, "render event: %s (CaelixResult %d)", what, static_cast<int>(result));
    }
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

// Publishes the packet's sequence: C# may reuse its memory from now on.
void PublishConsumed(uint64_t sequence)
{
    State& state = GetState();
    uint64_t consumed = state.last_consumed_sequence.load();
    while (sequence > consumed &&
           !state.last_consumed_sequence.compare_exchange_weak(consumed, sequence))
    {
    }
}

void ForgetSceneContents()
{
    g_rt.entities.clear();
    g_rt.pending.clear();
    g_rt.pending_bricks = 0;
    GetState().bricks_pending_plugin.store(0);
}

// Creates the shared scene with `encoding` when there is none.
bool EnsureScene(uint32_t encoding)
{
    State& state = GetState();
    if (state.scene == nullptr)
    {
        CaelixSceneDesc desc{};
        desc.encoding = encoding;
        const CaelixResult result = caelix_scene_create(state.device, &desc, &state.scene);
        if (result != CAELIX_OK)
        {
            state.scene = nullptr;
            Fail(result, "caelix_scene_create failed");
            return false;
        }
        Log(LogLevel::Info, "scene created (encoding %u)", encoding);
        g_rt.scene_encoding = encoding;
        g_rt.warned_encoding = false;
        state.scene_encoding.store(encoding);
        ForgetSceneContents();
    }
    else if (encoding != g_rt.scene_encoding && !g_rt.warned_encoding)
    {
        // The release event before this packet should have destroyed the scene (C# issues it when the
        // encoding changes). Without it the scene keeps its encoding.
        g_rt.warned_encoding = true;
        Log(LogLevel::Warning, "packet encoding %u differs from the scene's %u and no release event "
                               "recreated the scene; the scene keeps %u",
            encoding, g_rt.scene_encoding,
            g_rt.scene_encoding);
    }
    return true;
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
        Log(LogLevel::Info, "renderer created for view %u", view);
    }
    return true;
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

// Maps Unity's safe frame to the newest safe plugin frame.
uint64_t SafePluginFrame(uint64_t unity_safe)
{
    while (!g_rt.frames.empty() && g_rt.frames.front().second <= unity_safe)
    {
        g_rt.safe_plugin_frame = g_rt.frames.front().first;
        g_rt.frames.pop_front();
    }
    return g_rt.safe_plugin_frame;
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
// failure only drops the target (logged a few times), the frame still renders.
bool AccessOptionalTarget(const CaelixUnityTarget& target, const char* name, uint32_t width,
                          uint32_t height, CaelixTargetImage& out)
{
    out = CaelixTargetImage{};
    if (target.native_texture == nullptr && target.native_render_buffer == nullptr) return false;
    UnityVulkanImage image{};
    const char* problem = nullptr;
    if (!AccessTarget(target, image))
    {
        problem = "AccessTexture failed";
    }
    else if ((image.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0)
    {
        problem = "no STORAGE usage (RenderTexture.enableRandomWrite is off)";
    }
    else if (image.extent.width != width || image.extent.height != height)
    {
        problem = "its size differs from the colour target";
    }
    if (problem != nullptr)
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

// ---------------------------------------------------------------------------------------------
// Scene feed

// True when [offset, offset + count * stride) lies inside the packet and offset is aligned.
bool InPacket(const CaelixUnityPacket& packet, uint32_t offset, uint32_t count, uint32_t stride,
              uint32_t alignment)
{
    if (count == 0) return true;
    if (offset % alignment != 0 || offset < sizeof(CaelixUnityPacket)) return false;
    const uint64_t end = static_cast<uint64_t>(offset) + static_cast<uint64_t>(count) * stride;
    return end <= packet.total_bytes;
}

void UpdatePendingCount()
{
    uint64_t total = 0;
    for (const PendingBatch& batch : g_rt.pending) total += batch.keys.size() - batch.offset;
    g_rt.pending_bricks = total;
    GetState().bricks_pending_plugin.store(total);
}

void QueuePending(CaelixEntityId entity, uint32_t count, const CaelixBrickKey* keys,
                  const uint16_t* blocks)
{
    PendingBatch batch;
    batch.entity = entity;
    batch.keys.assign(keys, keys + count);
    batch.blocks.assign(blocks, blocks + static_cast<size_t>(count) * kBlocksPerBrick);
    g_rt.pending.push_back(std::move(batch));
}

// Uploads as much as the staging ring takes. Returns the number of bricks accepted; false in
// `ok` when the library returned an error.
uint32_t Upload(CaelixEntityId entity, uint32_t count, const CaelixBrickKey* keys,
                const uint16_t* blocks, bool& ok)
{
    State& state = GetState();
    uint32_t consumed = 0;
    const CaelixResult result =
        caelix_scene_upload_bricks(state.scene, entity, count, keys, blocks, &consumed);
    ok = result == CAELIX_OK;
    if (!ok) SceneError(result, "caelix_scene_upload_bricks failed");
    if (consumed > count) consumed = count;
    state.uploaded_total.fetch_add(consumed);
    return consumed;
}

// Uploads the pending queue in order until the staging ring is full.
void RetryPending()
{
    while (!g_rt.pending.empty())
    {
        PendingBatch& batch = g_rt.pending.front();
        const size_t remaining = batch.keys.size() - batch.offset;
        bool ok = true;
        const uint32_t consumed =
            Upload(batch.entity, static_cast<uint32_t>(remaining), batch.keys.data() + batch.offset,
                   batch.blocks.data() + batch.offset * kBlocksPerBrick, ok);
        batch.offset += consumed;
        if (batch.offset < batch.keys.size())
        {
            if (!ok)
            {
                // An invalid batch would block the queue forever: drop it (the error is logged).
                g_rt.pending.pop_front();
                continue;
            }
            break;
        }
        g_rt.pending.pop_front();
    }
}

// Uploads a packet batch, or queues it behind the pending bricks.
void UploadOrQueue(CaelixEntityId entity, uint32_t count, const CaelixBrickKey* keys,
                   const uint16_t* blocks)
{
    if (!g_rt.pending.empty())
    {
        QueuePending(entity, count, keys, blocks);
        return;
    }
    bool ok = true;
    const uint32_t consumed = Upload(entity, count, keys, blocks, ok);
    if (consumed < count && ok)
    {
        QueuePending(entity, count - consumed, keys + consumed,
                     blocks + static_cast<size_t>(consumed) * kBlocksPerBrick);
    }
}

void DropPendingOf(CaelixEntityId entity)
{
    for (auto it = g_rt.pending.begin(); it != g_rt.pending.end();)
    {
        it = it->entity == entity ? g_rt.pending.erase(it) : it + 1;
    }
}

// Removes the keys from the pending batches of `entity` (they were removed after being queued).
void FilterPending(CaelixEntityId entity, const std::vector<CaelixBrickKey>& removed)
{
    std::unordered_set<CaelixBrickKey, KeyHash, KeyEqual> set(removed.begin(), removed.end());
    for (PendingBatch& batch : g_rt.pending)
    {
        if (batch.entity != entity) continue;
        size_t write = batch.offset;
        for (size_t read = batch.offset; read < batch.keys.size(); ++read)
        {
            if (set.count(batch.keys[read]) != 0) continue;
            if (write != read)
            {
                batch.keys[write] = batch.keys[read];
                std::memcpy(batch.blocks.data() + write * kBlocksPerBrick,
                            batch.blocks.data() + read * kBlocksPerBrick, kBlocksPerBrick * 2);
            }
            ++write;
        }
        batch.keys.resize(write);
        batch.blocks.resize(write * kBlocksPerBrick);
    }
    for (auto it = g_rt.pending.begin(); it != g_rt.pending.end();)
    {
        it = it->offset >= it->keys.size() ? g_rt.pending.erase(it) : it + 1;
    }
}

void RemoveEntity(const Guid& guid)
{
    State& state = GetState();
    auto it = g_rt.entities.find(guid);
    if (it == g_rt.entities.end()) return;
    DropPendingOf(it->second);
    const CaelixResult result = caelix_scene_remove_entity(state.scene, it->second);
    if (result != CAELIX_OK) SceneError(result, "caelix_scene_remove_entity failed");
    g_rt.entities.erase(it);
}

void ResetScene()
{
    State& state = GetState();
    for (const auto& entry : g_rt.entities)
    {
        const CaelixResult result = caelix_scene_remove_entity(state.scene, entry.second);
        if (result != CAELIX_OK) SceneError(result, "caelix_scene_remove_entity failed (reset)");
    }
    const size_t removed = g_rt.entities.size();
    ForgetSceneContents();
    const CaelixResult result = caelix_scene_set_materials(state.scene, nullptr, 0);
    if (result != CAELIX_OK) SceneError(result, "caelix_scene_set_materials(default) failed (reset)");
    Log(LogLevel::Info, "scene reset: %zu entities removed, pending uploads dropped, built-in "
                        "materials restored",
        removed);
}

void ApplyMaterials(const CaelixUnityPacket& packet, const uint8_t* base)
{
    State& state = GetState();
    if (packet.material_mode == CAELIX_UNITY_MATERIALS_NONE) return;
    if (packet.material_count > kMaterialEntries)
    {
        SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "material_count is above 65536");
        return;
    }
    if (packet.material_mode == CAELIX_UNITY_MATERIALS_FULL)
    {
        if (!InPacket(packet, packet.materials_offset, packet.material_count, sizeof(CaelixMaterial), 4))
        {
            SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "material table outside the packet");
            return;
        }
        const auto* entries = packet.material_count != 0
                                  ? reinterpret_cast<const CaelixMaterial*>(base + packet.materials_offset)
                                  : nullptr;
        const CaelixResult result = caelix_scene_set_materials(state.scene, entries, packet.material_count);
        if (result != CAELIX_OK)
            SceneError(result, "caelix_scene_set_materials failed");
        else
            Log(LogLevel::Info, "materials: full table, %u entries", packet.material_count);
        return;
    }
    if (packet.material_mode != CAELIX_UNITY_MATERIALS_PARTIAL)
    {
        SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "unknown material_mode");
        return;
    }
    if (!InPacket(packet, packet.materials_offset, packet.material_count,
                  sizeof(CaelixUnityMaterialEntry), 4))
    {
        SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "material entries outside the packet");
        return;
    }
    // The built-in table first (the CPU copy is updated at once), then the entries over it.
    CaelixResult result = caelix_scene_set_materials(state.scene, nullptr, 0);
    if (result != CAELIX_OK)
    {
        SceneError(result, "caelix_scene_set_materials(default) failed");
        return;
    }
    std::vector<CaelixMaterial> table(kMaterialEntries);
    for (uint32_t code = 0; code < kMaterialEntries; ++code)
    {
        result = caelix_scene_get_material(state.scene, code, &table[code]);
        if (result != CAELIX_OK)
        {
            SceneError(result, "caelix_scene_get_material failed");
            return;
        }
    }
    const auto* entries = reinterpret_cast<const CaelixUnityMaterialEntry*>(base + packet.materials_offset);
    for (uint32_t i = 0; i < packet.material_count; ++i)
    {
        CaelixUnityMaterialEntry entry;
        std::memcpy(&entry, entries + i, sizeof(entry));
        if (entry.code >= kMaterialEntries)
        {
            SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "material entry code above 0xFFFF");
            return;
        }
        table[entry.code] = entry.material;
    }
    result = caelix_scene_set_materials(state.scene, table.data(), kMaterialEntries);
    if (result != CAELIX_OK)
        SceneError(result, "caelix_scene_set_materials failed");
    else
        Log(LogLevel::Info, "materials: %u entries over the built-in table", packet.material_count);
}

// Applies the packet's scene records. `base` is the packet memory (read in place).
void ApplyScene(const CaelixUnityPacket& packet, const uint8_t* base)
{
    State& state = GetState();
    const bool has_records = (packet.scene_flags & CAELIX_UNITY_SCENE_RESET) != 0 ||
                             packet.entity_count != 0 || packet.remove_count != 0 ||
                             packet.upload_count != 0 ||
                             packet.material_mode != CAELIX_UNITY_MATERIALS_NONE;
    if (has_records) state.scene_packets.fetch_add(1);

    if ((packet.scene_flags & CAELIX_UNITY_SCENE_RESET) != 0) ResetScene();

    if (!InPacket(packet, packet.entities_offset, packet.entity_count, sizeof(CaelixUnityEntity), 4) ||
        !InPacket(packet, packet.removes_offset, packet.remove_count, sizeof(CaelixUnityBrickRemove), 4) ||
        !InPacket(packet, packet.uploads_offset, packet.upload_count, sizeof(CaelixUnityUpload), 4))
    {
        SceneError(CAELIX_ERROR_INVALID_ARGUMENT, "scene records outside the packet: packet ignored");
        RetryPending();
        return;
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

    // Bricks of earlier packets first.
    RetryPending();

    // Brick removes, in runs of one entity.
    const auto* removes = reinterpret_cast<const CaelixUnityBrickRemove*>(base + packet.removes_offset);
    for (uint32_t i = 0; i < packet.remove_count;)
    {
        CaelixUnityBrickRemove first;
        std::memcpy(&first, removes + i, sizeof(first));
        std::vector<CaelixBrickKey>& keys = g_rt.remove_scratch;
        keys.clear();
        uint32_t j = i;
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
        const CaelixEntityId entity = ids[first.entity_index];
        if (!g_rt.pending.empty()) FilterPending(entity, keys);
        const CaelixResult result = caelix_scene_remove_bricks(state.scene, entity,
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
        UploadOrQueue(ids[upload.entity_index], upload.brick_count,
                      reinterpret_cast<const CaelixBrickKey*>(base + upload.keys_offset),
                      reinterpret_cast<const uint16_t*>(base + upload.blocks_offset));
    }
    UpdatePendingCount();

    ApplyMaterials(packet, base);
    if (has_records && !g_rt.scene_busy)
    {
        g_rt.scene_busy = true;
        g_rt.scene_busy_since = g_rt.plugin_frame;
    }
}

// Publishes the scene statistics and the GPU timings for C#. Logs once when the scene becomes
// idle (no pending bricks, no pending builds) after it had work.
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
    // The timings are those of view 0 (the Game view in C#).
    if (state.renderers[0] != nullptr)
    {
        CaelixGpuTimings timings{};
        caelix_renderer_get_gpu_timings(state.renderers[0], &timings);
        state.flush_ms.store(timings.flush_ms);
        state.trace_ms.store(timings.trace_ms);
        state.composite_ms.store(timings.composite_ms);
        state.denoise_ms.store(timings.denoise_ms);
        state.timings_frame.store(timings.frame_number);
    }
    const bool busy = g_rt.pending_bricks != 0 || stats.as_builds_pending != 0 || stats.pending_bricks != 0;
    if (g_rt.scene_busy && !busy)
    {
        g_rt.scene_busy = false;
        // Only a burst that lasted a while is worth a line: a world that changes every frame goes
        // busy and idle once per frame and would write one line per frame to the Editor log.
        constexpr uint64_t kIdleLogMinFrames = 30;
        if (g_rt.plugin_frame - g_rt.scene_busy_since < kIdleLogMinFrames) return;
        Log(LogLevel::Info, "scene idle after plugin frame %llu: %u entities, %u groups, %llu bricks "
                            "resident, %llu uploaded so far, records %.1f MiB, BLAS %.1f MiB",
            static_cast<unsigned long long>(g_rt.plugin_frame), stats.entities, stats.groups,
            static_cast<unsigned long long>(stats.bricks_resident),
            static_cast<unsigned long long>(state.uploaded_total.load()),
            static_cast<double>(stats.record_bytes) / (1 << 20),
            static_cast<double>(stats.blas_bytes) / (1 << 20));
    }
}

// Reads the fixed part of a packet. False (logged) when the size or the version differs.
bool ReadPacketHeader(const void* data, CaelixUnityPacket& packet)
{
    uint32_t size_and_version[2];
    std::memcpy(size_and_version, data, sizeof(size_and_version));
    if (size_and_version[0] != sizeof(CaelixUnityPacket) ||
        size_and_version[1] != CAELIX_UNITY_PACKET_VERSION)
    {
        if (ShouldLogError())
        {
            Log(LogLevel::Error, "packet size %u version %u, the plugin expects size %u version %u "
                                 "(C# and plugin differ)",
                size_and_version[0], size_and_version[1],
                static_cast<unsigned>(sizeof(CaelixUnityPacket)), CAELIX_UNITY_PACKET_VERSION);
        }
        return false;
    }
    std::memcpy(&packet, data, sizeof(packet));
    return true;
}

void RenderEvent(void* data)
{
    State& state = GetState();
    if (data == nullptr)
    {
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "no packet");
        return;
    }
    CaelixUnityPacket packet;
    if (!ReadPacketHeader(data, packet))
    {
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "packet size or version mismatch");
        return;  // the sequence cannot be trusted; C# reuses the slot after its stale timeout
    }
    const auto* base = static_cast<const uint8_t*>(data);

    if (state.device == nullptr)
    {
        PublishConsumed(packet.sequence);
        Fail(CAELIX_OK, "no adopted device (see the caelix_device_adopt log)");
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

    if (!EnsureScene(packet.encoding))
    {
        PublishConsumed(packet.sequence);
        return;
    }

    // Access calls may record commands and invalidate the recording state: all of them first.
    UnityVulkanImage color{};
    bool color_ok = AccessTarget(packet.color, color);
    if (!color_ok)
    {
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, "AccessTexture failed for the colour target");
    }
    else if ((color.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0)
    {
        color_ok = false;
        Fail(CAELIX_ERROR_INVALID_ARGUMENT,
             "the colour target has no STORAGE usage (RenderTexture.enableRandomWrite is off)");
    }
    CaelixRenderTargets targets{};
    if (color_ok)
    {
        AccessOptionalTarget(packet.depth, "depth", color.extent.width, color.extent.height, targets.depth);
        AccessOptionalTarget(packet.motion, "motion", color.extent.width, color.extent.height, targets.motion);
        AccessOptionalTarget(packet.albedo, "albedo", color.extent.width, color.extent.height, targets.albedo);
    }
    UnityVulkanRecordingState recording{};
    const bool recording_ok =
        state.vulkan->CommandRecordingState(&recording, kUnityVulkanGraphicsQueueAccess_DontCare);
    if (!recording_ok)
    {
        // The scene records are still applied (they must not be lost); flush and render wait.
        ApplyScene(packet, base);
        PublishConsumed(packet.sequence);
        Fail(CAELIX_OK, "CommandRecordingState failed");
        return;
    }
    if (!g_rt.logged_level)
    {
        g_rt.logged_level = true;
        Log(LogLevel::Info, "render event: command buffer %p level %s, render pass %p, framebuffer %p, "
                            "subpass %d, Unity frame %llu, safe %llu",
            static_cast<void*>(recording.commandBuffer),
            recording.commandBufferLevel == VK_COMMAND_BUFFER_LEVEL_PRIMARY ? "primary" : "secondary",
            static_cast<void*>(recording.renderPass), static_cast<void*>(recording.framebuffer),
            recording.subPassIndex, recording.currentFrameNumber, recording.safeFrameNumber);
    }

    const uint64_t frame = ++g_rt.plugin_frame;
    g_rt.frames.emplace_back(frame, recording.currentFrameNumber);
    uint64_t safe = SafePluginFrame(recording.safeFrameNumber);
    if (safe >= frame) safe = frame - 1;
    state.last_unity_frame.store(recording.currentFrameNumber);
    state.last_safe_frame.store(recording.safeFrameNumber);
    state.last_plugin_frame.store(frame);

    CaelixFrameDesc frame_desc{};
    frame_desc.command_buffer = reinterpret_cast<uint64_t>(recording.commandBuffer);
    frame_desc.frame_number = frame;
    frame_desc.safe_frame_number = safe;

    // Recycle first: the uploads below need the staging space of the completed frames.
    CaelixResult result = caelix_scene_recycle(state.scene, safe);
    if (result != CAELIX_OK)
    {
        SceneError(result, "caelix_scene_recycle failed");
    }
    ApplyScene(packet, base);
    PublishConsumed(packet.sequence);

    result = caelix_scene_flush(state.scene, &frame_desc);
    if (result != CAELIX_OK)
    {
        // OUT_OF_MEMORY leaves acceleration structure builds pending but records everything else.
        Fail(result, "caelix_scene_flush failed");
        if (result != CAELIX_ERROR_OUT_OF_MEMORY) return;
    }
    if (!color_ok)
    {
        PublishSceneStats();
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
    // The scene records are applied and flushed above whatever the view is; only the render needs
    // the view's renderer.
    if (packet.view >= CAELIX_UNITY_MAX_VIEWS)
    {
        char what[96];
        std::snprintf(what, sizeof(what), "packet view %u out of range (max %u)", packet.view,
                      CAELIX_UNITY_MAX_VIEWS - 1);
        Fail(CAELIX_ERROR_INVALID_ARGUMENT, what);
        PublishSceneStats();
        return;
    }
    if (!EnsureRenderer(packet.view))
    {
        PublishSceneStats();
        return;
    }
    result = caelix_renderer_render(state.renderers[packet.view], &frame_desc, &targets,
                                    &packet.camera, &packet.options);
    PublishSceneStats();
    if (result != CAELIX_OK)
    {
        Fail(result, "caelix_renderer_render failed");
        return;
    }
    state.renders.fetch_add(1);
    if (g_rt.access_logs < 4)
    {
        ++g_rt.access_logs;
        Log(LogLevel::Info, "render %llu: AccessTexture reported layout %d (VkImage %p)", frame,
            static_cast<int>(color.layout), static_cast<void*>(color.image));
    }
    if (!g_rt.logged_first_render)
    {
        g_rt.logged_first_render = true;
        Log(LogLevel::Info, "first render: %ux%u, VkFormat %d, layout at access %d, plugin frame "
                            "%llu, Unity frame %llu (safe %llu), mode %u",
            color.extent.width, color.extent.height,
            static_cast<int>(color.format), static_cast<int>(color.layout), frame,
            recording.currentFrameNumber, recording.safeFrameNumber, packet.options.mode);
    }
}

// Runs with graphics queue access (ConfigureEvent below): Unity has submitted its command buffers
// and keeps its worker threads off the queue, so waiting for the queue to be idle is legal here.
// `data` is NULL or the packet of the following render event.
void ReleaseEvent(void* data)
{
    State& state = GetState();
    bool recreate = false;
    uint32_t encoding = g_rt.scene_encoding;
    if (data != nullptr && state.scene != nullptr)
    {
        CaelixUnityPacket packet;
        if (ReadPacketHeader(data, packet) && packet.encoding != g_rt.scene_encoding)
        {
            recreate = true;
            encoding = packet.encoding;
        }
    }
    if (!HasRenderer() && !recreate)
    {
        g_rt.released_since_generation = true;
        return;
    }
    if (state.queue_wait_idle != nullptr)
    {
        state.queue_wait_idle(state.vk_queue);
    }
    if (recreate)
    {
        // The device is idle for this queue: renderers and scene can go now. The render events that
        // follow create them again with the packet's encoding.
        Log(LogLevel::Info, "encoding change %u -> %u: destroying renderers and scene (%zu entities)",
            g_rt.scene_encoding, encoding, g_rt.entities.size());
        for (CaelixRenderer*& renderer : state.renderers)
        {
            caelix_renderer_destroy(renderer);
            renderer = nullptr;
        }
        caelix_scene_destroy(state.scene);
        state.scene = nullptr;
        g_rt.scene_encoding = kNoScene;
        state.scene_encoding.store(kNoScene);
        ForgetSceneContents();
        state.scene_recreations.fetch_add(1);
    }
    else
    {
        for (CaelixRenderer* renderer : state.renderers)
        {
            if (renderer != nullptr) caelix_renderer_release_target_views(renderer);
        }
    }
    g_rt.released_since_generation = true;
    state.view_releases.fetch_add(1);
}

}  // namespace

void UNITY_INTERFACE_API OnPluginEvent(int event_id, void* data)
{
    const State& state = GetState();
    if (event_id == state.render_event_id)
    {
        RenderEvent(data);
    }
    else if (event_id == state.release_event_id)
    {
        ReleaseEvent(data);
    }
}

void ConfigureEvents()
{
    State& state = GetState();
    UnityVulkanPluginEventConfig render{};
    render.renderPassPrecondition = kUnityVulkanRenderPass_EnsureOutside;
    render.graphicsQueueAccess = kUnityVulkanGraphicsQueueAccess_DontCare;
    // We bind compute pipelines and descriptor sets: Unity must not assume its bindings survive.
    render.flags = kUnityVulkanEventConfigFlag_EnsurePreviousFrameSubmission |
                   kUnityVulkanEventConfigFlag_ModifiesCommandBuffersState;
    state.vulkan->ConfigureEvent(state.render_event_id, &render);

    UnityVulkanPluginEventConfig release{};
    release.renderPassPrecondition = kUnityVulkanRenderPass_EnsureOutside;
    release.graphicsQueueAccess = kUnityVulkanGraphicsQueueAccess_Allow;
    release.flags = kUnityVulkanEventConfigFlag_EnsurePreviousFrameSubmission |
                    kUnityVulkanEventConfigFlag_FlushCommandBuffers |
                    kUnityVulkanEventConfigFlag_SyncWorkerThreads;
    state.vulkan->ConfigureEvent(state.release_event_id, &release);
    Log(LogLevel::Info, "plugin events configured: render %d (outside render pass, no queue access), "
                        "release %d (queue access, flush)",
        state.render_event_id, state.release_event_id);
}

void DestroyRenderObjects()
{
    State& state = GetState();
    for (CaelixRenderer*& renderer : state.renderers)
    {
        caelix_renderer_destroy(renderer);
        renderer = nullptr;
    }
    caelix_scene_destroy(state.scene);
    state.scene = nullptr;
    state.scene_encoding.store(kNoScene);
    g_rt = RenderThreadState{};
    state.bricks_pending_plugin.store(0);
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
    const caelix_unity::State& state = caelix_unity::GetState();
    CaelixUnityStatus status{};
    status.size = sizeof(CaelixUnityStatus);
    status.packet_size = sizeof(CaelixUnityPacket);
    status.packet_version = CAELIX_UNITY_PACKET_VERSION;
    status.unity_renderer = state.unity_renderer.load();
    status.intercept_registered = state.intercept_registered.load();
    status.intercept_called = state.intercept_called.load();
    status.instance_patched = state.instance_patched.load();
    status.device_patched = state.device_patched.load();
    status.device_adopted = state.device_adopted.load();
    status.last_error = state.last_error.load();
    status.render_event_id = state.render_event_id;
    status.release_event_id = state.release_event_id;
    status.last_consumed_sequence = state.last_consumed_sequence.load();
    status.last_unity_frame = state.last_unity_frame.load();
    status.last_safe_frame = state.last_safe_frame.load();
    status.last_plugin_frame = state.last_plugin_frame.load();
    status.renders = state.renders.load();
    status.failed_events = state.failed_events.load();
    status.view_releases = state.view_releases.load();
    status.scene_encoding = state.scene_encoding.load();
    status.entities = state.scene_entities.load();
    status.groups = state.scene_groups.load();
    status.scene_recreations = state.scene_recreations.load();
    status.bricks_resident = state.bricks_resident.load();
    status.bricks_pending_plugin = state.bricks_pending_plugin.load();
    status.as_builds_pending = state.as_builds_pending.load();
    status.uploaded_total = state.uploaded_total.load();
    status.removed_total = state.removed_total.load();
    status.scene_packets = state.scene_packets.load();
    status.record_bytes = state.record_bytes.load();
    status.blas_bytes = state.blas_bytes.load();
    status.flush_ms = state.flush_ms.load();
    status.trace_ms = state.trace_ms.load();
    status.composite_ms = state.composite_ms.load();
    status.denoise_ms = state.denoise_ms.load();
    status.timings_frame = state.timings_frame.load();
    *out = status;
}
