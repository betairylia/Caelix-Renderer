// The plugin's C API for C# and the render-thread work of the plugin events.
//
// Frame numbers: the library needs one strictly increasing frame number per flush, and a safe
// frame number below it. Unity's currentFrameNumber is not strictly increasing per event (two
// cameras render in one Unity frame), so the plugin counts its own frames: every render event is
// plugin frame n + 1, and the plugin remembers the Unity frame of each one. The safe plugin frame
// is the newest plugin frame whose Unity frame is <= Unity's safeFrameNumber.
#include "plugin_state.h"

#include <caelix_unity.h>

#include <cstring>
#include <deque>
#include <utility>

namespace caelix_unity {
namespace {

// Render thread only.
struct RenderThreadState {
  std::deque<std::pair<uint64_t, uint64_t>> frames;  // (plugin frame, Unity frame), oldest first
  uint64_t plugin_frame = 0;
  uint64_t safe_plugin_frame = 0;
  uint32_t last_generation = 0;
  bool released_since_generation = true;
  bool logged_first_render = false;
  bool logged_access_fallback = false;
  bool logged_level = false;
  bool warned_encoding = false;
  uint32_t scene_encoding = 0;
  uint64_t error_logs = 0;
  uint32_t access_logs = 0;
};
RenderThreadState g_rt;

// Logs the first errors of a kind, then one in 600, so a broken setup does not flood the log.
bool ShouldLogError() {
  ++g_rt.error_logs;
  return g_rt.error_logs <= 10 || g_rt.error_logs % 600 == 0;
}

void Fail(CaelixResult result, const char* what) {
  State& state = GetState();
  state.failed_events.fetch_add(1);
  if (result != CAELIX_OK) {
    state.last_error.store(static_cast<int32_t>(result));
  }
  if (ShouldLogError()) {
    Log(LogLevel::Error, "render event: %s (CaelixResult %d)", what, static_cast<int>(result));
  }
}

bool EnsureRenderObjects(uint32_t encoding) {
  State& state = GetState();
  if (state.scene == nullptr) {
    CaelixSceneDesc desc{};
    desc.encoding = encoding;
    const CaelixResult result = caelix_scene_create(state.device, &desc, &state.scene);
    if (result != CAELIX_OK) {
      state.scene = nullptr;
      Fail(result, "caelix_scene_create failed");
      return false;
    }
    Log(LogLevel::Info, "scene created (encoding %u)", encoding);
    g_rt.scene_encoding = encoding;
  } else if (encoding != g_rt.scene_encoding && !g_rt.warned_encoding) {
    // The scene keeps the encoding of the first packet. P2 recreates it at a safe point.
    g_rt.warned_encoding = true;
    Log(LogLevel::Warning, "packet encoding %u differs from the scene's %u; the scene keeps %u",
        encoding, g_rt.scene_encoding, g_rt.scene_encoding);
  }
  if (state.renderer == nullptr) {
    const CaelixResult result = caelix_renderer_create(state.device, state.scene, &state.renderer);
    if (result != CAELIX_OK) {
      state.renderer = nullptr;
      Fail(result, "caelix_renderer_create failed");
      return false;
    }
    Log(LogLevel::Info, "renderer created");
  }
  return true;
}

// Maps Unity's safe frame to the newest safe plugin frame.
uint64_t SafePluginFrame(uint64_t unity_safe) {
  while (!g_rt.frames.empty() && g_rt.frames.front().second <= unity_safe) {
    g_rt.safe_plugin_frame = g_rt.frames.front().first;
    g_rt.frames.pop_front();
  }
  return g_rt.safe_plugin_frame;
}

bool AccessTarget(const CaelixUnityTarget& target, UnityVulkanImage& image) {
  State& state = GetState();
  const VkImageSubresource subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  // PipelineBarrier: Unity records a barrier from its tracked layout and access to GENERAL /
  // compute shader write and then tracks the image as GENERAL with that access. The library
  // leaves the image in GENERAL with its writes available to all later commands, so Unity's
  // tracked state stays true and its next use of the texture transitions from it correctly.
  if (target.native_texture != nullptr &&
      state.vulkan->AccessTexture(target.native_texture, &subresource, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                  kUnityVulkanResourceAccess_PipelineBarrier, &image)) {
    return true;
  }
  if (target.native_render_buffer != nullptr &&
      state.vulkan->AccessRenderBufferTexture(
          static_cast<UnityRenderBuffer>(target.native_render_buffer), &subresource,
          VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
          kUnityVulkanResourceAccess_PipelineBarrier, &image)) {
    if (!g_rt.logged_access_fallback) {
      g_rt.logged_access_fallback = true;
      Log(LogLevel::Warning, "AccessTexture refused the native texture; AccessRenderBufferTexture "
          "accepted the render buffer");
    }
    return true;
  }
  return false;
}

void RenderEvent(void* data) {
  State& state = GetState();
  if (data == nullptr) {
    Fail(CAELIX_ERROR_INVALID_ARGUMENT, "no packet");
    return;
  }
  // Copy first: C# may reuse the packet as soon as the sequence is published.
  CaelixUnityPacket packet;
  std::memcpy(&packet, data, sizeof(packet));
  if (packet.size != sizeof(CaelixUnityPacket) || packet.version != CAELIX_UNITY_PACKET_VERSION) {
    Fail(CAELIX_ERROR_INVALID_ARGUMENT, "packet size or version mismatch (C# and plugin differ)");
    return;
  }
  uint64_t consumed = state.last_consumed_sequence.load();
  while (packet.sequence > consumed &&
         !state.last_consumed_sequence.compare_exchange_weak(consumed, packet.sequence)) {
  }

  if (state.device == nullptr) {
    Fail(CAELIX_OK, "no adopted device (see the caelix_device_adopt log)");
    return;
  }
  if (packet.target_generation != g_rt.last_generation) {
    if (!g_rt.released_since_generation && ShouldLogError()) {
      Log(LogLevel::Warning, "target generation changed (%u -> %u) without a release event: a "
          "cached image view may refer to a destroyed image", g_rt.last_generation,
          packet.target_generation);
    }
    g_rt.last_generation = packet.target_generation;
  }
  g_rt.released_since_generation = false;

  if (!EnsureRenderObjects(packet.encoding)) {
    return;
  }

  // Access calls may record commands and invalidate the recording state: all of them first.
  UnityVulkanImage color{};
  if (!AccessTarget(packet.color, color)) {
    Fail(CAELIX_ERROR_INVALID_ARGUMENT, "AccessTexture failed for the colour target");
    return;
  }
  if ((color.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0) {
    Fail(CAELIX_ERROR_INVALID_ARGUMENT,
         "the colour target has no STORAGE usage (RenderTexture.enableRandomWrite is off)");
    return;
  }
  UnityVulkanRecordingState recording{};
  if (!state.vulkan->CommandRecordingState(&recording, kUnityVulkanGraphicsQueueAccess_DontCare)) {
    Fail(CAELIX_OK, "CommandRecordingState failed");
    return;
  }
  if (!g_rt.logged_level) {
    g_rt.logged_level = true;
    Log(LogLevel::Info, "render event: command buffer %p level %s, render pass %p, framebuffer %p, "
        "subpass %d, Unity frame %llu, safe %llu", static_cast<void*>(recording.commandBuffer),
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

  CaelixResult result = caelix_scene_recycle(state.scene, safe);
  if (result != CAELIX_OK) {
    Fail(result, "caelix_scene_recycle failed");
    return;
  }
  result = caelix_scene_flush(state.scene, &frame_desc);
  if (result != CAELIX_OK) {
    Fail(result, "caelix_scene_flush failed");
    return;
  }

  CaelixRenderTargets targets{};
  targets.color.image = reinterpret_cast<uint64_t>(color.image);
  targets.color.format = static_cast<uint32_t>(color.format);
  targets.color.width = color.extent.width;
  targets.color.height = color.extent.height;
  targets.color.layout = static_cast<uint32_t>(color.layout);
  if (color.extent.width != packet.color.width || color.extent.height != packet.color.height) {
    if (ShouldLogError()) {
      Log(LogLevel::Warning, "colour target is %ux%u, the packet says %ux%u; using the image size",
          color.extent.width, color.extent.height, packet.color.width, packet.color.height);
    }
  }
  result = caelix_renderer_render(state.renderer, &frame_desc, &targets, &packet.camera,
                                  &packet.options);
  if (result != CAELIX_OK) {
    Fail(result, "caelix_renderer_render failed");
    return;
  }
  state.renders.fetch_add(1);
  if (g_rt.access_logs < 4) {
    ++g_rt.access_logs;
    Log(LogLevel::Info, "render %llu: AccessTexture reported layout %d (VkImage %p)", frame,
        static_cast<int>(color.layout), static_cast<void*>(color.image));
  }
  if (!g_rt.logged_first_render) {
    g_rt.logged_first_render = true;
    Log(LogLevel::Info, "first render: %ux%u, VkFormat %d, layout at access %d, plugin frame "
        "%llu, Unity frame %llu (safe %llu), mode %u", color.extent.width, color.extent.height,
        static_cast<int>(color.format), static_cast<int>(color.layout), frame,
        recording.currentFrameNumber, recording.safeFrameNumber, packet.options.mode);
  }
}

// Runs with graphics queue access (ConfigureEvent below): Unity has submitted its command buffers
// and keeps its worker threads off the queue, so waiting for the queue to be idle is legal here.
void ReleaseEvent() {
  State& state = GetState();
  if (state.renderer == nullptr) {
    g_rt.released_since_generation = true;
    return;
  }
  if (state.queue_wait_idle != nullptr) {
    state.queue_wait_idle(state.vk_queue);
  }
  caelix_renderer_release_target_views(state.renderer);
  g_rt.released_since_generation = true;
  state.view_releases.fetch_add(1);
}

}  // namespace

void UNITY_INTERFACE_API OnPluginEvent(int event_id, void* data) {
  const State& state = GetState();
  if (event_id == state.render_event_id) {
    RenderEvent(data);
  } else if (event_id == state.release_event_id) {
    ReleaseEvent();
  }
}

void ConfigureEvents() {
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
      "release %d (queue access, flush)", state.render_event_id, state.release_event_id);
}

void DestroyRenderObjects() {
  State& state = GetState();
  caelix_renderer_destroy(state.renderer);
  state.renderer = nullptr;
  caelix_scene_destroy(state.scene);
  state.scene = nullptr;
  g_rt = RenderThreadState{};
}

}  // namespace caelix_unity

extern "C" CAELIX_UNITY_API void* caelix_unity_get_render_event_func(void) {
  return reinterpret_cast<void*>(&caelix_unity::OnPluginEvent);
}

extern "C" CAELIX_UNITY_API void caelix_unity_status(CaelixUnityStatus* out) {
  if (out == nullptr) {
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
  *out = status;
}
