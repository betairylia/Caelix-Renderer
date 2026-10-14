// The NRD settings glue of the renderer: executor lifetime, per-frame
// CommonSettings and ReblurSettings, and NRD info. NrdExecutor itself is in
// nrd_executor.cpp.
#include "render/renderer.h"

#include "core/log.h"
#include "render/renderer_internal.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <span>

namespace caelix {
namespace {

// CommonSettings::denoisingRange: pixels with a larger view z are not denoised.
// The worlds are at most 8192 units across (the 8K citadel), so 1e6 keeps every
// hit in range, and the miss view z (1e30) far out of it.
constexpr float kDenoisingRange = 1e6f;

// NRD's world-to-view matrix of `camera`: column-major, column vectors
// (view = M * world), rows right, up, forward (the camera's view space, +z forward,
// left-handed like the projection), translation -dot(axis, position).
void WorldToView(const CaelixCamera& camera, float m[16]) {
  const float* axes[3] = {camera.right, camera.up, camera.forward};
  for (uint32_t row = 0; row < 3; ++row) {
    const float* axis = axes[row];
    m[0 * 4 + row] = axis[0];
    m[1 * 4 + row] = axis[1];
    m[2 * 4 + row] = axis[2];
    m[3 * 4 + row] = -(axis[0] * camera.position[0] + axis[1] * camera.position[1] +
                       axis[2] * camera.position[2]);
  }
  m[3] = 0.0f;
  m[7] = 0.0f;
  m[11] = 0.0f;
  m[15] = 1.0f;
}

}  // namespace

NrdExecutor* Renderer::EnsureDenoiser(uint32_t width, uint32_t height, bool& error) {
  if (nrd_ != nullptr && nrd_->width() == width && nrd_->height() == height) {
    return nrd_.get();
  }
  if (nrd_ != nullptr) {
    retired_nrd_.push_back(RetiredDenoiser{nrd_last_frame_, std::move(nrd_)});
  }
  const nrd::DenoiserDesc denoiser{kReblurIdentifier, nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR};
  try {
    nrd_ = std::make_unique<NrdExecutor>(device_, width, height, std::span(&denoiser, 1));
  } catch (const NrdError& e) {
    CAELIX_LOGE("caelix_renderer_render: NRD: %s; denoising is disabled for this renderer",
                e.what());
    nrd_failed_ = true;
    error = true;
    return nullptr;
  }
  nrd_clear_ = true;
  nrd_history_valid_ = false;
  nrd_frame_index_ = 0;
  nrd_last_time_ = {};
  return nrd_.get();
}

bool Renderer::ConfigureDenoiser(NrdExecutor& executor, const CaelixCamera& camera,
                                 const CaelixRenderOptions& options, uint32_t width,
                                 uint32_t height, bool restart) {
  const CaelixCamera& prev = has_prev_camera_ ? prev_camera_ : camera;
  nrd::CommonSettings common{};
  WorldToView(camera, common.worldToViewMatrix);
  WorldToView(prev, common.worldToViewMatrixPrev);
  std::memcpy(common.viewToClipMatrix, camera.projection, sizeof(common.viewToClipMatrix));
  std::memcpy(common.viewToClipMatrixPrev, prev.projection, sizeof(common.viewToClipMatrixPrev));
  // The trace writes motion in pixels (xy) and viewZprev - viewZ (z).
  common.motionVectorScale[0] = 1.0f / static_cast<float>(width);
  common.motionVectorScale[1] = 1.0f / static_cast<float>(height);
  common.motionVectorScale[2] = 1.0f;
  common.isMotionVectorInWorldSpace = false;
  common.cameraJitter[0] = common.cameraJitter[1] = 0.0f;
  common.cameraJitterPrev[0] = common.cameraJitterPrev[1] = 0.0f;
  common.resourceSize[0] = common.resourceSizePrev[0] = common.rectSize[0] =
      common.rectSizePrev[0] = static_cast<uint16_t>(width);
  common.resourceSize[1] = common.resourceSizePrev[1] = common.rectSize[1] =
      common.rectSizePrev[1] = static_cast<uint16_t>(height);
  common.denoisingRange = kDenoisingRange;
  common.disocclusionThreshold = 0.01f;
  common.frameIndex = nrd_frame_index_;
  common.accumulationMode = nrd_clear_ ? nrd::AccumulationMode::CLEAR_AND_RESTART
                            : restart  ? nrd::AccumulationMode::RESTART
                                       : nrd::AccumulationMode::CONTINUE;
  common.enableValidation = options.debug_view == CAELIX_VIEW_NRD_VALIDATION;
  const auto now = std::chrono::steady_clock::now();
  if (nrd_last_time_ != std::chrono::steady_clock::time_point{}) {
    common.timeDeltaBetweenFrames =
        std::chrono::duration<float, std::milli>(now - nrd_last_time_).count();
  }  // else 0: NRD measures the time itself
  nrd_last_time_ = now;
  if (!executor.SetCommonSettings(common)) {
    return false;
  }
  nrd::ReblurSettings reblur{};
  reblur.hitDistanceParameters = HitDistanceParameters();
  reblur.enableAntiFirefly = true;
  reblur.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
  reblur.maxAccumulatedFrameNum = 30;
  if (!executor.SetReblurSettings(kReblurIdentifier, reblur)) {
    return false;
  }
  nrd_clear_ = false;
  ++nrd_frame_index_;
  return true;
}

void Renderer::GetNrdInfo(CaelixNrdInfo& out) const {
  out = CaelixNrdInfo{};
  const nrd::LibraryDesc& library = NrdExecutor::lib();
  out.version_major = library.versionMajor;
  out.version_minor = library.versionMinor;
  out.version_build = library.versionBuild;
  out.failed = nrd_failed_ ? 1u : 0u;
  if (nrd_ == nullptr) {
    return;
  }
  out.width = nrd_->width();
  out.height = nrd_->height();
  out.pipeline_count = nrd_->desc().pipelinesNum;
  out.permanent_texture_count = nrd_->permanent_count();
  out.transient_texture_count = nrd_->transient_count();
  out.permanent_bytes = nrd_->permanent_bytes();
  out.transient_bytes = nrd_->transient_bytes();
  out.dispatch_count = nrd_->last_dispatch_count();
}

}  // namespace caelix
