#pragma once
// NVIDIA DLSS Super Resolution and Ray Reconstruction through the NGX SDK (docs/DLSS.md). Only
// CAELIX_DLSS builds have these classes; include this header from .cpp files only (renderer.h
// forward-declares NgxRuntime and sees the generic Upscaler interface of upscaler.h).
#include "caelix/caelix.h"
#include "render/upscaler.h"

#if CAELIX_DLSS

#include "rhi/device.h"
#include "rhi/gpu_image.h"

#include <cstdint>
#include <stdexcept>

// NGX's opaque handle types (typedef struct X X in nvsdk_ngx_defs.h); the SDK headers stay in
// dlss_upscaler.cpp.
struct NVSDK_NGX_Parameter;
struct NVSDK_NGX_Handle;

namespace caelix {

// An error reported by NGX itself (an NVSDK_NGX_Result failure), as opposed to a Vulkan error
// (VulkanError). The renderer falls back to the nearest upscale after one.
class NgxError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// The NGX SDK instance of one adopted device: NVSDK_NGX_VULKAN_Init_with_ProjectID with the
// library's project id (engine type CUSTOM), the application data path %TEMP%\caelix-ngx (NGX's
// own log file goes there), the directory of caelix_renderer.dll as the feature DLL search path
// (the build copies nvngx_dlss.dll and nvngx_dlssd.dll there) and a log callback into CAELIX_LOG*;
// then the capability parameter map. Created lazily by the CaelixDevice (the first query or
// upscaled render) and destroyed before the RenderKernels and the Device, after every feature is
// released and the device is idle (destructor: the capability map, then
// NVSDK_NGX_VULKAN_Shutdown1). The NGX API is not thread safe; the ABI thread rule serializes
// every call of one device.
class NgxRuntime
{
public:
    // Throws NgxError when NGX does not start (no NVIDIA driver, a DLL or a device extension is
    // missing; the reason is logged).
    explicit NgxRuntime(Device& device);
    ~NgxRuntime();
    NgxRuntime(const NgxRuntime&) = delete;
    NgxRuntime& operator=(const NgxRuntime&) = delete;

    // Availability and the optimal render size of `kind` at `quality` (CaelixUpscaleQuality) for an
    // output_width x output_height output, from NVSDK_NGX_VULKAN_GetFeatureRequirements, the
    // capability map (Available, NeedsUpdatedDriver, MinDriverVersion*, FeatureInitResult) and
    // NGX_DLSS_GET_OPTIMAL_SETTINGS / NGX_DLSSD_GET_OPTIMAL_SETTINGS. The caller has checked the
    // arguments (kind is SR or RR, quality <= ULTRA_PERFORMANCE, output >= 64 x 32).
    void Query(UpscalerKind kind, uint32_t quality, uint32_t output_width, uint32_t output_height,
               CaelixUpscalerQueryResult& out);

    Device& device() const
    {
        return device_;
    }

private:
    Device& device_;
    NVSDK_NGX_Parameter* capabilities_ = nullptr;
};

// One DLSS feature (SR: NGX_VULKAN_CREATE_DLSS_EXT1, RR: NGX_VULKAN_CREATE_DLSSD_EXT1) with its own
// parameter map (NVSDK_NGX_VULKAN_AllocateParameters; the preset hint of the chosen quality mode
// is set on it before creation when desc.preset != 0) and, for SR without auto exposure, a 1 x 1
// R32F exposure image cleared to 1.0 in the creation command buffer (the colour arrives already
// exposed) and kept in SHADER_READ_ONLY_OPTIMAL. Create flags: IsHDR | MVLowRes, DepthInverted
// when desc.depth_inverted, AutoExposure for SR with desc.auto_exposure. RR: denoise mode
// DLUnified, roughness packed in normals.w, hardware depth.
// Destroy it only when no pending GPU work uses it (NVSDK_NGX_VULKAN_ReleaseFeature, the parameter
// map, the image); the NgxRuntime must outlive it.
class DlssUpscaler final : public Upscaler
{
public:
    // Records the feature creation (and the exposure clear) into `cmd`, which must be recording.
    // Throws NgxError when NGX refuses (logged), VulkanError when the exposure image cannot be
    // created.
    DlssUpscaler(NgxRuntime& runtime, const UpscalerDesc& desc, VkCommandBuffer cmd);
    ~DlssUpscaler() override;
    DlssUpscaler(const DlssUpscaler&) = delete;
    DlssUpscaler& operator=(const DlssUpscaler&) = delete;

    const UpscalerDesc& desc() const override
    {
        return desc_;
    }
    bool Evaluate(VkCommandBuffer cmd, const UpscalerInputs& in) override;
    bool failed() const override
    {
        return failed_;
    }

private:
    void Release();

    NgxRuntime& runtime_;
    UpscalerDesc desc_;
    NVSDK_NGX_Parameter* params_ = nullptr;
    NVSDK_NGX_Handle* feature_ = nullptr;
    GpuImage exposure_;  // SR without auto exposure: 1 x 1 R32F = 1.0
    bool failed_ = false;
};

// The Vulkan extensions of DLSS SR and RR (NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements
// / GetFeatureDeviceExtensionRequirements, the union of both features). The instance list is
// computed once; the device list at every call (valid until the next call). The strings are owned
// by the library.
void UpscalerInstanceExtensions(uint32_t& count, const char* const*& names);
CaelixResult UpscalerDeviceExtensions(VkInstance instance, VkPhysicalDevice physical_device,
                                      uint32_t& count, const char* const*& names);

}  // namespace caelix

#endif  // CAELIX_DLSS
