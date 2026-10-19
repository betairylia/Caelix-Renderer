// NVIDIA DLSS through the NGX SDK (docs/DLSS.md): the NGX runtime of a device, one DLSS feature
// (Super Resolution or Ray Reconstruction) and the Vulkan extension queries. The whole file is
// empty without CAELIX_DLSS.
#include "render/dlss_upscaler.h"

#if CAELIX_DLSS

#include "core/log.h"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
// NGX_DLSSD_GET_OPTIMAL_SETTINGS lives in the D3D flavour of the RR helpers; it only reads the
// parameter map (no D3D types).
#include <nvsdk_ngx_helpers_dlssd_d3d.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace caelix {
namespace {

// The NGX project id of the Caelix renderer (engine type CUSTOM). NGX requires a GUID-like string
// for applications without an NVIDIA application id; this one was generated once for the library
// (2026-10-20). Keep it stable: NGX keys its per-application settings and updates by it.
constexpr const char* kProjectId = "d8ec5ae2-3625-4c4a-ba69-224a61105aad";

const VkImageSubresourceRange kColorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

const char* ResultName(NVSDK_NGX_Result result)
{
    switch (result)
    {
        case NVSDK_NGX_Result_Success: return "Success";
        case NVSDK_NGX_Result_Fail: return "Fail";
        case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "FAIL_FeatureNotSupported";
        case NVSDK_NGX_Result_FAIL_PlatformError: return "FAIL_PlatformError";
        case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists: return "FAIL_FeatureAlreadyExists";
        case NVSDK_NGX_Result_FAIL_FeatureNotFound: return "FAIL_FeatureNotFound";
        case NVSDK_NGX_Result_FAIL_InvalidParameter: return "FAIL_InvalidParameter";
        case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall: return "FAIL_ScratchBufferTooSmall";
        case NVSDK_NGX_Result_FAIL_NotInitialized: return "FAIL_NotInitialized";
        case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat: return "FAIL_UnsupportedInputFormat";
        case NVSDK_NGX_Result_FAIL_RWFlagMissing: return "FAIL_RWFlagMissing";
        case NVSDK_NGX_Result_FAIL_MissingInput: return "FAIL_MissingInput";
        case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature: return "FAIL_UnableToInitializeFeature";
        case NVSDK_NGX_Result_FAIL_OutOfDate: return "FAIL_OutOfDate";
        case NVSDK_NGX_Result_FAIL_OutOfGPUMemory: return "FAIL_OutOfGPUMemory";
        case NVSDK_NGX_Result_FAIL_UnsupportedFormat: return "FAIL_UnsupportedFormat";
        case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath: return "FAIL_UnableToWriteToAppDataPath";
        case NVSDK_NGX_Result_FAIL_UnsupportedParameter: return "FAIL_UnsupportedParameter";
        case NVSDK_NGX_Result_FAIL_Denied: return "FAIL_Denied";
        case NVSDK_NGX_Result_FAIL_NotImplemented: return "FAIL_NotImplemented";
        default: return "unknown NVSDK_NGX_Result";
    }
}

std::string Narrow(const std::wstring& wide)
{
    if (wide.empty())
    {
        return {};
    }
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), nullptr, 0,
                                         nullptr, nullptr);
    std::string out(static_cast<size_t>(std::max(size, 0)), '\0');
    if (size > 0)
    {
        WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), out.data(), size, nullptr,
                            nullptr);
    }
    return out;
}

// NGX log lines -> CAELIX_LOG*. NGX's levels (ON, VERBOSE) say how much is logged, not how bad a
// line is: a line that NGX marks as an error or a warning keeps that level, every other line is
// debug output. Called from any thread (the library's logger is thread-safe).
void NVSDK_CONV NgxLogCallback(const char* message, NVSDK_NGX_Logging_Level /*level*/,
                               NVSDK_NGX_Feature /*component*/)
{
    if (message == nullptr)
    {
        return;
    }
    std::string line(message);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
    {
        line.pop_back();
    }
    std::string lower(line);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    CaelixLogLevel caelix_level = CAELIX_LOG_DEBUG;
    if (lower.find("[error]") != std::string::npos || lower.find("error:") != std::string::npos)
    {
        caelix_level = CAELIX_LOG_ERROR;
    }
    else if (lower.find("[warning]") != std::string::npos || lower.find("warning:") != std::string::npos)
    {
        caelix_level = CAELIX_LOG_WARNING;
    }
    ::caelix::log(caelix_level, "NGX: %s", line.c_str());
}

// Paths and common info shared by NGX's init and its discovery queries; built once, never moved
// (NVSDK_NGX_FeatureCommonInfo points into it).
struct NgxEnvironment
{
    std::wstring app_data_path;  // %TEMP%\caelix-ngx: NGX's own log file and temporary files
    std::wstring dll_dir;        // the directory of caelix_renderer.dll: the feature DLLs
    const wchar_t* paths[1] = {nullptr};
    NVSDK_NGX_FeatureCommonInfo common{};
};

const NgxEnvironment& Environment()
{
    static NgxEnvironment env;
    static std::once_flag once;
    std::call_once(once, [] {
        wchar_t temp[MAX_PATH + 1] = {};
        const DWORD temp_length = GetTempPathW(MAX_PATH + 1, temp);
        env.app_data_path = std::wstring(temp, temp_length) + L"caelix-ngx";
        if (!CreateDirectoryW(env.app_data_path.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            CAELIX_LOGW("NGX: cannot create the application data directory %s",
                        Narrow(env.app_data_path).c_str());
        }
        // The module that contains this code: caelix_renderer.dll.
        static const char anchor = 0;
        HMODULE module = nullptr;
        std::vector<wchar_t> path(32768, L'\0');
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(&anchor), &module))
        {
            const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
            const std::wstring file(path.data(), length);
            const size_t slash = file.find_last_of(L"\\/");
            env.dll_dir = slash == std::wstring::npos ? std::wstring(L".") : file.substr(0, slash);
        }
        env.paths[0] = env.dll_dir.c_str();
        env.common.PathListInfo.Path = env.paths;
        env.common.PathListInfo.Length = env.dll_dir.empty() ? 0u : 1u;
        env.common.InternalData = nullptr;
        env.common.LoggingInfo.LoggingCallback = &NgxLogCallback;
        env.common.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
        // Only our callback: NGX would otherwise print every line to stdout and write log files.
        env.common.LoggingInfo.DisableOtherLoggingSinks = true;
        CAELIX_LOGI("NGX: feature DLL directory %s, application data %s", Narrow(env.dll_dir).c_str(),
                    Narrow(env.app_data_path).c_str());
    });
    return env;
}

NVSDK_NGX_FeatureDiscoveryInfo Discovery(NVSDK_NGX_Feature feature)
{
    const NgxEnvironment& env = Environment();
    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = feature;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    info.Identifier.v.ProjectDesc.ProjectId = kProjectId;
    info.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    info.Identifier.v.ProjectDesc.EngineVersion = CAELIX_VERSION_STRING;
    info.ApplicationDataPath = env.app_data_path.c_str();
    info.FeatureInfo = &env.common;
    return info;
}

// CaelixUpscaleQuality -> NGX's execution mode.
NVSDK_NGX_PerfQuality_Value PerfQuality(uint32_t quality)
{
    switch (quality)
    {
        case CAELIX_UPSCALE_DLAA: return NVSDK_NGX_PerfQuality_Value_DLAA;
        case CAELIX_UPSCALE_QUALITY: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case CAELIX_UPSCALE_BALANCED: return NVSDK_NGX_PerfQuality_Value_Balanced;
        case CAELIX_UPSCALE_PERFORMANCE: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        default: return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
    }
}

// The preset hint parameter of `quality` for SR or RR.
const char* PresetParameter(UpscalerKind kind, uint32_t quality)
{
    const bool rr = kind == UpscalerKind::DlssRayReconstruction;
    switch (quality)
    {
        case CAELIX_UPSCALE_DLAA:
            return rr ? NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA
                      : NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA;
        case CAELIX_UPSCALE_QUALITY:
            return rr ? NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality
                      : NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality;
        case CAELIX_UPSCALE_BALANCED:
            return rr ? NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced
                      : NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced;
        case CAELIX_UPSCALE_PERFORMANCE:
            return rr ? NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance
                      : NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance;
        default:
            return rr ? NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance
                      : NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance;
    }
}

const char* KindName(UpscalerKind kind)
{
    return kind == UpscalerKind::DlssRayReconstruction ? "DLSS RR" : "DLSS SR";
}

// An image barrier of mip 0, layer 0.
VkImageMemoryBarrier2 ImageBarrier(VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
                                   VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                   VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = kColorRange;
    return barrier;
}

void RecordBarriers(VkCommandBuffer cmd, const VkMemoryBarrier2* memory, const VkImageMemoryBarrier2* images,
                    uint32_t image_count)
{
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = memory != nullptr ? 1u : 0u;
    dependency.pMemoryBarriers = memory;
    dependency.imageMemoryBarrierCount = image_count;
    dependency.pImageMemoryBarriers = images;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

NVSDK_NGX_Resource_VK Resource(const UpscalerImage& image, bool read_write)
{
    return NVSDK_NGX_Create_ImageView_Resource_VK(image.view, image.image, kColorRange, image.format, image.width,
                                                  image.height, read_write);
}

// Appends the extension names that `names` does not have yet.
void AddNames(std::vector<std::string>& names, const VkExtensionProperties* properties, uint32_t count)
{
    for (uint32_t i = 0; i < count && properties != nullptr; ++i)
    {
        const std::string name(properties[i].extensionName);
        if (std::find(names.begin(), names.end(), name) == names.end())
        {
            names.push_back(name);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// NgxRuntime

NgxRuntime::NgxRuntime(Device& device)
    : device_(device)
{
    const NgxEnvironment& env = Environment();
    // volk's vkGetInstanceProcAddr is the host's (Device::Adopt, volkInitializeCustom).
    const PFN_vkGetInstanceProcAddr gipa = vkGetInstanceProcAddr;
    const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(device.instance(), "vkGetDeviceProcAddr"));
    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, CAELIX_VERSION_STRING, env.app_data_path.c_str(),
        device.instance(), device.physical_device(), device.device(), gipa, gdpa, &env.common);
    if (NVSDK_NGX_FAILED(result))
    {
        CAELIX_LOGE("NGX: NVSDK_NGX_VULKAN_Init_with_ProjectID failed: %s (0x%08X); DLSS is unavailable (it "
                    "needs an NVIDIA GPU and driver and the device extensions of "
                    "caelix_upscaler_device_extensions; NGX log in %s)",
                    ResultName(result), static_cast<unsigned>(result), Narrow(env.app_data_path).c_str());
        throw NgxError("NVSDK_NGX_VULKAN_Init_with_ProjectID failed");
    }
    result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&capabilities_);
    if (NVSDK_NGX_FAILED(result) || capabilities_ == nullptr)
    {
        CAELIX_LOGE("NGX: NVSDK_NGX_VULKAN_GetCapabilityParameters failed: %s (0x%08X)", ResultName(result),
                    static_cast<unsigned>(result));
        capabilities_ = nullptr;
        NVSDK_NGX_VULKAN_Shutdown1(device.device());
        throw NgxError("NVSDK_NGX_VULKAN_GetCapabilityParameters failed");
    }
    CAELIX_LOGI("NGX initialized (SDK API 0x%X, project id %s)", static_cast<unsigned>(NVSDK_NGX_Version_API),
                kProjectId);
}

NgxRuntime::~NgxRuntime()
{
    if (capabilities_ != nullptr)
    {
        NVSDK_NGX_VULKAN_DestroyParameters(capabilities_);
        capabilities_ = nullptr;
    }
    const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Shutdown1(device_.device());
    if (NVSDK_NGX_FAILED(result))
    {
        CAELIX_LOGW("NGX: NVSDK_NGX_VULKAN_Shutdown1 failed: %s (0x%08X)", ResultName(result),
                    static_cast<unsigned>(result));
    }
}

void NgxRuntime::Query(UpscalerKind kind, uint32_t quality, uint32_t output_width, uint32_t output_height,
                       CaelixUpscalerQueryResult& out)
{
    out = CaelixUpscalerQueryResult{};
    const bool rr = kind == UpscalerKind::DlssRayReconstruction;
    const auto unsupported = [&](CaelixUpscalerUnsupportedReason reason, const char* why) {
        out.supported = 0;
        out.reason = reason;
        CAELIX_LOGI("upscaler query: %s at quality %u for %ux%u is unavailable: %s", KindName(kind), quality,
                    output_width, output_height, why);
    };

    // The system check of the feature (adapter, driver, OS).
    const NVSDK_NGX_FeatureDiscoveryInfo discovery =
        Discovery(rr ? NVSDK_NGX_Feature_RayReconstruction : NVSDK_NGX_Feature_SuperSampling);
    NVSDK_NGX_FeatureRequirement requirement{};
    const NVSDK_NGX_Result requirement_result = NVSDK_NGX_VULKAN_GetFeatureRequirements(
        device_.instance(), device_.physical_device(), &discovery, &requirement);
    const unsigned support =
        NVSDK_NGX_SUCCEED(requirement_result) ? static_cast<unsigned>(requirement.FeatureSupported) : 0u;

    // The capability map.
    int needs_driver = 0;
    unsigned int driver_major = 0;
    unsigned int driver_minor = 0;
    int available = 0;
    NVSDK_NGX_Parameter_GetI(capabilities_,
                             rr ? NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver
                                : NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
                             &needs_driver);
    NVSDK_NGX_Parameter_GetUI(capabilities_,
                              rr ? NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor
                                 : NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor,
                              &driver_major);
    NVSDK_NGX_Parameter_GetUI(capabilities_,
                              rr ? NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor
                                 : NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor,
                              &driver_minor);
    const NVSDK_NGX_Result available_result = NVSDK_NGX_Parameter_GetI(
        capabilities_,
        rr ? NVSDK_NGX_Parameter_SuperSamplingDenoising_Available : NVSDK_NGX_Parameter_SuperSampling_Available,
        &available);
    if (needs_driver != 0 || (support & NVSDK_NGX_FeatureSupportResult_DriverVersionUnsupported) != 0)
    {
        out.min_driver_major = driver_major;
        out.min_driver_minor = driver_minor;
        unsupported(CAELIX_UPSCALER_DRIVER_TOO_OLD, "the driver is too old");
        return;
    }
    if (support != 0 || NVSDK_NGX_FAILED(available_result) || available == 0)
    {
        unsupported(CAELIX_UPSCALER_FEATURE_UNAVAILABLE, "the GPU or the NGX runtime lacks the feature");
        return;
    }
    int init_result = 1;
    const NVSDK_NGX_Result init_query =
        NVSDK_NGX_Parameter_GetI(capabilities_,
                                 rr ? NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult
                                    : NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
                                 &init_result);
    if (NVSDK_NGX_SUCCEED(init_query) && init_result == 0)
    {
        unsupported(CAELIX_UPSCALER_FEATURE_UNAVAILABLE, "NGX denies the feature for this application");
        return;
    }

    // The optimal render size of the execution mode.
    unsigned int render_width = 0;
    unsigned int render_height = 0;
    unsigned int max_width = 0;
    unsigned int max_height = 0;
    unsigned int min_width = 0;
    unsigned int min_height = 0;
    float sharpness = 0.0f;
    const NVSDK_NGX_Result optimal =
        rr ? NGX_DLSSD_GET_OPTIMAL_SETTINGS(capabilities_, output_width, output_height, PerfQuality(quality),
                                            &render_width, &render_height, &max_width, &max_height, &min_width,
                                            &min_height, &sharpness)
           : NGX_DLSS_GET_OPTIMAL_SETTINGS(capabilities_, output_width, output_height, PerfQuality(quality),
                                           &render_width, &render_height, &max_width, &max_height, &min_width,
                                           &min_height, &sharpness);
    if (NVSDK_NGX_FAILED(optimal) || render_width == 0 || render_height == 0)
    {
        unsupported(CAELIX_UPSCALER_QUALITY_UNAVAILABLE, "no render size for this quality");
        return;
    }
    out.supported = 1;
    out.reason = CAELIX_UPSCALER_OK;
    out.render_width = render_width;
    out.render_height = render_height;
    out.render_min_width = min_width;
    out.render_min_height = min_height;
    out.render_max_width = max_width;
    out.render_max_height = max_height;
}

// ---------------------------------------------------------------------------
// DlssUpscaler

DlssUpscaler::DlssUpscaler(NgxRuntime& runtime, const UpscalerDesc& desc, VkCommandBuffer cmd)
    : runtime_(runtime), desc_(desc)
{
    Device& device = runtime_.device();
    const bool rr = desc.kind == UpscalerKind::DlssRayReconstruction;
    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_AllocateParameters(&params_);
    if (NVSDK_NGX_FAILED(result) || params_ == nullptr)
    {
        params_ = nullptr;
        CAELIX_LOGE("NGX: NVSDK_NGX_VULKAN_AllocateParameters failed: %s (0x%08X)", ResultName(result),
                    static_cast<unsigned>(result));
        throw NgxError("NVSDK_NGX_VULKAN_AllocateParameters failed");
    }
    if (desc.preset != 0)
    {
        NVSDK_NGX_Parameter_SetUI(params_, PresetParameter(desc.kind, desc.quality), desc.preset);
    }
    int flags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (desc.depth_inverted)
    {
        flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    }
    if (!rr && desc.auto_exposure)
    {
        flags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    }
    if (rr)
    {
        NVSDK_NGX_DLSSD_Create_Params create{};
        create.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
        create.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
        create.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_HW;
        create.InWidth = desc.render_width;
        create.InHeight = desc.render_height;
        create.InTargetWidth = desc.output_width;
        create.InTargetHeight = desc.output_height;
        create.InPerfQualityValue = PerfQuality(desc.quality);
        create.InFeatureCreateFlags = flags;
        create.InEnableOutputSubrects = false;
        result = NGX_VULKAN_CREATE_DLSSD_EXT1(device.device(), cmd, 1, 1, &feature_, params_, &create);
    }
    else
    {
        NVSDK_NGX_DLSS_Create_Params create{};
        create.Feature.InWidth = desc.render_width;
        create.Feature.InHeight = desc.render_height;
        create.Feature.InTargetWidth = desc.output_width;
        create.Feature.InTargetHeight = desc.output_height;
        create.Feature.InPerfQualityValue = PerfQuality(desc.quality);
        create.InFeatureCreateFlags = flags;
        create.InEnableOutputSubrects = false;
        result = NGX_VULKAN_CREATE_DLSS_EXT1(device.device(), cmd, 1, 1, &feature_, params_, &create);
    }
    if (NVSDK_NGX_FAILED(result) || feature_ == nullptr)
    {
        CAELIX_LOGE("NGX: %s feature creation (%ux%u -> %ux%u, quality %u, preset %u, flags 0x%X) failed: %s "
                    "(0x%08X)",
                    KindName(desc.kind), desc.render_width, desc.render_height, desc.output_width,
                    desc.output_height, desc.quality, desc.preset, static_cast<unsigned>(flags), ResultName(result),
                    static_cast<unsigned>(result));
        feature_ = nullptr;
        Release();
        throw NgxError("DLSS feature creation failed");
    }

    // SR with a fixed exposure: the colour is already multiplied by CaelixRenderOptions::exposure,
    // so the exposure DLSS divides by is 1.
    if (!rr && !desc.auto_exposure)
    {
        try
        {
            exposure_ = GpuImage(device.allocator(), device.device(), 1, 1, VK_FORMAT_R32_SFLOAT,
                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                 "dlss_exposure");
        }
        catch (...)
        {
            Release();
            throw;
        }
        const VkImageMemoryBarrier2 to_transfer =
            ImageBarrier(exposure_.image(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT);
        RecordBarriers(cmd, nullptr, &to_transfer, 1);
        VkClearColorValue one{};
        one.float32[0] = 1.0f;
        vkCmdClearColorImage(cmd, exposure_.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &one, 1, &kColorRange);
        const VkImageMemoryBarrier2 to_read = ImageBarrier(
            exposure_.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);
        RecordBarriers(cmd, nullptr, &to_read, 1);
    }
    CAELIX_LOGI("NGX: %s feature created: %ux%u -> %ux%u, quality %u, preset %u, depth %s, exposure %s",
                KindName(desc.kind), desc.render_width, desc.render_height, desc.output_width, desc.output_height,
                desc.quality, desc.preset, desc.depth_inverted ? "inverted" : "normal",
                rr ? "not used (RR)" : (desc.auto_exposure ? "auto" : "fixed 1.0"));
}

DlssUpscaler::~DlssUpscaler()
{
    Release();
}

void DlssUpscaler::Release()
{
    if (feature_ != nullptr)
    {
        const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_ReleaseFeature(feature_);
        if (NVSDK_NGX_FAILED(result))
        {
            CAELIX_LOGW("NGX: NVSDK_NGX_VULKAN_ReleaseFeature failed: %s (0x%08X)", ResultName(result),
                        static_cast<unsigned>(result));
        }
        feature_ = nullptr;
    }
    if (params_ != nullptr)
    {
        NVSDK_NGX_VULKAN_DestroyParameters(params_);
        params_ = nullptr;
    }
    exposure_.Reset();
}

bool DlssUpscaler::Evaluate(VkCommandBuffer cmd, const UpscalerInputs& in)
{
    if (failed_ || feature_ == nullptr)
    {
        return false;
    }
    const bool rr = desc_.kind == UpscalerKind::DlssRayReconstruction;
    const UpscalerImage* inputs[7] = {&in.color, &in.depth, &in.motion};
    uint32_t input_count = 3;
    if (rr)
    {
        inputs[input_count++] = &in.diffuse_albedo;
        inputs[input_count++] = &in.specular_albedo;
        inputs[input_count++] = &in.normal_roughness;
        inputs[input_count++] = &in.specular_hit_distance;
    }

    // The renderer's compute writes -> NGX's reads (a global barrier); every input GENERAL ->
    // SHADER_READ_ONLY_OPTIMAL (NGX samples its inputs); the output stays GENERAL (a storage image),
    // after every earlier compute access of it (the previous frame's encode read).
    VkImageMemoryBarrier2 barriers[8];
    uint32_t barrier_count = 0;
    for (uint32_t i = 0; i < input_count; ++i)
    {
        barriers[barrier_count++] =
            ImageBarrier(inputs[i]->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    }
    barriers[barrier_count++] = ImageBarrier(
        in.output.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    before.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    before.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    before.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    before.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    RecordBarriers(cmd, &before, barriers, barrier_count);

    NVSDK_NGX_Resource_VK color = Resource(in.color, false);
    NVSDK_NGX_Resource_VK depth = Resource(in.depth, false);
    NVSDK_NGX_Resource_VK motion = Resource(in.motion, false);
    NVSDK_NGX_Resource_VK output = Resource(in.output, true);
    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
    if (rr)
    {
        NVSDK_NGX_Resource_VK diffuse = Resource(in.diffuse_albedo, false);
        NVSDK_NGX_Resource_VK specular = Resource(in.specular_albedo, false);
        NVSDK_NGX_Resource_VK normals = Resource(in.normal_roughness, false);
        NVSDK_NGX_Resource_VK hit_distance = Resource(in.specular_hit_distance, false);
        // NGX's "row major, left multiplication" matrices have the memory layout of our
        // column-major, column-vector ones: passed as they are.
        float world_to_view[16];
        float view_to_clip[16];
        std::memcpy(world_to_view, in.world_to_view, sizeof(world_to_view));
        std::memcpy(view_to_clip, in.view_to_clip, sizeof(view_to_clip));
        NVSDK_NGX_VK_DLSSD_Eval_Params eval{};
        eval.pInDiffuseAlbedo = &diffuse;
        eval.pInSpecularAlbedo = &specular;
        eval.pInNormals = &normals;
        eval.pInRoughness = &normals;  // packed in normals.w (InRoughnessMode = Packed)
        eval.pInColor = &color;
        eval.pInOutput = &output;
        eval.pInDepth = &depth;
        eval.pInMotionVectors = &motion;
        eval.InJitterOffsetX = in.jitter[0];
        eval.InJitterOffsetY = in.jitter[1];
        eval.InRenderSubrectDimensions.Width = desc_.render_width;
        eval.InRenderSubrectDimensions.Height = desc_.render_height;
        eval.InReset = in.reset ? 1 : 0;
        eval.InMVScaleX = 1.0f;
        eval.InMVScaleY = 1.0f;
        eval.InPreExposure = 1.0f;
        eval.InExposureScale = 1.0f;
        eval.pInSpecularHitDistance = &hit_distance;
        eval.pInWorldToViewMatrix = world_to_view;
        eval.pInViewToClipMatrix = view_to_clip;
        eval.InFrameTimeDeltaInMsec = in.frame_time_ms;
        result = NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd, feature_, params_, &eval);
    }
    else
    {
        NVSDK_NGX_Resource_VK exposure{};
        NVSDK_NGX_VK_DLSS_Eval_Params eval{};
        eval.Feature.pInColor = &color;
        eval.Feature.pInOutput = &output;
        eval.pInDepth = &depth;
        eval.pInMotionVectors = &motion;
        eval.InJitterOffsetX = in.jitter[0];
        eval.InJitterOffsetY = in.jitter[1];
        eval.InRenderSubrectDimensions.Width = desc_.render_width;
        eval.InRenderSubrectDimensions.Height = desc_.render_height;
        eval.InReset = in.reset ? 1 : 0;
        eval.InMVScaleX = 1.0f;
        eval.InMVScaleY = 1.0f;
        eval.InPreExposure = 1.0f;
        eval.InExposureScale = 1.0f;
        eval.InFrameTimeDeltaInMsec = in.frame_time_ms;
        if (exposure_.valid())
        {
            const UpscalerImage image{exposure_.image(), exposure_.view(), exposure_.format(), 1, 1};
            exposure = Resource(image, false);
            eval.pInExposureTexture = &exposure;
        }
        result = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, feature_, params_, &eval);
    }

    // NGX's writes -> the encode kernel's reads; the inputs back to GENERAL for the next kernels.
    // Recorded also after a failed evaluate (the inputs were transitioned above).
    barrier_count = 0;
    for (uint32_t i = 0; i < input_count; ++i)
    {
        barriers[barrier_count++] =
            ImageBarrier(inputs[i]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    barriers[barrier_count++] = ImageBarrier(
        in.output.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    RecordBarriers(cmd, nullptr, barriers, barrier_count);

    if (NVSDK_NGX_FAILED(result))
    {
        CAELIX_LOGE("NGX: %s evaluate failed: %s (0x%08X); the renderer falls back to the nearest upscale",
                    KindName(desc_.kind), ResultName(result), static_cast<unsigned>(result));
        failed_ = true;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Extension queries

void UpscalerInstanceExtensions(uint32_t& count, const char* const*& names)
{
    struct List
    {
        std::vector<std::string> names;
        std::vector<const char*> pointers;
    };
    static const List list = [] {
        List out;
        for (const NVSDK_NGX_Feature feature :
             {NVSDK_NGX_Feature_SuperSampling, NVSDK_NGX_Feature_RayReconstruction})
        {
            const NVSDK_NGX_FeatureDiscoveryInfo discovery = Discovery(feature);
            uint32_t extension_count = 0;
            VkExtensionProperties* properties = nullptr;
            const NVSDK_NGX_Result result =
                NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&discovery, &extension_count, &properties);
            if (NVSDK_NGX_FAILED(result))
            {
                CAELIX_LOGW("NGX: GetFeatureInstanceExtensionRequirements (feature %d) failed: %s (0x%08X)",
                            static_cast<int>(feature), ResultName(result), static_cast<unsigned>(result));
                continue;
            }
            AddNames(out.names, properties, extension_count);
        }
        for (const std::string& name : out.names)
        {
            out.pointers.push_back(name.c_str());
        }
        return out;
    }();
    count = static_cast<uint32_t>(list.pointers.size());
    names = list.pointers.empty() ? nullptr : list.pointers.data();
}

CaelixResult UpscalerDeviceExtensions(VkInstance instance, VkPhysicalDevice physical_device, uint32_t& count,
                                      const char* const*& names)
{
    static std::mutex mutex;
    static std::vector<std::string> storage;
    static std::vector<const char*> pointers;
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<std::string> fresh;
    CaelixResult status = CAELIX_OK;
    for (const NVSDK_NGX_Feature feature : {NVSDK_NGX_Feature_SuperSampling, NVSDK_NGX_Feature_RayReconstruction})
    {
        const NVSDK_NGX_FeatureDiscoveryInfo discovery = Discovery(feature);
        uint32_t extension_count = 0;
        VkExtensionProperties* properties = nullptr;
        const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            instance, physical_device, &discovery, &extension_count, &properties);
        if (NVSDK_NGX_FAILED(result))
        {
            CAELIX_LOGE("NGX: GetFeatureDeviceExtensionRequirements (feature %d) failed: %s (0x%08X)",
                        static_cast<int>(feature), ResultName(result), static_cast<unsigned>(result));
            status = CAELIX_ERROR_VULKAN;
            continue;
        }
        AddNames(fresh, properties, extension_count);
    }
    if (status != CAELIX_OK)
    {
        fresh.clear();
    }
    storage = std::move(fresh);
    pointers.clear();
    for (const std::string& name : storage)
    {
        pointers.push_back(name.c_str());
    }
    count = static_cast<uint32_t>(pointers.size());
    names = pointers.empty() ? nullptr : pointers.data();
    return status;
}

}  // namespace caelix

#endif  // CAELIX_DLSS
