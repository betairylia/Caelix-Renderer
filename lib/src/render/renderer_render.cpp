// Renderer::Render: validation, descriptors, frame constants, barriers and the
// pass sequence (sky SH, trace, NRD, composite, upscaler, encode).
#include "render/renderer.h"

#include "core/log.h"
#include "render/frame_constants.h"
#include "render/renderer_internal.h"
#include "scene/scene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <span>

namespace caelix {
namespace {

// [numthreads(8, 8, 1)] in caelix_path_trace.hlsl, caelix_composite.hlsl and caelix_upscale_encode.hlsl.
constexpr uint32_t kWorkgroupSize = 8;
// The smallest output of an upscaler (DLSS: 64 x 32).
constexpr uint32_t kMinUpscaleOutputWidth = 64;
constexpr uint32_t kMinUpscaleOutputHeight = 32;
constexpr uint32_t kDebugViewCount = 11;  // CAELIX_VIEW_ALBEDO .. CAELIX_VIEW_NRD_SPEC
constexpr uint32_t kModeCount = 3;
constexpr float kDefaultSunAngularRadius = 0.00873f;  // 0.5 degrees

bool FormatAccepted(const TargetSpec& spec, VkFormat format)
{
    if (spec.formats[0] == VK_FORMAT_UNDEFINED)
    {
        return true;  // any storage format
    }
    for (const VkFormat accepted : spec.formats)
    {
        if (accepted != VK_FORMAT_UNDEFINED && accepted == format)
        {
            return true;
        }
    }
    return false;
}

// UNORM colour formats: the composite writes sRGB-encoded values into them.
bool IsUnormFormat(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16B16A16_UNORM:
            return true;
        default:
            return false;
    }
}

// The barriers of one vkCmdPipelineBarrier2: image barriers (any layout -> GENERAL; mip 0,
// `layer_count` layers) and at most one global memory barrier.
class Barriers
{
public:
    void Image(VkImage image, VkImageLayout old_layout, VkPipelineStageFlags2 src_stage,
               VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
               uint32_t layer_count = 1)
    {
        VkImageMemoryBarrier2& barrier = images_[image_count_++];
        barrier = VkImageMemoryBarrier2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = src_stage;
        barrier.srcAccessMask = src_access;
        barrier.dstStageMask = dst_stage;
        barrier.dstAccessMask = dst_access;
        barrier.oldLayout = old_layout;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = kColorRange;
        barrier.subresourceRange.layerCount = layer_count;
    }
    void Memory(VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
    {
        memory_ = VkMemoryBarrier2{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memory_.srcStageMask = src_stage;
        memory_.srcAccessMask = src_access;
        memory_.dstStageMask = dst_stage;
        memory_.dstAccessMask = dst_access;
        has_memory_ = true;
    }
    void Record(VkCommandBuffer cmd) const
    {
        if (image_count_ == 0 && !has_memory_)
        {
            return;
        }
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = has_memory_ ? 1u : 0u;
        dependency.pMemoryBarriers = &memory_;
        dependency.imageMemoryBarrierCount = image_count_;
        dependency.pImageMemoryBarriers = images_;
        vkCmdPipelineBarrier2(cmd, &dependency);
    }

private:
    // The targets, the internal targets, the fallback sky cube and the upscaler's output.
    VkImageMemoryBarrier2 images_[std::size(kTargetSpecs) + std::size(kInternalSpecs) + 2] = {};
    uint32_t image_count_ = 0;
    VkMemoryBarrier2 memory_{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    bool has_memory_ = false;
};

void Copy3(const float in[3], float out[4])
{
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
    out[3] = 0.0f;
}

UpscalerImage ImageOf(const GpuImage& image)
{
    return UpscalerImage{image.image(), image.view(), image.format(), image.width(), image.height()};
}

}  // namespace

CaelixResult Renderer::Render(const CaelixFrameDesc& frame, const CaelixRenderTargets& targets,
                              const CaelixCamera& camera, const CaelixRenderOptions& options)
{
    constexpr const char* kFunction = "caelix_renderer_render";
    if (frame.command_buffer == 0)
    {
        CAELIX_LOGE("%s: the frame's command buffer is null", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (frame.safe_frame_number >= frame.frame_number)
    {
        CAELIX_LOGE("%s: safe_frame_number %llu must be below frame_number %llu", kFunction,
                    static_cast<unsigned long long>(frame.safe_frame_number),
                    static_cast<unsigned long long>(frame.frame_number));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    // One render per renderer per frame: the frame slot, the previous camera, the NRD
    // history and the accumulation all advance once per frame number.
    if (has_rendered_ && frame.frame_number <= last_frame_)
    {
        CAELIX_LOGE("%s: frame_number %llu is not above frame %llu of this renderer's previous render "
                    "(a renderer renders at most once per frame)",
                    kFunction, static_cast<unsigned long long>(frame.frame_number),
                    static_cast<unsigned long long>(last_frame_));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.debug_view >= kDebugViewCount)
    {
        CAELIX_LOGE("%s: unknown debug view %u", kFunction, options.debug_view);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.mode >= kModeCount)
    {
        CAELIX_LOGE("%s: unknown mode %u", kFunction, options.mode);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.bounce_count > CAELIX_MAX_BOUNCES)
    {
        CAELIX_LOGE("%s: bounce_count %u is above the maximum %u", kFunction, options.bounce_count,
                    CAELIX_MAX_BOUNCES);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.bounce_count_transparent > CAELIX_MAX_BOUNCES_TRANSPARENT)
    {
        CAELIX_LOGE("%s: bounce_count_transparent %u is above the maximum %u", kFunction,
                    options.bounce_count_transparent, CAELIX_MAX_BOUNCES_TRANSPARENT);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(options.fog_density) || options.fog_density < 0.0f)
    {
        CAELIX_LOGE("%s: fog_density %g must be finite and >= 0", kFunction,
                    static_cast<double>(options.fog_density));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(options.emission_scale) || options.emission_scale < 0.0f)
    {
        CAELIX_LOGE("%s: emission_scale %g must be finite and >= 0", kFunction,
                    static_cast<double>(options.emission_scale));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.sky.image != 0 &&
        (options.sky.format == VK_FORMAT_UNDEFINED || options.sky.size == 0 ||
         options.sky.mip_levels == 0 || options.sky.layout == VK_IMAGE_LAYOUT_UNDEFINED))
    {
        CAELIX_LOGE("%s: the sky cubemap needs a format, a size, mip levels and a layout (VkFormat %u, "
                    "size %u, mip_levels %u, VkImageLayout %u)",
                    kFunction, options.sky.format, options.sky.size, options.sky.mip_levels,
                    options.sky.layout);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.upscaler > CAELIX_UPSCALER_DLSS_RR)
    {
        CAELIX_LOGE("%s: unknown upscaler %u", kFunction, options.upscaler);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    const bool lighting = options.mode != CAELIX_MODE_PRIMARY;
    // The host's sky cubemap (lighting modes only; the primary mode never samples the sky).
    const bool sky = lighting && options.sky.image != 0;
    // The upscaler (path trace mode with a colour target): the colour target has the output size,
    // every other present target and the internal targets the render size of the query.
    const bool upscaling = options.mode == CAELIX_MODE_PATH_TRACE && targets.color.image != 0 &&
                           options.upscaler != CAELIX_UPSCALER_NONE;
    uint32_t render_width = 0;
    uint32_t render_height = 0;
    if (upscaling)
    {
        if (options.upscale_quality > CAELIX_UPSCALE_ULTRA_PERFORMANCE)
        {
            CAELIX_LOGE("%s: unknown upscale_quality %u", kFunction, options.upscale_quality);
            return CAELIX_ERROR_INVALID_ARGUMENT;
        }
        if (targets.color.width < kMinUpscaleOutputWidth || targets.color.height < kMinUpscaleOutputHeight)
        {
            CAELIX_LOGE("%s: the colour target is %ux%u; an upscaler needs at least %ux%u", kFunction,
                        targets.color.width, targets.color.height, kMinUpscaleOutputWidth, kMinUpscaleOutputHeight);
            return CAELIX_ERROR_INVALID_ARGUMENT;
        }
        const CaelixResult query = QueryUpscaler(options.upscaler, options.upscale_quality, targets.color.width,
                                                 targets.color.height, render_width, render_height);
        if (query != CAELIX_OK)
        {
            return query;
        }
    }
    // DLSS RR replaces NRD and must not see checkerboard rendering (RR guide 3.5).
    const bool ray_reconstruction = upscaling && options.upscaler == CAELIX_UPSCALER_DLSS_RR;
    const uint32_t flags =
        ray_reconstruction ? (options.flags & ~CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD) : options.flags;

    // The present targets (image != 0), in binding order.
    const CaelixTargetImage* const all[kTargetCount] = {&targets.color, &targets.albedo,
                                                        &targets.normal, &targets.depth,
                                                        &targets.motion, &targets.view_z};
    uint32_t present[kTargetCount];  // indices into `all`
    uint32_t present_count = 0;
    uint32_t mask = 0;
    for (uint32_t i = 0; i < kTargetCount; ++i)
    {
        const CaelixTargetImage& target = *all[i];
        const TargetSpec& spec = kTargetSpecs[i];
        if (target.image == 0)
        {
            continue;
        }
        if (target.width == 0 || target.height == 0 || target.format == VK_FORMAT_UNDEFINED)
        {
            CAELIX_LOGE("%s: the %s target needs a format and a nonzero size", kFunction, spec.name);
            return CAELIX_ERROR_INVALID_ARGUMENT;
        }
        if (upscaling)
        {
            const uint32_t expected_width = i == 0 ? targets.color.width : render_width;
            const uint32_t expected_height = i == 0 ? targets.color.height : render_height;
            if (target.width != expected_width || target.height != expected_height)
            {
                CAELIX_LOGE("%s: the %s target is %ux%u; with upscaler %u at quality %u it must be %ux%u (the "
                            "render size of caelix_upscaler_query for the %ux%u colour target)",
                            kFunction, spec.name, target.width, target.height, options.upscaler,
                            options.upscale_quality, expected_width, expected_height, targets.color.width,
                            targets.color.height);
                return CAELIX_ERROR_INVALID_ARGUMENT;
            }
        }
        else if (present_count > 0)
        {
            const CaelixTargetImage& first = *all[present[0]];
            if (target.width != first.width || target.height != first.height)
            {
                CAELIX_LOGE("%s: the %s target is %ux%u, the %s target %ux%u; every target must have the "
                            "same size",
                            kFunction, spec.name, target.width, target.height,
                            kTargetSpecs[present[0]].name, first.width, first.height);
                return CAELIX_ERROR_INVALID_ARGUMENT;
            }
        }
        for (uint32_t j = 0; j < present_count; ++j)
        {
            if (all[present[j]]->image == target.image)
            {
                CAELIX_LOGE("%s: the %s and %s targets are the same image", kFunction,
                            kTargetSpecs[present[j]].name, spec.name);
                return CAELIX_ERROR_INVALID_ARGUMENT;
            }
        }
        if (!FormatAccepted(spec, static_cast<VkFormat>(target.format)))
        {
            CAELIX_LOGE("%s: VkFormat %u is not an accepted format of the %s target (see "
                        "CaelixRenderTargets)",
                        kFunction, target.format, spec.name);
            return CAELIX_ERROR_INVALID_ARGUMENT;
        }
        present[present_count++] = i;
        mask |= spec.bit;
    }
    if (present_count == 0)
    {
        CAELIX_LOGE("%s: no target is present (every target image is 0)", kFunction);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    for (uint32_t p = 0; p < present_count; ++p)
    {
        if (!FormatSupported(static_cast<VkFormat>(all[present[p]]->format)))
        {
            return CAELIX_ERROR_UNSUPPORTED;
        }
    }
    if (targets.normal.image != 0 && targets.normal.format == VK_FORMAT_R8G8B8A8_UNORM)
    {
        mask |= kTargetNormalUnorm;
    }
    // The render size (the size of the trace, the internal targets and every geometry target).
    const uint32_t width = upscaling ? render_width : all[present[0]]->width;
    const uint32_t height = upscaling ? render_height : all[present[0]]->height;
    const bool has_color = targets.color.image != 0;
    const uint32_t output_width = has_color ? targets.color.width : width;
    const uint32_t output_height = has_color ? targets.color.height : height;
    const bool composite = lighting && has_color;
    // The delta checkerboard (path trace mode only) needs the cross resolve after the composite.
    const bool cross_resolve = composite && options.mode == CAELIX_MODE_PATH_TRACE &&
                               (flags & CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD) != 0;

    const VkCommandBuffer cmd = HandleFromU64<VkCommandBuffer>(frame.command_buffer);
    const uint64_t safe = frame.safe_frame_number;
    timer_.Resolve(safe);
    composite_timer_.Resolve(safe);
    denoise_timer_.Resolve(safe);
    upscale_timer_.Resolve(safe);
    FreeRetired(safe);

    // Internal targets of the lighting modes and the NRD executor (before anything is
    // recorded: the allocations can throw).
    bool restart_accumulation = options.reset_accumulation != 0 || options.mode != last_mode_;
    if (lighting && EnsureInternalTargets(width, height, frame.frame_number))
    {
        restart_accumulation = true;
    }
    // Light NEE (the lighting modes with a light tree; 0 = it does not run this frame).
    const uint32_t light_nee_flags = LightNeeFlags(lighting);
    bool nrd_error = false;
    NrdExecutor* denoiser = nullptr;
    if (options.mode == CAELIX_MODE_PATH_TRACE && options.denoise != 0 && composite && !nrd_failed_ &&
        !ray_reconstruction)
    {
        denoiser = EnsureDenoiser(width, height, nrd_error);
    }
    else if (ray_reconstruction && nrd_ != nullptr)
    {
        // RR replaces NRD: the instance (about 160 MiB at 2048x1152) is released once its last
        // frame is safe; a later SR frame creates a new one (its history is stale anyway).
        RetireDenoiser();
    }

    FrameSlot* slot = AcquireSlot(frame.frame_number, safe);
    if (slot == nullptr)
    {
        CAELIX_LOGE("%s: more than %u frames in flight (safe frame %llu, frame %llu)", kFunction,
                    kMaxSlots, static_cast<unsigned long long>(safe),
                    static_cast<unsigned long long>(frame.frame_number));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }

    // The upscaler of these options (created at its first frame: the NGX feature creation is
    // recorded into this command buffer; an NGX error here: the fallback for this desc).
    bool upscale_error = false;
    Upscaler* upscaler = nullptr;
    if (upscaling)
    {
        UpscalerDesc desc;
        desc.kind = static_cast<UpscalerKind>(options.upscaler);
        desc.quality = options.upscale_quality;
        desc.preset = options.upscale_preset;
        desc.render_width = width;
        desc.render_height = height;
        desc.output_width = output_width;
        desc.output_height = output_height;
        desc.depth_inverted = ProjectionDepthInverted(camera);
        desc.auto_exposure =
            !ray_reconstruction && (options.upscale_flags & CAELIX_UPSCALE_FLAG_AUTO_EXPOSURE) != 0;
        upscaler = EnsureUpscaler(desc, cmd, frame.frame_number, upscale_error);
    }

    // NRD settings of this frame (an NRD error here: no denoising, raw output).
    if (denoiser != nullptr)
    {
        const bool restart = options.reset_accumulation != 0 || !nrd_history_valid_;
        if (!ConfigureDenoiser(*denoiser, camera, options, width, height, restart))
        {
            nrd_failed_ = true;
            nrd_error = true;
            denoiser = nullptr;
        }
    }

    // Descriptors: the slot's previous frame has completed, so its sets are free.
    // Only the TLAS (when the scene has one), the present targets and, in the
    // lighting modes, the internal targets are written; the other bindings keep
    // stale or no descriptors, which PARTIALLY_BOUND allows because the kernels do
    // not touch them (has_tlas, target_mask, mode). The composite set is written
    // after the NRD dispatches are recorded: it reads NRD's outputs only when they
    // were.
    const VkAccelerationStructureKHR tlas = scene_.tlas();
    VkWriteDescriptorSetAccelerationStructureKHR as_write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    as_write.accelerationStructureCount = 1;
    as_write.pAccelerationStructures = &tlas;
    // The trace set (TLAS, targets, internal targets, sky image + sampler) and the sky
    // set (sky image + sampler) in one update; the composite set in a second one.
    constexpr uint32_t kMaxWrites = kTraceBindingCount + 2;
    static_assert(kMaxWrites >= kCompositeBindingCount);
    VkDescriptorImageInfo image_infos[kMaxWrites];
    VkWriteDescriptorSet writes[kMaxWrites];
    uint32_t write_count = 0;
    const auto write_descriptor = [&](VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                                      const VkDescriptorImageInfo& info) {
        image_infos[write_count] = info;
        VkWriteDescriptorSet& write = writes[write_count];
        write = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = type;
        write.pImageInfo = &image_infos[write_count];
        ++write_count;
    };
    const auto write_image = [&](VkDescriptorSet set, uint32_t binding, VkImageView view) {
        write_descriptor(set, binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                         {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL});
    };
    // The sky cubemap: the host's (in its layout) or the device's fallback cube (GENERAL,
    // never sampled: sky_cubemap = 0); the sampler is the device's.
    const GpuImage& fallback_sky = kernels_.fallback_sky();
    const VkImageView sky_view =
        sky ? SkyViewFor(options.sky, frame.frame_number, safe) : fallback_sky.view();
    const VkImageLayout sky_layout =
        sky ? static_cast<VkImageLayout>(options.sky.layout) : VK_IMAGE_LAYOUT_GENERAL;
    const auto write_sky = [&](VkDescriptorSet set, uint32_t binding) {
        write_descriptor(set, binding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                         {VK_NULL_HANDLE, sky_view, sky_layout});
        write_descriptor(set, binding + 1, VK_DESCRIPTOR_TYPE_SAMPLER,
                         {kernels_.sky_sampler(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
    };
    if (tlas != VK_NULL_HANDLE)
    {
        VkWriteDescriptorSet& write = writes[write_count++];
        write = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.pNext = &as_write;
        write.dstSet = slot->set;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    }
    VkImage images[kTargetCount];
    VkImageLayout layouts[kTargetCount];
    VkImageView color_view = VK_NULL_HANDLE;
    for (uint32_t p = 0; p < present_count; ++p)
    {
        const CaelixTargetImage& target = *all[present[p]];
        images[p] = HandleFromU64<VkImage>(target.image);
        layouts[p] = static_cast<VkImageLayout>(target.layout);
        const VkImageView view = ViewFor(target, frame.frame_number, safe);
        if (present[p] == 0)
        {
            color_view = view;
        }
        write_image(slot->set, 1 + present[p], view);
    }
    if (lighting)
    {
        for (uint32_t i = 0; i < kTraceInternalCount; ++i)
        {
            write_image(slot->set, 1 + kTargetCount + i, internal_[i].view());
        }
    }
    write_sky(slot->set, kTraceSkyBinding);
    if (sky)
    {
        write_sky(slot->sky_set, 0);
    }
    vkUpdateDescriptorSets(device_.device(), write_count, writes, 0, nullptr);

    // Frame constants (written to the slot's buffer below, before the submit).
    FrameConstants constants{};
    Copy3(camera.position, constants.position);
    Copy3(camera.right, constants.right);
    Copy3(camera.up, constants.up);
    Copy3(camera.forward, constants.forward);
    constants.size[0] = width;
    constants.size[1] = height;
    constants.output_size[0] = output_width;
    constants.output_size[1] = output_height;
    constants.upscale_src_size[0] = width;  // the fallback's source; the upscaler's output below
    constants.upscale_src_size[1] = height;
    constants.upscaler = upscaling ? options.upscaler : 0u;
    constants.upscale = upscaling ? 1u : 0u;
    constants.jitter[0] = camera.jitter[0];
    constants.jitter[1] = camera.jitter[1];
    constants.tan_half_fov_y = camera.tan_half_fov_y;
    constants.aspect = camera.aspect;
    constants.near_z = camera.near_z;
    constants.far_z = camera.far_z;
    constants.page_table = scene_.page_table_address();
    constants.slot_tables = scene_.slot_tables_address();
    constants.group_infos = scene_.group_infos_address();
    constants.entity_infos = scene_.entity_infos_address();
    constants.materials = scene_.materials_address();
    constants.subsurface_table = scene_.subsurface_address();
    constants.light_tree = scene_.light_tree_address();
    constants.light_nee_flags = light_nee_flags;
    constants.records_per_page_shift = scene_.records_per_page_shift();
    constants.record_bytes = scene_.record_bytes();
    constants.group_shift = scene_.group_shift();
    constants.encoding = scene_.encoding();
    constants.debug_view = options.debug_view;
    constants.flags = flags;
    float sun[3] = {options.sun_dir[0], options.sun_dir[1], options.sun_dir[2]};
    const float length = std::sqrt(sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]);
    if (length > 0.0f && std::isfinite(length))
    {
        sun[0] /= length;
        sun[1] /= length;
        sun[2] /= length;
    }
    else
    {
        sun[0] = 0.0f;
        sun[1] = 1.0f;
        sun[2] = 0.0f;
    }
    Copy3(sun, constants.sun_dir);
    Copy3(options.sky_color, constants.sky_color);
    // CaelixCamera::projection is column-major; the constants hold its rows.
    for (uint32_t row = 0; row < 4; ++row)
    {
        for (uint32_t column = 0; column < 4; ++column)
        {
            constants.projection[row][column] = camera.projection[column * 4 + row];
        }
    }
    if (has_prev_camera_)
    {
        Copy3(prev_camera_.position, constants.prev_position);
        Copy3(prev_camera_.right, constants.prev_right);
        Copy3(prev_camera_.up, constants.prev_up);
        Copy3(prev_camera_.forward, constants.prev_forward);
        constants.prev_tan_half_fov_y = prev_camera_.tan_half_fov_y;
        constants.prev_aspect = prev_camera_.aspect;
        constants.has_prev_camera = 1;
    }
    constants.target_mask = mask;
    constants.miss_depth = options.miss_depth;
    constants.miss_view_z = options.miss_view_z;
    constants.has_tlas = tlas != VK_NULL_HANDLE ? 1u : 0u;
    // Lighting.
    const uint32_t accum_count = restart_accumulation ? 0u : accum_count_;
    constants.mode = options.mode;
    constants.bounce_count = options.bounce_count;
    constants.bounce_count_transparent = options.bounce_count_transparent;
    constants.frame_seed = options.frame_seed;
    constants.accum_count = accum_count;
    constants.exposure = options.exposure != 0.0f ? options.exposure : 1.0f;
    Copy3(options.sun_radiance, constants.sun_radiance);
    Copy3(options.sky_zenith, constants.sky_zenith);
    Copy3(options.sky_horizon, constants.sky_horizon);
    Copy3(options.sky_ground, constants.sky_ground);
    const float sun_radius =
        options.sun_angular_radius > 0.0f ? options.sun_angular_radius : kDefaultSunAngularRadius;
    constants.sun_cos_radius = std::cos(sun_radius);
    // 2 pi (1 - cos r), written as 4 pi sin^2(r / 2) in double: 1 - cos r of a float cancels for
    // a small disk (0.5 degrees: 1 - cos r = 3.8e-5, float spacing near 1 is 6e-8).
    const double half_sin = std::sin(0.5 * static_cast<double>(sun_radius));
    constants.sun_solid_angle = static_cast<float>(4.0 * 3.14159265358979323846 * half_sin * half_sin);
    constants.output_srgb = IsUnormFormat(static_cast<VkFormat>(targets.color.format)) ? 1u : 0u;
    constants.cross_resolve = cross_resolve ? 1u : 0u;
    constants.t_max = kTMax;
    // The kernels' REBLUR_FrontEnd_GetNormHitDist uses the same A, B, C as the denoiser.
    const nrd::ReblurHitDistanceParameters hit_distance = HitDistanceParameters();
    constants.hit_dist_params[0] = hit_distance.A;
    constants.hit_dist_params[1] = hit_distance.B;
    constants.hit_dist_params[2] = hit_distance.C;
    // Fog (lighting modes), emission scale (every mode), sky cubemap (lighting modes).
    if (lighting)
    {
        Copy3(options.fog_color, constants.fog_color);
        constants.fog_color[3] = options.fog_density;
    }
    constants.emission_scale = options.emission_scale > 0.0f ? options.emission_scale : 1.0f;
    constants.sky_sh = sky_sh_buffer_.address();
    if (sky)
    {
        constants.sky_cubemap = 1;
        constants.sky_max_lod = static_cast<float>(options.sky.mip_levels - 1);
        // The mip whose face is about 32 px (the SH kernel's 32 x 32 sampling per face).
        constants.sky_sh_lod = std::clamp(std::log2(static_cast<float>(options.sky.size)) - 5.0f, 0.0f,
                                          constants.sky_max_lod);
    }

    // Any earlier use of the host targets (the host's acquire chains into
    // ALL_COMMANDS) -> the kernels' writes. Lighting modes: the internal targets'
    // first use (UNDEFINED -> GENERAL), else a memory barrier that orders the
    // previous frame's kernel reads and writes before this frame's (also the previous
    // frame's reads of sky_sh_buffer_ before this frame's SH write). The first render
    // call on the device (any renderer) moves the device's fallback sky cube to GENERAL.
    {
        Barriers before;
        for (uint32_t p = 0; p < present_count; ++p)
        {
            before.Image(images[p], layouts[p], VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        if (lighting)
        {
            if (internal_needs_init_)
            {
                for (uint32_t i = 0; i < kInternalCount; ++i)
                {
                    before.Image(internal_[i].image(), VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                }
            }
            before.Memory(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        if (kernels_.TakeFallbackSkyInit())
        {
            before.Image(fallback_sky.image(), VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE,
                         VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, fallback_sky.layers());
        }
        // The upscaler's new output image: written by the upscaler (any stage), read by the encode.
        if (upscaling && upscale_out_needs_init_)
        {
            before.Image(upscale_out_.image(), VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE,
                         VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        before.Record(cmd);
    }
    // Every kernel below binds its own set as set 0 and the slot's frame set (the frame constants,
    // written to the slot's buffer before the submit) as set kFrameSetIndex = 1.
    static_assert(kFrameSetIndex == 1);
    const auto bind_sets = [&](const ComputePipeline& kernel, VkDescriptorSet own_set) {
        const VkDescriptorSet sets[2] = {own_set, slot->frame_set};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.layout(), 0, 2, sets, 0, nullptr);
    };

    const uint32_t groups_x = (width + kWorkgroupSize - 1) / kWorkgroupSize;
    const uint32_t groups_y = (height + kWorkgroupSize - 1) / kWorkgroupSize;
    timer_.Begin(cmd, frame.frame_number, safe);
    // Sky SH (host cubemap): one workgroup projects the cubemap into sky_sh_buffer_; the
    // barrier orders that write before the trace's reads. Inside the trace timestamps.
    if (sky)
    {
        bind_sets(kernels_.sky_sh(), slot->sky_set);
        vkCmdDispatch(cmd, 1, 1, 1);
        Barriers sh_written;
        sh_written.Memory(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        sh_written.Record(cmd);
    }
    // Trace (the variant without the light NEE code when this frame runs none).
    const ComputePipeline& trace = light_nee_flags != 0 ? kernels_.trace() : kernels_.trace_no_light_nee();
    bind_sets(trace, slot->set);
    vkCmdDispatch(cmd, groups_x, groups_y, 1);
    timer_.End(cmd);

    // NRD (path trace mode with denoising). Its first dispatch waits for the trace's
    // writes, its last barrier orders its outputs before the composite.
    bool denoised = false;
    if (denoiser != nullptr)
    {
        NrdExecutor::Bindings bindings;
        bindings.view_z = internal_[kInternalViewZ].view();
        bindings.normal_roughness = internal_[kInternalNormalRoughness].view();
        bindings.motion = internal_[kInternalMotion].view();
        bindings.diff_in = internal_[kInternalDiff].view();
        bindings.spec_in = internal_[kInternalSpec].view();
        bindings.diff_out = internal_[kInternalDiffOut].view();
        bindings.spec_out = internal_[kInternalSpecOut].view();
        bindings.validation = internal_[kInternalValidation].view();
        const nrd::Identifier identifier = kReblurIdentifier;
        const bool timed = denoise_timer_.Begin(cmd, frame.frame_number, safe);
        denoised = denoiser->Denoise(cmd, slot->index, bindings, std::span(&identifier, 1));
        if (timed)
        {
            denoise_timer_.End(cmd);
        }
        if (!denoised)
        {
            nrd_failed_ = true;
            nrd_error = true;
        }
        nrd_last_frame_ = frame.frame_number;
    }
    nrd_history_valid_ = denoised;
    constants.denoised = denoised ? 1u : 0u;

    // Composite (lighting modes with a colour target). Bindings 3 and 4 hold NRD's
    // outputs when this frame was denoised, else the raw radiance.
    // Binding 7 is the motion of the primary hit (CAELIX_VIEW_MOTION); 9 the upscale input
    // (written instead of the colour target when an upscaler runs); 10 and 11 the sky.
    if (composite)
    {
        write_count = 0;
        write_image(slot->composite_set, 0, color_view);
        const uint32_t composite_internals[] = {
            kInternalAlbedo, kInternalDeterministic, denoised ? kInternalDiffOut : kInternalDiff,
            denoised ? kInternalSpecOut : kInternalSpec, kInternalAccum, kInternalValidation,
            kInternalMotion, kInternalResolve, kInternalUpscaleIn};
        static_assert(std::size(composite_internals) + 1 == kCompositeStorageCount);
        for (uint32_t i = 0; i < std::size(composite_internals); ++i)
        {
            write_image(slot->composite_set, 1 + i, internal_[composite_internals[i]].view());
        }
        write_sky(slot->composite_set, kCompositeSkyBinding);
        vkUpdateDescriptorSets(device_.device(), write_count, writes, 0, nullptr);

        Barriers between;
        between.Memory(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        between.Record(cmd);
        bind_sets(kernels_.composite(), slot->composite_set);
        composite_timer_.Begin(cmd, frame.frame_number, safe);
        vkCmdDispatch(cmd, groups_x, groups_y, 1);
        if (cross_resolve)
        {
            // The composite wrote the linear colour into the resolve image; the cross resolve
            // reads it (and the neighbours' checkerboard flags) and writes the host target.
            // Same descriptor sets; its time is part of composite_ms.
            Barriers resolve;
            resolve.Memory(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            resolve.Record(cmd);
            bind_sets(kernels_.cross_resolve(), slot->composite_set);
            vkCmdDispatch(cmd, groups_x, groups_y, 1);
        }
        composite_timer_.End(cmd);
    }

    // Upscaler (renderer_upscale.cpp): the evaluate (its own barriers before and after: the
    // composite's writes -> its reads, its writes -> the encode's reads), then the encode kernel
    // writes the colour target from its output; without a working upscaler the encode kernel reads
    // the upscale input (nearest upscale). Both between the upscale timestamps.
    if (upscaling)
    {
        const bool timed = upscale_timer_.Begin(cmd, frame.frame_number, safe);
        bool evaluated = false;
        if (upscaler != nullptr)
        {
            UpscalerInputs in;
            in.color = ImageOf(internal_[kInternalUpscaleIn]);
            in.depth = ImageOf(internal_[kInternalDepth]);
            in.motion = ImageOf(internal_[kInternalMotion]);
            in.output = ImageOf(upscale_out_);
            if (ray_reconstruction)
            {
                in.diffuse_albedo = ImageOf(internal_[kInternalRrDiffuseAlbedo]);
                in.specular_albedo = ImageOf(internal_[kInternalRrSpecularAlbedo]);
                in.normal_roughness = ImageOf(internal_[kInternalRrNormalRoughness]);
                in.specular_hit_distance = ImageOf(internal_[kInternalRrHitDistance]);
                WorldToView(camera, in.world_to_view);
                std::memcpy(in.view_to_clip, camera.projection, sizeof(in.view_to_clip));
            }
            const auto clamp_jitter = [](float value) {
                return std::isfinite(value) ? std::clamp(value, -0.5f, 0.5f) : 0.0f;
            };
            in.jitter[0] = clamp_jitter(camera.jitter[0]);
            in.jitter[1] = clamp_jitter(camera.jitter[1]);
            in.reset = upscaler_new_ || (options.upscale_flags & CAELIX_UPSCALE_FLAG_RESET_HISTORY) != 0 ||
                       options.reset_accumulation != 0 || !has_prev_camera_;
            const auto now = std::chrono::steady_clock::now();
            if (upscale_last_time_ != std::chrono::steady_clock::time_point{})
            {
                in.frame_time_ms = std::chrono::duration<float, std::milli>(now - upscale_last_time_).count();
            }
            upscale_last_time_ = now;
            evaluated = upscaler->Evaluate(cmd, in);
            upscaler_new_ = false;
            if (!evaluated)
            {
                upscaler_failed_ = true;
                upscale_error = true;
            }
        }
        else
        {
            // The fallback: the composite's writes of the upscale input -> the encode's reads.
            Barriers fallback;
            fallback.Memory(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            fallback.Record(cmd);
        }
        if (evaluated)
        {
            constants.upscale_src_size[0] = output_width;
            constants.upscale_src_size[1] = output_height;
        }
        write_count = 0;
        write_image(slot->encode_set, 0, color_view);
        write_image(slot->encode_set, 1,
                    evaluated ? upscale_out_.view() : internal_[kInternalUpscaleIn].view());
        write_image(slot->encode_set, 2, internal_[kInternalAlbedo].view());
        vkUpdateDescriptorSets(device_.device(), write_count, writes, 0, nullptr);
        // NGX changed the command buffer state: everything is bound again.
        bind_sets(kernels_.upscale_encode(), slot->encode_set);
        vkCmdDispatch(cmd, (output_width + kWorkgroupSize - 1) / kWorkgroupSize,
                      (output_height + kWorkgroupSize - 1) / kWorkgroupSize, 1);
        if (timed)
        {
            upscale_timer_.End(cmd);
        }
        upscaler_last_frame_ = frame.frame_number;
        upscale_out_needs_init_ = false;
    }

    // The constants are complete (the GPU reads them when the command buffer executes).
    std::memcpy(slot->constants.mapped(), &constants, sizeof(constants));
    slot->constants.FlushHostWrites(0, sizeof(constants));

    // The kernels' writes -> all later commands (the host's use of its targets).
    {
        Barriers after;
        for (uint32_t p = 0; p < present_count; ++p)
        {
            after.Image(images[p], VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        }
        after.Record(cmd);
    }

    prev_camera_ = camera;
    has_prev_camera_ = true;
    last_frame_ = frame.frame_number;
    has_rendered_ = true;
    last_mode_ = options.mode;
    if (lighting)
    {
        internal_needs_init_ = false;
        internal_last_frame_ = frame.frame_number;
        accum_count_ =
            (options.mode == CAELIX_MODE_REFERENCE && composite) ? accum_count + 1 : accum_count;
    }
    // An NRD or NGX error is reported once, by the call in which it happened; the frame is
    // recorded (raw output, nearest upscale) all the same.
    return nrd_error || upscale_error ? CAELIX_ERROR_VULKAN : CAELIX_OK;
}

}  // namespace caelix
