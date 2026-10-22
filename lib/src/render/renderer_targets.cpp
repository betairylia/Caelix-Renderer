// Renderer creation, frame slots, target views, internal targets, ReSTIR GI buffers and options,
// and timings.
#include "render/renderer.h"

#include "core/log.h"
#include "render/frame_constants.h"
#include "render/renderer_internal.h"
#include "scene/scene.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iterator>
#include <utility>

namespace caelix {

Renderer::Renderer(Device& device, RenderKernels& kernels, Scene& scene, std::function<NgxRuntime*()> ngx)
    : device_(device), kernels_(kernels), scene_(scene)
{
#if CAELIX_DLSS
    ngx_ = std::move(ngx);
#else
    static_cast<void>(ngx);
#endif
}

CaelixResult Renderer::Create(Device& device, RenderKernels& kernels, Scene& scene,
                              std::function<NgxRuntime*()> ngx, std::unique_ptr<Renderer>& out)
{
    out.reset();
    std::unique_ptr<Renderer> renderer(new Renderer(device, kernels, scene, std::move(ngx)));
    const VkDevice vk_device = device.device();
    static_assert(std::size(kTargetSpecs) == kTargetCount);
    static_assert(std::size(kInternalSpecs) == kInternalCount);

    // The set layouts, pipelines, sky sampler and fallback sky cube are the device's
    // (RenderKernels); the renderer owns its frame slots, the sky SH buffer and timers.
    // 9 float4 written by the sky SH kernel and read by the trace kernel.
    renderer->sky_sh_buffer_ = GpuBuffer::DeviceLocalWithUsage(
        device.allocator(), vk_device, 9 * 4 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    for (uint32_t i = 0; i < kInitialSlots; ++i)
    {
        renderer->AddSlot();
    }
    renderer->timer_ = GpuTimer(device);
    renderer->composite_timer_ = GpuTimer(device);
    renderer->denoise_timer_ = GpuTimer(device);
    renderer->upscale_timer_ = GpuTimer(device);
    renderer->gi_timer_ = GpuTimer(device);
    renderer->light_timer_ = GpuTimer(device);
    CAELIX_LOGI("renderer created (shared kernels), %u frame slots", kInitialSlots);
    out = std::move(renderer);
    return CAELIX_OK;
}

Renderer::~Renderer()
{
    const VkDevice device = device_.device();
    ReleaseTargetViews();
    for (const std::unique_ptr<FrameSlot>& slot : slots_)
    {
        if (slot->pool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device, slot->pool, nullptr);  // frees the sets too
        }
    }
    slots_.clear();
    // The device is idle (caelix_renderer_destroy contract): internal_, retired_, the NRD
    // executors, the upscalers and sky_sh_buffer_ are destroyed with the members (before the
    // device's NGX runtime, which outlives every renderer). The layouts, pipelines, sky sampler
    // and fallback sky belong to the RenderKernels.
}

void Renderer::AddSlot()
{
    const VkDevice device = device_.device();
    auto slot = std::make_unique<FrameSlot>();
    slot->index = static_cast<uint32_t>(slots_.size());
    // Four sets: trace, composite, sky SH (each has one sampled image and one sampler) and the
    // upscale encode set (storage images only); the trace set also has the light cells (a storage
    // buffer).
    const VkDescriptorPoolSize sizes[5] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         kTargetCount + kTraceInternalCount + kCompositeStorageCount + kEncodeBindingCount},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
    };
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 4;
    pool_info.poolSizeCount = 5;
    pool_info.pPoolSizes = sizes;
    VkResult result = vkCreateDescriptorPool(device, &pool_info, nullptr, &slot->pool);
    if (result != VK_SUCCESS)
    {
        slot->pool = VK_NULL_HANDLE;
        CAELIX_LOGE("vkCreateDescriptorPool failed: %d", static_cast<int>(result));
        throw VulkanError("vkCreateDescriptorPool failed", result);
    }
    // Pushed now so that the destructor destroys the pool if a later call throws.
    slots_.push_back(std::move(slot));
    FrameSlot& added = *slots_.back();
    const VkDescriptorSetLayout layouts[4] = {kernels_.trace_layout(), kernels_.composite_layout(),
                                              kernels_.sky_layout(), kernels_.encode_layout()};
    VkDescriptorSet sets[4] = {};
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = added.pool;
    set_info.descriptorSetCount = 4;
    set_info.pSetLayouts = layouts;
    result = vkAllocateDescriptorSets(device, &set_info, sets);
    if (result != VK_SUCCESS)
    {
        CAELIX_LOGE("vkAllocateDescriptorSets failed: %d", static_cast<int>(result));
        throw VulkanError("vkAllocateDescriptorSets failed", result);
    }
    added.set = sets[0];
    added.composite_set = sets[1];
    added.sky_set = sets[2];
    added.encode_set = sets[3];
    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
    allocation_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    added.constants = GpuBuffer(device_.allocator(), device, sizeof(FrameConstants),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                allocation_info, true);
}

Renderer::FrameSlot* Renderer::AcquireSlot(uint64_t frame, uint64_t safe_frame)
{
    const uint32_t count = static_cast<uint32_t>(slots_.size());
    for (uint32_t i = 0; i < count; ++i)
    {
        const uint32_t index = (next_slot_ + i) % count;
        FrameSlot& slot = *slots_[index];
        if (slot.frame <= safe_frame)
        {
            next_slot_ = (index + 1) % count;
            slot.frame = frame;
            return &slot;
        }
    }
    if (count >= kMaxSlots)
    {
        return nullptr;
    }
    AddSlot();
    CAELIX_LOGI("renderer: %zu frame slots (more frames in flight)", slots_.size());
    FrameSlot& slot = *slots_.back();
    slot.frame = frame;
    next_slot_ = 0;
    return &slot;
}

bool Renderer::FormatSupported(VkFormat format)
{
    for (const FormatSupport& known : formats_)
    {
        if (known.format == format)
        {
            return known.supported;
        }
    }
    VkFormatProperties3 properties3{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
    properties.pNext = &properties3;
    vkGetPhysicalDeviceFormatProperties2(device_.physical_device(), format, &properties);
    const VkFormatFeatureFlags2 needed =
        VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
    const bool supported = (properties3.optimalTilingFeatures & needed) == needed;
    formats_.push_back(FormatSupport{format, supported});
    if (!supported)
    {
        CAELIX_LOGE("caelix_renderer_render: VkFormat %d does not support storage image writes "
                    "without a format qualifier (optimal tiling)",
                    static_cast<int>(format));
    }
    return supported;
}

VkImageView Renderer::ViewFor(const CaelixTargetImage& target, uint64_t frame,
                              uint64_t safe_frame)
{
    return CachedViewFor(HandleFromU64<VkImage>(target.image), static_cast<VkFormat>(target.format),
                         false, 1, frame, safe_frame);
}

VkImageView Renderer::SkyViewFor(const CaelixSkyCubemap& sky, uint64_t frame, uint64_t safe_frame)
{
    return CachedViewFor(HandleFromU64<VkImage>(sky.image), static_cast<VkFormat>(sky.format), true,
                         sky.mip_levels, frame, safe_frame);
}

VkImageView Renderer::CachedViewFor(VkImage image, VkFormat format, bool cube,
                                    uint32_t mip_levels, uint64_t frame, uint64_t safe_frame)
{
    for (CachedView& cached : views_)
    {
        if (cached.image == image && cached.format == format && cached.cube == cube &&
            cached.mip_levels == mip_levels)
        {
            cached.last_frame = frame;
            return cached.view;
        }
    }
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, cube ? 6u : 1u};
    // A target's view is used as a storage image only, a cubemap's as a sampled image only
    // (the image may have other usages whose format support differs, for example sRGB
    // attachments).
    VkImageViewUsageCreateInfo usage_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    usage_info.usage = cube ? VK_IMAGE_USAGE_SAMPLED_BIT : VK_IMAGE_USAGE_STORAGE_BIT;
    view_info.pNext = &usage_info;
    VkImageView view = VK_NULL_HANDLE;
    const VkResult result = vkCreateImageView(device_.device(), &view_info, nullptr, &view);
    if (result != VK_SUCCESS)
    {
        CAELIX_LOGE("vkCreateImageView failed: %d", static_cast<int>(result));
        throw VulkanError("vkCreateImageView failed", result);
    }
    views_.push_back(CachedView{image, format, cube, mip_levels, view, frame});
    CAELIX_LOGD("renderer: %s view created for image 0x%llx (%zu cached)", cube ? "cube" : "image",
                static_cast<unsigned long long>(HandleToU64(image)), views_.size());

    // Over the cap: destroy the least recently used view whose last frame is safe.
    // When every other view is still in flight, the cache stays larger for now.
    if (views_.size() > kMaxCachedViews)
    {
        size_t oldest = views_.size();
        for (size_t i = 0; i + 1 < views_.size(); ++i)
        {  // never the view just created
            if (views_[i].last_frame <= safe_frame &&
                (oldest == views_.size() || views_[i].last_frame < views_[oldest].last_frame))
            {
                oldest = i;
            }
        }
        if (oldest < views_.size())
        {
            vkDestroyImageView(device_.device(), views_[oldest].view, nullptr);
            views_.erase(views_.begin() + static_cast<std::ptrdiff_t>(oldest));
        }
    }
    return view;
}

void Renderer::ReleaseTargetViews()
{
    for (const CachedView& cached : views_)
    {
        vkDestroyImageView(device_.device(), cached.view, nullptr);
    }
    views_.clear();
}

bool Renderer::EnsureInternalTargets(uint32_t width, uint32_t height, uint64_t frame)
{
    GpuImage& first = internal_[0];
    if (first.valid() && first.width() == width && first.height() == height)
    {
        return false;
    }
    if (first.valid())
    {
        RetiredImages retired;
        retired.frame = internal_last_frame_;
        retired.images = std::move(internal_);
        retired_.push_back(std::move(retired));
    }
    // On a throw the images created so far are destroyed with `fresh`, and the next
    // lighting render tries again.
    InternalImages fresh;
    for (uint32_t i = 0; i < kInternalCount; ++i)
    {
        fresh[i] = GpuImage(device_.allocator(), device_.device(), width, height,
                            kInternalSpecs[i].format,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            kInternalSpecs[i].name);
    }
    internal_ = std::move(fresh);
    internal_last_frame_ = frame;
    internal_needs_init_ = true;
    CAELIX_LOGI("renderer: internal targets created at %ux%u", width, height);
    return true;
}

bool Renderer::EnsureGiBuffers(uint32_t width, uint32_t height, uint64_t frame)
{
    if (gi_buffers_[0].valid() && gi_width_ == width && gi_height_ == height)
    {
        return false;
    }
    if (gi_buffers_[0].valid())
    {
        RetiredGiBuffers retired;
        retired.frame = gi_last_frame_;
        retired.buffers = std::move(gi_buffers_);
        retired_gi_.push_back(std::move(retired));
        gi_reservoir_bytes_ = 0;
    }
    // 48 bytes per pixel (GiReservoir, shaders/render/caelix_restir_gi.hlsl). On a throw the
    // buffers created so far are destroyed with `fresh`, and the next ReSTIR render tries again.
    constexpr VkDeviceSize kReservoirBytes = 48;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * height * kReservoirBytes;
    GiBuffers fresh;
    for (GpuBuffer& buffer : fresh)
    {
        buffer = GpuBuffer::DeviceLocalWithUsage(
            device_.allocator(), device_.device(), bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    }
    gi_buffers_ = std::move(fresh);
    gi_index_ = {kGiInitial, kGiPrev, kGiTemp, kGiOut};
    gi_width_ = width;
    gi_height_ = height;
    gi_reservoir_bytes_ = bytes * kGiBufferCount;
    gi_last_frame_ = frame;
    gi_history_valid_ = false;
    CAELIX_LOGI("renderer: ReSTIR GI reservoirs created at %ux%u (%.1f MiB)", width, height,
                static_cast<double>(gi_reservoir_bytes_) / (1024.0 * 1024.0));
    return true;
}

void Renderer::EnsureLightCells(uint32_t capacity, uint32_t reservoirs, uint64_t frame)
{
    if (light_cells_.valid() && light_cell_capacity_ == capacity && light_cell_reservoirs_ == reservoirs)
    {
        return;
    }
    // LightCellBytes(reservoirs) per cell and the 16 bytes of usage counters at the tail
    // (render/caelix_light_sample.hlsl), and the counters' readback ring. On a throw the old buffers stay.
    GpuBuffer fresh = GpuBuffer::DeviceLocal(device_.allocator(), device_.device(),
                                             LightCellBytes(reservoirs) * capacity + kLightCellCounterBytes);
    GpuBuffer fresh_readback = GpuBuffer::HostReadback(device_.allocator(), device_.device(),
                                                       kLightCellCounterBytes * kLightCounterSlots);
    if (light_cells_.valid())
    {
        retired_light_cells_.push_back(RetiredLightCells{light_cells_last_frame_, std::move(light_cells_),
                                                         std::move(light_counter_readback_)});
    }
    light_cells_ = std::move(fresh);
    light_counter_readback_ = std::move(fresh_readback);
    light_counter_frames_.fill(0);
    light_cell_capacity_ = capacity;
    light_cell_reservoirs_ = reservoirs;
    light_cells_need_clear_ = true;
    light_cells_last_frame_ = frame;
    CAELIX_LOGI("renderer: %u light cells of %u reservoirs created (%.1f MiB)", capacity, reservoirs,
                static_cast<double>(light_cells_.size()) / (1024.0 * 1024.0));
}

CaelixResult Renderer::SetGiOptions(const CaelixGiOptions& options)
{
    constexpr const char* kFunction = "caelix_renderer_set_gi_options";
    constexpr uint32_t kKnownFlags = CAELIX_GI_FLAG_TEMPORAL | CAELIX_GI_FLAG_SPATIAL | CAELIX_GI_FLAG_VISIBILITY |
                                     CAELIX_GI_FLAG_UNBIASED | CAELIX_GI_FLAG_DIFFUSE_ONLY |
                                     CAELIX_GI_FLAG_LIGHT_NEE | CAELIX_GI_FLAG_LIGHT_CELL_JITTER |
                                     CAELIX_GI_FLAG_LIGHT_MIS | CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL |
                                     CAELIX_GI_FLAG_LIGHT_CELL_ORIENTED | CAELIX_GI_FLAG_PAIRWISE_MIS;
    if (options.method > CAELIX_GI_RESTIR)
    {
        CAELIX_LOGE("%s: unknown method %u", kFunction, options.method);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if ((options.flags & ~kKnownFlags) != 0)
    {
        CAELIX_LOGE("%s: unknown flags 0x%x", kFunction, options.flags & ~kKnownFlags);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.spatial_samples > 8)
    {
        CAELIX_LOGE("%s: spatial_samples %u is above the maximum 8", kFunction, options.spatial_samples);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.spatial_passes > 2)
    {
        CAELIX_LOGE("%s: spatial_passes %u is above the maximum 2", kFunction, options.spatial_passes);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(options.spatial_radius) || options.spatial_radius < 0.0f)
    {
        CAELIX_LOGE("%s: spatial_radius %g must be finite and >= 0", kFunction,
                    static_cast<double>(options.spatial_radius));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(options.depth_threshold) || options.depth_threshold < 0.0f)
    {
        CAELIX_LOGE("%s: depth_threshold %g must be finite and >= 0", kFunction,
                    static_cast<double>(options.depth_threshold));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.light_cell_capacity != 0 &&
        (!std::has_single_bit(options.light_cell_capacity) || options.light_cell_capacity < (1u << 10) ||
         options.light_cell_capacity > (1u << 20)))
    {
        CAELIX_LOGE("%s: light_cell_capacity %u must be 0 or a power of two in 1024 .. 1048576", kFunction,
                    options.light_cell_capacity);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.light_candidates > 64)
    {
        CAELIX_LOGE("%s: light_candidates %u is above the maximum 64", kFunction, options.light_candidates);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(options.light_level_footprint) || options.light_level_footprint < 0.0f)
    {
        CAELIX_LOGE("%s: light_level_footprint %g must be finite and >= 0", kFunction,
                    static_cast<double>(options.light_level_footprint));
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.light_temporal_frames > 1024)
    {
        CAELIX_LOGE("%s: light_temporal_frames %u is above the maximum 1024", kFunction,
                    options.light_temporal_frames);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.light_nee_candidates > 32)
    {
        CAELIX_LOGE("%s: light_nee_candidates %u is above the maximum 32", kFunction, options.light_nee_candidates);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    if (options.light_cell_reservoirs != 0 &&
        (!std::has_single_bit(options.light_cell_reservoirs) || options.light_cell_reservoirs > 256))
    {
        CAELIX_LOGE("%s: light_cell_reservoirs %u must be 0 or a power of two in 1 .. 256", kFunction,
                    options.light_cell_reservoirs);
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    // The cell table is one storage buffer descriptor (trace set binding 21), and the kernels address
    // it with 32-bit byte offsets: it must fit in maxStorageBufferRange (at most 4 GiB - 1).
    const uint32_t capacity =
        options.light_cell_capacity != 0 ? options.light_cell_capacity : kDefaultLightCellCapacity;
    const uint32_t reservoirs =
        options.light_cell_reservoirs != 0 ? options.light_cell_reservoirs : kDefaultLightCellReservoirs;
    const VkDeviceSize cell_table_bytes = LightCellBytes(reservoirs) * capacity + kLightCellCounterBytes;
    if (cell_table_bytes > device_.max_storage_buffer_range())
    {
        CAELIX_LOGE("%s: %u light cells of %u reservoirs need %llu bytes, above the device's "
                    "maxStorageBufferRange %u",
                    kFunction, capacity, reservoirs, static_cast<unsigned long long>(cell_table_bytes),
                    device_.max_storage_buffer_range());
        return CAELIX_ERROR_INVALID_ARGUMENT;
    }
    // Reservoirs of another method or other reuse flags are not a valid history.
    if (options.reset != 0 || options.method != gi_options_.method || options.flags != gi_options_.flags)
    {
        gi_history_valid_ = false;
    }
    // The light cell history (CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL) is dropped by reset only: the cell
    // targets do not depend on the method or the other flags, and a fill without the flag stores M 0.
    if (options.reset != 0)
    {
        light_cell_reset_ = true;
    }
    gi_options_ = options;
    gi_options_.reset = 0;
    return CAELIX_OK;
}

void Renderer::GetGiStats(CaelixGiStats& out) const
{
    out = CaelixGiStats{};
    out.reservoir_bytes = gi_reservoir_bytes_;
    // The frame of GetTimings (trace_ms); 0 when that frame ran no GI pass.
    const uint64_t frame = timer_.latest_frame();
    float gi_ms = 0.0f;
    if (frame != 0 && gi_timer_.MsForFrame(frame, gi_ms))
    {
        out.gi_ms = gi_ms;
    }
    out.light_cell_bytes = light_cells_.size();
    float light_ms = 0.0f;
    if (frame != 0 && light_timer_.MsForFrame(frame, light_ms))
    {
        out.light_fill_ms = light_ms;
    }
    // The light cell counters of the same frame: its readback slot, when that slot still holds it
    // (the frame is safe, so the copy has completed).
    const uint32_t slot = static_cast<uint32_t>(frame % kLightCounterSlots);
    if (frame != 0 && light_counter_readback_.valid() && light_counter_frames_[slot] == frame)
    {
        const VkDeviceSize offset = static_cast<VkDeviceSize>(slot) * kLightCellCounterBytes;
        light_counter_readback_.InvalidateHostReads(offset, kLightCellCounterBytes);
        uint32_t counters[4];
        static_assert(sizeof(counters) <= kLightCellCounterBytes);
        std::memcpy(counters, light_counter_readback_.mapped() + offset, sizeof(counters));
        out.light_cells_live = counters[0];
        out.light_cells_active = counters[1];
        out.light_cells_inserted = counters[2];
        out.light_cell_lookups_failed = counters[3];
    }
}

void Renderer::FreeRetired(uint64_t safe_frame)
{
    std::erase_if(retired_, [safe_frame](const RetiredImages& retired) {
        return retired.frame <= safe_frame;
    });
    std::erase_if(retired_gi_, [safe_frame](const RetiredGiBuffers& retired) {
        return retired.frame <= safe_frame;
    });
    std::erase_if(retired_light_cells_, [safe_frame](const RetiredLightCells& retired) {
        return retired.frame <= safe_frame;
    });
    std::erase_if(retired_nrd_, [safe_frame](const RetiredDenoiser& retired) {
        return retired.frame <= safe_frame;
    });
    std::erase_if(retired_upscalers_, [safe_frame](const RetiredUpscaler& retired) {
        return retired.frame <= safe_frame;
    });
}

void Renderer::GetTimings(CaelixGpuTimings& out) const
{
    out = CaelixGpuTimings{};
    out.frame_number = timer_.latest_frame();
    if (out.frame_number == 0)
    {
        return;
    }
    out.trace_ms = timer_.latest_ms();
    float flush_ms = 0.0f;
    if (scene_.FlushMsForFrame(out.frame_number, flush_ms))
    {
        out.flush_ms = flush_ms;
    }
    float composite_ms = 0.0f;
    if (composite_timer_.MsForFrame(out.frame_number, composite_ms))
    {
        out.composite_ms = composite_ms;
    }
    float denoise_ms = 0.0f;
    if (denoise_timer_.MsForFrame(out.frame_number, denoise_ms))
    {
        out.denoise_ms = denoise_ms;
    }
    float upscale_ms = 0.0f;
    if (upscale_timer_.MsForFrame(out.frame_number, upscale_ms))
    {
        out.upscale_ms = upscale_ms;
    }
}

}  // namespace caelix
