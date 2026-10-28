// Renderer creation, frame slots, target views, internal targets and timings.
#include "render/renderer.h"

#include "core/log.h"
#include "render/frame_constants.h"
#include "render/renderer_internal.h"
#include "scene/scene.h"

#include <algorithm>
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
    // Five sets: trace, composite, sky SH (each has one sampled image and one sampler; the
    // composite set also the background, a second sampled image), the upscale encode set (storage
    // images only) and the frame set (one uniform buffer).
    constexpr uint32_t kSetCount = 5;
    const VkDescriptorPoolSize sizes[5] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         kTargetCount + kTraceInternalCount + kCompositeStorageCount + kEncodeBindingCount},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
    };
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = kSetCount;
    pool_info.poolSizeCount = static_cast<uint32_t>(std::size(sizes));
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
    const VkDescriptorSetLayout layouts[kSetCount] = {kernels_.trace_layout(), kernels_.composite_layout(),
                                                      kernels_.sky_layout(), kernels_.encode_layout(),
                                                      kernels_.frame_layout()};
    VkDescriptorSet sets[kSetCount] = {};
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = added.pool;
    set_info.descriptorSetCount = kSetCount;
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
    added.frame_set = sets[4];
    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
    allocation_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    // Read only as the uniform buffer of the frame set (no device address).
    added.constants = GpuBuffer(device_.allocator(), device, sizeof(FrameConstants),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, allocation_info, false);
    // The slot keeps its constants buffer for its whole life, so the frame set is written once.
    const VkDescriptorBufferInfo constants_info{added.constants.buffer(), 0, sizeof(FrameConstants)};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = added.frame_set;
    write.dstBinding = kFrameConstantsBinding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &constants_info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
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
                         false, false, 1, frame, safe_frame);
}

VkImageView Renderer::SkyViewFor(const CaelixSkyCubemap& sky, uint64_t frame, uint64_t safe_frame)
{
    return CachedViewFor(HandleFromU64<VkImage>(sky.image), static_cast<VkFormat>(sky.format), true, true,
                         sky.mip_levels, frame, safe_frame);
}

VkImageView Renderer::BackgroundViewFor(const CaelixTargetImage& background, uint64_t frame,
                                        uint64_t safe_frame)
{
    return CachedViewFor(HandleFromU64<VkImage>(background.image), static_cast<VkFormat>(background.format),
                         false, true, 1, frame, safe_frame);
}

VkImageView Renderer::CachedViewFor(VkImage image, VkFormat format, bool cube, bool sampled,
                                    uint32_t mip_levels, uint64_t frame, uint64_t safe_frame)
{
    sampled = sampled || cube;
    for (CachedView& cached : views_)
    {
        if (cached.image == image && cached.format == format && cached.cube == cube &&
            cached.sampled == sampled && cached.mip_levels == mip_levels)
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
    // A target's view is used as a storage image only, a cubemap's and the background's as a
    // sampled image only (the image may have other usages whose format support differs, for
    // example sRGB attachments).
    VkImageViewUsageCreateInfo usage_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    usage_info.usage = sampled ? VK_IMAGE_USAGE_SAMPLED_BIT : VK_IMAGE_USAGE_STORAGE_BIT;
    view_info.pNext = &usage_info;
    VkImageView view = VK_NULL_HANDLE;
    const VkResult result = vkCreateImageView(device_.device(), &view_info, nullptr, &view);
    if (result != VK_SUCCESS)
    {
        CAELIX_LOGE("vkCreateImageView failed: %d", static_cast<int>(result));
        throw VulkanError("vkCreateImageView failed", result);
    }
    views_.push_back(CachedView{image, format, cube, sampled, mip_levels, view, frame});
    CAELIX_LOGD("renderer: %s view created for image 0x%llx (%zu cached)",
                cube ? "cube" : (sampled ? "sampled image" : "image"),
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

bool Renderer::EnsureInternalTargets(uint32_t width, uint32_t height, uint32_t hint_shift, uint64_t frame)
{
    GpuImage& first = internal_[0];
    if (first.valid() && first.width() == width && first.height() == height && internal_hint_shift_ == hint_shift)
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
    // The light NEE hint images: one texel per patch x patch pixels (docs/LIGHT_NEE.md section 12.1).
    const uint32_t patch = 1u << hint_shift;
    const uint32_t hint_width = (width + patch - 1) >> hint_shift;
    const uint32_t hint_height = (height + patch - 1) >> hint_shift;
    for (uint32_t i = 0; i < kInternalCount; ++i)
    {
        const bool hint = i == kInternalLightHintA || i == kInternalLightHintB;
        fresh[i] = GpuImage(device_.allocator(), device_.device(), hint ? hint_width : width,
                            hint ? hint_height : height, kInternalSpecs[i].format,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            kInternalSpecs[i].name);
    }
    internal_ = std::move(fresh);
    internal_last_frame_ = frame;
    internal_needs_init_ = true;
    internal_hint_shift_ = hint_shift;
    // The new hint images hold no history: the next render with the hint reads 0.5 (kLightNeeHintReset).
    hint_history_valid_ = false;
    CAELIX_LOGI("renderer: internal targets created at %ux%u (light hint %ux%u, patch %u)", width, height,
                hint_width, hint_height, patch);
    return true;
}

void Renderer::FreeRetired(uint64_t safe_frame)
{
    std::erase_if(retired_, [safe_frame](const RetiredImages& retired) {
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
