// Renderer creation, frame slots, target views, internal targets and timings.
#include "render/renderer.h"

#include "core/log.h"
#include "render/frame_constants.h"
#include "render/renderer_internal.h"
#include "rhi/shader_loader.h"
#include "scene/scene.h"

#include <algorithm>
#include <iterator>

namespace caelix {
namespace {

// A descriptor set layout of `count` bindings, every one PARTIALLY_BOUND, compute
// stage. `types[i]` is the type of binding i.
VkDescriptorSetLayout CreateSetLayout(VkDevice device, uint32_t count,
                                      const VkDescriptorType* types)
{
    VkDescriptorSetLayoutBinding bindings[16];
    VkDescriptorBindingFlags binding_flags[16];
    for (uint32_t i = 0; i < count; ++i)
    {
        bindings[i] = {i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        binding_flags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    flags_info.bindingCount = count;
    flags_info.pBindingFlags = binding_flags;
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.pNext = &flags_info;
    layout_info.bindingCount = count;
    layout_info.pBindings = bindings;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    const VkResult result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout);
    if (result != VK_SUCCESS)
    {
        CAELIX_LOGE("vkCreateDescriptorSetLayout failed: %d", static_cast<int>(result));
        throw VulkanError("vkCreateDescriptorSetLayout failed", result);
    }
    return layout;
}

}  // namespace

Renderer::Renderer(Device& device, Scene& scene)
    : device_(device), scene_(scene)
{
}

CaelixResult Renderer::Create(Device& device, Scene& scene, std::unique_ptr<Renderer>& out)
{
    out.reset();
    std::unique_ptr<Renderer> renderer(new Renderer(device, scene));
    const VkDevice vk_device = device.device();
    static_assert(std::size(kTargetSpecs) == kTargetCount);
    static_assert(std::size(kInternalSpecs) == kInternalCount);

    // Trace set: binding 0 the TLAS, bindings 1..6 the host targets, 7..13 the
    // internal targets, 14 the sky cubemap, 15 its sampler. Every binding is
    // PARTIALLY_BOUND (descriptorBindingPartiallyBound, a device requirement): a render
    // call writes only the TLAS when there is one, the targets that are present, the
    // internal targets in the lighting modes and the sky bindings, and the kernel
    // touches only those.
    VkDescriptorType trace_types[kTraceBindingCount];
    std::fill(trace_types, trace_types + kTraceBindingCount, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    trace_types[0] = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    trace_types[kTraceSkyBinding] = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    trace_types[kTraceSkyBinding + 1] = VK_DESCRIPTOR_TYPE_SAMPLER;
    renderer->set_layout_ = CreateSetLayout(vk_device, kTraceBindingCount, trace_types);
    // Composite set: binding 0 the host colour target, 1..8 internal targets, 9 the sky
    // cubemap, 10 its sampler. The cross resolve kernel uses the same layout and set.
    VkDescriptorType composite_types[kCompositeBindingCount];
    std::fill(composite_types, composite_types + kCompositeBindingCount,
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    composite_types[kCompositeSkyBinding] = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    composite_types[kCompositeSkyBinding + 1] = VK_DESCRIPTOR_TYPE_SAMPLER;
    renderer->composite_layout_ = CreateSetLayout(vk_device, kCompositeBindingCount, composite_types);
    // Sky SH set: binding 0 the host's cubemap, 1 the sampler.
    const VkDescriptorType sky_types[2] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                           VK_DESCRIPTOR_TYPE_SAMPLER};
    renderer->sky_layout_ = CreateSetLayout(vk_device, 2, sky_types);

    const std::vector<uint32_t> trace_spirv = LoadShader("caelix_path_trace");
    renderer->trace_ = ComputePipeline(vk_device, trace_spirv, sizeof(TracePush), "caelix_path_trace",
                                       renderer->set_layout_);
    const std::vector<uint32_t> composite_spirv = LoadShader("caelix_composite");
    renderer->composite_ = ComputePipeline(vk_device, composite_spirv, sizeof(TracePush),
                                           "caelix_composite", renderer->composite_layout_);
    const std::vector<uint32_t> resolve_spirv = LoadShader("caelix_cross_resolve");
    renderer->cross_resolve_ = ComputePipeline(vk_device, resolve_spirv, sizeof(TracePush),
                                               "caelix_cross_resolve", renderer->composite_layout_);
    const std::vector<uint32_t> sky_sh_spirv = LoadShader("caelix_sky_sh");
    renderer->sky_sh_ = ComputePipeline(vk_device, sky_sh_spirv, sizeof(TracePush), "caelix_sky_sh",
                                        renderer->sky_layout_);

    // The sky sampler: trilinear, clamp to edge (a cube view samples across faces anyway),
    // every mip (the kernels clamp the lod to sky_max_lod).
    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;
    const VkResult sampler_result =
        vkCreateSampler(vk_device, &sampler_info, nullptr, &renderer->sky_sampler_);
    if (sampler_result != VK_SUCCESS)
    {
        renderer->sky_sampler_ = VK_NULL_HANDLE;
        CAELIX_LOGE("vkCreateSampler failed: %d", static_cast<int>(sampler_result));
        throw VulkanError("vkCreateSampler failed", sampler_result);
    }
    // Bound in every sky binding when the host gives no cubemap, so that every sampled-image
    // descriptor is valid; never sampled (sky_cubemap = 0). The first render call moves it
    // to GENERAL.
    renderer->fallback_sky_ = GpuImage::Cube(device.allocator(), vk_device, 1,
                                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT,
                                             1, "fallback_sky");
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
    CAELIX_LOGI("renderer created: kernels caelix_path_trace, caelix_composite, "
                "caelix_cross_resolve and caelix_sky_sh, %u frame slots",
                kInitialSlots);
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
    trace_ = ComputePipeline();
    composite_ = ComputePipeline();
    cross_resolve_ = ComputePipeline();
    sky_sh_ = ComputePipeline();
    if (set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
    }
    if (composite_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, composite_layout_, nullptr);
    }
    if (sky_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, sky_layout_, nullptr);
    }
    if (sky_sampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, sky_sampler_, nullptr);
    }
    // The device is idle (caelix_renderer_destroy contract): internal_, retired_,
    // fallback_sky_ and sky_sh_buffer_ are destroyed with the members.
}

void Renderer::AddSlot()
{
    const VkDevice device = device_.device();
    auto slot = std::make_unique<FrameSlot>();
    slot->index = static_cast<uint32_t>(slots_.size());
    // Three sets: trace, composite, sky SH; each has one sampled image and one sampler.
    const VkDescriptorPoolSize sizes[4] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         kTargetCount + kTraceInternalCount + kCompositeStorageCount},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 3},
    };
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 3;
    pool_info.poolSizeCount = 4;
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
    const VkDescriptorSetLayout layouts[3] = {set_layout_, composite_layout_, sky_layout_};
    VkDescriptorSet sets[3] = {};
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = added.pool;
    set_info.descriptorSetCount = 3;
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

void Renderer::FreeRetired(uint64_t safe_frame)
{
    std::erase_if(retired_, [safe_frame](const RetiredImages& retired) {
        return retired.frame <= safe_frame;
    });
    std::erase_if(retired_nrd_, [safe_frame](const RetiredDenoiser& retired) {
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
}

}  // namespace caelix
