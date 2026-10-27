#pragma once
#include "rhi/device.h"
#include "rhi/gpu_image.h"
#include "rhi/pipeline.h"

#include <NRD.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caelix {

// Descriptor bindings of the kernel sets, shared by the set layouts (RenderKernels)
// and the descriptor writes (Renderer). Each of the six frame-constant kernels binds its
// own set below as set 0 and the frame set (binding kFrameConstantsBinding = the frame
// slot's FrameConstants as a uniform buffer, render/frame_constants.h) as set
// kFrameSetIndex.
//
// Trace set (also bound by the voxel id kernel): binding 0 the TLAS, bindings 1..7 the host
// targets (7 = the voxel id target), bindings 8..19 the internal targets of the lighting modes
// (Renderer::Internal order up to kInternalAccum: the NRD G-buffer and radiance, the device depth
// and the four DLSS RR guides), binding 20 the sky cubemap (sampled image), 21 its sampler.
constexpr uint32_t kTargetCount = 7;          // descriptor bindings 1..7
constexpr uint32_t kTraceInternalCount = 12;  // descriptor bindings 8..19
constexpr uint32_t kTraceSkyBinding = 1 + kTargetCount + kTraceInternalCount;
constexpr uint32_t kTraceBindingCount = kTraceSkyBinding + 2;
// Composite set (also bound by the cross resolve kernel): binding 0 the host colour
// target, bindings 1..8 internal targets (8 = the resolve image), binding 9 the upscale
// input (the composite's linear output when an upscaler runs), binding 10 the sky
// cubemap (sampled image), 11 its sampler.
constexpr uint32_t kCompositeStorageCount = 10;  // bindings 0..9
constexpr uint32_t kCompositeSkyBinding = kCompositeStorageCount;
constexpr uint32_t kCompositeBindingCount = kCompositeSkyBinding + 2;
// Encode set (caelix_upscale_encode, upscaled frames only): binding 0 the host colour target,
// binding 1 the source (the upscaler's output, or the upscale input on the fallback), binding 2
// the internal albedo (coverage in alpha).
constexpr uint32_t kEncodeBindingCount = 3;
// The most bindings of one set (the arrays of CreateSetLayout).
constexpr uint32_t kMaxSetBindings = 32;

// The NRD objects that every NrdExecutor of a device shares: the samplers, the two
// set layouts, the pipeline layout and the pipelines, and the parts of the
// nrd::InstanceDesc they were built from. NRD 4.18's InstanceCreationDesc has no
// resolution (the size goes through CommonSettings::resourceSize per frame), so
// instances with the same denoiser list produce the same pipelines;
// RenderKernels::NrdKernelsFor checks every recorded field of a later instance's
// desc against these.
struct NrdKernels
{
    std::vector<VkSampler> samplers;                           // immutable in set 1
    VkDescriptorSetLayout resources_layout = VK_NULL_HANDLE;  // set 0
    VkDescriptorSetLayout constants_layout = VK_NULL_HANDLE;  // set 1
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    std::vector<VkPipeline> pipelines;  // one per nrd::PipelineDesc, in desc order
    uint32_t texture_binding_base = 0;  // set 0: first SAMPLED_IMAGE binding
    uint32_t storage_binding_base = 0;  // set 0: first STORAGE_IMAGE binding
    uint32_t constants_binding = 0;     // set 1: the UNIFORM_BUFFER_DYNAMIC binding

    // The desc the objects were built from (the fields a later instance must match).
    uint32_t pipelines_num = 0;
    std::vector<std::string> shader_identifiers;  // pipelines[i].shaderIdentifier
    uint32_t samplers_num = 0;
    std::vector<nrd::Sampler> sampler_kinds;
    nrd::DescriptorPoolDesc descriptor_pool_desc{};
    uint32_t constant_buffer_max_data_size = 0;
    uint32_t resources_base_register_index = 0;
    uint32_t constant_buffer_register_index = 0;
    uint32_t samplers_base_register_index = 0;
    uint32_t resources_space_index = 0;
    uint32_t constant_buffer_and_samplers_space_index = 0;
    std::string shader_entry_point;
};

// The immutable GPU objects of the render kernels, created once per adopted device
// and borrowed by every Renderer and NrdExecutor of that device:
// - the descriptor set layouts of the trace set, the composite set (also used by the
//   cross resolve kernel), the sky SH set (binding 0 the cubemap, 1 the sampler), the
//   encode set and the frame set (one uniform buffer), every binding PARTIALLY_BOUND;
// - the compute pipelines caelix_path_trace, caelix_path_trace_no_light_nee, caelix_voxel_id
//   (trace set), caelix_composite, caelix_cross_resolve, caelix_sky_sh and caelix_upscale_encode
//   (set 0 their own set, set kFrameSetIndex the frame set; no push constants);
// - the sky sampler (linear, mipmapped, clamp to edge) and the fallback sky cube (1 x 1
//   RGBA8), bound in every sky binding when the host gives no cubemap. The first render
//   call on the device moves the cube from UNDEFINED to GENERAL (TakeFallbackSkyInit);
// - the NRD objects (NrdKernels), built at the first NrdKernelsFor call.
// Every library call on one device runs on one host thread (the ABI thread rule), so
// nothing here locks. Destroy it only when the device is idle.
class RenderKernels
{
public:
    // Throws VulkanError (Vulkan failures) or the shader loader's errors.
    explicit RenderKernels(Device& device);
    ~RenderKernels();
    RenderKernels(const RenderKernels&) = delete;
    RenderKernels& operator=(const RenderKernels&) = delete;

    VkDescriptorSetLayout trace_layout() const
    {
        return trace_layout_;
    }
    VkDescriptorSetLayout composite_layout() const
    {
        return composite_layout_;
    }
    VkDescriptorSetLayout sky_layout() const
    {
        return sky_layout_;
    }
    VkDescriptorSetLayout encode_layout() const
    {
        return encode_layout_;
    }
    VkDescriptorSetLayout frame_layout() const
    {
        return frame_layout_;
    }
    // The trace kernel with the light NEE (frames that run it) and its variant without it (every frame
    // without light NEE; shaders/render/caelix_path_trace_no_light_nee.hlsl).
    const ComputePipeline& trace() const
    {
        return trace_;
    }
    const ComputePipeline& trace_no_light_nee() const
    {
        return trace_no_light_nee_;
    }
    // The voxel id target at the output size (frames with an upscaler and a voxel id target;
    // shaders/render/caelix_voxel_id.hlsl, trace set).
    const ComputePipeline& voxel_id() const
    {
        return voxel_id_;
    }
    const ComputePipeline& composite() const
    {
        return composite_;
    }
    const ComputePipeline& cross_resolve() const
    {
        return cross_resolve_;
    }
    const ComputePipeline& sky_sh() const
    {
        return sky_sh_;
    }
    const ComputePipeline& upscale_encode() const
    {
        return upscale_encode_;
    }
    VkSampler sky_sampler() const
    {
        return sky_sampler_;
    }
    const GpuImage& fallback_sky() const
    {
        return fallback_sky_;
    }
    // True once: the caller records the UNDEFINED -> GENERAL barrier of fallback_sky()
    // into its command buffer.
    bool TakeFallbackSkyInit()
    {
        const bool init = fallback_sky_needs_init_;
        fallback_sky_needs_init_ = false;
        return init;
    }

    // The shared NRD objects. The first call builds them from `desc` (throws NrdError for
    // NRD failures, VulkanError for Vulkan failures; nothing is kept then, and the next
    // call tries again); a later call checks that `desc` matches the recorded fields and
    // throws NrdError("NRD instance desc differs from the shared NRD kernels") on a
    // mismatch (the first differing field is logged).
    const NrdKernels& NrdKernelsFor(const nrd::InstanceDesc& desc);

private:
    void BuildNrdKernels(const nrd::InstanceDesc& desc, NrdKernels& out);
    void DestroyNrdKernels(NrdKernels& kernels);

    Device& device_;
    VkDescriptorSetLayout trace_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout composite_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sky_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout encode_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout frame_layout_ = VK_NULL_HANDLE;  // the frame constants (uniform buffer)
    ComputePipeline trace_;                   // shaders/render/caelix_path_trace.hlsl, trace set
    ComputePipeline trace_no_light_nee_;      // caelix_path_trace_no_light_nee.hlsl, trace set
    ComputePipeline voxel_id_;                // shaders/render/caelix_voxel_id.hlsl, trace set
    ComputePipeline composite_;               // shaders/render/caelix_composite.hlsl, composite set
    ComputePipeline cross_resolve_;           // shaders/render/caelix_cross_resolve.hlsl, composite set
    ComputePipeline sky_sh_;                  // shaders/render/caelix_sky_sh.hlsl, sky set
    ComputePipeline upscale_encode_;          // shaders/render/caelix_upscale_encode.hlsl, encode set
    VkSampler sky_sampler_ = VK_NULL_HANDLE;  // linear, mipmapped, clamp to edge
    GpuImage fallback_sky_;                   // 1 x 1 RGBA8 cube bound without a host cubemap
    bool fallback_sky_needs_init_ = true;     // fallback_sky_ still in VK_IMAGE_LAYOUT_UNDEFINED
    std::unique_ptr<NrdKernels> nrd_;         // null until the first NrdKernelsFor
};

}  // namespace caelix
