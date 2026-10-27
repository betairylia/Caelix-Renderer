// The per-device render kernels: set layouts, compute pipelines, the sky sampler and the
// fallback sky cube, and the shared NRD objects.
#include "render/render_kernels.h"

#include "core/log.h"
#include "render/frame_constants.h"
#include "render/nrd_executor.h"
#include "rhi/shader_loader.h"

#include <algorithm>
#include <cstring>

namespace caelix {
namespace {

// A descriptor set layout of `count` bindings, every one PARTIALLY_BOUND, compute
// stage. `types[i]` is the type of binding i.
VkDescriptorSetLayout CreateSetLayout(VkDevice device, uint32_t count,
                                      const VkDescriptorType* types)
{
    VkDescriptorSetLayoutBinding bindings[kMaxSetBindings];
    VkDescriptorBindingFlags binding_flags[kMaxSetBindings];
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

void Check(VkResult result, const char* call)
{
    if (result != VK_SUCCESS)
    {
        CAELIX_LOGE("NRD kernels: %s failed: %d", call, static_cast<int>(result));
        throw VulkanError(call, result);
    }
}

}  // namespace

RenderKernels::RenderKernels(Device& device)
    : device_(device)
{
    static_assert(kTraceBindingCount <= kMaxSetBindings && kCompositeBindingCount <= kMaxSetBindings &&
                      kEncodeBindingCount <= kMaxSetBindings,
                  "CreateSetLayout handles at most kMaxSetBindings bindings");
    const VkDevice vk_device = device.device();
    try
    {
        // Trace set: binding 0 the TLAS, bindings 1..7 the host targets, 8..19 the
        // internal targets, 20 the sky cubemap, 21 its sampler. Every binding is
        // PARTIALLY_BOUND (descriptorBindingPartiallyBound, a device requirement): a render
        // call writes only the TLAS when there is one, the targets that are present, the
        // internal targets in the lighting modes and the sky bindings, and the kernel
        // touches only those.
        VkDescriptorType trace_types[kTraceBindingCount];
        std::fill(trace_types, trace_types + kTraceBindingCount, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        trace_types[0] = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        trace_types[kTraceSkyBinding] = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        trace_types[kTraceSkyBinding + 1] = VK_DESCRIPTOR_TYPE_SAMPLER;
        trace_layout_ = CreateSetLayout(vk_device, kTraceBindingCount, trace_types);
        // Composite set: binding 0 the host colour target, 1..8 internal targets, 9 the upscale
        // input, 10 the sky cubemap, 11 its sampler, 12 the host background. The cross resolve
        // kernel uses the same layout and set.
        VkDescriptorType composite_types[kCompositeBindingCount];
        std::fill(composite_types, composite_types + kCompositeBindingCount,
                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        composite_types[kCompositeSkyBinding] = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        composite_types[kCompositeSkyBinding + 1] = VK_DESCRIPTOR_TYPE_SAMPLER;
        composite_types[kCompositeBackgroundBinding] = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        composite_layout_ = CreateSetLayout(vk_device, kCompositeBindingCount, composite_types);
        // Sky SH set: binding 0 the host's cubemap, 1 the sampler.
        const VkDescriptorType sky_types[2] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                               VK_DESCRIPTOR_TYPE_SAMPLER};
        sky_layout_ = CreateSetLayout(vk_device, 2, sky_types);
        // Encode set: three storage images (host colour, source, internal albedo).
        const VkDescriptorType encode_types[kEncodeBindingCount] = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        encode_layout_ = CreateSetLayout(vk_device, kEncodeBindingCount, encode_types);
        // Frame set: binding 0 the frame slot's FrameConstants (uniform buffer), set kFrameSetIndex of
        // every kernel below. A uniform buffer is read-only, so the compiler may re-read a field
        // where it is used instead of keeping the whole struct live in registers across the traces
        // (as it did for the struct loaded through a buffer device address).
        static_assert(kFrameSetIndex == 1 && kFrameConstantsBinding == 0,
                      "the pipelines below put their own set at 0 and the frame set at 1");
        const VkDescriptorType frame_types[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        frame_layout_ = CreateSetLayout(vk_device, 1, frame_types);

        // No push constants: everything per frame is in the frame constants.
        const auto create = [&](const char* name, VkDescriptorSetLayout own_layout) {
            const std::vector<uint32_t> spirv = LoadShader(name);
            const VkDescriptorSetLayout layouts[2] = {own_layout, frame_layout_};
            return ComputePipeline(vk_device, spirv, 0, name, layouts);
        };
        trace_ = create("caelix_path_trace", trace_layout_);
        trace_no_light_nee_ = create("caelix_path_trace_no_light_nee", trace_layout_);
        voxel_id_ = create("caelix_voxel_id", trace_layout_);
        composite_ = create("caelix_composite", composite_layout_);
        cross_resolve_ = create("caelix_cross_resolve", composite_layout_);
        sky_sh_ = create("caelix_sky_sh", sky_layout_);
        upscale_encode_ = create("caelix_upscale_encode", encode_layout_);

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
        const VkResult sampler_result = vkCreateSampler(vk_device, &sampler_info, nullptr, &sky_sampler_);
        if (sampler_result != VK_SUCCESS)
        {
            sky_sampler_ = VK_NULL_HANDLE;
            CAELIX_LOGE("vkCreateSampler failed: %d", static_cast<int>(sampler_result));
            throw VulkanError("vkCreateSampler failed", sampler_result);
        }
        // Bound in every sky binding when the host gives no cubemap, so that every sampled-image
        // descriptor is valid; never sampled (sky_cubemap = 0). The first render call on the
        // device moves it to GENERAL.
        fallback_sky_ = GpuImage::Cube(device.allocator(), vk_device, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                       VK_IMAGE_USAGE_SAMPLED_BIT, 1, "fallback_sky");
    }
    catch (...)
    {
        // The destructor does not run for a constructor that throws: release the raw handles
        // here (the pipelines and the image destroy themselves).
        trace_ = ComputePipeline();
        trace_no_light_nee_ = ComputePipeline();
        voxel_id_ = ComputePipeline();
        composite_ = ComputePipeline();
        cross_resolve_ = ComputePipeline();
        sky_sh_ = ComputePipeline();
        upscale_encode_ = ComputePipeline();
        if (sky_sampler_ != VK_NULL_HANDLE)
        {
            vkDestroySampler(vk_device, sky_sampler_, nullptr);
        }
        for (const VkDescriptorSetLayout layout :
             {trace_layout_, composite_layout_, sky_layout_, encode_layout_, frame_layout_})
        {
            if (layout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(vk_device, layout, nullptr);
            }
        }
        throw;
    }
    CAELIX_LOGI("render kernels created: caelix_path_trace, caelix_path_trace_no_light_nee, caelix_voxel_id, "
                "caelix_composite, caelix_cross_resolve, caelix_sky_sh, caelix_upscale_encode");
}

RenderKernels::~RenderKernels()
{
    const VkDevice device = device_.device();
    if (nrd_ != nullptr)
    {
        DestroyNrdKernels(*nrd_);
        nrd_.reset();
    }
    trace_ = ComputePipeline();
    trace_no_light_nee_ = ComputePipeline();
    voxel_id_ = ComputePipeline();
    composite_ = ComputePipeline();
    cross_resolve_ = ComputePipeline();
    sky_sh_ = ComputePipeline();
    upscale_encode_ = ComputePipeline();
    if (trace_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, trace_layout_, nullptr);
    }
    if (composite_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, composite_layout_, nullptr);
    }
    if (sky_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, sky_layout_, nullptr);
    }
    if (encode_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, encode_layout_, nullptr);
    }
    if (frame_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, frame_layout_, nullptr);
    }
    if (sky_sampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, sky_sampler_, nullptr);
    }
    // The device is idle (destroy contract): fallback_sky_ is destroyed with the members.
}

const NrdKernels& RenderKernels::NrdKernelsFor(const nrd::InstanceDesc& desc)
{
    if (nrd_ == nullptr)
    {
        auto built = std::make_unique<NrdKernels>();
        try
        {
            BuildNrdKernels(desc, *built);
        }
        catch (...)
        {
            DestroyNrdKernels(*built);
            throw;
        }
        nrd_ = std::move(built);
        CAELIX_LOGI("NRD kernels created: %u pipelines, %u samplers (shared by every NRD instance "
                    "of the device)",
                    nrd_->pipelines_num, nrd_->samplers_num);
        return *nrd_;
    }

    // A later instance: every field the shared objects were built from must match.
    const NrdKernels& k = *nrd_;
    const char* differs = nullptr;
    uint32_t differs_index = 0;  // the entry of an array field
    if (desc.pipelinesNum != k.pipelines_num)
    {
        differs = "pipelinesNum";
    }
    else if (desc.samplersNum != k.samplers_num)
    {
        differs = "samplersNum";
    }
    else if (desc.constantBufferMaxDataSize != k.constant_buffer_max_data_size)
    {
        differs = "constantBufferMaxDataSize";
    }
    else if (desc.resourcesBaseRegisterIndex != k.resources_base_register_index)
    {
        differs = "resourcesBaseRegisterIndex";
    }
    else if (desc.constantBufferRegisterIndex != k.constant_buffer_register_index)
    {
        differs = "constantBufferRegisterIndex";
    }
    else if (desc.samplersBaseRegisterIndex != k.samplers_base_register_index)
    {
        differs = "samplersBaseRegisterIndex";
    }
    else if (desc.resourcesSpaceIndex != k.resources_space_index ||
             desc.constantBufferAndSamplersSpaceIndex != k.constant_buffer_and_samplers_space_index)
    {
        differs = "register space indices";
    }
    else if (desc.shaderEntryPoint == nullptr || k.shader_entry_point != desc.shaderEntryPoint)
    {
        differs = "shaderEntryPoint";
    }
    else if (std::memcmp(&desc.descriptorPoolDesc, &k.descriptor_pool_desc,
                         sizeof(nrd::DescriptorPoolDesc)) != 0)
    {
        differs = "descriptorPoolDesc";
    }
    for (uint32_t i = 0; differs == nullptr && i < k.samplers_num; ++i)
    {
        if (desc.samplers[i] != k.sampler_kinds[i])
        {
            differs = "samplers";
            differs_index = i;
        }
    }
    for (uint32_t i = 0; differs == nullptr && i < k.pipelines_num; ++i)
    {
        if (k.shader_identifiers[i] != desc.pipelines[i].shaderIdentifier)
        {
            differs = "pipelines[].shaderIdentifier";
            differs_index = i;
        }
    }
    if (differs != nullptr)
    {
        CAELIX_LOGE("NRD instance desc differs from the shared NRD kernels: %s (entry %u for an array)",
                    differs, differs_index);
        throw NrdError("NRD instance desc differs from the shared NRD kernels");
    }
    return k;
}

void RenderKernels::BuildNrdKernels(const nrd::InstanceDesc& desc, NrdKernels& out)
{
    const VkDevice vk_device = device_.device();
    const nrd::SPIRVBindingOffsets& offsets = NrdExecutor::lib().spirvBindingOffsets;

    // The fields a later instance must match.
    out.pipelines_num = desc.pipelinesNum;
    out.shader_identifiers.reserve(desc.pipelinesNum);
    for (uint32_t i = 0; i < desc.pipelinesNum; ++i)
    {
        out.shader_identifiers.emplace_back(desc.pipelines[i].shaderIdentifier);
    }
    out.samplers_num = desc.samplersNum;
    out.sampler_kinds.assign(desc.samplers, desc.samplers + desc.samplersNum);
    out.descriptor_pool_desc = desc.descriptorPoolDesc;
    out.constant_buffer_max_data_size = desc.constantBufferMaxDataSize;
    out.resources_base_register_index = desc.resourcesBaseRegisterIndex;
    out.constant_buffer_register_index = desc.constantBufferRegisterIndex;
    out.samplers_base_register_index = desc.samplersBaseRegisterIndex;
    out.resources_space_index = desc.resourcesSpaceIndex;
    out.constant_buffer_and_samplers_space_index = desc.constantBufferAndSamplersSpaceIndex;
    out.shader_entry_point = desc.shaderEntryPoint != nullptr ? desc.shaderEntryPoint : "";

    // Samplers, immutable in set 1.
    for (uint32_t i = 0; i < desc.samplersNum; ++i)
    {
        const bool linear = desc.samplers[i] == nrd::Sampler::LINEAR_CLAMP;
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.magFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        info.minFilter = info.magFilter;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.maxLod = 0.0f;
        VkSampler sampler = VK_NULL_HANDLE;
        Check(vkCreateSampler(vk_device, &info, nullptr, &sampler), "vkCreateSampler");
        out.samplers.push_back(sampler);
    }

    // Set 0: sampled and storage images, every binding PARTIALLY_BOUND.
    const nrd::DescriptorPoolDesc& pool_desc = desc.descriptorPoolDesc;
    out.texture_binding_base = offsets.textureOffset + desc.resourcesBaseRegisterIndex;
    out.storage_binding_base = offsets.storageTextureAndBufferOffset + desc.resourcesBaseRegisterIndex;
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < pool_desc.perSetTexturesMaxNum; ++i)
        {
            bindings.push_back({out.texture_binding_base + i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                                VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        }
        for (uint32_t i = 0; i < pool_desc.perSetStorageTexturesMaxNum; ++i)
        {
            bindings.push_back({out.storage_binding_base + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        }
        const std::vector<VkDescriptorBindingFlags> flags(bindings.size(),
                                                          VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
        VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        flags_info.bindingCount = static_cast<uint32_t>(flags.size());
        flags_info.pBindingFlags = flags.data();
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.pNext = &flags_info;
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        Check(vkCreateDescriptorSetLayout(vk_device, &info, nullptr, &out.resources_layout),
              "vkCreateDescriptorSetLayout (NRD resources)");
    }
    // Set 1: immutable samplers and the dynamic constant buffer.
    out.constants_binding = offsets.constantBufferOffset + desc.constantBufferRegisterIndex;
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < desc.samplersNum; ++i)
        {
            bindings.push_back({offsets.samplerOffset + desc.samplersBaseRegisterIndex + i,
                                VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                &out.samplers[i]});
        }
        bindings.push_back({out.constants_binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                            VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        Check(vkCreateDescriptorSetLayout(vk_device, &info, nullptr, &out.constants_layout),
              "vkCreateDescriptorSetLayout (NRD constants)");
    }
    // NRD's register spaces are the descriptor set indices.
    if (desc.resourcesSpaceIndex != 0 || desc.constantBufferAndSamplersSpaceIndex != 1)
    {
        throw NrdError("NRD register spaces are not (resources 0, constants 1)");
    }
    {
        const VkDescriptorSetLayout layouts[2] = {out.resources_layout, out.constants_layout};
        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 2;
        info.pSetLayouts = layouts;
        Check(vkCreatePipelineLayout(vk_device, &info, nullptr, &out.pipeline_layout),
              "vkCreatePipelineLayout (NRD)");
    }

    // Pipelines, created in one call.
    std::vector<VkShaderModule> modules(desc.pipelinesNum, VK_NULL_HANDLE);
    std::vector<VkComputePipelineCreateInfo> infos(desc.pipelinesNum);
    VkResult result = VK_SUCCESS;
    for (uint32_t i = 0; i < desc.pipelinesNum && result == VK_SUCCESS; ++i)
    {
        const nrd::ComputeShaderDesc& spirv = desc.pipelines[i].computeShaderSPIRV;
        if (spirv.bytecode == nullptr || spirv.size == 0)
        {
            for (VkShaderModule module : modules)
            {
                vkDestroyShaderModule(vk_device, module, nullptr);
            }
            throw NrdError("NRD library has no embedded SPIR-V (NRD_EMBEDS_SPIRV_SHADERS)");
        }
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = static_cast<size_t>(spirv.size);
        module_info.pCode = static_cast<const uint32_t*>(spirv.bytecode);
        result = vkCreateShaderModule(vk_device, &module_info, nullptr, &modules[i]);
        VkComputePipelineCreateInfo& info = infos[i];
        info = VkComputePipelineCreateInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = modules[i];
        info.stage.pName = desc.shaderEntryPoint;
        info.layout = out.pipeline_layout;
    }
    if (result == VK_SUCCESS)
    {
        out.pipelines.assign(desc.pipelinesNum, VK_NULL_HANDLE);
        result = vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, desc.pipelinesNum, infos.data(),
                                          nullptr, out.pipelines.data());
    }
    for (VkShaderModule module : modules)
    {
        vkDestroyShaderModule(vk_device, module, nullptr);
    }
    Check(result, "vkCreateComputePipelines (NRD)");
}

void RenderKernels::DestroyNrdKernels(NrdKernels& kernels)
{
    const VkDevice device = device_.device();
    for (VkPipeline pipeline : kernels.pipelines)
    {
        if (pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
    }
    kernels.pipelines.clear();
    if (kernels.pipeline_layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, kernels.pipeline_layout, nullptr);
        kernels.pipeline_layout = VK_NULL_HANDLE;
    }
    if (kernels.resources_layout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, kernels.resources_layout, nullptr);
        kernels.resources_layout = VK_NULL_HANDLE;
    }
    if (kernels.constants_layout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, kernels.constants_layout, nullptr);
        kernels.constants_layout = VK_NULL_HANDLE;
    }
    for (VkSampler sampler : kernels.samplers)
    {
        vkDestroySampler(device, sampler, nullptr);
    }
    kernels.samplers.clear();
}

}  // namespace caelix
