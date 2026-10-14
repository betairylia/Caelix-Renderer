#include "nrd_executor.h"

#include "log.h"

#include <algorithm>
#include <cstring>
#include <string>

namespace caelix {
namespace {

const VkImageSubresourceRange kColorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

const char* ResultName(nrd::Result result) {
  switch (result) {
    case nrd::Result::SUCCESS: return "SUCCESS";
    case nrd::Result::FAILURE: return "FAILURE";
    case nrd::Result::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case nrd::Result::UNSUPPORTED: return "UNSUPPORTED";
    case nrd::Result::NON_UNIQUE_IDENTIFIER: return "NON_UNIQUE_IDENTIFIER";
    default: return "unknown nrd::Result";
  }
}

// nrd::Format -> VkFormat, the whole enum. NRD's R10_G10_B10_A2 / R11_G11_B10 /
// R9_G9_B9_E5 name the components from the least significant bit, Vulkan's
// packed formats from the most significant one.
VkFormat ToVkFormat(nrd::Format format) {
  switch (format) {
    case nrd::Format::R8_UNORM: return VK_FORMAT_R8_UNORM;
    case nrd::Format::R8_SNORM: return VK_FORMAT_R8_SNORM;
    case nrd::Format::R8_UINT: return VK_FORMAT_R8_UINT;
    case nrd::Format::R8_SINT: return VK_FORMAT_R8_SINT;
    case nrd::Format::RG8_UNORM: return VK_FORMAT_R8G8_UNORM;
    case nrd::Format::RG8_SNORM: return VK_FORMAT_R8G8_SNORM;
    case nrd::Format::RG8_UINT: return VK_FORMAT_R8G8_UINT;
    case nrd::Format::RG8_SINT: return VK_FORMAT_R8G8_SINT;
    case nrd::Format::RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case nrd::Format::RGBA8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case nrd::Format::RGBA8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case nrd::Format::RGBA8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case nrd::Format::RGBA8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case nrd::Format::R16_UNORM: return VK_FORMAT_R16_UNORM;
    case nrd::Format::R16_SNORM: return VK_FORMAT_R16_SNORM;
    case nrd::Format::R16_UINT: return VK_FORMAT_R16_UINT;
    case nrd::Format::R16_SINT: return VK_FORMAT_R16_SINT;
    case nrd::Format::R16_SFLOAT: return VK_FORMAT_R16_SFLOAT;
    case nrd::Format::RG16_UNORM: return VK_FORMAT_R16G16_UNORM;
    case nrd::Format::RG16_SNORM: return VK_FORMAT_R16G16_SNORM;
    case nrd::Format::RG16_UINT: return VK_FORMAT_R16G16_UINT;
    case nrd::Format::RG16_SINT: return VK_FORMAT_R16G16_SINT;
    case nrd::Format::RG16_SFLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case nrd::Format::RGBA16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case nrd::Format::RGBA16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case nrd::Format::RGBA16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case nrd::Format::RGBA16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    case nrd::Format::RGBA16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case nrd::Format::R32_UINT: return VK_FORMAT_R32_UINT;
    case nrd::Format::R32_SINT: return VK_FORMAT_R32_SINT;
    case nrd::Format::R32_SFLOAT: return VK_FORMAT_R32_SFLOAT;
    case nrd::Format::RG32_UINT: return VK_FORMAT_R32G32_UINT;
    case nrd::Format::RG32_SINT: return VK_FORMAT_R32G32_SINT;
    case nrd::Format::RG32_SFLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case nrd::Format::RGB32_UINT: return VK_FORMAT_R32G32B32_UINT;
    case nrd::Format::RGB32_SINT: return VK_FORMAT_R32G32B32_SINT;
    case nrd::Format::RGB32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case nrd::Format::RGBA32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
    case nrd::Format::RGBA32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
    case nrd::Format::RGBA32_SFLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case nrd::Format::R10_G10_B10_A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case nrd::Format::R10_G10_B10_A2_UINT: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case nrd::Format::R11_G11_B10_UFLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case nrd::Format::R9_G9_B9_E5_UFLOAT: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    default: return VK_FORMAT_UNDEFINED;
  }
}

uint32_t DivideUp(uint32_t value, uint32_t divisor) { return (value + divisor - 1) / divisor; }

void Check(VkResult result, const char* call) {
  if (result != VK_SUCCESS) {
    CAELIX_LOGE("NRD executor: %s failed: %d", call, static_cast<int>(result));
    throw VulkanError(call, result);
  }
}

}  // namespace

NrdExecutor::NrdExecutor(Device& device, uint32_t width, uint32_t height,
                         std::span<const nrd::DenoiserDesc> denoisers)
    : device_(device), width_(width), height_(height) {
  const VkDevice vk_device = device.device();
  try {
    const nrd::LibraryDesc& library = lib();
    // The kernels pack normals and roughness with the NRD.hlsli of the same build
    // (NRDConfig.hlsli); the library must agree.
    if (library.normalEncoding != nrd::NormalEncoding::R10_G10_B10_A2_UNORM ||
        library.roughnessEncoding != nrd::RoughnessEncoding::LINEAR) {
      throw NrdError("NRD library built with an unexpected normal or roughness encoding");
    }
    if (width > UINT16_MAX || height > UINT16_MAX) {
      throw NrdError("NRD resource sizes are 16 bit");
    }
    nrd::InstanceCreationDesc creation{};
    creation.denoisers = denoisers.data();
    creation.denoisersNum = static_cast<uint32_t>(denoisers.size());
    const nrd::Result created = nrd::CreateInstance(creation, instance_);
    if (created != nrd::Result::SUCCESS || instance_ == nullptr) {
      instance_ = nullptr;
      CAELIX_LOGE("nrd::CreateInstance failed: %s", ResultName(created));
      throw NrdError("nrd::CreateInstance failed");
    }
    desc_ = nrd::GetInstanceDesc(*instance_);
    const nrd::InstanceDesc& desc = *desc_;
    const nrd::SPIRVBindingOffsets& offsets = library.spirvBindingOffsets;

    // Samplers, immutable in set 1.
    for (uint32_t i = 0; i < desc.samplersNum; ++i) {
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
      samplers_.push_back(sampler);
    }

    // Set 0: sampled and storage images, every binding PARTIALLY_BOUND.
    const nrd::DescriptorPoolDesc& pool_desc = desc.descriptorPoolDesc;
    texture_binding_base_ = offsets.textureOffset + desc.resourcesBaseRegisterIndex;
    storage_binding_base_ = offsets.storageTextureAndBufferOffset + desc.resourcesBaseRegisterIndex;
    {
      std::vector<VkDescriptorSetLayoutBinding> bindings;
      for (uint32_t i = 0; i < pool_desc.perSetTexturesMaxNum; ++i) {
        bindings.push_back({texture_binding_base_ + i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                            VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
      }
      for (uint32_t i = 0; i < pool_desc.perSetStorageTexturesMaxNum; ++i) {
        bindings.push_back({storage_binding_base_ + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
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
      Check(vkCreateDescriptorSetLayout(vk_device, &info, nullptr, &resources_layout_),
            "vkCreateDescriptorSetLayout (NRD resources)");
    }
    // Set 1: immutable samplers and the dynamic constant buffer.
    const uint32_t constants_binding = offsets.constantBufferOffset + desc.constantBufferRegisterIndex;
    {
      std::vector<VkDescriptorSetLayoutBinding> bindings;
      for (uint32_t i = 0; i < desc.samplersNum; ++i) {
        bindings.push_back({offsets.samplerOffset + desc.samplersBaseRegisterIndex + i,
                            VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                            &samplers_[i]});
      }
      bindings.push_back({constants_binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                          VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
      VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      info.bindingCount = static_cast<uint32_t>(bindings.size());
      info.pBindings = bindings.data();
      Check(vkCreateDescriptorSetLayout(vk_device, &info, nullptr, &constants_layout_),
            "vkCreateDescriptorSetLayout (NRD constants)");
    }
    // NRD's register spaces are the descriptor set indices.
    if (desc.resourcesSpaceIndex != 0 || desc.constantBufferAndSamplersSpaceIndex != 1) {
      throw NrdError("NRD register spaces are not (resources 0, constants 1)");
    }
    {
      const VkDescriptorSetLayout layouts[2] = {resources_layout_, constants_layout_};
      VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      info.setLayoutCount = 2;
      info.pSetLayouts = layouts;
      Check(vkCreatePipelineLayout(vk_device, &info, nullptr, &pipeline_layout_),
            "vkCreatePipelineLayout (NRD)");
    }

    // Pipelines, created in one call.
    {
      std::vector<VkShaderModule> modules(desc.pipelinesNum, VK_NULL_HANDLE);
      std::vector<VkComputePipelineCreateInfo> infos(desc.pipelinesNum);
      VkResult result = VK_SUCCESS;
      for (uint32_t i = 0; i < desc.pipelinesNum && result == VK_SUCCESS; ++i) {
        const nrd::ComputeShaderDesc& spirv = desc.pipelines[i].computeShaderSPIRV;
        if (spirv.bytecode == nullptr || spirv.size == 0) {
          for (VkShaderModule module : modules) {
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
        info.layout = pipeline_layout_;
      }
      if (result == VK_SUCCESS) {
        pipelines_.assign(desc.pipelinesNum, VK_NULL_HANDLE);
        result = vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, desc.pipelinesNum,
                                          infos.data(), nullptr, pipelines_.data());
      }
      for (VkShaderModule module : modules) {
        vkDestroyShaderModule(vk_device, module, nullptr);
      }
      Check(result, "vkCreateComputePipelines (NRD)");
    }

    // Pools.
    const auto create_pool = [&](const nrd::TextureDesc* textures, uint32_t count, const char* kind,
                                 std::vector<GpuImage>& out) {
      for (uint32_t i = 0; i < count; ++i) {
        const VkFormat format = ToVkFormat(textures[i].format);
        if (format == VK_FORMAT_UNDEFINED) {
          throw NrdError("NRD pool texture with an unknown nrd::Format");
        }
        const uint32_t factor = std::max<uint32_t>(1u, textures[i].downsampleFactor);
        const std::string name = std::string("NRD ") + kind + " pool " + std::to_string(i);
        out.emplace_back(device.allocator(), vk_device, DivideUp(width, factor),
                         DivideUp(height, factor), format,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, name.c_str());
      }
    };
    create_pool(desc.permanentPool, desc.permanentPoolSize, "permanent", permanent_);
    create_pool(desc.transientPool, desc.transientPoolSize, "transient", transient_);

    // Constant ring: one aligned block per set, setsMaxNum sets per frame slot.
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device(), &properties);
    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        1, properties.limits.minUniformBufferOffsetAlignment);
    constants_stride_ =
        (static_cast<VkDeviceSize>(std::max(desc.constantBufferMaxDataSize, 16u)) + alignment - 1) /
        alignment * alignment;
    if (constants_stride_ > properties.limits.maxUniformBufferRange) {
      throw NrdError("NRD constant buffer is larger than maxUniformBufferRange");
    }
    {
      VmaAllocationCreateInfo allocation{};
      allocation.usage = VMA_MEMORY_USAGE_AUTO;
      allocation.flags =
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
      constants_ = GpuBuffer(device.allocator(), vk_device,
                             constants_stride_ * pool_desc.setsMaxNum * kMaxSlots,
                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, allocation, false);
    }
    {
      const VkDescriptorPoolSize sizes[2] = {
          {VK_DESCRIPTOR_TYPE_SAMPLER, std::max(1u, desc.samplersNum)},
          {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
      };
      VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      info.maxSets = 1;
      info.poolSizeCount = 2;
      info.pPoolSizes = sizes;
      Check(vkCreateDescriptorPool(vk_device, &info, nullptr, &constants_pool_),
            "vkCreateDescriptorPool (NRD constants)");
      VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      allocate.descriptorPool = constants_pool_;
      allocate.descriptorSetCount = 1;
      allocate.pSetLayouts = &constants_layout_;
      Check(vkAllocateDescriptorSets(vk_device, &allocate, &constants_set_),
            "vkAllocateDescriptorSets (NRD constants)");
      const VkDescriptorBufferInfo buffer{constants_.buffer(), 0, constants_stride_};
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = constants_set_;
      write.dstBinding = constants_binding;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      write.pBufferInfo = &buffer;
      vkUpdateDescriptorSets(vk_device, 1, &write, 0, nullptr);
    }
  } catch (...) {
    Release();
    throw;
  }
  CAELIX_LOGI("NRD %u.%u.%u instance created at %ux%u: %u pipelines, %u permanent textures "
              "(%.1f MiB), %u transient textures (%.1f MiB), constant block %llu bytes",
              lib().versionMajor, lib().versionMinor, lib().versionBuild, width, height,
              desc_->pipelinesNum, permanent_count(),
              static_cast<double>(permanent_bytes()) / (1024.0 * 1024.0), transient_count(),
              static_cast<double>(transient_bytes()) / (1024.0 * 1024.0),
              static_cast<unsigned long long>(constants_stride_));
}

NrdExecutor::~NrdExecutor() { Release(); }

void NrdExecutor::Release() {
  const VkDevice device = device_.device();
  for (VkDescriptorPool& pool : slot_pools_) {
    if (pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, pool, nullptr);
      pool = VK_NULL_HANDLE;
    }
  }
  if (constants_pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, constants_pool_, nullptr);  // frees constants_set_
    constants_pool_ = VK_NULL_HANDLE;
    constants_set_ = VK_NULL_HANDLE;
  }
  for (VkPipeline pipeline : pipelines_) {
    if (pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, pipeline, nullptr);
    }
  }
  pipelines_.clear();
  if (pipeline_layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
    pipeline_layout_ = VK_NULL_HANDLE;
  }
  if (resources_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, resources_layout_, nullptr);
    resources_layout_ = VK_NULL_HANDLE;
  }
  if (constants_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, constants_layout_, nullptr);
    constants_layout_ = VK_NULL_HANDLE;
  }
  for (VkSampler sampler : samplers_) {
    vkDestroySampler(device, sampler, nullptr);
  }
  samplers_.clear();
  permanent_.clear();
  transient_.clear();
  constants_.Reset();
  if (instance_ != nullptr) {
    nrd::DestroyInstance(*instance_);
    instance_ = nullptr;
    desc_ = nullptr;
  }
}

VkDeviceSize NrdExecutor::permanent_bytes() const {
  VkDeviceSize bytes = 0;
  for (const GpuImage& image : permanent_) {
    bytes += image.allocation_bytes();
  }
  return bytes;
}

VkDeviceSize NrdExecutor::transient_bytes() const {
  VkDeviceSize bytes = 0;
  for (const GpuImage& image : transient_) {
    bytes += image.allocation_bytes();
  }
  return bytes;
}

void NrdExecutor::Fail(const char* call, nrd::Result result) {
  CAELIX_LOGE("%s failed: %s; denoising is disabled for this renderer", call, ResultName(result));
  failed_ = true;
}

bool NrdExecutor::SetCommonSettings(const nrd::CommonSettings& settings) {
  if (failed_) {
    return false;
  }
  const nrd::Result result = nrd::SetCommonSettings(*instance_, settings);
  if (result != nrd::Result::SUCCESS) {
    Fail("nrd::SetCommonSettings", result);
    return false;
  }
  return true;
}

bool NrdExecutor::SetReblurSettings(nrd::Identifier identifier, const nrd::ReblurSettings& settings) {
  if (failed_) {
    return false;
  }
  const nrd::Result result = nrd::SetDenoiserSettings(*instance_, identifier, &settings);
  if (result != nrd::Result::SUCCESS) {
    Fail("nrd::SetDenoiserSettings", result);
    return false;
  }
  return true;
}

VkDescriptorPool NrdExecutor::SlotPool(uint32_t frame_slot) {
  VkDescriptorPool& pool = slot_pools_[frame_slot];
  if (pool != VK_NULL_HANDLE) {
    Check(vkResetDescriptorPool(device_.device(), pool, 0), "vkResetDescriptorPool (NRD)");
    return pool;
  }
  const nrd::DescriptorPoolDesc& pool_desc = desc_->descriptorPoolDesc;
  // A set of a PARTIALLY_BOUND layout still takes every binding of the layout from the pool.
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
       std::max(1u, pool_desc.setsMaxNum * pool_desc.perSetTexturesMaxNum)},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       std::max(1u, pool_desc.setsMaxNum * pool_desc.perSetStorageTexturesMaxNum)},
  };
  VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  info.maxSets = pool_desc.setsMaxNum;
  info.poolSizeCount = 2;
  info.pPoolSizes = sizes;
  Check(vkCreateDescriptorPool(device_.device(), &info, nullptr, &pool),
        "vkCreateDescriptorPool (NRD slot)");
  return pool;
}

VkImageView NrdExecutor::Resolve(const nrd::ResourceDesc& resource, const Bindings& bindings) const {
  switch (resource.type) {
    case nrd::ResourceType::PERMANENT_POOL:
      return resource.indexInPool < permanent_.size() ? permanent_[resource.indexInPool].view()
                                                      : VK_NULL_HANDLE;
    case nrd::ResourceType::TRANSIENT_POOL:
      return resource.indexInPool < transient_.size() ? transient_[resource.indexInPool].view()
                                                      : VK_NULL_HANDLE;
    case nrd::ResourceType::IN_MV: return bindings.motion;
    case nrd::ResourceType::IN_NORMAL_ROUGHNESS: return bindings.normal_roughness;
    case nrd::ResourceType::IN_VIEWZ: return bindings.view_z;
    case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST: return bindings.diff_in;
    case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST: return bindings.spec_in;
    case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST: return bindings.diff_out;
    case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST: return bindings.spec_out;
    case nrd::ResourceType::OUT_VALIDATION: return bindings.validation;
    default: return VK_NULL_HANDLE;
  }
}

bool NrdExecutor::Denoise(VkCommandBuffer cmd, uint32_t frame_slot, const Bindings& bindings,
                          std::span<const nrd::Identifier> identifiers) {
  if (failed_) {
    return false;
  }
  if (frame_slot >= kMaxSlots) {
    CAELIX_LOGE("NRD executor: frame slot %u is out of range (%u slots)", frame_slot, kMaxSlots);
    failed_ = true;
    return false;
  }
  const nrd::DispatchDesc* dispatches = nullptr;
  uint32_t dispatch_count = 0;
  const nrd::Result result =
      nrd::GetComputeDispatches(*instance_, identifiers.data(),
                                static_cast<uint32_t>(identifiers.size()), dispatches, dispatch_count);
  if (result != nrd::Result::SUCCESS) {
    Fail("nrd::GetComputeDispatches", result);
    return false;
  }
  const nrd::InstanceDesc& desc = *desc_;
  if (dispatch_count > desc.descriptorPoolDesc.setsMaxNum) {
    CAELIX_LOGE("NRD executor: %u dispatches, but setsMaxNum is %u", dispatch_count,
                desc.descriptorPoolDesc.setsMaxNum);
    failed_ = true;
    return false;
  }

  // Descriptor writes of every dispatch, checked before anything is recorded.
  const VkDevice device = device_.device();
  const VkDescriptorPool pool = SlotPool(frame_slot);
  std::vector<VkDescriptorSet> sets(dispatch_count, VK_NULL_HANDLE);
  if (dispatch_count > 0) {
    const std::vector<VkDescriptorSetLayout> layouts(dispatch_count, resources_layout_);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool;
    allocate.descriptorSetCount = dispatch_count;
    allocate.pSetLayouts = layouts.data();
    Check(vkAllocateDescriptorSets(device, &allocate, sets.data()),
          "vkAllocateDescriptorSets (NRD resources)");
  }
  size_t resource_total = 0;
  for (uint32_t i = 0; i < dispatch_count; ++i) {
    resource_total += dispatches[i].resourcesNum;
  }
  std::vector<VkDescriptorImageInfo> image_infos;
  std::vector<VkWriteDescriptorSet> writes;
  image_infos.reserve(resource_total);  // the writes point into it: no reallocation
  writes.reserve(resource_total);
  for (uint32_t i = 0; i < dispatch_count; ++i) {
    const nrd::DispatchDesc& dispatch = dispatches[i];
    const nrd::PipelineDesc& pipeline = desc.pipelines[dispatch.pipelineIndex];
    uint32_t n = 0;  // index into dispatch.resources
    for (uint32_t r = 0; r < pipeline.resourceRangesNum; ++r) {
      const nrd::ResourceRangeDesc& range = pipeline.resourceRanges[r];
      const bool storage = range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;
      for (uint32_t j = 0; j < range.descriptorsNum; ++j, ++n) {
        if (n >= dispatch.resourcesNum) {
          CAELIX_LOGE("NRD executor: dispatch '%s' lists %u resources, its pipeline more",
                      dispatch.name != nullptr ? dispatch.name : "?", dispatch.resourcesNum);
          failed_ = true;
          return false;
        }
        const nrd::ResourceDesc& resource = dispatch.resources[n];
        const VkImageView view = Resolve(resource, bindings);
        if (view == VK_NULL_HANDLE) {
          CAELIX_LOGE("NRD executor: dispatch '%s' needs resource %s (pool index %u), which the "
                      "executor does not provide",
                      dispatch.name != nullptr ? dispatch.name : "?",
                      nrd::GetResourceTypeString(resource.type),
                      static_cast<unsigned>(resource.indexInPool));
          failed_ = true;
          return false;
        }
        image_infos.push_back({VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL});
        VkWriteDescriptorSet& write = writes.emplace_back();
        write = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = sets[i];
        write.dstBinding = (storage ? storage_binding_base_ : texture_binding_base_) + j;
        write.descriptorCount = 1;
        write.descriptorType =
            storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &image_infos.back();
      }
    }
  }
  if (!writes.empty()) {
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  // Constants: this slot's part of the ring; a block is written only when the
  // dispatch's data differs from the previous dispatch's.
  const VkDeviceSize slot_base =
      constants_stride_ * desc.descriptorPoolDesc.setsMaxNum * frame_slot;
  VkDeviceSize used = 0;
  std::vector<uint32_t> offsets(dispatch_count, 0);
  uint32_t current = static_cast<uint32_t>(slot_base);
  for (uint32_t i = 0; i < dispatch_count; ++i) {
    const nrd::DispatchDesc& dispatch = dispatches[i];
    const bool reuse = i > 0 && dispatch.constantBufferDataMatchesPreviousDispatch;
    if (dispatch.constantBufferDataSize > 0 && !reuse) {
      current = static_cast<uint32_t>(slot_base + used);
      std::memcpy(constants_.mapped() + current, dispatch.constantBufferData,
                  std::min<size_t>(dispatch.constantBufferDataSize, constants_stride_));
      used += constants_stride_;
    }
    offsets[i] = current;
  }
  if (used > 0) {
    constants_.FlushHostWrites(slot_base, used);
  }

  // Recording.
  std::vector<VkImageMemoryBarrier2> init_barriers;
  if (needs_init_) {
    for (std::vector<GpuImage>* pool_images : {&permanent_, &transient_}) {
      for (const GpuImage& image : *pool_images) {
        VkImageMemoryBarrier2& barrier = init_barriers.emplace_back();
        barrier = VkImageMemoryBarrier2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        barrier.srcAccessMask = VK_ACCESS_2_NONE;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.image();
        barrier.subresourceRange = kColorRange;
      }
    }
    needs_init_ = false;
  }
  VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  memory.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  for (uint32_t i = 0; i < dispatch_count; ++i) {
    const nrd::DispatchDesc& dispatch = dispatches[i];
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    if (i == 0) {
      dependency.imageMemoryBarrierCount = static_cast<uint32_t>(init_barriers.size());
      dependency.pImageMemoryBarriers = init_barriers.data();
    }
    vkCmdPipelineBarrier2(cmd, &dependency);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[dispatch.pipelineIndex]);
    const VkDescriptorSet bound[2] = {sets[i], constants_set_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 2, bound, 1,
                            &offsets[i]);
    vkCmdDispatch(cmd, dispatch.gridWidth, dispatch.gridHeight, 1);
  }
  // NRD's outputs -> the caller's next compute pass (the composite reads them).
  {
    VkMemoryBarrier2 after{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    after.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    after.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    after.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    after.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &after;
    vkCmdPipelineBarrier2(cmd, &dependency);
  }
  last_dispatch_count_ = dispatch_count;
  return true;
}

}  // namespace caelix
