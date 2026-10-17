#pragma once
#include "rhi/device.h"
#include "rhi/gpu_buffer.h"
#include "rhi/gpu_image.h"

#include <NRD.h>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace caelix {

class RenderKernels;
struct NrdKernels;

// An error reported by the NRD library itself (nrd::Result), as opposed to a
// Vulkan error (VulkanError). The renderer stops denoising after one.
class NrdError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Runs NVIDIA NRD (Real-Time Denoisers) on Vulkan without NRI: the library gives
// a list of compute dispatches with their resources and constants
// (nrd::GetComputeDispatches); this class owns the per-instance Vulkan objects
// behind them and records them.
//
// Per device (NrdKernels, built by RenderKernels::NrdKernelsFor from the first
// instance's nrd::InstanceDesc and nrd::GetLibraryDesc, never hard-coded; every
// executor of the device borrows them, and a later instance's desc must match):
// - One pipeline layout shared by every NRD pipeline (NRD's recommendation for a
//   shared root signature). Set 0 (NRD's resources space): perSetTexturesMaxNum
//   SAMPLED_IMAGE bindings from spirvBindingOffsets.textureOffset +
//   resourcesBaseRegisterIndex and perSetStorageTexturesMaxNum STORAGE_IMAGE
//   bindings from storageTextureAndBufferOffset + resourcesBaseRegisterIndex, all
//   PARTIALLY_BOUND. Set 1 (constant buffer and samplers space): the immutable
//   samplers at samplerOffset + samplersBaseRegisterIndex + i (nearest or linear,
//   CLAMP_TO_EDGE) and one UNIFORM_BUFFER_DYNAMIC at constantBufferOffset +
//   constantBufferRegisterIndex.
// - One compute pipeline per nrd::PipelineDesc (computeShaderSPIRV, entry point
//   InstanceDesc::shaderEntryPoint).
//
// Per instance (owned by the executor, recreated with it on a size change):
// - The nrd::Instance and its desc.
// - One GpuImage per pool TextureDesc (permanent and transient), SAMPLED |
//   STORAGE, size DivideUp(size, downsampleFactor). Every pool image is moved to
//   VK_IMAGE_LAYOUT_GENERAL once (at the first Denoise) and stays there, for both
//   sampled and storage use: the executor needs no layout tracking, and GENERAL
//   costs nothing on NVIDIA. The user images (Bindings) must be in GENERAL too.
// - A host-visible, persistently mapped constant buffer ring:
//   align(constantBufferMaxDataSize, minUniformBufferOffsetAlignment) * setsMaxNum
//   per frame slot, kMaxSlots slots. Set 1 is written once (the whole ring) and
//   bound with a dynamic offset per dispatch, so it is allocated once, not per
//   frame; its descriptor pool is the executor's.
// - One descriptor pool per frame slot (setsMaxNum sets of the set 0 layout),
//   created at the slot's first use and reset at the start of every Denoise with
//   that slot. The caller reuses a slot only when its previous frame is complete.
//
// Per dispatch: one set 0 filled from DispatchDesc::resources (the resources of
// the pipeline's ranges in range order; a TEXTURE range goes to the sampled image
// bindings, a STORAGE_TEXTURE range to the storage image bindings; PERMANENT_POOL
// and TRANSIENT_POOL resolve to the pool image `indexInPool`, every other type to
// the caller's Bindings), then one global memory barrier (compute shader writes ->
// compute shader sampled reads, storage reads and writes) and the dispatch of
// (gridWidth, gridHeight, 1) groups. The barrier is conservative: a dispatch that
// does not read the previous dispatch's outputs still waits for it. Finer barriers
// (per resource, only on a read-after-write) would let independent dispatches
// overlap; not done yet. After the last dispatch: one barrier compute writes ->
// compute reads (the caller's composite).
//
// Errors: a failing nrd:: call is logged at error level and makes failed() true;
// the call then returns false and records nothing. The constructor throws NrdError
// for NRD failures (including a desc that does not match the shared NrdKernels) and
// VulkanError for Vulkan failures. Destroy the executor only when no pending GPU
// work uses it; the RenderKernels must outlive it.
class NrdExecutor
{
public:
    // Frame slots of the constant ring and the descriptor pools (the renderer's maximum).
    static constexpr uint32_t kMaxSlots = 8;

    // The caller's images by nrd::ResourceType; each view must be usable as a sampled
    // and a storage image, in VK_IMAGE_LAYOUT_GENERAL.
    struct Bindings
    {
        VkImageView view_z = VK_NULL_HANDLE;            // IN_VIEWZ
        VkImageView normal_roughness = VK_NULL_HANDLE;  // IN_NORMAL_ROUGHNESS
        VkImageView motion = VK_NULL_HANDLE;            // IN_MV
        VkImageView diff_in = VK_NULL_HANDLE;           // IN_DIFF_RADIANCE_HITDIST
        VkImageView spec_in = VK_NULL_HANDLE;           // IN_SPEC_RADIANCE_HITDIST
        VkImageView diff_out = VK_NULL_HANDLE;          // OUT_DIFF_RADIANCE_HITDIST
        VkImageView spec_out = VK_NULL_HANDLE;          // OUT_SPEC_RADIANCE_HITDIST
        VkImageView validation = VK_NULL_HANDLE;        // OUT_VALIDATION
    };

    NrdExecutor(Device& device, RenderKernels& kernels, uint32_t width, uint32_t height,
                std::span<const nrd::DenoiserDesc> denoisers);
    ~NrdExecutor();
    NrdExecutor(const NrdExecutor&) = delete;
    NrdExecutor& operator=(const NrdExecutor&) = delete;

    // nrd::SetCommonSettings once per frame. False on an NRD error (logged, failed()).
    bool SetCommonSettings(const nrd::CommonSettings& settings);
    // nrd::SetDenoiserSettings for a REBLUR denoiser. False on an NRD error.
    bool SetReblurSettings(nrd::Identifier identifier, const nrd::ReblurSettings& settings);
    // nrd::GetComputeDispatches for `identifiers` and records them into `cmd` with the
    // resources of frame slot `frame_slot` (< kMaxSlots). False (nothing recorded)
    // when failed() is or becomes true.
    bool Denoise(VkCommandBuffer cmd, uint32_t frame_slot, const Bindings& bindings,
                 std::span<const nrd::Identifier> identifiers);

    bool failed() const
    {
        return failed_;
    }
    uint32_t width() const
    {
        return width_;
    }
    uint32_t height() const
    {
        return height_;
    }
    const nrd::InstanceDesc& desc() const
    {
        return *desc_;
    }
    static const nrd::LibraryDesc& lib()
    {
        return *nrd::GetLibraryDesc();
    }
    uint32_t permanent_count() const
    {
        return static_cast<uint32_t>(permanent_.size());
    }
    uint32_t transient_count() const
    {
        return static_cast<uint32_t>(transient_.size());
    }
    VkDeviceSize permanent_bytes() const;
    VkDeviceSize transient_bytes() const;
    // Dispatches recorded by the newest Denoise.
    uint32_t last_dispatch_count() const
    {
        return last_dispatch_count_;
    }

private:
    void Release();
    void Fail(const char* call, nrd::Result result);
    VkDescriptorPool SlotPool(uint32_t frame_slot);
    // The image view of one dispatch resource, or VK_NULL_HANDLE for a type the
    // executor does not provide.
    VkImageView Resolve(const nrd::ResourceDesc& resource, const Bindings& bindings) const;

    Device& device_;
    const NrdKernels* kernels_ = nullptr;  // the device's shared NRD objects (RenderKernels)
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    nrd::Instance* instance_ = nullptr;
    const nrd::InstanceDesc* desc_ = nullptr;
    std::vector<GpuImage> permanent_;
    std::vector<GpuImage> transient_;
    bool needs_init_ = true;             // pool images still in VK_IMAGE_LAYOUT_UNDEFINED
    GpuBuffer constants_;                // the constant ring
    VkDeviceSize constants_stride_ = 0;  // aligned constant block size
    VkDescriptorPool constants_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet constants_set_ = VK_NULL_HANDLE;
    std::array<VkDescriptorPool, kMaxSlots> slot_pools_{};
    uint32_t last_dispatch_count_ = 0;
    bool failed_ = false;
};

}  // namespace caelix
