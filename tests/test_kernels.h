// Host-visible buffers with a device address and compute kernels with push constants only, shared
// by the GPU tests that run probe kernels or the library's own kernels from the build's shader
// folder (light_tree_test.cpp, hash_table_test.cpp, light_nee_test.cpp).
#pragma once
#include "test_device.h"

#include <cstdint>

namespace test {

uint32_t FindMemoryType(Gpu& gpu, uint32_t type_bits, VkMemoryPropertyFlags wanted);

// A host-visible, coherent, persistently mapped buffer with a device address (storage and indirect
// usage). The size is rounded up to 16 bytes.
struct HostBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    uint8_t* data = nullptr;
    VkDeviceSize size = 0;

    template <typename T>
    T* As() const
    {
        return reinterpret_cast<T*>(data);
    }
};

HostBuffer CreateHostBuffer(Gpu& gpu, VkDeviceSize size);
void DestroyHostBuffer(Gpu& gpu, HostBuffer& buffer);

// A compute kernel <shader_dir>/<name>.spv with push constants only.
class ComputeKernel
{
public:
    ComputeKernel(Gpu& gpu, const char* shader_dir, const char* name, uint32_t push_bytes);
    ~ComputeKernel();
    ComputeKernel(const ComputeKernel&) = delete;
    ComputeKernel& operator=(const ComputeKernel&) = delete;

    // The dispatch alone.
    void Record(VkCommandBuffer cmd, const void* push, uint32_t groups) const;
    // vkCmdDispatchIndirect of the VkDispatchIndirectCommand at `buffer` + `offset`.
    void RecordIndirect(VkCommandBuffer cmd, const void* push, VkBuffer buffer, VkDeviceSize offset) const;
    // The dispatch, then compute writes -> compute (and indirect) and host reads.
    void Dispatch(VkCommandBuffer cmd, const void* push, uint32_t groups) const;

private:
    Gpu& gpu_;
    uint32_t push_bytes_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// Compute and transfer writes -> compute reads and writes, indirect reads and host reads.
void ComputeBarrier(VkCommandBuffer cmd);

}  // namespace test
