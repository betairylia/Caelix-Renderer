#pragma once
#include "rhi/device.h"
#include "rhi/gpu_buffer.h"
#include "rhi/gpu_timer.h"
#include "rhi/hash_table_layout.h"
#include "rhi/pipeline.h"

#include <cstdint>
#include <vector>

namespace caelix {

// The reusable GPU hash table (docs/HASH_TABLE.md; first written for the light cells, removed
// 2026-10-24): 64-bit keys -> indices of a payload pool that the user owns and allocates. The table
// owns the key slots, the free stack, the counters and their readback, the active list and the
// indirect dispatch of the user's pass, and the maintenance kernels (shaders/hash/). The kernels use it
// through shaders/common/caelix_hash_table.hlsl; it knows only the payload header contract of
// hash_table_layout.h (state, touched, key words).
//
// Per frame, on one command buffer: RecordMaintenance (evict by age, rebuild the key table from the
// live payloads, write the active list and the dispatch) -> the user's pass over the active list
// (vkCmdDispatchIndirect on dispatch_buffer()) -> the user's lookups (inserts pop from the free
// stack) -> RecordReadback. Resolve reads the counters of frames that are safe.
//
// Key slots: the power of two at or above kHashKeySlotsPerPayload x capacity. Every buffer is
// device-local and addressed through its device address. Destroy the table only when no pending GPU
// work uses it.
class HashTable
{
public:
    struct Desc
    {
        const char* name = "hash table";  // for the log
        uint32_t payload_capacity = 0;    // 1 .. kHashMaxPayloads
        uint32_t payload_stride = 0;      // bytes per payload: a multiple of 4 that holds the header
        uint32_t key_words = 0;           // 1 .. kHashMaxKeyWords
    };

    // Allocates the buffers and creates the kernels. Throws VulkanError, std::invalid_argument for a
    // desc outside the ranges above, or the shader loader's errors.
    HashTable(Device& device, const Desc& desc);
    ~HashTable() = default;
    HashTable(const HashTable&) = delete;
    HashTable& operator=(const HashTable&) = delete;

    // Records the maintenance of the user's frame counter `frame` (>= 2, +1 per frame; the lookups of
    // the frame touch with the same value) for the pool at `payloads`; payloads with frame - touched
    // > max_age are freed. The first call also initialises the free stack (every payload free: the
    // pool's state words are written). Barriers: earlier compute and transfer accesses -> the
    // maintenance; its writes -> later compute reads and writes and the indirect dispatch read.
    // `items_per_group` (>= 1): active payloads per workgroup of the user's pass (its dispatch).
    // `frame_number` / `safe_frame` time it (maintenance_ms).
    void RecordMaintenance(VkCommandBuffer cmd, VkDeviceAddress payloads, uint32_t frame, uint32_t max_age,
                           uint32_t items_per_group, uint64_t frame_number, uint64_t safe_frame);
    // Copies the counters to a readback slot of `frame_number` (after the user's last pass that counts
    // into them, before the next RecordMaintenance); skipped while every slot is pending.
    void RecordReadback(VkCommandBuffer cmd, uint64_t frame_number);
    // Reads the readback slots and the timer of frames <= safe_frame.
    void Resolve(uint64_t safe_frame);

    // The newest counters read back and their frame number (0 = none yet).
    const HashCounters& counters() const
    {
        return counters_;
    }
    uint64_t counters_frame() const
    {
        return counters_frame_;
    }
    // GPU time of the newest measured maintenance; 0 before the first.
    float maintenance_ms() const
    {
        return timer_.latest_ms();
    }

    VkDeviceAddress keys_address() const
    {
        return keys_.address();
    }
    VkDeviceAddress stack_address() const
    {
        return stack_.address();
    }
    VkDeviceAddress counters_address() const
    {
        return counters_buffer_.address();
    }
    VkDeviceAddress active_address() const
    {
        return active_.address();
    }
    VkDeviceAddress dispatch_address() const
    {
        return dispatch_.address();
    }
    VkBuffer dispatch_buffer() const
    {
        return dispatch_.buffer();
    }
    uint32_t key_mask() const
    {
        return key_slots_ - 1;
    }
    uint32_t payload_capacity() const
    {
        return desc_.payload_capacity;
    }
    // GPU memory of the table (without the user's pool).
    uint64_t bytes() const;

private:
    struct ReadbackSlot
    {
        GpuBuffer buffer;
        uint64_t frame = 0;
        bool pending = false;
    };

    Device& device_;
    Desc desc_;
    uint32_t key_slots_ = 0;
    GpuBuffer keys_;
    GpuBuffer stack_;
    GpuBuffer counters_buffer_;
    GpuBuffer active_;
    GpuBuffer dispatch_;
    bool initialised_ = false;
    std::vector<ReadbackSlot> readback_;
    HashCounters counters_{};
    uint64_t counters_frame_ = 0;
    ComputePipeline init_;
    ComputePipeline begin_;
    ComputePipeline maintain_;
    ComputePipeline finish_;
    GpuTimer timer_;
};

}  // namespace caelix
