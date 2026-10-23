// The reusable GPU hash table: buffers, the maintenance recording and the counter readback.
#include "rhi/hash_table.h"

#include "core/log.h"
#include "rhi/shader_loader.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>

namespace caelix {
namespace {

constexpr uint32_t kReadbackSlots = 8;  // frames in flight whose counters can be read back

void Barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
             VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

constexpr VkAccessFlags2 kStorage =
    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

}  // namespace

HashTable::HashTable(Device& device, const Desc& desc)
    : device_(device), desc_(desc)
{
    if (desc.payload_capacity == 0 || desc.payload_capacity > kHashMaxPayloads || desc.key_words == 0 ||
        desc.key_words > kHashMaxKeyWords || desc.payload_stride % 4 != 0 ||
        desc.payload_stride < kHashPayloadKey + 4 * desc.key_words)
    {
        CAELIX_LOGE("%s: invalid hash table desc (capacity %u, stride %u, key words %u)", desc.name,
                    desc.payload_capacity, desc.payload_stride, desc.key_words);
        throw std::invalid_argument("invalid hash table desc");
    }
    key_slots_ = std::bit_ceil(desc.payload_capacity * kHashKeySlotsPerPayload);
    const VmaAllocator allocator = device.allocator();
    const VkDevice vk_device = device.device();
    keys_ = GpuBuffer::DeviceLocal(allocator, vk_device, uint64_t{key_slots_} * kHashSlotBytes);
    stack_ = GpuBuffer::DeviceLocal(allocator, vk_device, HashStackBytes(desc.payload_capacity));
    counters_buffer_ = GpuBuffer::DeviceLocal(allocator, vk_device, kHashCounterBytes);
    active_ =
        GpuBuffer::DeviceLocal(allocator, vk_device, uint64_t{desc.payload_capacity} * sizeof(uint32_t));
    dispatch_ =
        GpuBuffer::DeviceLocal(allocator, vk_device, kHashDispatchBytes, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    readback_.resize(kReadbackSlots);
    for (ReadbackSlot& slot : readback_)
    {
        slot.buffer = GpuBuffer::HostReadback(allocator, vk_device, kHashCounterBytes);
    }
    const auto create = [&](const char* name) {
        return ComputePipeline(vk_device, LoadShader(name), sizeof(HashPush), name);
    };
    init_ = create("caelix_hash_init");
    begin_ = create("caelix_hash_begin");
    maintain_ = create("caelix_hash_maintain");
    finish_ = create("caelix_hash_finish");
    timer_ = GpuTimer(device);
    CAELIX_LOGI("%s: %u payloads of %u bytes, %u key slots", desc.name, desc.payload_capacity,
                desc.payload_stride, key_slots_);
}

uint64_t HashTable::bytes() const
{
    return keys_.size() + stack_.size() + counters_buffer_.size() + active_.size() + dispatch_.size();
}

void HashTable::RecordMaintenance(VkCommandBuffer cmd, VkDeviceAddress payloads, uint32_t frame,
                                  uint32_t max_age, uint32_t items_per_group, uint64_t frame_number,
                                  uint64_t safe_frame)
{
    HashPush push{};
    push.keys = keys_.address();
    push.stack = stack_.address();
    push.counters = counters_buffer_.address();
    push.payloads = payloads;
    push.active = active_.address();
    push.dispatch = dispatch_.address();
    push.key_mask = key_slots_ - 1;
    push.payload_capacity = desc_.payload_capacity;
    push.payload_stride = desc_.payload_stride;
    push.key_words = desc_.key_words;
    push.frame = frame;
    push.max_age = max_age;
    push.items_per_group = std::max(items_per_group, 1u);
    const uint32_t payload_groups = (desc_.payload_capacity + kHashThreads - 1) / kHashThreads;

    const bool timed = timer_.Begin(cmd, frame_number, safe_frame);
    // Earlier lookups (pops, inserts, touches), the user's pass, the readback copy -> this pass.
    Barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            kStorage | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            kStorage | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    if (!initialised_)
    {
        // Every payload free, the free stack 0 .. capacity - 1.
        init_.Dispatch(cmd, &push, sizeof(push), payload_groups);
        Barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kStorage);
        initialised_ = true;
    }
    begin_.Dispatch(cmd, &push, sizeof(push), 1);
    vkCmdFillBuffer(cmd, keys_.buffer(), 0, VK_WHOLE_SIZE, 0);
    Barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kStorage);
    maintain_.Dispatch(cmd, &push, sizeof(push), payload_groups);
    Barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kStorage);
    finish_.Dispatch(cmd, &push, sizeof(push), 1);
    // The table, the active list and the dispatch -> the user's pass (indirect) and the lookups.
    Barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            kStorage | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    if (timed)
    {
        timer_.End(cmd);
    }
}

void HashTable::RecordReadback(VkCommandBuffer cmd, uint64_t frame_number)
{
    for (ReadbackSlot& slot : readback_)
    {
        if (slot.pending)
        {
            continue;
        }
        Barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        const VkBufferCopy region{0, 0, kHashCounterBytes};
        vkCmdCopyBuffer(cmd, counters_buffer_.buffer(), slot.buffer.buffer(), 1, &region);
        Barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        slot.frame = frame_number;
        slot.pending = true;
        return;
    }
}

void HashTable::Resolve(uint64_t safe_frame)
{
    timer_.Resolve(safe_frame);
    for (ReadbackSlot& slot : readback_)
    {
        if (!slot.pending || slot.frame > safe_frame)
        {
            continue;
        }
        slot.pending = false;
        slot.buffer.InvalidateHostReads(0, kHashCounterBytes);
        if (slot.frame >= counters_frame_)
        {
            std::memcpy(&counters_, slot.buffer.mapped(), sizeof(counters_));
            counters_frame_ = slot.frame;
        }
    }
}

}  // namespace caelix
