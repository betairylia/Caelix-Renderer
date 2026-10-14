#pragma once
#include "rhi/gpu_buffer.h"

#include <array>
#include <cstdint>
#include <vector>

namespace caelix {

// What a block of acceleration structure storage holds; the allocator keeps one
// byte counter per kind for the stats.
enum class AsKind : uint8_t { kBlas = 0, kTlas = 1 };

// A block of acceleration structure storage inside one chunk buffer.
struct AsBlock {
  static constexpr uint32_t kNoChunk = 0xFFFFFFFFu;

  uint32_t chunk = kNoChunk;
  uint32_t size_class = 0;
  VkDeviceSize offset = 0;  // multiple of kMinBlockBytes, so of 256 (the AS offset rule)
  VkDeviceSize size = 0;    // power of two, >= BlockAllocator::kMinBlockBytes
  AsKind kind = AsKind::kBlas;
  bool valid() const { return size != 0; }
};

// Storage for acceleration structures. An acceleration structure must never move:
// a TLAS refers to its BLASes by device address. So unlike RowPool there is no
// growth by reallocation: storage comes in chunks of kChunkBytes device-local
// buffers (ACCELERATION_STRUCTURE_STORAGE | SHADER_DEVICE_ADDRESS) that are only
// added. Blocks come from power-of-two size classes from kMinBlockBytes, with one
// LIFO free list per class per chunk and a bump pointer at the end of each chunk;
// a new chunk is created when no chunk has room. A block larger than a chunk gets
// a chunk of its own size. Freed blocks are never merged or split.
//
// FreeAfter() defers the release: a TLAS of a frame in flight may still refer to
// the old BLAS. Recycle(safe) destroys the acceleration structure handle of every
// block whose frame <= safe and then returns the block to its free list.
//
// Not copyable or movable (the scene owns one). The destructor destroys the
// handles of pending frees and every chunk; the caller destroys the handles of
// live blocks first, with the device idle.
class BlockAllocator {
 public:
  static constexpr VkDeviceSize kChunkBytes = 64ull << 20;
  static constexpr VkDeviceSize kMinBlockBytes = 4096;

  BlockAllocator() = default;
  ~BlockAllocator();
  BlockAllocator(const BlockAllocator&) = delete;
  BlockAllocator& operator=(const BlockAllocator&) = delete;

  void Init(VmaAllocator allocator, VkDevice device);

  // Throws VulkanError (or std::bad_alloc) when a new chunk cannot be created; the
  // allocator is unchanged then.
  AsBlock Allocate(VkDeviceSize bytes, AsKind kind);
  // Releases a block at once. Only for a block no recorded command uses.
  void Free(const AsBlock& block);
  // Releases the block (and destroys `handle`, which may be null) when the safe
  // frame reaches `frame`.
  void FreeAfter(const AsBlock& block, uint64_t frame, VkAccelerationStructureKHR handle);
  void Recycle(uint64_t safe_frame);

  VkBuffer buffer(const AsBlock& block) const { return chunks_[block.chunk].buffer.buffer(); }

  // Bytes of all chunk buffers.
  uint64_t chunk_bytes() const { return chunk_bytes_; }
  uint32_t chunk_count() const { return static_cast<uint32_t>(chunks_.size()); }
  // Bytes of the blocks of one kind that are allocated, pending frees included.
  uint64_t allocated_bytes(AsKind kind) const { return allocated_[static_cast<size_t>(kind)]; }

 private:
  struct Chunk {
    GpuBuffer buffer;
    VkDeviceSize bump = 0;
    std::vector<std::vector<VkDeviceSize>> free_by_class;  // offsets, LIFO
  };
  struct PendingFree {
    uint64_t frame;
    AsBlock block;
    VkAccelerationStructureKHR handle;
  };

  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  std::vector<Chunk> chunks_;
  std::vector<PendingFree> pending_;
  uint64_t chunk_bytes_ = 0;
  std::array<uint64_t, 2> allocated_{};
};

// Creates an acceleration structure of `type` that covers the whole block.
// Throws VulkanError.
VkAccelerationStructureKHR CreateAccelerationStructure(VkDevice device, VkBuffer buffer,
                                                       const AsBlock& block,
                                                       VkAccelerationStructureTypeKHR type);
VkDeviceAddress AccelerationStructureAddress(VkDevice device, VkAccelerationStructureKHR handle);

}  // namespace caelix
