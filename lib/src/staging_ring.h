#pragma once
#include "gpu_buffer.h"

#include <cstdint>
#include <deque>
#include <vector>

namespace caelix {

// A persistently mapped, host-visible upload buffer used as a ring.
//
// TryAllocate bump-allocates contiguous space (wrapping to offset 0 when the
// span does not fit before the end and the oldest live span has moved past it).
// Everything allocated since the last CloseFrame belongs to the open frame;
// CloseFrame(frame) tags it with the frame number of the flush that copies it.
// Recycle(safe) releases tagged spans whose frame <= safe, oldest first.
//
// The last kRowReserveBytes are reserved for slot table rows: record uploads
// (Purpose::kRecords) may fill only capacity - kRowReserveBytes, row writes
// (Purpose::kRows) may fill all of it. So a flush can always stage the rows of
// the bricks that filled the ring.
class StagingRing {
 public:
  static constexpr uint64_t kRowReserveBytes = 1ull << 20;

  enum class Purpose { kRecords, kRows };

  struct Span {
    uint64_t offset = 0;
    uint8_t* data = nullptr;
  };

  StagingRing() = default;
  StagingRing(VmaAllocator allocator, VkDevice device, uint64_t capacity);

  // Returns false when the ring has no room; nothing changes then.
  bool TryAllocate(uint64_t bytes, uint64_t alignment, Purpose purpose, Span& out);
  void CloseFrame(uint64_t frame);
  void Recycle(uint64_t safe_frame);

  // Makes the host writes since the last call visible to the device.
  void FlushHostWrites();

  VkBuffer buffer() const { return buffer_.buffer(); }
  uint64_t capacity() const { return capacity_; }
  uint64_t used() const { return used_; }

 private:
  struct ClosedFrame {
    uint64_t frame;
    uint64_t bytes;  // live bytes of the frame, alignment and wrap padding included
    uint64_t end;    // head position when the frame was closed
  };
  struct WrittenRange {
    uint64_t begin, end;
  };

  GpuBuffer buffer_;
  uint64_t capacity_ = 0;
  // Live bytes are [tail_, head_) modulo capacity_; used_ disambiguates
  // empty from full when head_ == tail_. head_ may equal capacity_.
  uint64_t head_ = 0;
  uint64_t tail_ = 0;
  uint64_t used_ = 0;
  uint64_t open_bytes_ = 0;
  std::deque<ClosedFrame> closed_;
  std::vector<WrittenRange> written_;  // host writes not yet flushed, merged when adjacent
};

}  // namespace caelix
