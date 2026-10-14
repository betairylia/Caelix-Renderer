#include "staging_ring.h"

namespace caelix {
namespace {

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

StagingRing::StagingRing(VmaAllocator allocator, VkDevice device, uint64_t capacity)
    : buffer_(GpuBuffer::HostUpload(allocator, device, capacity)), capacity_(capacity) {}

bool StagingRing::TryAllocate(uint64_t bytes, uint64_t alignment, Purpose purpose, Span& out) {
  const uint64_t limit =
      purpose == Purpose::kRecords ? capacity_ - kRowReserveBytes : capacity_;
  if (bytes == 0 || bytes > limit) {
    return false;
  }
  if (used_ == 0) {
    head_ = 0;
    tail_ = 0;
  }
  uint64_t offset = 0;
  uint64_t padding = 0;
  const bool wrapped = used_ > 0 && head_ <= tail_;
  const uint64_t aligned = AlignUp(head_, alignment);
  if (!wrapped) {
    // Free space: [head_, capacity_) and [0, tail_).
    if (aligned + bytes <= capacity_) {
      offset = aligned;
      padding = aligned - head_;
    } else if (bytes <= tail_) {
      offset = 0;
      padding = capacity_ - head_;
    } else {
      return false;
    }
  } else {
    // Free space: [head_, tail_).
    if (aligned + bytes > tail_) {
      return false;
    }
    offset = aligned;
    padding = aligned - head_;
  }
  if (used_ + padding + bytes > limit) {
    return false;
  }
  used_ += padding + bytes;
  open_bytes_ += padding + bytes;
  head_ = offset + bytes;

  if (!written_.empty() && written_.back().end == offset) {
    written_.back().end = offset + bytes;
  } else {
    written_.push_back({offset, offset + bytes});
  }
  out.offset = offset;
  out.data = buffer_.mapped() + offset;
  return true;
}

void StagingRing::CloseFrame(uint64_t frame) {
  if (open_bytes_ == 0) {
    return;
  }
  closed_.push_back({frame, open_bytes_, head_});
  open_bytes_ = 0;
}

void StagingRing::Recycle(uint64_t safe_frame) {
  while (!closed_.empty() && closed_.front().frame <= safe_frame) {
    used_ -= closed_.front().bytes;
    tail_ = closed_.front().end == capacity_ ? 0 : closed_.front().end;
    closed_.pop_front();
  }
  if (used_ == 0) {
    head_ = 0;
    tail_ = 0;
  }
}

void StagingRing::FlushHostWrites() {
  for (const WrittenRange& range : written_) {
    buffer_.FlushHostWrites(range.begin, range.end - range.begin);
  }
  written_.clear();
}

}  // namespace caelix
