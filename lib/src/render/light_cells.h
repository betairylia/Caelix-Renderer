#pragma once
#include "caelix/caelix.h"
#include "render/light_cells_layout.h"
#include "rhi/device.h"
#include "rhi/gpu_buffer.h"
#include "rhi/gpu_timer.h"
#include "rhi/hash_table.h"
#include "rhi/pipeline.h"

#include <cstdint>
#include <memory>

namespace caelix {

// The light cells of one renderer (docs/LIGHT_NEE.md section 3): a payload pool of cells (1088 bytes
// each) in a hash table (rhi/hash_table.h), the fill kernel, the cell-size controller and the
// counters. Per frame, before the trace: Record (the table's maintenance: eviction, active list, key
// rebuild; then the fill over the active cells); the trace looks cells up and inserts missing ones;
// after the trace: RecordReadback. Resolve reads the counters of safe frames and runs the controller.
//
// The light frame (light_frame(), the value the trace writes into a cell's touched word) is the
// renderer's own counter: +1 per recorded frame, starting at 2.
class LightCells
{
public:
    // The options with every 0 replaced by its default (CaelixLightNeeOptions).
    struct Settings
    {
        uint32_t payload_capacity = 0;
        float footprint = 0.0f;
        uint32_t reservoir_mode = 0;  // kLightCellModeOff / Rotate
        uint32_t candidates = 0;      // 1 or 2
        uint32_t max_age = 0;
        uint32_t flags = 0;  // CAELIX_LIGHT_NEE_FLAG_*
    };

    // Allocates the pool (zero-filled at the first Record) and the table; throws VulkanError or the
    // shader loader's errors.
    LightCells(Device& device, uint32_t payload_capacity);
    LightCells(const LightCells&) = delete;
    LightCells& operator=(const LightCells&) = delete;

    // The maintenance and the fill of this frame (light_frame() advances first). `light_tree` is the
    // scene's descriptor (nonzero). Ends with the fill's writes visible to the trace.
    void Record(VkCommandBuffer cmd, VkDeviceAddress light_tree, const Settings& settings,
                uint32_t frame_seed, uint64_t frame_number, uint64_t safe_frame);
    // After the trace of the frame: the counters to a readback slot.
    void RecordReadback(VkCommandBuffer cmd, uint64_t frame_number);
    // The counters and timers of frames <= safe_frame; every kControllerFrames frames the controller
    // (docs/LIGHT_NEE.md section 3.5) adjusts the footprint scale from the newest counters.
    void Resolve(uint64_t safe_frame, const Settings& settings);

    void GetStats(CaelixLightNeeStats& out) const;

    uint32_t light_frame() const
    {
        return frame_;
    }
    // The level-0 footprint threshold of the trace: footprint x the controller's scale.
    float footprint(const Settings& settings) const
    {
        return settings.footprint * scale_;
    }
    VkDeviceAddress payloads_address() const
    {
        return payloads_.address();
    }
    const HashTable& table() const
    {
        return *table_;
    }
    uint32_t payload_capacity() const
    {
        return table_->payload_capacity();
    }
    uint64_t bytes() const
    {
        return payloads_.size() + table_->bytes();
    }

    static constexpr uint32_t kControllerFrames = 16;

private:
    Device& device_;
    std::unique_ptr<HashTable> table_;
    GpuBuffer payloads_;
    bool payloads_cleared_ = false;
    ComputePipeline fill_;
    GpuTimer fill_timer_;
    uint32_t frame_ = 1;  // the first Record makes it 2
    float scale_ = 1.0f;  // the controller's footprint scale, in [kMinScale, 1]
    uint64_t controller_frame_ = 0;  // the counters frame of the last controller step
    uint32_t last_mode_ = 0;
    uint32_t last_candidates_ = 0;
};

}  // namespace caelix
