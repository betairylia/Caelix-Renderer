#pragma once
#include "caelix/caelix.h"
#include "scene/material_pack.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace caelix {

// The C ABI struct is the host's view; the GPU layouts (kMaterial*, kSubsurface*) are in
// scene/material_pack.h.
static_assert(sizeof(CaelixMaterial) == 112);
static_assert(offsetof(CaelixMaterial, albedo) == 0);
static_assert(offsetof(CaelixMaterial, smoothness) == 12);
static_assert(offsetof(CaelixMaterial, emission) == 16);
static_assert(offsetof(CaelixMaterial, metallic) == 28);
static_assert(offsetof(CaelixMaterial, ior) == 32);
static_assert(offsetof(CaelixMaterial, extinction) == 36);
static_assert(offsetof(CaelixMaterial, sss_model) == 40);
static_assert(offsetof(CaelixMaterial, sss_probability) == 44);
static_assert(offsetof(CaelixMaterial, sss_albedo) == 48);
static_assert(offsetof(CaelixMaterial, sss_anisotropy) == 60);
static_assert(offsetof(CaelixMaterial, sss_steps) == 64);
static_assert(offsetof(CaelixMaterial, sss_params) == 68);
static_assert(offsetof(CaelixMaterial, texture_set) == 100);
static_assert(offsetof(CaelixMaterial, connect_group) == 104);
static_assert(offsetof(CaelixMaterial, reserved) == 108);

// The CPU copy of a scene's material table: one CaelixMaterial per 16-bit block
// code (the ABI truth, returned by caelix_scene_get_material), plus its two packed
// GPU tables (scene/material_pack.h): the 32-byte hot entries that the trace
// kernels read (shaders/common/caelix_material.hlsl) and the 32-byte subsurface
// entries. The scene uploads the chunks that changed at the next flush.
//
// Built-in content per encoding (see caelix.h, "Materials"): the packed decode of
// opaque and emissive codes (formerly in the shader), the glass palette for the
// transparent codes of the packed encoding, pastel hash colours for the block-ID
// encoding. Every transparent code is filled, face bits included, with the entry
// of its id, so a raw dump of the table is consistent; the kernels look a
// transparent code up without its face bits anyway (CanonicalCode).
class MaterialTable
{
public:
    static constexpr uint32_t kEntries = 65536;
    static constexpr uint32_t kChunkEntries = 1024;  // upload granularity: 32 KiB per table
    static constexpr uint32_t kChunks = kEntries / kChunkEntries;
    static_assert(kChunks == 64, "the dirty mask is one uint64_t");
    static constexpr uint32_t kEntryWords = 8;  // both GPU entries are 8 words
    static_assert(kMaterialBytes == kEntryWords * 4 && kSubsurfaceBytes == kEntryWords * 4);
    static constexpr uint64_t kHotTableBytes = static_cast<uint64_t>(kEntries) * kMaterialBytes;
    static constexpr uint64_t kSubsurfaceTableBytes = static_cast<uint64_t>(kEntries) * kSubsurfaceBytes;
    static constexpr uint64_t kHotChunkBytes = static_cast<uint64_t>(kChunkEntries) * kMaterialBytes;
    static constexpr uint64_t kSubsurfaceChunkBytes = static_cast<uint64_t>(kChunkEntries) * kSubsurfaceBytes;

    // Builds the built-in table of `encoding` (0 packed, 1 block-ID); every chunk is dirty.
    explicit MaterialTable(uint32_t encoding);

    // The code the kernels look up for a raw block code: transparent codes lose
    // their face bits (packed: & 0xFF, block-ID: & 0x1FF), opaque codes stay.
    uint32_t CanonicalCode(uint32_t code) const;
    const CaelixMaterial& Entry(uint32_t code) const
    {
        return entries_[code];
    }
    // The packed GPU tables, kHotTableBytes and kSubsurfaceTableBytes.
    const uint8_t* hot_data() const
    {
        return reinterpret_cast<const uint8_t*>(hot_.data());
    }
    const uint8_t* subsurface_data() const
    {
        return reinterpret_cast<const uint8_t*>(subsurface_.data());
    }
    uint32_t encoding() const
    {
        return encoding_;
    }
    // The table has an SSS entry: an opaque code with sss_model CAELIX_SSS_MODEL_HG, sss_probability > 0
    // and sss_steps > 0 (docs/SUBSURFACE.md section 1). Transmissive shadow rays then pass SSS voxels
    // (FrameConstants::subsurface_present).
    bool has_subsurface() const
    {
        return subsurface_count_ > 0;
    }

    // Checks one entry (finite floats within the half range, ior > 0, extinction >= 0,
    // the subsurface and texture ranges of caelix.h). False: `reason` names the first problem.
    static bool Valid(const CaelixMaterial& material, const char*& reason);

    // caelix_scene_set_materials: entries [0, count) from the host, the rest the
    // built-in default. The caller has validated the entries.
    void Replace(const CaelixMaterial* entries, uint32_t count);
    // caelix_scene_set_material_entries: one entry at `code` (< kEntries), stored with
    // reserved zeroed. The caller has validated the code and the entry.
    void Set(uint32_t code, const CaelixMaterial& material);
    // caelix_scene_set_glass_palette (packed only): palette ids 1 .. count - 1, then
    // every transparent code of those ids. The caller has validated the entries.
    void SetGlassPalette(const CaelixMaterial* entries, uint32_t count);

    // Bit c = chunk c ([c * kChunkEntries, (c + 1) * kChunkEntries)) changed since
    // it was last staged.
    uint64_t dirty_chunks() const
    {
        return dirty_chunks_;
    }
    void MarkStaged(uint32_t chunk)
    {
        dirty_chunks_ &= ~(1ull << chunk);
    }

private:
    // The built-in entry of `code` with the current glass palette.
    CaelixMaterial BuiltIn(uint32_t code) const;
    // Stores `material` at `code` (reserved zeroed), packs its GPU entries and keeps
    // subsurface_count_; does not mark dirty.
    void Store(uint32_t code, const CaelixMaterial& material);
    // The entry `material` at `code` is an SSS entry (see has_subsurface).
    bool IsSubsurface(uint32_t code, const CaelixMaterial& material) const;
    void MarkDirty(uint32_t first_code, uint32_t end_code);

    uint32_t encoding_ = 0;
    std::array<CaelixMaterial, 256> palette_{};  // packed: the glass palette by id
    std::vector<CaelixMaterial> entries_;        // kEntries
    std::vector<uint32_t> hot_;                  // kEntries * kEntryWords: the GPU hot table
    std::vector<uint32_t> subsurface_;           // kEntries * kEntryWords: the GPU subsurface table
    uint64_t dirty_chunks_ = ~0ull;
    uint32_t subsurface_count_ = 0;  // entries for which IsSubsurface holds
};

}  // namespace caelix
