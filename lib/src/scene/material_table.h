#pragma once
#include "caelix/caelix.h"

#include <array>
#include <cstdint>
#include <vector>

namespace caelix {

static_assert(sizeof(CaelixMaterial) == 48, "CaelixMaterial is the 48-byte GPU table entry");

// The CPU copy of a scene's material table: one CaelixMaterial per 16-bit block
// code, the layout of the GPU buffer that the trace kernels read
// (shaders/common/caelix_material.hlsl). The scene uploads the chunks that
// changed at the next flush.
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
    static constexpr uint32_t kChunkEntries = 1024;  // upload granularity: 48 KiB
    static constexpr uint32_t kChunks = kEntries / kChunkEntries;
    static_assert(kChunks == 64, "the dirty mask is one uint64_t");
    static constexpr uint64_t kBytes = static_cast<uint64_t>(kEntries) * sizeof(CaelixMaterial);
    static constexpr uint64_t kChunkBytes = static_cast<uint64_t>(kChunkEntries) * sizeof(CaelixMaterial);

    // Builds the built-in table of `encoding` (0 packed, 1 block-ID); every chunk is dirty.
    explicit MaterialTable(uint32_t encoding);

    // The code the kernels look up for a raw block code: transparent codes lose
    // their face bits (packed: & 0xFF, block-ID: & 0x1FF), opaque codes stay.
    uint32_t CanonicalCode(uint32_t code) const;
    const CaelixMaterial& Entry(uint32_t code) const
    {
        return entries_[code];
    }
    const CaelixMaterial* data() const
    {
        return entries_.data();
    }
    uint32_t encoding() const
    {
        return encoding_;
    }

    // Checks one entry (finite values, ior > 0, extinction >= 0). False: `reason`
    // names the first problem.
    static bool Valid(const CaelixMaterial& material, const char*& reason);

    // caelix_scene_set_materials: entries [0, count) from the host, the rest the
    // built-in default. The caller has validated the entries.
    void Replace(const CaelixMaterial* entries, uint32_t count);
    // caelix_scene_set_material_entries: one entry at `code` (< kEntries), stored with pad
    // zeroed. The caller has validated the code and the entry.
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
    void MarkDirty(uint32_t first_code, uint32_t end_code);

    uint32_t encoding_ = 0;
    std::array<CaelixMaterial, 256> palette_{};  // packed: the glass palette by id
    std::vector<CaelixMaterial> entries_;        // kEntries
    uint64_t dirty_chunks_ = ~0ull;
};

}  // namespace caelix
