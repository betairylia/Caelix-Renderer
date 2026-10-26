#pragma once
#include "caelix/caelix.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace caelix {

// The GPU keeps the 64-byte C ABI struct as two tables (shaders/common/caelix_material.hlsl):
// - the main table, kMaterialBytes per code = the struct's first 48 bytes, read at these offsets: the
//   16-byte rows (albedo, smoothness), (emission, metallic), (ior, extinction, sss_anisotropy, sss_steps);
// - the subsurface table, kSubsurfaceRowBytes per code = the struct's last 16 bytes (sss_albedo,
//   sss_probability), read only by the subsurface walk (FrameConstants::subsurface_table; experiment
//   branch sss-expr). The main kernels keep the 48-byte stride of the main table.
constexpr uint32_t kMaterialBytes = 48;
constexpr uint32_t kMaterialAlbedo = 0;
constexpr uint32_t kMaterialEmission = 16;
constexpr uint32_t kMaterialIor = 32;
constexpr uint32_t kMaterialSssAnisotropy = 40;
constexpr uint32_t kMaterialSssSteps = 44;
constexpr uint32_t kSubsurfaceRowBytes = 16;
// Limits of the subsurface fields of an entry with sss_probability > 0 (docs/SSS_EXPERIMENT.md).
constexpr uint32_t kMaterialSssMaxSteps = 4096;
constexpr float kMaterialSssMaxAnisotropy = 0.99f;
static_assert(sizeof(CaelixMaterial) == kMaterialBytes + kSubsurfaceRowBytes,
              "CaelixMaterial is the main table row followed by the subsurface row");
static_assert(offsetof(CaelixMaterial, albedo) == kMaterialAlbedo);
static_assert(offsetof(CaelixMaterial, emission) == kMaterialEmission);
static_assert(offsetof(CaelixMaterial, ior) == kMaterialIor);
static_assert(offsetof(CaelixMaterial, sss_anisotropy) == kMaterialSssAnisotropy);
static_assert(offsetof(CaelixMaterial, sss_steps) == kMaterialSssSteps);
static_assert(offsetof(CaelixMaterial, sss_albedo) == kMaterialBytes);
static_assert(offsetof(CaelixMaterial, sss_probability) == kMaterialBytes + 12);

// The CPU copy of a scene's material table: one CaelixMaterial per 16-bit block
// code. The GPU holds it as two buffers, the main table and the subsurface table
// (see kMaterialBytes; shaders/common/caelix_material.hlsl): the scene uploads the
// chunks that changed at the next flush, each chunk split into its rows of both
// tables by WriteGpuChunk.
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
    static constexpr uint32_t kChunkEntries = 1024;  // upload granularity: 48 + 16 KiB
    static constexpr uint32_t kChunks = kEntries / kChunkEntries;
    static_assert(kChunks == 64, "the dirty mask is one uint64_t");
    // The two GPU tables and one chunk of each.
    static constexpr uint64_t kMainBytes = static_cast<uint64_t>(kEntries) * kMaterialBytes;
    static constexpr uint64_t kSubsurfaceBytes = static_cast<uint64_t>(kEntries) * kSubsurfaceRowBytes;
    static constexpr uint64_t kChunkMainBytes = static_cast<uint64_t>(kChunkEntries) * kMaterialBytes;
    static constexpr uint64_t kChunkSubsurfaceBytes = static_cast<uint64_t>(kChunkEntries) * kSubsurfaceRowBytes;
    // Staging bytes of one chunk: its main rows, then its subsurface rows.
    static constexpr uint64_t kChunkBytes = kChunkMainBytes + kChunkSubsurfaceBytes;

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

    // Checks one entry (finite values, ior > 0, extinction >= 0, sss_probability in [0, 1]; with
    // sss_probability > 0: sss_steps 1 .. kMaterialSssMaxSteps, sss_albedo in [0, 1], sss_anisotropy in
    // [-kMaterialSssMaxAnisotropy, kMaterialSssMaxAnisotropy]). False: `reason` names the first problem.
    static bool Valid(const CaelixMaterial& material, const char*& reason);

    // caelix_scene_set_materials: entries [0, count) from the host, the rest the
    // built-in default. The caller has validated the entries.
    void Replace(const CaelixMaterial* entries, uint32_t count);
    // caelix_scene_set_material_entries: one entry at `code` (< kEntries). The caller has validated
    // the code and the entry.
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
    // Writes chunk `chunk` as the GPU sees it: kChunkMainBytes of main rows to `main_rows`, then
    // kChunkSubsurfaceBytes of subsurface rows to `subsurface_rows`.
    void WriteGpuChunk(uint32_t chunk, uint8_t* main_rows, uint8_t* subsurface_rows) const;
    // True when an OPAQUE code has sss_probability > 0 (the renderer then binds the trace kernels with
    // the subsurface walk). Kept up to date by Replace and Set; SetGlassPalette writes transparent
    // codes only, which never count.
    bool has_subsurface() const
    {
        return subsurface_codes_ != 0;
    }

private:
    // The built-in entry of `code` with the current glass palette.
    CaelixMaterial BuiltIn(uint32_t code) const;
    void MarkDirty(uint32_t first_code, uint32_t end_code);
    // 1 when `code` is opaque and its entry `material` has sss_probability > 0 (subsurface_codes_).
    uint32_t SubsurfaceCode(uint32_t code, const CaelixMaterial& material) const;

    uint32_t encoding_ = 0;
    std::array<CaelixMaterial, 256> palette_{};  // packed: the glass palette by id
    std::vector<CaelixMaterial> entries_;        // kEntries
    uint64_t dirty_chunks_ = ~0ull;
    uint32_t subsurface_codes_ = 0;  // opaque codes with sss_probability > 0 (the built-in tables have none)
};

}  // namespace caelix
