// The built-in material tables and the host's replacements (caelix_scene_set_materials,
// caelix_scene_set_glass_palette).
#include "scene/material_table.h"

#include "scene/record_layout.h"

#include <cmath>
#include <cstring>

namespace caelix {
namespace {

#include "scene/glass_palette_default.inc"

// shaders/render/caelix_shading.hlsl HashU32 (the block-ID pastel colours).
uint32_t HashU32(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return value;
}

// shaders/render/caelix_shading.hlsl Pastel.
void Pastel(uint32_t hash, float out[3])
{
    out[0] = 0.35f + 0.6f * (static_cast<float>(hash & 0xFFu) / 255.0f);
    out[1] = 0.35f + 0.6f * (static_cast<float>((hash >> 8) & 0xFFu) / 255.0f);
    out[2] = 0.35f + 0.6f * (static_cast<float>((hash >> 16) & 0xFFu) / 255.0f);
}

// An opaque entry: albedo, smoothness 0, metallic 0, ior 1.5, extinction 0, no emission.
CaelixMaterial Opaque(float r, float g, float b)
{
    CaelixMaterial m{};
    m.albedo[0] = r;
    m.albedo[1] = g;
    m.albedo[2] = b;
    m.ior = 1.5f;
    return m;
}

bool Finite(const float* values, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (!std::isfinite(values[i]))
        {
            return false;
        }
    }
    return true;
}

}  // namespace

MaterialTable::MaterialTable(uint32_t encoding)
    : encoding_(encoding)
{
    std::memcpy(palette_.data(), kDefaultGlassPalette, sizeof(kDefaultGlassPalette));
    entries_.resize(kEntries);
    for (uint32_t code = 0; code < kEntries; ++code)
    {
        entries_[code] = BuiltIn(code);
    }
}

uint32_t MaterialTable::CanonicalCode(uint32_t code) const
{
    return BlockMaterialCode(code & 0xFFFFu, encoding_);
}

CaelixMaterial MaterialTable::BuiltIn(uint32_t code) const
{
    if (encoding_ == kEncodingPacked)
    {
        if ((code & kPackedOpaqueMask) == 0)
        {
            return palette_[code & kPackedIdMask];
        }
        if ((code & 0x8000u) != 0)
        {
            return Opaque(static_cast<float>((code >> 10) & 31u) / 31.0f,
                          static_cast<float>((code >> 5) & 31u) / 31.0f,
                          static_cast<float>(code & 31u) / 31.0f);
        }
        CaelixMaterial m = Opaque(static_cast<float>((code >> 10) & 15u) / 15.0f,
                                  static_cast<float>((code >> 6) & 15u) / 15.0f,
                                  static_cast<float>((code >> 2) & 15u) / 15.0f);
        const float strength = std::exp2(3.0f * static_cast<float>(code & 3u));
        for (int c = 0; c < 3; ++c)
        {
            m.emission[c] = m.albedo[c] * strength;
        }
        return m;
    }
    // Block-ID.
    float rgb[3];
    if ((code & kBlockIdOpaqueMask) != 0)
    {
        Pastel(HashU32(code & 0x7FFFu), rgb);
        return Opaque(rgb[0], rgb[1], rgb[2]);
    }
    const uint32_t id = code & kBlockIdIdMask;
    if (id == 0)
    {
        return kDefaultGlassPalette[0];  // air: the vacuum entry
    }
    Pastel(HashU32(id | 0x10000u), rgb);
    return Opaque(rgb[0], rgb[1], rgb[2]);
}

bool MaterialTable::Valid(const CaelixMaterial& material, const char*& reason)
{
    if (!Finite(material.albedo, 3) || !Finite(&material.smoothness, 1) ||
        !Finite(material.emission, 3) || !Finite(&material.metallic, 1) ||
        !Finite(&material.ior, 1) || !Finite(&material.extinction, 1) ||
        !Finite(&material.sss_anisotropy, 1) || !Finite(material.sss_albedo, 3) ||
        !Finite(&material.sss_probability, 1) || !Finite(material.sggx_s, 6) ||
        !Finite(&material.sggx_density, 1) || !Finite(&material.sggx_transmission, 1) ||
        !Finite(material.sggx_transmission_albedo, 3))
    {
        reason = "a value is not finite";
        return false;
    }
    if (!(material.ior > 0.0f))
    {
        reason = "ior is not above 0";
        return false;
    }
    if (material.extinction < 0.0f)
    {
        reason = "extinction is negative";
        return false;
    }
    if (material.sss_probability < 0.0f || material.sss_probability > 1.0f)
    {
        reason = "sss_probability is outside [0, 1]";
        return false;
    }
    if (material.sss_probability > 0.0f)
    {
        if (material.sss_steps < 1 || material.sss_steps > kMaterialSssMaxSteps)
        {
            reason = "sss_steps is outside 1 .. 4096";
            return false;
        }
        for (int c = 0; c < 3; ++c)
        {
            if (material.sss_albedo[c] < 0.0f || material.sss_albedo[c] > 1.0f)
            {
                reason = "an sss_albedo component is outside [0, 1]";
                return false;
            }
        }
        if (material.sss_anisotropy < -kMaterialSssMaxAnisotropy ||
            material.sss_anisotropy > kMaterialSssMaxAnisotropy)
        {
            reason = "sss_anisotropy is outside [-0.99, 0.99]";
            return false;
        }
    }
    if (material.sggx_density < 0.0f)
    {
        reason = "sggx_density is negative";
        return false;
    }
    if (material.sggx_density > 0.0f)
    {
        if (!(material.sss_probability > 0.0f))
        {
            reason = "sggx_density > 0 needs sss_probability > 0";
            return false;
        }
        if (material.sggx_transmission < 0.0f || material.sggx_transmission > 1.0f)
        {
            reason = "sggx_transmission is outside [0, 1]";
            return false;
        }
        for (int c = 0; c < 3; ++c)
        {
            if (material.sggx_transmission_albedo[c] < 0.0f || material.sggx_transmission_albedo[c] > 1.0f)
            {
                reason = "an sggx_transmission_albedo component is outside [0, 1]";
                return false;
            }
        }
        // Sylvester's criterion on S (xx, yy, zz, xy, xz, yz), in double.
        const double xx = material.sggx_s[0];
        const double yy = material.sggx_s[1];
        const double zz = material.sggx_s[2];
        const double xy = material.sggx_s[3];
        const double xz = material.sggx_s[4];
        const double yz = material.sggx_s[5];
        const double det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz);
        if (!(xx > 0.0) || !(xx * yy - xy * xy > 0.0) || !(det > 0.0))
        {
            reason = "sggx_s is not positive definite";
            return false;
        }
    }
    return true;
}

void MaterialTable::MarkDirty(uint32_t first_code, uint32_t end_code)
{
    if (end_code <= first_code)
    {
        return;
    }
    const uint32_t first = first_code / kChunkEntries;
    const uint32_t last = (end_code - 1) / kChunkEntries;
    for (uint32_t chunk = first; chunk <= last; ++chunk)
    {
        dirty_chunks_ |= 1ull << chunk;
    }
}

uint32_t MaterialTable::SubsurfaceCode(uint32_t code, const CaelixMaterial& material) const
{
    return BlockIsOpaque(code, encoding_) && material.sss_probability > 0.0f ? 1u : 0u;
}

void MaterialTable::Replace(const CaelixMaterial* entries, uint32_t count)
{
    subsurface_codes_ = 0;
    for (uint32_t code = 0; code < kEntries; ++code)
    {
        entries_[code] = code < count ? entries[code] : BuiltIn(code);
        subsurface_codes_ += SubsurfaceCode(code, entries_[code]);
    }
    MarkDirty(0, kEntries);
}

void MaterialTable::Set(uint32_t code, const CaelixMaterial& material)
{
    subsurface_codes_ -= SubsurfaceCode(code, entries_[code]);
    entries_[code] = material;
    subsurface_codes_ += SubsurfaceCode(code, material);
    MarkDirty(code, code + 1);
}

void MaterialTable::WriteGpuChunk(uint32_t chunk, uint8_t* main_rows, uint8_t* subsurface_rows,
                                  uint8_t* sggx_rows) const
{
    const uint32_t first = chunk * kChunkEntries;
    for (uint32_t i = 0; i < kChunkEntries; ++i)
    {
        const uint8_t* entry = reinterpret_cast<const uint8_t*>(&entries_[first + i]);
        std::memcpy(main_rows + static_cast<size_t>(i) * kMaterialBytes, entry, kMaterialBytes);
        std::memcpy(subsurface_rows + static_cast<size_t>(i) * kSubsurfaceRowBytes, entry + kMaterialBytes,
                    kSubsurfaceRowBytes);
        uint8_t* const sggx_row = sggx_rows + static_cast<size_t>(i) * kSggxRowBytes;
        std::memcpy(sggx_row, entry + kMaterialBytes + kSubsurfaceRowBytes, kSggxRowBytes);
        // pad_sggx: the GPU row's last word is 0.
        std::memset(sggx_row + kSggxRowBytes - sizeof(float), 0, sizeof(float));
    }
}

void MaterialTable::SetGlassPalette(const CaelixMaterial* entries, uint32_t count)
{
    for (uint32_t id = 1; id < count; ++id)
    {
        palette_[id] = entries[id];
    }
    // Transparent codes are 0x0000..0x3FFF: id = code & 0xFF, face bits 8..13.
    for (uint32_t code = 0; code < 0x4000u; ++code)
    {
        const uint32_t id = code & 0xFFu;
        if (id >= 1 && id < count)
        {
            entries_[code] = palette_[id];
        }
    }
    MarkDirty(0, 0x4000u);
}

}  // namespace caelix
