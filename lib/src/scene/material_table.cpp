// The built-in material tables and the host's replacements (caelix_scene_set_materials,
// caelix_scene_set_glass_palette).
#include "scene/material_table.h"

#include <cmath>
#include <cstring>

namespace caelix {
namespace {

#include "scene/glass_palette_default.inc"

constexpr uint32_t kEncodingPacked = 0;

// shaders/render/caelix_shading.hlsl HashU32 (the block-ID pastel colours).
uint32_t HashU32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7FEB352Du;
  value ^= value >> 15;
  value *= 0x846CA68Bu;
  value ^= value >> 16;
  return value;
}

// shaders/render/caelix_shading.hlsl Pastel.
void Pastel(uint32_t hash, float out[3]) {
  out[0] = 0.35f + 0.6f * (static_cast<float>(hash & 0xFFu) / 255.0f);
  out[1] = 0.35f + 0.6f * (static_cast<float>((hash >> 8) & 0xFFu) / 255.0f);
  out[2] = 0.35f + 0.6f * (static_cast<float>((hash >> 16) & 0xFFu) / 255.0f);
}

// An opaque entry: albedo, smoothness 0, metallic 0, ior 1.5, extinction 0, no emission.
CaelixMaterial Opaque(float r, float g, float b) {
  CaelixMaterial m{};
  m.albedo[0] = r;
  m.albedo[1] = g;
  m.albedo[2] = b;
  m.ior = 1.5f;
  return m;
}

bool Finite(const float* values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

MaterialTable::MaterialTable(uint32_t encoding) : encoding_(encoding) {
  std::memcpy(palette_.data(), kDefaultGlassPalette, sizeof(kDefaultGlassPalette));
  entries_.resize(kEntries);
  for (uint32_t code = 0; code < kEntries; ++code) {
    entries_[code] = BuiltIn(code);
  }
}

uint32_t MaterialTable::CanonicalCode(uint32_t code) const {
  code &= 0xFFFFu;
  if (encoding_ == kEncodingPacked) {
    return (code & 0xC000u) != 0 ? code : (code & 0xFFu);
  }
  return (code & 0x8000u) != 0 ? code : (code & 0x1FFu);
}

CaelixMaterial MaterialTable::BuiltIn(uint32_t code) const {
  if (encoding_ == kEncodingPacked) {
    if ((code & 0xC000u) == 0) {
      return palette_[code & 0xFFu];
    }
    if ((code & 0x8000u) != 0) {
      return Opaque(static_cast<float>((code >> 10) & 31u) / 31.0f,
                    static_cast<float>((code >> 5) & 31u) / 31.0f,
                    static_cast<float>(code & 31u) / 31.0f);
    }
    CaelixMaterial m = Opaque(static_cast<float>((code >> 10) & 15u) / 15.0f,
                              static_cast<float>((code >> 6) & 15u) / 15.0f,
                              static_cast<float>((code >> 2) & 15u) / 15.0f);
    const float strength = std::exp2(3.0f * static_cast<float>(code & 3u));
    for (int c = 0; c < 3; ++c) {
      m.emission[c] = m.albedo[c] * strength;
    }
    return m;
  }
  // Block-ID.
  float rgb[3];
  if ((code & 0x8000u) != 0) {
    Pastel(HashU32(code & 0x7FFFu), rgb);
    return Opaque(rgb[0], rgb[1], rgb[2]);
  }
  const uint32_t id = code & 0x1FFu;
  if (id == 0) {
    return kDefaultGlassPalette[0];  // air: the vacuum entry
  }
  Pastel(HashU32(id | 0x10000u), rgb);
  return Opaque(rgb[0], rgb[1], rgb[2]);
}

bool MaterialTable::Valid(const CaelixMaterial& material, const char*& reason) {
  if (!Finite(material.albedo, 3) || !Finite(&material.smoothness, 1) ||
      !Finite(material.emission, 3) || !Finite(&material.metallic, 1) ||
      !Finite(&material.ior, 1) || !Finite(&material.extinction, 1)) {
    reason = "a value is not finite";
    return false;
  }
  if (!(material.ior > 0.0f)) {
    reason = "ior is not above 0";
    return false;
  }
  if (material.extinction < 0.0f) {
    reason = "extinction is negative";
    return false;
  }
  return true;
}

void MaterialTable::MarkDirty(uint32_t first_code, uint32_t end_code) {
  if (end_code <= first_code) {
    return;
  }
  const uint32_t first = first_code / kChunkEntries;
  const uint32_t last = (end_code - 1) / kChunkEntries;
  for (uint32_t chunk = first; chunk <= last; ++chunk) {
    dirty_chunks_ |= 1ull << chunk;
  }
}

void MaterialTable::Replace(const CaelixMaterial* entries, uint32_t count) {
  for (uint32_t code = 0; code < kEntries; ++code) {
    CaelixMaterial m = code < count ? entries[code] : BuiltIn(code);
    m.pad[0] = 0.0f;
    m.pad[1] = 0.0f;
    entries_[code] = m;
  }
  MarkDirty(0, kEntries);
}

void MaterialTable::SetGlassPalette(const CaelixMaterial* entries, uint32_t count) {
  for (uint32_t id = 1; id < count; ++id) {
    palette_[id] = entries[id];
    palette_[id].pad[0] = 0.0f;
    palette_[id].pad[1] = 0.0f;
  }
  // Transparent codes are 0x0000..0x3FFF: id = code & 0xFF, face bits 8..13.
  for (uint32_t code = 0; code < 0x4000u; ++code) {
    const uint32_t id = code & 0xFFu;
    if (id >= 1 && id < count) {
      entries_[code] = palette_[id];
    }
  }
  MarkDirty(0, 0x4000u);
}

}  // namespace caelix
