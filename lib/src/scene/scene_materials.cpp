// The scene's material table: caelix_scene_set_materials, caelix_scene_set_glass_palette and
// caelix_scene_get_material. The upload is Scene::StageMaterials (scene_flush.cpp).
#include "scene/scene.h"

#include "core/log.h"

namespace caelix {
namespace {

constexpr uint32_t kEncodingPacked = 0;

// Checks entries [first, count). Logs the first invalid one.
bool ValidEntries(const char* function, const CaelixMaterial* entries, uint32_t first,
                  uint32_t count) {
  for (uint32_t i = first; i < count; ++i) {
    const char* reason = nullptr;
    if (!MaterialTable::Valid(entries[i], reason)) {
      CAELIX_LOGE("%s: entry %u is not valid: %s", function, i, reason);
      return false;
    }
  }
  return true;
}

}  // namespace

CaelixResult Scene::SetMaterials(const CaelixMaterial* entries, uint32_t count) {
  constexpr const char* kFunction = "caelix_scene_set_materials";
  if (count > MaterialTable::kEntries) {
    CAELIX_LOGE("%s: count %u is above %u", kFunction, count, MaterialTable::kEntries);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (count > 0 && entries == nullptr) {
    CAELIX_LOGE("%s: entries is null", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (!ValidEntries(kFunction, entries, 0, count)) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  materials_->Replace(entries, count);
  return CAELIX_OK;
}

CaelixResult Scene::SetGlassPalette(const CaelixMaterial* entries, uint32_t count) {
  constexpr const char* kFunction = "caelix_scene_set_glass_palette";
  if (encoding_ != kEncodingPacked) {
    CAELIX_LOGE("%s: the scene uses the block-ID encoding; the glass palette belongs to the packed "
                "encoding (use caelix_scene_set_materials)",
                kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (count > 256) {
    CAELIX_LOGE("%s: count %u is above 256", kFunction, count);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (count > 0 && entries == nullptr) {
    CAELIX_LOGE("%s: entries is null", kFunction);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  if (!ValidEntries(kFunction, entries, 1, count)) {
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  materials_->SetGlassPalette(entries, count);
  return CAELIX_OK;
}

CaelixResult Scene::GetMaterial(uint32_t code, CaelixMaterial& out) const {
  if (code > 0xFFFFu) {
    CAELIX_LOGE("caelix_scene_get_material: code 0x%X is above 0xFFFF", code);
    return CAELIX_ERROR_INVALID_ARGUMENT;
  }
  out = materials_->Entry(materials_->CanonicalCode(code));
  return CAELIX_OK;
}

}  // namespace caelix
