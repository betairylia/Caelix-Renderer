/* Compile check: the public header is valid C99 with no warnings. Touches every
   struct and every function declaration of the ABI. Not run. */
#include "caelix/caelix.h"

#include <stddef.h>

typedef void (*AnyFn)(void);
typedef CaelixResult (*SceneCreateFn)(CaelixDevice*, const CaelixSceneDesc*, CaelixScene**);

size_t caelix_abi_c99_touch(void);

/* CaelixMaterial is the GPU table layout: 48 bytes (C99 has no static_assert). */
typedef char caelix_material_is_48_bytes[(sizeof(CaelixMaterial) == 48) ? 1 : -1];

size_t caelix_abi_c99_touch(void) {
  CaelixVersion version = {0, 0, 0, CAELIX_ABI_VERSION};
  CaelixDeviceRequirements requirements = {0, 0, NULL, CAELIX_FEATURE_RAY_QUERY};
  CaelixDeviceAdoptDesc adopt = {NULL, 0, 0, 0, 0, 0};
  CaelixTargetImage target = {0, 0, 0, 0, 0};
  CaelixSceneDesc scene_desc = {0, 0, 0, 0};
  CaelixEntityDesc entity_desc = {{0}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f}};
  CaelixBrickKey key = {-1, 0, 1};
  CaelixFrameDesc frame = {0, 1, 0};
  CaelixSceneStats stats = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  CaelixBrickInfo info = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  CaelixCamera camera = {{0.0f, 0.0f, 0.0f}, 0.0f, {1.0f, 0.0f, 0.0f}, 0.0f, {0.0f, 1.0f, 0.0f},
                         0.0f, {0.0f, 0.0f, 1.0f}, 0.0f, 0.577f, 1.0f, 0.1f, 1000.0f,
                         {1.732f, 0.0f, 0.0f, 0.0f, 0.0f, 1.732f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.1f, 0.0f}};
  CaelixRenderOptions options = {CAELIX_VIEW_ALBEDO,
                                 CAELIX_RENDER_FLAG_SHADING | CAELIX_RENDER_FLAG_SKY_ON_MISS,
                                 {0.0f, 1.0f, 0.0f}, 0.0f, {0.5f, 0.7f, 0.9f}, 0.0f, 0.0f, 1e30f,
                                 CAELIX_MODE_PATH_TRACE, CAELIX_MAX_BOUNCES, 0, 1, 1.0f,
                                 {20.0f, 19.0f, 18.0f}, 0.0f, {0.2f, 0.4f, 0.9f}, {0.7f, 0.8f, 0.95f},
                                 {0.25f, 0.22f, 0.2f}, 1, CAELIX_MAX_BOUNCES_TRANSPARENT};
  CaelixMaterial material = {{1.0f, 1.0f, 1.0f}, 1.0f, {0.0f, 0.0f, 0.0f}, 0.0f, 1.55f, 0.0f,
                             {0.0f, 0.0f}};
  CaelixRenderTargets targets = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0},
                                 {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
  CaelixGpuTimings timings = {0.0f, 0.0f, 0, 0.0f, 0.0f};
  CaelixNrdInfo nrd_info = {4, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  CaelixDeviceFeatureBits read_bit = CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT;
  CaelixRenderMode mode = CAELIX_MODE_REFERENCE;
  CaelixDebugView view = CAELIX_VIEW_NRD_SPEC;
  CaelixRenderer* renderer = NULL;
  CaelixEntityId entity = CAELIX_INVALID_ENTITY;
  CaelixLogCallback log_callback = NULL;
  CaelixLogLevel level = CAELIX_LOG_INFO;
  CaelixResult result = CAELIX_OK;
  SceneCreateFn create = caelix_scene_create;
  size_t sum = sizeof(version) + sizeof(requirements) + sizeof(adopt) + sizeof(target) +
               sizeof(scene_desc) + sizeof(entity_desc) + sizeof(key) + sizeof(frame) +
               sizeof(stats) + sizeof(info) + sizeof(entity) + sizeof(log_callback) +
               sizeof(level) + sizeof(result) + sizeof(camera) + sizeof(options) +
               sizeof(targets) + sizeof(timings) + sizeof(view) + sizeof(renderer) +
               sizeof(mode) + sizeof(nrd_info) + sizeof(read_bit) + sizeof(material) +
               sizeof(caelix_material_is_48_bytes);
  /* Take the address of every function so a declaration typo fails to compile. Casts between
     function pointer types are valid C; the pointers are never called. */
  AnyFn functions[27];
  functions[0] = (AnyFn)caelix_get_version;
  functions[1] = (AnyFn)caelix_version_string;
  functions[2] = (AnyFn)caelix_set_log_callback;
  functions[3] = (AnyFn)caelix_get_device_requirements;
  functions[4] = (AnyFn)caelix_device_adopt;
  functions[5] = (AnyFn)caelix_device_destroy;
  functions[6] = (AnyFn)caelix_renderer_create;
  functions[7] = (AnyFn)caelix_scene_destroy;
  functions[8] = (AnyFn)caelix_scene_add_entity;
  functions[9] = (AnyFn)caelix_scene_remove_entity;
  functions[10] = (AnyFn)caelix_scene_set_entity_transform;
  functions[11] = (AnyFn)caelix_scene_upload_bricks;
  functions[12] = (AnyFn)caelix_scene_remove_bricks;
  functions[13] = (AnyFn)caelix_scene_flush;
  functions[14] = (AnyFn)caelix_scene_recycle;
  functions[15] = (AnyFn)caelix_scene_get_stats;
  functions[16] = (AnyFn)caelix_scene_query_brick;
  functions[17] = (AnyFn)caelix_scene_get_tlas;
  functions[18] = (AnyFn)caelix_renderer_destroy;
  functions[19] = (AnyFn)caelix_renderer_render;
  functions[20] = (AnyFn)caelix_renderer_release_target_views;
  functions[21] = (AnyFn)caelix_renderer_get_gpu_timings;
  functions[22] = (AnyFn)caelix_get_device_requirements;
  functions[23] = (AnyFn)caelix_renderer_get_nrd_info;
  functions[24] = (AnyFn)caelix_scene_set_materials;
  functions[25] = (AnyFn)caelix_scene_set_glass_palette;
  functions[26] = (AnyFn)caelix_scene_get_material;
  sum += sizeof(functions) + (create != NULL ? 1u : 0u) + (functions[22] != NULL ? 1u : 0u);
  sum += (size_t)key.x + (size_t)frame.frame_number + (size_t)info.resident;
  return sum;
}
