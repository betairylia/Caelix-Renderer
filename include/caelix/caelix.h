#ifndef CAELIX_H
#define CAELIX_H
/*
 * Caelix renderer C ABI. C99, no Vulkan headers.
 *
 * Vulkan objects cross this ABI as plain integers and pointers so that a host
 * does not need the Vulkan headers to bind the library:
 *   - Every Vulkan handle, dispatchable (VkInstance, VkDevice, VkQueue,
 *     VkCommandBuffer, ...) and non-dispatchable (VkImage, ...), is a uint64_t
 *     that holds the raw handle value.
 *   - Vulkan function pointers are void*.
 *   - Vulkan enums (VkFormat, VkImageLayout, ...) are uint32_t; the doc comment
 *     of each field names the Vulkan type.
 *
 * Ownership model: the HOST creates and owns the Vulkan instance, device,
 * queues, swapchain and command buffers. The library adopts the host's device
 * (caelix_device_adopt) and records its work into command buffers that the
 * host provides. The library never submits or presents.
 *
 * Thread rule: calls on one object (a CaelixDevice, a CaelixScene, a
 * CaelixRenderer) are externally synchronized by the host. The library does not
 * lock: the host must not call two functions on the same scene (or on a scene,
 * its renderer and its device) at the same time from different threads.
 */
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32)
  #if defined(CAELIX_BUILD_SHARED)
    #define CAELIX_API __declspec(dllexport)
  #else
    #define CAELIX_API __declspec(dllimport)
  #endif
#else
  #define CAELIX_API __attribute__((visibility("default")))
#endif

/* Bumped on every breaking change of this header. Exception: version 2 is still in development;
   the finalize step and the acceleration structure step appended fields to CaelixSceneDesc
   (encoding, scratch_bytes; 0 = default), CaelixSceneStats and CaelixBrickInfo, and added
   caelix_scene_get_tlas, without a bump. The renderer step removed caelix_clear_target and added
   the renderer API (caelix_renderer_*) and CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT, also
   without a bump. The optional-targets step appended CaelixCamera::projection,
   CaelixRenderOptions::miss_depth / miss_view_z and the albedo, normal, depth, motion and view_z
   targets of CaelixRenderTargets, and made the colour target optional, also without a bump. The
   lighting step (6a) appended the path-tracing fields to CaelixRenderOptions (mode .. frame_seed),
   CAELIX_RENDER_FLAG_SKY_ON_MISS and CaelixGpuTimings::composite_ms, also without a bump. The sun
   NEE step added CAELIX_RENDER_FLAG_SUN_NEE (a new flag bit, no struct change). The NRD step (6b)
   added CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT, the debug views CAELIX_VIEW_NRD_*,
   CaelixNrdInfo and caelix_renderer_get_nrd_info, renamed CaelixGpuTimings::pad to denoise_ms
   (same layout) and made CaelixRenderOptions::denoise active, also without a bump. The motion
   view step added CAELIX_VIEW_MOTION (a new enum value, no struct change). The materials step (6d)
   added CaelixMaterial, caelix_scene_set_materials, caelix_scene_set_glass_palette,
   caelix_scene_get_material, CaelixSceneStats::material_table_bytes,
   CaelixRenderOptions::bounce_count_transparent, CAELIX_MAX_BOUNCES_TRANSPARENT and the debug views
   CAELIX_VIEW_MATERIAL and CAELIX_VIEW_NRD_SPEC, also without a bump; its follow-up added
   CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD (a new flag bit) and the RGBA formats of the motion target
   (w = the checkerboard flag), also without a bump. The Unity plugin step (P3) appended
   CaelixCamera::jitter and added CAELIX_RENDER_FLAG_FLIP_Y (a new flag bit), also without a
   bump. A host
   built against an older version-2 header must be rebuilt (the library reads and writes the larger
   structs). */
#define CAELIX_ABI_VERSION 2u

typedef struct CaelixVersion {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
  uint32_t abi;   /* == CAELIX_ABI_VERSION of the built library */
} CaelixVersion;

/* Fills *out with the library's version. Never fails. */
CAELIX_API void caelix_get_version(CaelixVersion* out);
/* Static string like "0.1.0", owned by the library. */
CAELIX_API const char* caelix_version_string(void);

/* Result of every fallible ABI call. */
typedef enum CaelixResult {
  CAELIX_OK = 0,
  CAELIX_ERROR_INVALID_ARGUMENT = 1,
  CAELIX_ERROR_UNSUPPORTED = 2,      /* the adopted device lacks a requirement; see log */
  CAELIX_ERROR_VULKAN = 3,           /* a Vulkan call failed; see log */
  CAELIX_ERROR_OUT_OF_MEMORY = 4
} CaelixResult;

/* ---------------------------------------------------------------------------
 * Logging
 * ------------------------------------------------------------------------- */

typedef enum CaelixLogLevel {
  CAELIX_LOG_DEBUG = 0,
  CAELIX_LOG_INFO = 1,
  CAELIX_LOG_WARNING = 2,
  CAELIX_LOG_ERROR = 3
} CaelixLogLevel;

/* `message` is a NUL-terminated UTF-8 string, valid only during the call. */
typedef void (*CaelixLogCallback)(CaelixLogLevel level, const char* message, void* user_data);
/* Thread-safe to set; messages may come from any thread the library is called on. NULL disables logging. */
CAELIX_API void caelix_set_log_callback(CaelixLogCallback callback, void* user_data);

/* ---------------------------------------------------------------------------
 * Device requirements and adoption
 * ------------------------------------------------------------------------- */

/* Device features the library needs ENABLED on the adopted device. Vulkan 1.3 core features are
   listed too because the host must still enable them at vkCreateDevice. */
typedef enum CaelixDeviceFeatureBits {
  CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS   = 1u << 0, /* VkPhysicalDeviceVulkan12Features::bufferDeviceAddress */
  CAELIX_FEATURE_DESCRIPTOR_INDEXING     = 1u << 1, /* runtimeDescriptorArray, shaderStorageBufferArrayNonUniformIndexing, descriptorBindingPartiallyBound, descriptorBindingVariableDescriptorCount */
  CAELIX_FEATURE_SYNCHRONIZATION2        = 1u << 2, /* VkPhysicalDeviceVulkan13Features::synchronization2 */
  CAELIX_FEATURE_DYNAMIC_RENDERING       = 1u << 3, /* VkPhysicalDeviceVulkan13Features::dynamicRendering */
  CAELIX_FEATURE_TIMELINE_SEMAPHORE      = 1u << 4, /* VkPhysicalDeviceVulkan12Features::timelineSemaphore */
  CAELIX_FEATURE_ACCELERATION_STRUCTURE  = 1u << 5, /* VkPhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure */
  CAELIX_FEATURE_RAY_QUERY               = 1u << 6, /* VkPhysicalDeviceRayQueryFeaturesKHR::rayQuery */
  CAELIX_FEATURE_SHADER_INT64            = 1u << 7, /* VkPhysicalDeviceFeatures::shaderInt64 */
  CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT     = 1u << 8, /* VkPhysicalDeviceVulkan12Features::scalarBlockLayout */
  CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT = 1u << 9, /* VkPhysicalDeviceFeatures::shaderStorageImageWriteWithoutFormat */
  CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT = 1u << 10  /* VkPhysicalDeviceFeatures::shaderStorageImageReadWithoutFormat
                                                                   (NRD's temporal stabilization and validation passes) */
} CaelixDeviceFeatureBits;

typedef struct CaelixDeviceRequirements {
  uint32_t api_version;                 /* VK_MAKE_API_VERSION(0,1,3,0) */
  uint32_t device_extension_count;
  const char* const* device_extensions; /* static strings owned by the library */
  uint32_t feature_flags;               /* CaelixDeviceFeatureBits */
} CaelixDeviceRequirements;

/* Fills *out with what the host must request at vkCreateInstance / vkCreateDevice.
   Call before creating the device. Never fails; a NULL `out` is ignored. */
CAELIX_API void caelix_get_device_requirements(CaelixDeviceRequirements* out);

typedef struct CaelixDeviceAdoptDesc {
  void*    get_instance_proc_addr;  /* PFN_vkGetInstanceProcAddr */
  uint64_t instance;                /* VkInstance */
  uint64_t physical_device;         /* VkPhysicalDevice */
  uint64_t device;                  /* VkDevice */
  uint64_t graphics_queue;          /* VkQueue, graphics + compute capable */
  uint32_t graphics_queue_family;
} CaelixDeviceAdoptDesc;

typedef struct CaelixDevice CaelixDevice;  /* opaque */

/* Adopts the host's device. Checks that the physical device supports every requirement of
   caelix_get_device_requirements, and that its subgroups support quad operations in compute shaders
   (VkPhysicalDeviceVulkan11Properties: subgroupSupportedStages has COMPUTE, subgroupSupportedOperations
   has QUAD; NRD's shaders use QuadReadAcrossX/Y; a property, nothing to enable). A missing item is
   logged and returns CAELIX_ERROR_UNSUPPORTED.
   The library loads Vulkan through `get_instance_proc_addr`, so it uses the host's loader and
   layers. One adopted device per process at a time.
   The host keeps ownership of every Vulkan object in the desc and must keep them alive until caelix_device_destroy. */
CAELIX_API CaelixResult caelix_device_adopt(const CaelixDeviceAdoptDesc* desc, CaelixDevice** out_device);
/* Releases the library's objects on the device (never the host's objects). NULL is ignored.
   The host must ensure the device is idle (vkDeviceWaitIdle) before calling this. */
CAELIX_API void caelix_device_destroy(CaelixDevice* device);

/* ---------------------------------------------------------------------------
 * Render targets
 * ------------------------------------------------------------------------- */

/* An image that the library writes (see CaelixRenderTargets). The host owns it; the library only
   creates and caches image views of it (see caelix_renderer_release_target_views). */
typedef struct CaelixTargetImage {
  uint64_t image;   /* VkImage */
  uint32_t format;  /* VkFormat */
  uint32_t width;
  uint32_t height;
  uint32_t layout;  /* VkImageLayout of the image when the library starts recording */
} CaelixTargetImage;

/* ---------------------------------------------------------------------------
 * Scene: voxel bricks on the GPU
 *
 * The HOST owns the raw voxel data. A scene stores only what the GPU needs: one
 * GPU record per resident brick, a slot table per render group, and a CPU index
 * from brick keys to records. The host uploads raw bricks; the library converts
 * each one into a record in a host-visible staging ring, and caelix_scene_flush
 * records the copies from the ring to the GPU pools into the host's command
 * buffer.
 *
 * Bricks are 8x8x8 blocks. A brick key is the entity-local block coordinate
 * shifted right by 3 (floor). Bricks are grouped in render groups of
 * (1 << group_shift)^3 bricks; every group has its own slot table with dense
 * brick ids.
 *
 * Finalization: caelix_scene_flush runs the "finalize" kernel over every brick uploaded since the
 * previous flush and every resident neighbour (6-neighbourhood) of an uploaded or removed brick. It
 * fills the record header (coarse occupancy, tight bounds, micro occupancy), adds face bits to
 * transparent voxels of the payload and writes the brick's AABB row. A brick is finalized at the
 * flush after its upload (or, when the staging ring is full, at a later flush); the host must not
 * read records or AABB rows before that flush has completed on the GPU.
 *
 * Lossless rule: the payload keeps the uploaded block values. Opaque values stay unchanged;
 * transparent values (no opaque bit) get face bits in the bits above their id, so
 * raw = opaque ? value : (value & id_mask) recovers the uploaded value, and finalizing a brick
 * again gives the same record. Uploaded transparent values must carry no bits above the id mask.
 *
 * Block encodings (CaelixSceneDesc::encoding):
 *   0 packed scene colour: opaque when (value & 0xC000) != 0; transparent id = value & 0xFF,
 *     face bits at bits 8..13.
 *   1 block-ID: opaque when (value & 0x8000) != 0; transparent id = value & 0x1FF, face bits at
 *     bits 9..14.
 * Face bit d (order +X, -X, +Y, -Y, +Z, -Z) of a transparent voxel is set when its neighbour in
 * direction d is not opaque and has a different id. Air (id 0) next to glass gets face bits too.
 *
 * Record pool pages never move: a record keeps its page and offset until the
 * brick is removed.
 *
 * Acceleration structures: every render group has one bottom-level acceleration structure (BLAS)
 * over its AABB rows (procedural AABB geometry, one primitive per dense brick id; a free id or a
 * brick without occupied voxels has a NaN row and is an inactive primitive). The scene has one
 * top-level acceleration structure (TLAS) with one instance per group that has a BLAS; the instance
 * transform is the entity transform (the AABBs are in entity-local block coordinates) and
 * instanceCustomIndex is the scene-wide group index. caelix_scene_flush builds them after the
 * finalize kernel: a group's BLAS is rebuilt when its bricks or AABB rows change, the TLAS when a
 * BLAS was built, a group was freed, or an entity was moved or removed. Builds share a scratch
 * buffer (CaelixSceneDesc::scratch_bytes) and the TLAS instances go through the staging ring; the
 * builds that do not fit wait for a later flush. A brick is traceable in the commands recorded
 * after the flush that finalizes it when both budgets allowed its group's build; otherwise after a
 * later flush. CaelixSceneStats::as_builds_pending == 0 means that every acceleration structure
 * matches the bricks the host has uploaded and that flushes have finalized.
 *
 * Frame contract: the host calls caelix_scene_flush once per frame that has
 * scene changes (or every frame), with a strictly increasing frame number and
 * the newest frame number that has completed on the GPU. Staging space and
 * replaced GPU buffers are released only when their frame is <= that number.
 * ------------------------------------------------------------------------- */

typedef struct CaelixScene CaelixScene;  /* opaque */

typedef struct CaelixSceneDesc {
  uint64_t staging_bytes;      /* host-visible upload ring. 0 = 64 MiB. Must be >= 4 MiB. */
  uint32_t records_per_page;   /* record pool page size in bricks, power of two. 0 = 1 << 18. */
  uint32_t group_shift;        /* render group edge in bricks = 1 << group_shift. 0 = 4 (16^3). Max 8.
                                  Only 1..5 are implemented in this version: 6..8 return
                                  CAELIX_ERROR_UNSUPPORTED (dense brick ids are 16-bit). */
  uint32_t block_bytes;        /* bytes per block value. 0 = 2. Only 2 is implemented in this version;
                                  4 returns CAELIX_ERROR_UNSUPPORTED. */
  uint32_t encoding;           /* block encoding for finalization: 0 = packed scene colour (the
                                  default, what Caelix worlds use), 1 = block-ID. Other values
                                  return CAELIX_ERROR_INVALID_ARGUMENT. See the encodings above. */
  uint64_t scratch_bytes;      /* scratch buffer of the acceleration structure builds of one flush.
                                  0 = 128 MiB. A flush builds as many BLASes as fit (the TLAS
                                  first); the rest wait. The buffer grows when the TLAS and the
                                  largest single BLAS do not fit together. */
} CaelixSceneDesc;

/* Creates an empty scene on an adopted device. `desc` may be NULL (all defaults).
   The scene keeps a reference to `device`: destroy every scene before caelix_device_destroy. */
CAELIX_API CaelixResult caelix_scene_create(CaelixDevice* device, const CaelixSceneDesc* desc,
                                            CaelixScene** out_scene);
/* Destroys the scene and all its GPU buffers. NULL is ignored.
   The host must guarantee the device is idle, or that safe_frame_number has passed the last flush. */
CAELIX_API void caelix_scene_destroy(CaelixScene* scene);

/* Entity ids are small indices. The id of a removed entity is reused by a later add. */
typedef uint32_t CaelixEntityId;
#define CAELIX_INVALID_ENTITY 0xFFFFFFFFu

typedef struct CaelixEntityDesc {
  uint8_t guid[16];      /* stored for the host; the library does not interpret it */
  float position[3];     /* world position of the entity origin */
  float rotation[4];     /* unit quaternion x, y, z, w */
} CaelixEntityDesc;

/* Adds an entity with no bricks. */
CAELIX_API CaelixResult caelix_scene_add_entity(CaelixScene* scene, const CaelixEntityDesc* desc,
                                                CaelixEntityId* out_id);
/* Removes the entity and frees all its bricks and groups. */
CAELIX_API CaelixResult caelix_scene_remove_entity(CaelixScene* scene, CaelixEntityId entity);
CAELIX_API CaelixResult caelix_scene_set_entity_transform(CaelixScene* scene, CaelixEntityId entity,
                                                          const float position[3],
                                                          const float rotation[4]);

/* Entity-local brick key = block >> 3 per axis. Each axis must be in [-2^20, 2^20). */
typedef struct CaelixBrickKey {
  int32_t x, y, z;
} CaelixBrickKey;

/* Raw bricks, borrowed for the duration of the call. `blocks` holds count * 512 block values of
   block_bytes each; voxel index = x + 8*y + 64*z. A brick with the same key is replaced. The call
   accepts as many bricks as the staging ring can hold and writes the number to *consumed; when
   *consumed < count the host must call caelix_scene_flush (and present/submit that frame) before
   uploading the rest. The staging space of a flushed frame is free again when the host reports that
   frame as completed (caelix_scene_recycle or a later flush). Returns CAELIX_OK even when
   *consumed == 0. Every key is checked before any brick is accepted: one key out of range returns
   CAELIX_ERROR_INVALID_ARGUMENT and accepts nothing. When a later error stops the call (for example
   CAELIX_ERROR_OUT_OF_MEMORY for a new record page), *consumed still counts the bricks accepted
   before it. */
CAELIX_API CaelixResult caelix_scene_upload_bricks(CaelixScene* scene, CaelixEntityId entity,
                                                   uint32_t count, const CaelixBrickKey* keys,
                                                   const void* blocks, uint32_t* consumed);
/* Removes bricks. Keys that the entity does not have are ignored. A group whose last brick is
   removed is freed. */
CAELIX_API CaelixResult caelix_scene_remove_bricks(CaelixScene* scene, CaelixEntityId entity,
                                                   uint32_t count, const CaelixBrickKey* keys);

typedef struct CaelixFrameDesc {
  uint64_t command_buffer;     /* VkCommandBuffer in the recording state, graphics queue */
  uint64_t frame_number;       /* strictly increasing across flushes. Must be > safe_frame_number,
                                  so the first frame number is 1 or higher. */
  uint64_t safe_frame_number;  /* every frame <= this has completed on the GPU. 0 = none yet. */
} CaelixFrameDesc;

/* Records into the command buffer all copies from staging to the pools for everything uploaded or
   removed since the previous flush, then the finalize kernel over the changed bricks and their
   neighbours, then the acceleration structure builds (BLASes, then the TLAS), with barriers: one
   before the copies (all earlier commands -> all commands), transfer -> compute before the kernel,
   transfer and compute -> all later commands after it, BLAS builds -> TLAS build, and TLAS build ->
   all later commands. When it records work, two timestamp queries bracket it (reported as
   CaelixGpuTimings::flush_ms). Recycles staging spans and acceleration structures whose
   frame <= safe. With nothing to do it records nothing. A frame_number that is not greater than the previous flush's,
   or a safe_frame_number >= frame_number, returns CAELIX_ERROR_INVALID_ARGUMENT. When an
   acceleration structure cannot be allocated, the flush records everything else, the builds stay
   pending, and it returns CAELIX_ERROR_OUT_OF_MEMORY. */
CAELIX_API CaelixResult caelix_scene_flush(CaelixScene* scene, const CaelixFrameDesc* desc);

/* Tells the scene that every frame <= safe_frame_number has completed on the GPU and releases the
   staging spans and old GPU buffers of those frames now. Optional: caelix_scene_flush does the same.
   Use it when the host waits for a frame and then uploads before the next flush, for example in a
   loop that streams more bricks than the staging ring holds. A value lower than an earlier one has
   no effect. */
CAELIX_API CaelixResult caelix_scene_recycle(CaelixScene* scene, uint64_t safe_frame_number);

/* Fields are only appended. finalized_bricks_total .. grid_bytes were added to ABI version 2 by the
   finalize step, blas_count .. as_builds_pending by the acceleration structure step. */
typedef struct CaelixSceneStats {
  uint32_t entities, groups;                               /* live entities and render groups */
  uint64_t bricks_resident, bricks_known;                  /* with a GPU record; with a slot */
  uint64_t record_bytes, slot_table_bytes, staging_bytes;  /* GPU allocation sizes */
  uint64_t uploaded_bricks_total, pending_bricks;          /* pending = staged, not yet flushed */
  uint64_t record_pages;
  uint64_t finalized_bricks_total;  /* finalize kernel invocations (bricks) recorded so far */
  uint64_t reverse_table_bytes;     /* GPU allocation sizes of the finalize lookup tables: */
  uint64_t aabb_bytes;              /*   reverse tables, AABB rows, group grids */
  uint64_t grid_bytes;
  uint64_t blas_count;              /* live groups that have a BLAS */
  uint64_t blas_bytes;              /* BLAS storage blocks, including blocks whose release waits for
                                       a safe frame */
  uint64_t tlas_bytes;              /* TLAS storage block(s), same rule */
  uint64_t scratch_bytes;           /* capacity of the build scratch buffer */
  uint64_t as_builds_total;         /* BLAS and TLAS builds recorded so far */
  uint64_t as_builds_pending;       /* groups whose BLAS is out of date (waiting for finalization,
                                       scratch or ring space), plus 1 when the TLAS is out of date */
  uint64_t material_table_bytes;    /* GPU size of the material table (65536 * 48 = 3 MiB) */
} CaelixSceneStats;

/* Fills *out. A NULL argument is ignored. */
CAELIX_API void caelix_scene_get_stats(const CaelixScene* scene, CaelixSceneStats* out);

/* Inspection and debug aid: where the GPU record, the slot table row, the AABB row, the group's
   BLAS and the entity's info row of a brick live. The GPU
   contents match the last flush; the buffers can change at the next upload or flush (the slot table
   grows by reallocation), so use the handles only until then. An unknown brick reports
   resident = 0 and 0 in every other field.
   Slot table row (16 bytes): uint32 pos_state (bits 0..23 group-local position, bits 24..31 state:
   0 free, 1 resident, 2 summary-only, 3 requested), uint32 summary, uint32 record_index
   (0xFFFFFFFF when not resident), uint32 last_used. */
typedef struct CaelixBrickInfo {
  uint32_t resident;                  /* 1 when the brick has a GPU record */
  uint32_t group_index;               /* scene-wide render group index */
  uint32_t group_local_id;            /* dense brick id inside the group = slot table row */
  uint32_t record_index;              /* global record index = page << log2(records_per_page) | slot */
  uint64_t record_page_address;       /* VkDeviceAddress of the record's page buffer */
  uint64_t record_page_buffer;        /* VkBuffer of the record's page */
  uint64_t record_page_offset_bytes;  /* byte offset of the record inside its page */
  uint64_t slot_table_buffer;         /* VkBuffer that holds every slot table of the scene */
  uint64_t slot_row_offset_bytes;     /* byte offset of the brick's slot table row */
  uint64_t aabb_buffer;               /* VkBuffer that holds every AABB row of the scene */
  uint64_t aabb_row_offset_bytes;     /* byte offset of the brick's AABB row (24 bytes: float min[3],
                                         float max[3], entity-local block coordinates, all NaN
                                         when the brick has no occupied voxel) */
  uint64_t blas_address;              /* VkDeviceAddress of the group's BLAS, 0 when not built yet */
  uint64_t blas_handle;               /* VkAccelerationStructureKHR of the group's BLAS, 0 when not
                                         built yet. Owned by the library. */
  uint64_t entity_info_buffer;        /* VkBuffer that holds the entity info rows */
  uint64_t entity_info_offset_bytes;  /* byte offset of the entity's info row (128 bytes: int32
                                         grid_min[3], uint32 grid_base, int32 grid_dims[3], uint32
                                         pad, float cur_rows[3][4], float prev_rows[3][4]; the rows
                                         are the object-to-world matrix, rotation | position, of the
                                         current and of the previous frame's transform) */
} CaelixBrickInfo;

CAELIX_API CaelixResult caelix_scene_query_brick(const CaelixScene* scene, CaelixEntityId entity,
                                                 CaelixBrickKey key, CaelixBrickInfo* out);

/* The scene's TLAS: *out_handle = its VkAccelerationStructureKHR, *out_address = its
   VkDeviceAddress, both 0 before the first TLAS build. This is the handle the renderer binds. The
   library owns it: the host must not destroy it. The TLAS may be rebuilt into a new handle at any
   flush, so query it after the flush whose commands use it. NULL out pointers return
   CAELIX_ERROR_INVALID_ARGUMENT. */
CAELIX_API CaelixResult caelix_scene_get_tlas(const CaelixScene* scene, uint64_t* out_handle,
                                              uint64_t* out_address);

/* ---------------------------------------------------------------------------
 * Materials
 *
 * Every scene has a material table: 65536 CaelixMaterial entries, one per 16-bit block code, in a
 * device-local GPU buffer that the lighting modes read (Unity's g_Materials). A transparent code is
 * looked up with its face bits removed (code & id mask, see the encodings above), so only the entry
 * of the face-free code is used; an opaque code is looked up as it is.
 *
 * Built-in content, by encoding:
 *   packed: an opaque code (bit 15) is albedo rgb5 / 31 (r = bits 14..10, g = 9..5, b = 4..0); an
 *     emissive code (bits 15..14 = 01) is albedo rgb4 / 15 (r = bits 13..10, g = 9..6, b = 5..2)
 *     and emission albedo * 2^(3 * (code & 3)); both smoothness 0, metallic 0, ior 1.5,
 *     extinction 0. A transparent code (bits 15..14 = 00) is glass palette entry code & 0xFF: the
 *     built-in palette is Caelix's glass-palette.json (ids 1..132: tint, ior, extinction,
 *     smoothness; metallic 0, no emission); id 0 (air) and the reserved ids 133..255 are albedo 1,
 *     smoothness 1, ior 1, extinction 0 (Unity's GetSharedGlass default).
 *   block-ID: an opaque code is a pastel hash colour of code & 0x7FFF; a transparent id 1..511 a
 *     pastel hash colour of the id; both smoothness 0, metallic 0, ior 1.5, no emission. Air (id 0)
 *     is the vacuum entry of the packed table. The host is expected to replace the table.
 * Changes take effect at the next caelix_scene_flush, which uploads the changed part of the table
 * through the staging ring (a part that does not fit waits for a later flush). Until the first
 * upload has completed the GPU table is all zero (black, no emission).
 * ------------------------------------------------------------------------- */

/* One material table entry. 48 bytes, the layout of the GPU table. All values are linear. */
typedef struct CaelixMaterial {
  float albedo[3];    /* diffuse / base colour; for a transparent medium its tint */
  float smoothness;   /* 0..1. NRD roughness = 1 - smoothness. A transparent entry with smoothness
                         >= 0.99 is a delta interface (reflection and refraction in the delta chain,
                         CaelixRenderOptions::bounce_count_transparent); a rougher one is shaded
                         stochastically (still a perfect reflection or refraction, as in Unity) */
  float emission[3];  /* emitted radiance */
  float metallic;     /* 0..1; opaque entries: specular chance lerp(metallic, 1, fresnel * smoothness) */
  float ior;          /* index of refraction; must be > 0 (transparent entries use it at interfaces,
                         opaque entries for the Fresnel term of the specular chance) */
  float extinction;   /* transparent entries: per voxel unit; a segment of length t inside the medium
                         transmits exp(-(1 - albedo) * t * extinction). Must be >= 0. */
  float pad[2];       /* ignored; stored as 0 */
} CaelixMaterial;

/* Replaces the whole material table: entries [0, count) come from `entries`, entries [count, 65536)
   get the built-in default of the scene's encoding (packed: with the current glass palette).
   count <= 65536 (0 restores the default table).
   Every value must be finite, ior > 0 and extinction >= 0: an invalid entry returns
   CAELIX_ERROR_INVALID_ARGUMENT (logged, the table is unchanged). Entries of transparent codes with
   face bits are stored but never read. Packed scenes: a later caelix_scene_set_glass_palette
   rewrites the transparent codes. */
CAELIX_API CaelixResult caelix_scene_set_materials(CaelixScene* scene, const CaelixMaterial* entries,
                                                   uint32_t count);
/* Packed encoding only (a block-ID scene returns CAELIX_ERROR_INVALID_ARGUMENT): replaces glass
   palette ids 1 .. count - 1 by entries[1 .. count - 1] (entries[0] is ignored: air stays the vacuum
   entry; ids >= count keep their current palette value) and rewrites the table entries of every
   transparent code with such an id. count <= 256. Same value checks as caelix_scene_set_materials. */
CAELIX_API CaelixResult caelix_scene_set_glass_palette(CaelixScene* scene,
                                                       const CaelixMaterial* entries, uint32_t count);
/* Inspection: the current table entry of a raw block code (code <= 0xFFFF; face bits of a
   transparent code are removed first, as the kernels do). The value is the CPU copy: it reaches the
   GPU at the next flush. */
CAELIX_API CaelixResult caelix_scene_get_material(const CaelixScene* scene, uint32_t code,
                                                  CaelixMaterial* out);

/* ---------------------------------------------------------------------------
 * Renderer: traces a scene into the host's images
 *
 * A renderer draws one scene. caelix_renderer_render records a compute dispatch that traces one
 * primary ray per pixel through the scene's TLAS with an inline ray query (VK_KHR_ray_query), runs
 * the brick DDA on every procedural candidate, and writes the result of the hit into the targets
 * that the host passes (CaelixRenderTargets): a colour image and the optional geometry targets
 * albedo, normal, depth, motion and view_z. Every target is optional, but at least one must be
 * present.
 *
 * Modes (CaelixRenderOptions::mode):
 *   CAELIX_MODE_PRIMARY    the colour target shows a debug view (CaelixDebugView) of the primary hit.
 *                          One dispatch (trace).
 *   CAELIX_MODE_PATH_TRACE path-traced lighting at 1 sample per pixel with the shading rules of the
 *                          Unity path tracer (primary emission, cosine-weighted diffuse bounces,
 *                          Russian roulette, procedural sky with a sun disk on a miss, optional
 *                          next-event estimation of the sun: CAELIX_RENDER_FLAG_SUN_NEE). Two
 *                          dispatches: trace (G-buffer, internal radiance targets) and composite
 *                          (colour = emission + albedo * incident radiance, exposure, sRGB encode for
 *                          UNORM targets). With CaelixRenderOptions::denoise = 1 (and a colour
 *                          target) NVIDIA NRD (REBLUR_DIFFUSE_SPECULAR) denoises the diffuse and
 *                          specular radiance between the two dispatches; with 0 the colour target
 *                          holds the raw 1 spp estimate.
 *   CAELIX_MODE_REFERENCE  as CAELIX_MODE_PATH_TRACE, but the composite accumulates the raw frames
 *                          into an internal RGBA32F image (running mean since the last reset) and
 *                          writes the mean. Use a static camera and a new frame_seed every frame.
 *                          Never denoised.
 * In the two lighting modes the renderer owns internal targets (about 90 bytes per pixel) sized to
 * the host's targets; they are created at the first lighting render and recreated when the target
 * size changes. The NRD instance (its own pools, see caelix_renderer_get_nrd_info) is created at the
 * first denoised render and recreated (history lost) when the target size changes: NRD has no
 * resize. Its 14 compute pipelines are created again too, a one-time CPU cost per size change.
 *
 * G-buffer mode: a host with its own deferred pipeline passes only the geometry targets (no colour
 * target). The library then writes no lighting at all: per pixel, the material albedo, the world
 * normal, the device depth of the host's projection, the screen-space motion vector and the linear
 * view depth, with coverage (1 on a hit, 0 on a miss) in the alpha channel of albedo and normal.
 *
 * Mapping to NVIDIA NRD (the renderer's own NRD instance uses internal copies of these targets; a
 * host with its own NRD integration can use the host targets the same way):
 *   view_z -> IN_VIEWZ (positive linear view depth; give NRD the same view and projection).
 *   motion -> IN_MV with CommonSettings::motionVectorScale = (1 / width, 1 / height) and
 *             isMotionVectorInWorldSpace = false: NRD's uvPrev = uv + mv * scale matches
 *             motion = previous_pixel - current_pixel (both use a top-left origin, y down).
 *   normal -> IN_NORMAL_ROUGHNESS after packing with a roughness (NRD_FrontEnd_PackNormalAndRoughness).
 *   albedo -> not an NRD input: demodulate the radiance by it before the denoiser and multiply after.
 *
 * Frame contract: call caelix_renderer_render AFTER caelix_scene_flush of the same frame, into the
 * same command buffer and with the same CaelixFrameDesc: the flush builds the TLAS that the render
 * binds, and the TLAS handle can change at any flush. The renderer keeps its per-frame resources
 * (descriptor set, constants, timestamp queries) in a ring of 3 slots that grows to at most 8
 * frames in flight; a slot is reused only when its frame is <= safe_frame_number.
 *
 * Layout contract: every present target is received in its CaelixTargetImage::layout and left in
 * VK_IMAGE_LAYOUT_GENERAL. The last barrier of the render call makes the writes available to all
 * later commands (ALL_COMMANDS, memory read and write); the host transitions the images from
 * GENERAL for present or for its own use.
 *
 * Previous frame: the renderer keeps the camera of its previous render call and uses it, with the
 * entity's previous transform (CaelixBrickInfo: prev_rows), for the motion target. The first render
 * call of a renderer has no previous camera and writes motion 0.
 * ------------------------------------------------------------------------- */

typedef struct CaelixRenderer CaelixRenderer;  /* opaque */

/* Creates a renderer for `scene` (which must belong to `device`). The renderer keeps references to
   both: destroy it before the scene and the device. */
CAELIX_API CaelixResult caelix_renderer_create(CaelixDevice* device, CaelixScene* scene,
                                               CaelixRenderer** out_renderer);
/* Destroys the renderer and its GPU objects (pipeline, descriptor pools, constants, timestamp
   queries, cached image views). NULL is ignored. The device must be idle. */
CAELIX_API void caelix_renderer_destroy(CaelixRenderer* renderer);

/* The camera of one render call, in world space. The voxel world is Y up and uses Unity's
   convention: +X right, +Y up, +Z forward. right, up and forward must form an orthonormal basis.
   Pixel (px, py), row 0 = the TOP row of the image, traces the ray from `position` in direction
     s   = (px + 0.5 + jitter[0], py + 0.5 + jitter[1])          (the sample position, pixels)
     ndc = (s.x / width * 2 - 1, 1 - s.y / height * 2)
     dir = normalize(right * ndc.x * tan_half_fov_y * aspect + up * ndc.y * tan_half_fov_y + forward)
   (CAELIX_RENDER_FLAG_FLIP_Y changes only where the result is stored, not this mapping.) */
typedef struct CaelixCamera {
  float position[3];
  float pad0;
  float right[3];
  float pad1;
  float up[3];
  float pad2;
  float forward[3];
  float pad3;
  float tan_half_fov_y;  /* tangent of half the VERTICAL field of view */
  float aspect;          /* width / height */
  float near_z, far_z;   /* distance range of the depth debug view only (no clipping) */
  /* The host's projection, column-major 4x4 (row r, column c = projection[c * 4 + r]):
     clip = projection * (view.x, view.y, view.z, 1) with the view space of this camera,
       view = (dot(p - position, right), dot(p - position, up), dot(p - position, forward)).
     It must describe the same frustum as tan_half_fov_y and aspect; the library does not check.
     Used only for the depth target (device depth = clip.z / clip.w). A normal-Z host passes, for
     example, clip.z = far / (far - near) * (z - near), clip.w = z; a reverse-Z infinite host
     clip.z = near, clip.w = z. Pass the projection WITHOUT jitter (NRD's viewToClipMatrix). */
  float projection[16];
  /* Sub-pixel offset of the primary sample in pixels, x right and y DOWN (the row order of the
     unflipped image), typically in [-0.5, 0.5]; {0, 0} = the pixel centre. For temporal
     anti-aliasing the host varies it per frame. It moves the primary rays only: the motion target
     stays the motion of the surface itself (previous pixel - the sample position, both through
     the unjittered pinhole mappings), so a static camera over a static scene writes motion 0 for
     any jitter. NRD receives it as CommonSettings::cameraJitter (and the previous call's as
     cameraJitterPrev), clamped to [-0.5, 0.5]. A host that renders into a y-up image
     (CAELIX_RENDER_FLAG_FLIP_Y) still passes y down here. */
  float jitter[2];
} CaelixCamera;

/* CaelixRenderOptions::mode */
typedef enum CaelixRenderMode {
  CAELIX_MODE_PRIMARY = 0,     /* debug views of the primary hit (CaelixDebugView) */
  CAELIX_MODE_PATH_TRACE = 1,  /* path-traced lighting, raw 1 sample per pixel */
  CAELIX_MODE_REFERENCE = 2    /* path-traced lighting, accumulated over frames (reference image) */
} CaelixRenderMode;

/* Largest CaelixRenderOptions::bounce_count (Unity's path tracer has the same limit). */
#define CAELIX_MAX_BOUNCES 14u
/* Largest CaelixRenderOptions::bounce_count_transparent. */
#define CAELIX_MAX_BOUNCES_TRANSPARENT 15u

typedef enum CaelixDebugView {
  CAELIX_VIEW_ALBEDO = 0,  /* material albedo; Lambert-shaded with CAELIX_RENDER_FLAG_SHADING */
  CAELIX_VIEW_NORMAL = 1,  /* world normal * 0.5 + 0.5 */
  CAELIX_VIEW_DEPTH = 2,   /* grey (t - near_z) / (far_z - near_z), clamped to [0, 1] */
  CAELIX_VIEW_GROUP = 3,   /* hash colour of the render group of the hit */
  CAELIX_VIEW_BRICK = 4,   /* hash colour of the brick of the hit (group and dense brick id) */
  CAELIX_VIEW_STEPS = 5,   /* heat map of the procedural candidates of the ray, 0 .. 64 */
  /* Views 6, 7, 8 and 10 apply to CAELIX_MODE_PATH_TRACE (CAELIX_MODE_PRIMARY shows the albedo view
     for them, CAELIX_MODE_REFERENCE ignores debug_view); view 9 to CAELIX_MODE_PRIMARY (the lighting
     modes ignore it). */
  CAELIX_VIEW_NRD_VALIDATION = 6, /* NRD's validation layer (CommonSettings::enableValidation) blended
                                     over the image by its alpha (after the sRGB encode); the normal
                                     image when the frame was not denoised */
  CAELIX_VIEW_NRD_DIFF = 7,       /* the denoised diffuse radiance without the albedo (the raw diffuse
                                     radiance when the frame was not denoised), exposure applied */
  CAELIX_VIEW_MOTION = 8,         /* the motion vector of the primary hit (the motion target's value):
                                     rgb = (saturate(motion.x / 32 * 0.5 + 0.5),
                                     saturate(motion.y / 32 * 0.5 + 0.5), 0.5 * coverage), alpha =
                                     coverage, written as is (no exposure, no sRGB encode). A hit
                                     without motion is grey (0.5, 0.5, 0.5), a miss (0.5, 0.5, 0);
                                     +-32 px saturate. */
  CAELIX_VIEW_MATERIAL = 9,       /* CAELIX_MODE_PRIMARY: the material table's albedo of the primary
                                     hit, mixed 50 % with blue (0.2, 0.45, 1) for a transparent code
                                     and 50 % with white for an emissive entry (any emission > 0);
                                     an air code is magenta. Lambert-shaded with
                                     CAELIX_RENDER_FLAG_SHADING. */
  CAELIX_VIEW_NRD_SPEC = 10       /* CAELIX_MODE_PATH_TRACE: the denoised specular radiance without
                                     the albedo (the raw specular radiance when the frame was not
                                     denoised), exposure applied; like CAELIX_VIEW_NRD_DIFF */
} CaelixDebugView;

/* CaelixRenderOptions::flags */
#define CAELIX_RENDER_FLAG_SHADING (1u << 0) /* CAELIX_MODE_PRIMARY only: Lambert shading with sun_dir
                                                on the albedo view */
#define CAELIX_RENDER_FLAG_SKY_ON_MISS (1u << 1) /* lighting modes only: the colour target shows the
                                                    procedural sky where the primary ray misses (alpha
                                                    stays 0). Without it a miss is (0, 0, 0, 0), for a
                                                    host that composites its own sky. */
#define CAELIX_RENDER_FLAG_SUN_NEE (1u << 2) /* lighting modes only: next-event estimation of the sun.
                                                Every diffuse path vertex that has a next segment
                                                samples a direction in the sun cone and traces a
                                                shadow ray; the rays sampled from such a vertex see
                                                the sky without the sun disk (no double counting).
                                                Primary misses still show the disk. Without it the
                                                sun is found only by chance (the sky includes the
                                                disk for every ray). */
#define CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD (1u << 3) /* CAELIX_MODE_PATH_TRACE only: at the FIRST
                                                transparent interface of the delta chain
                                                (bounce_count_transparent), pixels with even
                                                px + py refract and odd ones reflect instead of a
                                                random Fresnel-weighted choice; the chosen branch
                                                carries the weight 2 (1 - F) or 2 F, and the pixel is
                                                flagged (motion target w = 1). Total internal
                                                reflection stays unsplit. With a colour target, the
                                                composite is followed by the cross resolve: a
                                                flagged pixel becomes 0.5 * itself + 0.125 * each
                                                flagged orthogonal neighbour (renormalized) on the
                                                linear colour, before the sRGB encode. Each pixel then
                                                shows the same surface every frame (no flip of its
                                                G-buffer between reflection and refraction), at the
                                                cost of a one-pixel blend on glass. Later interfaces
                                                keep the random choice. CAELIX_MODE_REFERENCE ignores
                                                the flag (a fixed per-pixel branch would converge to
                                                a checkerboard, not to the image). Unity parity. */
#define CAELIX_RENDER_FLAG_FLIP_Y (1u << 4) /* every mode: the HOST targets (CaelixRenderTargets) are
                                                stored bottom row first: the result of pixel
                                                (px, py) is written at (px, height - 1 - py), for a
                                                host whose images have row 0 at the bottom (Unity
                                                render textures). The motion target's y is
                                                negated (y grows upward in the stored image, so
                                                motion stays previous - current in the stored
                                                rows), and so is the motion debug view. Ray
                                                generation, the random stream, the jitter and the
                                                internal targets (also NRD's inputs) are not
                                                flipped: an image rendered with the flag is the
                                                image rendered without it, mirrored in y. */

/* Zero-initialized options are valid: CAELIX_MODE_PRIMARY, the albedo view, no flags. */
typedef struct CaelixRenderOptions {
  uint32_t debug_view;  /* CaelixDebugView; read in CAELIX_MODE_PRIMARY (views 0..5, 9) and
                           CAELIX_MODE_PATH_TRACE (views 6, 7, 8, 10), checked in every mode */
  uint32_t flags;       /* CAELIX_RENDER_FLAG_* */
  float sun_dir[3];     /* world-space direction TOWARD the sun; the library normalizes it ((0, 1, 0)
                           when zero). Every mode: the Lambert term of the primary mode and the sun
                           disk of the lighting modes. */
  float pad;
  float sky_color[3];   /* CAELIX_MODE_PRIMARY: colour of rays that hit nothing (every view) */
  float pad2;
  float miss_depth;     /* depth target value of a miss: 1.0 for a normal-Z host, 0.0 for reverse Z */
  float miss_view_z;    /* view_z target value of a miss, for example 1e30 */

  /* Lighting (CAELIX_MODE_PATH_TRACE and CAELIX_MODE_REFERENCE). All radiance values are linear RGB. */
  uint32_t mode;               /* CaelixRenderMode */
  uint32_t bounce_count;       /* stochastic bounces after the primary hit, 0 .. CAELIX_MAX_BOUNCES
                                  (Unity default 4). 0 = primary emission only. */
  uint32_t denoise;            /* CAELIX_MODE_PATH_TRACE with a colour target: 1 = NRD denoises the
                                  frame (REBLUR_DIFFUSE_SPECULAR), 0 = raw 1 spp. Ignored in the other
                                  modes. */
  uint32_t reset_accumulation; /* 1 = restart the CAELIX_MODE_REFERENCE accumulation, and drop the NRD
                                  history (accumulation mode RESTART), with this frame. Both are also
                                  restarted by a mode change, a target size change and the first
                                  render; the NRD history also after a frame without denoising.
                                  Reference mode: set it when the camera or the scene changed. Path
                                  trace mode: NRD follows camera and entity motion through the motion
                                  vectors; set it only for a cut (a teleport, a new scene). */
  float exposure;              /* the composite multiplies the colour (sky included) by it before the
                                  sRGB encode; 0 is treated as 1 */
  float sun_radiance[3];       /* radiance of the sun disk (constant over the disk). The irradiance
                                  of a surface facing the sun is sun_radiance * 2 pi (1 - cos r),
                                  r = sun_angular_radius: about 2.39e-4 * sun_radiance at 0.5 degrees */
  float sun_angular_radius;    /* angular radius of the sun disk in radians; 0 = 0.00873 (0.5 degrees) */
  float sky_zenith[3];         /* sky radiance straight up */
  float sky_horizon[3];        /* sky radiance at the horizon */
  float sky_ground[3];         /* radiance straight down (below the horizon) */
  uint32_t frame_seed;         /* random seed of the frame: the per-pixel random stream is a function of
                                  the pixel and this value only. Increment it every frame. */
  uint32_t bounce_count_transparent; /* lighting modes: the delta chain of the primary ray, 0 ..
                                  CAELIX_MAX_BOUNCES_TRANSPARENT (Unity default 5; 0 = off). The camera
                                  ray crosses up to this many smooth transparent interfaces (materials
                                  with smoothness >= 0.99): at each it reflects or refracts, chosen at
                                  random with the Fresnel reflectance as the reflection probability
                                  (Schlick; total internal reflection always reflects), and the
                                  medium's extinction attenuates every segment inside it. The first hit
                                  that is not such an interface (an opaque or rough surface, or any hit
                                  after the budget) is the primary surface for everything that follows
                                  (primary surface replacement): albedo and normal targets, shading,
                                  emission and bounces, all times the chain's transmission; the depth
                                  and view_z targets get the distance along the whole chain, measured
                                  along the camera ray. A chain that ends in the sky writes the sky
                                  radiance times the transmission (coverage 1). Opaque surfaces always
                                  end the chain (smooth metals reflect through the stochastic
                                  specular lobe, as in Unity). */
} CaelixRenderOptions;

/* The images of one render call. A target is PRESENT when its `image` is nonzero; an absent target
   is not written. At least one target must be present, and every present target must have the same
   width and height and a different image.
   Every present image needs STORAGE usage (the kernel writes it as a storage image without a format
   qualifier, so its format must support VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT with
   optimal tiling). It arrives in CaelixTargetImage::layout and is left in VK_IMAGE_LAYOUT_GENERAL.
   Every pixel of a present target is written on every render call, a miss included. */
typedef struct CaelixRenderTargets {
  /* Any storage format, typically RGBA8, BGRA8, RGBA16F or RGBA32F.
     CAELIX_MODE_PRIMARY: rgb = the debug view of CaelixRenderOptions::debug_view; a miss is
     sky_color. alpha = 1 everywhere (opaque, also on a miss, so that a host can present the image
     directly; the coverage is in the albedo and normal targets).
     Lighting modes: rgb = premultiplied colour (times exposure), a = coverage (1 hit, 0 miss). A miss
     is the procedural sky in the primary direction with CAELIX_RENDER_FLAG_SKY_ON_MISS (alpha still
     0), else (0, 0, 0, 0). For a UNORM format (RGBA8, BGRA8, A2B10G10R10, ...) the library encodes
     rgb linear -> sRGB before the write; float formats receive linear values.
     A lighting render without a colour target runs the trace only (no composite, no accumulation). */
  CaelixTargetImage color;
  /* R8G8B8A8_UNORM or R16G16B16A16_SFLOAT. rgb = the material albedo as stored in the block (sRGB
     encoded, no shading), a = coverage (1 hit, 0 miss). A miss is (0, 0, 0, 0). */
  CaelixTargetImage albedo;
  /* R16G16B16A16_SFLOAT, or R8G8B8A8_UNORM (then xyz is stored as n * 0.5 + 0.5). xyz = world-space
     face normal (unit length), w = coverage. A miss is (0, 0, 0, 0). */
  CaelixTargetImage normal;
  /* R32_SFLOAT (a storage image, not a depth attachment). Device depth clip.z / clip.w of
     CaelixCamera::projection at the hit; a miss is CaelixRenderOptions::miss_depth. */
  CaelixTargetImage depth;
  /* R16G16_SFLOAT or R32G32_SFLOAT; or R16G16B16A16_SFLOAT / R32G32B32A32_SFLOAT, which also receive
     z = 0 and w = the delta checkerboard flag (1 = this pixel traced one branch of the split at the
     first transparent interface, see CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD, and needs a cross
     resolve with its flagged neighbours; 0 otherwise). xy = previous_pixel - current_pixel, in
     pixels of the current target size: the hit point is moved back by the entity's previous transform and projected with the
     previous render call's camera (the unjittered pinhole mapping of the ray generation); the current pixel is
     the sample position (px + 0.5 + jitter[0], py + 0.5 + jitter[1]); with CAELIX_RENDER_FLAG_FLIP_Y
     y is negated. Example: a camera that moved +1 unit along `right` sees
     a static point move LEFT on the screen, so motion.x > 0 (the point was to the right).
     0 on a miss, on the first render call of the renderer, and when the point was behind the
     previous camera. */
  CaelixTargetImage motion;
  /* R32_SFLOAT. Linear view depth dot(hit - position, forward) (NRD's IN_VIEWZ); a miss is
     CaelixRenderOptions::miss_view_z. */
  CaelixTargetImage view_z;
} CaelixRenderTargets;

/* Records the frame's rendering into frame->command_buffer: one barrier per present target from its
   layout to GENERAL (one vkCmdPipelineBarrier2), the trace dispatch (8 x 8 pixels per workgroup)
   between two timestamps, in the lighting modes a barrier and the composite dispatch (8 x 8, also
   between two timestamps), and one barrier from the writes to all later commands. Before the
   scene's first TLAS build the dispatch traces nothing and writes the miss values. The renderer
   creates one image view per target image and caches it (16 views; the least recently used view
   is destroyed once its last frame is safe). A successful call stores the camera as the previous
   camera of the next call.
   Returns CAELIX_ERROR_INVALID_ARGUMENT for a null argument, no present target, a present target
   without format or size, present targets of different sizes, one image in two targets, a geometry
   target format outside its list above, an unknown debug view, an unknown mode, a bounce_count above
   CAELIX_MAX_BOUNCES, a bounce_count_transparent above CAELIX_MAX_BOUNCES_TRANSPARENT, a frame number not above safe_frame_number, or a ninth frame in flight;
   CAELIX_ERROR_OUT_OF_MEMORY when the internal targets of the lighting modes or the NRD instance
   cannot be allocated (nothing is recorded); CAELIX_ERROR_UNSUPPORTED when a target format
   cannot be written as a storage image without a format qualifier.
   Denoising (CaelixRenderOptions::denoise): between the trace and the composite the call records
   NRD's dispatches (about 30 per frame, each after a global compute-to-compute memory barrier)
   between two timestamps. When NRD itself reports an error (instance creation, settings, dispatch
   list), the error is logged, the frame is recorded without denoising (raw 1 spp in the colour
   target) and this call returns CAELIX_ERROR_VULKAN; every later call of this renderer renders
   without denoising and returns CAELIX_OK (CaelixNrdInfo::failed is 1). */
CAELIX_API CaelixResult caelix_renderer_render(CaelixRenderer* renderer, const CaelixFrameDesc* frame,
                                               const CaelixRenderTargets* targets,
                                               const CaelixCamera* camera,
                                               const CaelixRenderOptions* options);

/* Destroys every cached image view. Call it before the host destroys target images, for example
   when it recreates its swapchain: a new image can get the handle of a destroyed one, and the
   cache would then hand out a view of the destroyed image. Every frame that used the views must
   have completed (for example after vkDeviceWaitIdle). NULL is ignored. */
CAELIX_API void caelix_renderer_release_target_views(CaelixRenderer* renderer);

/* GPU times of the newest frame whose timestamps were read back. The renderer reads them in a
   render call whose safe_frame_number has reached that frame, so the values trail the current
   frame by the number of frames in flight. */
typedef struct CaelixGpuTimings {
  float trace_ms;         /* the trace dispatch of that frame */
  float flush_ms;         /* caelix_scene_flush of that frame; 0 when the flush recorded no work */
  uint64_t frame_number;  /* the frame the values belong to; 0 = none yet */
  float composite_ms;     /* the composite dispatch of that frame; 0 when it had none (primary mode) */
  float denoise_ms;       /* NRD's dispatches of that frame; 0 when it was not denoised */
} CaelixGpuTimings;

/* Fills *out. A NULL argument is ignored. All zero when the queue has no timestamp support. */
CAELIX_API void caelix_renderer_get_gpu_timings(const CaelixRenderer* renderer,
                                                CaelixGpuTimings* out);

/* The NRD library and the renderer's NRD instance. The instance fields are 0 until the first
   denoised render created the instance; they describe the newest instance (a target size change
   replaces it). */
typedef struct CaelixNrdInfo {
  uint32_t version_major;            /* NRD library version (nrd::GetLibraryDesc) */
  uint32_t version_minor;
  uint32_t version_build;
  uint32_t width, height;            /* size of the instance; 0 = no instance yet */
  uint32_t pipeline_count;           /* compute pipelines of the instance */
  uint32_t permanent_texture_count;  /* NRD's permanent pool (history, kept between frames) */
  uint32_t transient_texture_count;  /* NRD's transient pool (scratch within a frame) */
  uint64_t permanent_bytes;          /* GPU memory of the permanent pool images */
  uint64_t transient_bytes;          /* GPU memory of the transient pool images */
  uint32_t dispatch_count;           /* NRD dispatches of the newest denoised frame */
  uint32_t failed;                   /* 1 after an NRD error: the renderer no longer denoises */
} CaelixNrdInfo;

/* Fills *out. Returns CAELIX_ERROR_INVALID_ARGUMENT (logged) for a NULL argument. */
CAELIX_API CaelixResult caelix_renderer_get_nrd_info(const CaelixRenderer* renderer,
                                                     CaelixNrdInfo* out);

/* Reserved for later (not declared yet):
 *   caelix_scene_upload_brick_summaries - per-brick summaries for bricks that are known but not
 *                                         resident (slot state "summary-only").
 *   caelix_scene_poll_brick_requests    - bricks that the GPU asked for (slot state "requested").
 */

#ifdef __cplusplus
}
#endif
#endif
