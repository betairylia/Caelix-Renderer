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
   bump. The sky/fog/emission step appended emission_scale, fog_density, fog_color, pad3 and sky
   (CaelixSkyCubemap) to CaelixRenderOptions, also without a bump. The plugin refactor step added
   CaelixMaterialEntry and caelix_scene_set_material_entries, appended
   CaelixSceneStats::upload_bricks_free and made a second caelix_renderer_render of one renderer with a
   frame_number not above its previous one an error (CAELIX_ERROR_INVALID_ARGUMENT), also without a
   bump. The DLSS step appended upscaler, upscale_quality, upscale_preset and upscale_flags to
   CaelixRenderOptions (176 -> 192 bytes) and CaelixGpuTimings::upscale_ms (24 -> 32 bytes), and
   added CaelixUpscaler, CaelixUpscaleQuality, CAELIX_UPSCALE_FLAG_*, the upscaler query
   (CaelixUpscalerQueryDesc, CaelixUpscalerQueryResult, CaelixUpscalerUnsupportedReason,
   caelix_upscaler_query), the extension lists (caelix_upscaler_instance_extensions,
   caelix_upscaler_device_extensions), the jitter helpers (caelix_upscaler_jitter_phases,
   caelix_jitter_halton) and CaelixUpscalerInfo with caelix_renderer_get_upscaler_info, also without
   a bump. The GI experiment step added CaelixGiMethod, CAELIX_GI_FLAG_*, CaelixGiOptions,
   caelix_renderer_set_gi_options, CaelixGiStats and caelix_renderer_get_gi_stats (no existing struct
   changed), also without a bump. The light list step (docs/LIGHT_LIST.md) added CaelixLightOptions,
   caelix_scene_set_light_options, CaelixLightStats, caelix_scene_get_light_stats and
   CAELIX_GI_FLAG_LIGHT_NEE, and appended the light cell fields to the experimental CaelixGiOptions
   (32 -> 48 bytes) and CaelixGiStats (16 -> 32 bytes), also without a bump. The ReSTIR DI step
   (docs/RESTIR_DI.md) added CAELIX_GI_FLAG_RESTIR_DI and renamed CaelixGiOptions::light_reserved to
   di_candidates (same size). The light cell jitter added CAELIX_GI_FLAG_LIGHT_CELL_JITTER. The light MIS step
   (docs/LIGHT_MIS.md) added CAELIX_GI_FLAG_LIGHT_MIS (no struct changed). The light cell temporal reuse
   added CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL and appended light_temporal_frames to CaelixGiOptions
   (48 -> 52 bytes). The light cell usage counters appended four uint32 fields to CaelixGiStats
   (32 -> 48 bytes). The light cell push-back added CAELIX_GI_FLAG_LIGHT_CELL_PUSH, appended
   light_push_confidence to CaelixGiOptions (52 -> 56 bytes) and two uint32 fields to CaelixGiStats
   (48 -> 56 bytes). The per-vertex resampling of the light NEE appended light_nee_candidates to
   CaelixGiOptions (56 -> 60 bytes) and added CAELIX_GI_FLAG_LIGHT_PROXY_VERTEX. The oriented light cells
   added CAELIX_GI_FLAG_LIGHT_CELL_ORIENTED and shrank a light cell to 256 bytes (8 reservoirs,
   CaelixGiStats::light_cell_bytes changes accordingly). The light cell size step (2026-10-22) made a
   level 0 cell 32 voxels (128 at level 1) and appended light_cell_reservoirs to CaelixGiOptions
   (60 -> 64 bytes; the cell is 64 + 24 x reservoirs bytes). The pairwise MIS step added
   CAELIX_GI_FLAG_PAIRWISE_MIS (no struct changed). A host built against
   an older version-2 header must be rebuilt (the library reads and writes the larger structs). */
#define CAELIX_ABI_VERSION 2u

typedef struct CaelixVersion
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    uint32_t abi; /* == CAELIX_ABI_VERSION of the built library */
} CaelixVersion;

/* Fills *out with the library's version. Never fails. */
CAELIX_API void caelix_get_version(CaelixVersion* out);
/* Static string like "0.1.0", owned by the library. */
CAELIX_API const char* caelix_version_string(void);

/* Result of every fallible ABI call. */
typedef enum CaelixResult
{
    CAELIX_OK = 0,
    CAELIX_ERROR_INVALID_ARGUMENT = 1,
    CAELIX_ERROR_UNSUPPORTED = 2, /* the adopted device lacks a requirement; see log */
    CAELIX_ERROR_VULKAN = 3,      /* a Vulkan call failed; see log */
    CAELIX_ERROR_OUT_OF_MEMORY = 4
} CaelixResult;

/* ---------------------------------------------------------------------------
 * Logging
 * ------------------------------------------------------------------------- */

typedef enum CaelixLogLevel
{
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
typedef enum CaelixDeviceFeatureBits
{
    CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS = 1u << 0,              /* VkPhysicalDeviceVulkan12Features::bufferDeviceAddress */
    CAELIX_FEATURE_DESCRIPTOR_INDEXING = 1u << 1,                /* runtimeDescriptorArray, shaderStorageBufferArrayNonUniformIndexing, descriptorBindingPartiallyBound, descriptorBindingVariableDescriptorCount */
    CAELIX_FEATURE_SYNCHRONIZATION2 = 1u << 2,                   /* VkPhysicalDeviceVulkan13Features::synchronization2 */
    CAELIX_FEATURE_DYNAMIC_RENDERING = 1u << 3,                  /* VkPhysicalDeviceVulkan13Features::dynamicRendering */
    CAELIX_FEATURE_TIMELINE_SEMAPHORE = 1u << 4,                 /* VkPhysicalDeviceVulkan12Features::timelineSemaphore */
    CAELIX_FEATURE_ACCELERATION_STRUCTURE = 1u << 5,             /* VkPhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure */
    CAELIX_FEATURE_RAY_QUERY = 1u << 6,                          /* VkPhysicalDeviceRayQueryFeaturesKHR::rayQuery */
    CAELIX_FEATURE_SHADER_INT64 = 1u << 7,                       /* VkPhysicalDeviceFeatures::shaderInt64 */
    CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT = 1u << 8,                /* VkPhysicalDeviceVulkan12Features::scalarBlockLayout */
    CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT = 1u << 9, /* VkPhysicalDeviceFeatures::shaderStorageImageWriteWithoutFormat */
    CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT = 1u << 10  /* VkPhysicalDeviceFeatures::shaderStorageImageReadWithoutFormat
                                                                     (NRD's temporal stabilization and validation passes) */
} CaelixDeviceFeatureBits;

typedef struct CaelixDeviceRequirements
{
    uint32_t api_version; /* VK_MAKE_API_VERSION(0,1,3,0) */
    uint32_t device_extension_count;
    const char* const* device_extensions; /* static strings owned by the library */
    uint32_t feature_flags;               /* CaelixDeviceFeatureBits */
} CaelixDeviceRequirements;

/* Fills *out with what the host must request at vkCreateInstance / vkCreateDevice.
   Call before creating the device. Never fails; a NULL `out` is ignored. */
CAELIX_API void caelix_get_device_requirements(CaelixDeviceRequirements* out);

typedef struct CaelixDeviceAdoptDesc
{
    void* get_instance_proc_addr; /* PFN_vkGetInstanceProcAddr */
    uint64_t instance;            /* VkInstance */
    uint64_t physical_device;     /* VkPhysicalDevice */
    uint64_t device;              /* VkDevice */
    uint64_t graphics_queue;      /* VkQueue, graphics + compute capable */
    uint32_t graphics_queue_family;
} CaelixDeviceAdoptDesc;

typedef struct CaelixDevice CaelixDevice; /* opaque */

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
typedef struct CaelixTargetImage
{
    uint64_t image;  /* VkImage */
    uint32_t format; /* VkFormat */
    uint32_t width;
    uint32_t height;
    uint32_t layout; /* VkImageLayout of the image when the library starts recording */
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

typedef struct CaelixScene CaelixScene; /* opaque */

typedef struct CaelixSceneDesc
{
    uint64_t staging_bytes;    /* host-visible upload ring. 0 = 64 MiB. Must be >= 4 MiB. */
    uint32_t records_per_page; /* record pool page size in bricks, power of two. 0 = 1 << 18. */
    uint32_t group_shift;      /* render group edge in bricks = 1 << group_shift. 0 = 4 (16^3). Max 8.
                                  Only 1..5 are implemented in this version: 6..8 return
                                  CAELIX_ERROR_UNSUPPORTED (dense brick ids are 16-bit). */
    uint32_t block_bytes;      /* bytes per block value. 0 = 2. Only 2 is implemented in this version;
                                  4 returns CAELIX_ERROR_UNSUPPORTED. */
    uint32_t encoding;         /* block encoding for finalization: 0 = packed scene colour (the
                                  default, what Caelix worlds use), 1 = block-ID. Other values
                                  return CAELIX_ERROR_INVALID_ARGUMENT. See the encodings above. */
    uint64_t scratch_bytes;    /* scratch buffer of the acceleration structure builds of one flush.
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

typedef struct CaelixEntityDesc
{
    uint8_t guid[16];  /* stored for the host; the library does not interpret it */
    float position[3]; /* world position of the entity origin */
    float rotation[4]; /* unit quaternion x, y, z, w */
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
typedef struct CaelixBrickKey
{
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

typedef struct CaelixFrameDesc
{
    uint64_t command_buffer;    /* VkCommandBuffer in the recording state, graphics queue */
    uint64_t frame_number;      /* strictly increasing across flushes. Must be > safe_frame_number,
                                   so the first frame number is 1 or higher. */
    uint64_t safe_frame_number; /* every frame <= this has completed on the GPU. 0 = none yet. */
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
   finalize step, blas_count .. as_builds_pending by the acceleration structure step,
   material_table_bytes by the materials step, upload_bricks_free by the plugin refactor step. */
typedef struct CaelixSceneStats
{
    uint32_t entities, groups;                              /* live entities and render groups */
    uint64_t bricks_resident, bricks_known;                 /* with a GPU record; with a slot */
    uint64_t record_bytes, slot_table_bytes, staging_bytes; /* GPU allocation sizes */
    uint64_t uploaded_bricks_total, pending_bricks;         /* pending = staged, not yet flushed */
    uint64_t record_pages;
    uint64_t finalized_bricks_total; /* finalize kernel invocations (bricks) recorded so far */
    uint64_t reverse_table_bytes;    /* GPU allocation sizes of the finalize lookup tables: */
    uint64_t aabb_bytes;             /*   reverse tables, AABB rows, group grids */
    uint64_t grid_bytes;
    uint64_t blas_count;           /* live groups that have a BLAS */
    uint64_t blas_bytes;           /* BLAS storage blocks, including blocks whose release waits for
                                      a safe frame */
    uint64_t tlas_bytes;           /* TLAS storage block(s), same rule */
    uint64_t scratch_bytes;        /* capacity of the build scratch buffer */
    uint64_t as_builds_total;      /* BLAS and TLAS builds recorded so far */
    uint64_t as_builds_pending;    /* groups whose BLAS is out of date (waiting for finalization,
                                      scratch or ring space), plus 1 when the TLAS is out of date */
    uint64_t material_table_bytes; /* GPU size of the material table (65536 * 48 = 3 MiB) */
    uint64_t upload_bricks_free; /* upper bound of the bricks that caelix_scene_upload_bricks accepts
                                    before the next flush, from the free staging space minus the row
                                    reserve (wrap fragmentation can make the real number smaller;
                                    *consumed is the truth). Read it after a flush to size the next
                                    frame's uploads. */
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
typedef struct CaelixBrickInfo
{
    uint32_t resident;                 /* 1 when the brick has a GPU record */
    uint32_t group_index;              /* scene-wide render group index */
    uint32_t group_local_id;           /* dense brick id inside the group = slot table row */
    uint32_t record_index;             /* global record index = page << log2(records_per_page) | slot */
    uint64_t record_page_address;      /* VkDeviceAddress of the record's page buffer */
    uint64_t record_page_buffer;       /* VkBuffer of the record's page */
    uint64_t record_page_offset_bytes; /* byte offset of the record inside its page */
    uint64_t slot_table_buffer;        /* VkBuffer that holds every slot table of the scene */
    uint64_t slot_row_offset_bytes;    /* byte offset of the brick's slot table row */
    uint64_t aabb_buffer;              /* VkBuffer that holds every AABB row of the scene */
    uint64_t aabb_row_offset_bytes;    /* byte offset of the brick's AABB row (24 bytes: float min[3],
                                          float max[3], entity-local block coordinates, all NaN
                                          when the brick has no occupied voxel) */
    uint64_t blas_address;             /* VkDeviceAddress of the group's BLAS, 0 when not built yet */
    uint64_t blas_handle;              /* VkAccelerationStructureKHR of the group's BLAS, 0 when not
                                          built yet. Owned by the library. */
    uint64_t entity_info_buffer;       /* VkBuffer that holds the entity info rows */
    uint64_t entity_info_offset_bytes; /* byte offset of the entity's info row (128 bytes: int32
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
typedef struct CaelixMaterial
{
    float albedo[3];   /* diffuse / base colour; for a transparent medium its tint */
    float smoothness;  /* 0..1. NRD roughness = 1 - smoothness. A transparent entry with smoothness
                          >= 0.99 is a delta interface (reflection and refraction in the delta chain,
                          CaelixRenderOptions::bounce_count_transparent); a rougher one is shaded
                          stochastically (still a perfect reflection or refraction, as in Unity) */
    float emission[3]; /* emitted radiance */
    float metallic;    /* 0..1; opaque entries: specular chance lerp(metallic, 1, fresnel * smoothness) */
    float ior;         /* index of refraction; must be > 0 (transparent entries use it at interfaces,
                          opaque entries for the Fresnel term of the specular chance) */
    float extinction;  /* transparent entries: per voxel unit; a segment of length t inside the medium
                          transmits exp(-(1 - albedo) * t * extinction). Must be >= 0. */
    float pad[2];      /* ignored; stored as 0 */
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

/* One entry of caelix_scene_set_material_entries: a 16-bit block code and its table entry. 52 bytes. */
typedef struct CaelixMaterialEntry
{
    uint32_t code; /* 0 .. 0xFFFF */
    CaelixMaterial material;
} CaelixMaterialEntry;

/* Writes entries[i].material at table entry entries[i].code, over the CURRENT table (the other
   entries keep their values; nothing is reset to the built-in default). Every code must be
   <= 0xFFFF and every material valid (same checks as caelix_scene_set_materials): one bad entry
   returns CAELIX_ERROR_INVALID_ARGUMENT (logged) and changes nothing. count 0 is a no-op. Like the
   other setters, the change reaches the GPU at the next caelix_scene_flush. For a transparent code
   write the face-free code (the kernels look up code & id mask); an entry stored at a code with face
   bits is never read. */
CAELIX_API CaelixResult caelix_scene_set_material_entries(CaelixScene* scene,
                                                          const CaelixMaterialEntry* entries,
                                                          uint32_t count);
/* Inspection: the current table entry of a raw block code (code <= 0xFFFF; face bits of a
   transparent code are removed first, as the kernels do). The value is the CPU copy: it reaches the
   GPU at the next flush. */
CAELIX_API CaelixResult caelix_scene_get_material(const CaelixScene* scene, uint32_t code,
                                                  CaelixMaterial* out);

/* ---------------------------------------------------------------------------
 * Light list (experimental, docs/LIGHT_LIST.md)
 *
 * A material is a "list light" when the luminance of its emission (0.2126 r + 0.7152 g + 0.0722 b,
 * before CaelixRenderOptions::emission_scale) is >= min_emission. The finalize kernel writes one
 * light word per brick (the luminance sum of its alive list-light voxels), and every flush that
 * finalized bricks or freed brick ids rebuilds the scene's light table: one 448-byte entry per brick
 * with list lights (per face direction a mask of the voxels whose face is open and its weight, the
 * brick's origin; docs/LIGHT_FACES.md) and a CDF over their weights. Renderers with CAELIX_GI_FLAG_LIGHT_NEE sample it. Bricks beyond max_bricks
 * are dropped (counted, never an error). */
typedef struct CaelixLightOptions
{
    float min_emission;  /* 0 = 20.0: list light when luminance(material.emission) >= this (packed
                            encoding: an emissive voxel of strength 2 (64 x albedo) with an albedo
                            luminance of at least 0.3125, and every bright strength 3 voxel) */
    uint32_t max_bricks; /* 0 = 65536; 1 .. 1048576: the light table's capacity in bricks */
    float nee_min_distance; /* 0 = 8.0 world units: light points closer than this to the receiver are
                               left to the BSDF path (its hits of list lights closer than this keep
                               their emission). Area sampling with 1/d^2 makes the near field an
                               extremely heavy-tailed NEE estimate (bistro lost 14 % after 512
                               frames); the split is unbiased. ReSTIR DI applies the same split. */
    uint32_t reserved;
} CaelixLightOptions;

/* Scene-wide. The light table is built only once this function was called (a host that never calls
   it, the Unity plugin today, pays nothing beyond the 4-byte light word and entry map word per AABB
   row); pass zeros for the defaults. min_emission applies to bricks finalized after the call and
   to the next table rebuild; bricks finalized before keep their light word until they are
   finalized again (call before the uploads). A new max_bricks reallocates the table at the next
   flush. Validation: min_emission not finite or negative, max_bricks above 1048576 ->
   CAELIX_ERROR_INVALID_ARGUMENT (logged; the previous options stay). */
CAELIX_API CaelixResult caelix_scene_set_light_options(CaelixScene* scene, const CaelixLightOptions* options);

typedef struct CaelixLightStats
{
    uint32_t bricks;         /* entries of the table after the newest rebuild whose frame is safe
                                (read back from the GPU; 0 before the first one) */
    uint32_t bricks_dropped; /* bricks with list lights beyond max_bricks in that rebuild */
    uint64_t table_bytes;    /* GPU size of the light table (64 + 452 bytes per max_bricks entry) plus
                                the light words (4 bytes per AABB row) */
    uint64_t rebuilds;       /* flushes that rebuilt the table so far */
} CaelixLightStats;

/* Fills *out; a null argument is ignored. */
CAELIX_API void caelix_scene_get_light_stats(const CaelixScene* scene, CaelixLightStats* out);

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
 * Sky and fog of the lighting modes: the host can pass a sky cubemap per render call
 * (CaelixRenderOptions::sky, CaelixSkyCubemap); without it the procedural sky (sky_zenith,
 * sky_horizon, sky_ground) is used. CaelixRenderOptions::fog_density / fog_color make the air a
 * homogeneous absorbing and emitting medium for every segment that ends on a surface;
 * emission_scale multiplies every material's emission in every mode.
 * In the two lighting modes the renderer owns internal targets (about 120 bytes per pixel) sized to
 * the host's targets (the render size with an upscaler); they are created at the first lighting
 * render and recreated when the target size changes. The NRD instance (its own pools, see
 * caelix_renderer_get_nrd_info) is created at the first denoised render and recreated (history
 * lost) when the target size changes: NRD has no resize.
 *
 * Upscalers (CaelixRenderOptions::upscaler, CAELIX_MODE_PATH_TRACE with a colour target only;
 * docs/DLSS.md): NVIDIA DLSS Super Resolution (CAELIX_UPSCALER_DLSS_SR, keeps NRD as the denoiser)
 * or DLSS Ray Reconstruction (CAELIX_UPSCALER_DLSS_RR, replaces NRD). The renderer then traces at
 * the RENDER size and the upscaler writes the OUTPUT size:
 *   - the colour target has the output size; every other present target (albedo, normal, depth,
 *     motion, view_z) has the render size that caelix_upscaler_query reports for that upscaler,
 *     quality and output size. A mismatch returns CAELIX_ERROR_INVALID_ARGUMENT with a log line
 *     naming the expected size. The internal targets have the render size. The output must be at
 *     least 64 x 32.
 *   - CAELIX_MODE_PRIMARY, CAELIX_MODE_REFERENCE and a render without a colour target ignore the
 *     upscaler (every present target then has one size, as without an upscaler).
 *   - DLSS RR ignores CaelixRenderOptions::denoise and forces CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD
 *     off (the RR guide forbids checkerboard rendering). DLSS SR keeps both.
 *   - The debug views of the path trace mode are upscaled like the colour (NRD's validation overlay
 *     is not drawn; the motion view is not sRGB-encoded).
 *   - CAELIX_ERROR_UNSUPPORTED when the upscaler is not available (the build has no DLSS, NGX does
 *     not start, the GPU or driver lacks the feature, or the quality is unavailable): query first.
 *   - When NGX reports an error (feature creation or evaluation), the frame is recorded with the
 *     fallback (nearest upscale of the render-size colour), the call returns CAELIX_ERROR_VULKAN
 *     once, and later calls return CAELIX_OK with CaelixUpscalerInfo::failed = 1 until the upscaler
 *     options (upscaler, quality, preset, sizes, auto exposure, depth direction) change.
 *   - The upscaler keeps a history: CAELIX_UPSCALE_FLAG_RESET_HISTORY, reset_accumulation, the
 *     first frame of a new upscaler and the first render call of the renderer drop it.
 *   - Jitter: vary CaelixCamera::jitter every frame (caelix_jitter_halton with
 *     caelix_upscaler_jitter_phases phases); the upscaler receives it as its jitter offset.
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
 * frames in flight; a slot is reused only when its frame is <= safe_frame_number. Several renderers
 * may render the same frame (one flush, one frame desc, N render calls, for example one renderer per
 * view); one renderer renders at most once per frame number: a second call with a frame_number not
 * above its previous successful call's returns CAELIX_ERROR_INVALID_ARGUMENT.
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

typedef struct CaelixRenderer CaelixRenderer; /* opaque */

/* Creates a renderer for `scene` (which must belong to `device`). The renderer keeps references to
   both: destroy it before the scene and the device. The first renderer of a device also creates the
   device's shared kernels (descriptor set layouts, compute pipelines, the sky sampler and the
   fallback sky cube; the NRD pipelines at the first denoised render); every later renderer borrows
   them, and they live until caelix_device_destroy. */
CAELIX_API CaelixResult caelix_renderer_create(CaelixDevice* device, CaelixScene* scene,
                                               CaelixRenderer** out_renderer);
/* Destroys the renderer and its GPU objects (descriptor pools, constants, timestamp queries, cached
   image views, internal targets, NRD instance; the shared kernels stay with the device). NULL is
   ignored. The device must be idle. */
CAELIX_API void caelix_renderer_destroy(CaelixRenderer* renderer);

/* The camera of one render call, in world space. The voxel world is Y up and uses Unity's
   convention: +X right, +Y up, +Z forward. right, up and forward must form an orthonormal basis.
   Pixel (px, py), row 0 = the TOP row of the image, traces the ray from `position` in direction
     s   = (px + 0.5 + jitter[0], py + 0.5 + jitter[1])          (the sample position, pixels)
     ndc = (s.x / width * 2 - 1, 1 - s.y / height * 2)
     dir = normalize(right * ndc.x * tan_half_fov_y * aspect + up * ndc.y * tan_half_fov_y + forward)
   (CAELIX_RENDER_FLAG_FLIP_Y changes only where the result is stored, not this mapping.) */
typedef struct CaelixCamera
{
    float position[3];
    float pad0;
    float right[3];
    float pad1;
    float up[3];
    float pad2;
    float forward[3];
    float pad3;
    float tan_half_fov_y; /* tangent of half the VERTICAL field of view */
    float aspect;         /* width / height */
    float near_z, far_z;  /* distance range of the depth debug view only (no clipping) */
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
typedef enum CaelixRenderMode
{
    CAELIX_MODE_PRIMARY = 0,    /* debug views of the primary hit (CaelixDebugView) */
    CAELIX_MODE_PATH_TRACE = 1, /* path-traced lighting, raw 1 sample per pixel */
    CAELIX_MODE_REFERENCE = 2   /* path-traced lighting, accumulated over frames (reference image) */
} CaelixRenderMode;

/* Largest CaelixRenderOptions::bounce_count (Unity's path tracer has the same limit). */
#define CAELIX_MAX_BOUNCES 14u
/* Largest CaelixRenderOptions::bounce_count_transparent. */
#define CAELIX_MAX_BOUNCES_TRANSPARENT 15u

typedef enum CaelixDebugView
{
    CAELIX_VIEW_ALBEDO = 0, /* material albedo; Lambert-shaded with CAELIX_RENDER_FLAG_SHADING */
    CAELIX_VIEW_NORMAL = 1, /* world normal * 0.5 + 0.5 */
    CAELIX_VIEW_DEPTH = 2,  /* grey (t - near_z) / (far_z - near_z), clamped to [0, 1] */
    CAELIX_VIEW_GROUP = 3,  /* hash colour of the render group of the hit */
    CAELIX_VIEW_BRICK = 4,  /* hash colour of the brick of the hit (group and dense brick id) */
    CAELIX_VIEW_STEPS = 5,  /* heat map of the procedural candidates of the ray, 0 .. 64 */
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

/* A cubemap of the sky for the lighting modes, given per render call like a target. A cube image
   (VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, 6 layers in the Vulkan face order +X, -X, +Y, -Y, +Z,
   -Z, SAMPLED usage, a colour format that supports linear filtering), sampled with the world
   direction (Y up, +Z forward, as CaelixCamera). It must NOT contain the sun disk: the sun stays
   CaelixRenderOptions::sun_dir / sun_radiance (with CAELIX_RENDER_FLAG_SUN_NEE). Where it is used:
   a primary miss and a delta chain that ends in the sky sample mip 0; a specular bounce that
   misses samples mip roughness * (mip_levels - 1) (roughness = 1 - smoothness of the lobe);
   a diffuse bounce that misses receives irradiance(normal) / pi from a 9-coefficient spherical
   harmonic (L2) that the renderer projects from the cubemap every render call (one compute
   dispatch over a 32 x 32 x 6 sampling of the cubemap, part of trace_ms): a low-noise,
   slightly biased estimate (irradiance times cosine-weighted sky visibility). The library creates
   and caches a cube image view (as for the targets) and does NOT transition the image: the host
   gives it in `layout`, with its own writes made visible to compute-shader sampled reads, and
   keeps it there for the whole render call. With image 0 the renderer
   uses the procedural sky (sky_zenith, sky_horizon, sky_ground) everywhere, as before. */
typedef struct CaelixSkyCubemap
{
    uint64_t image;      /* VkImage; 0 = no cubemap */
    uint32_t format;     /* VkFormat */
    uint32_t layout;     /* VkImageLayout the image is in during the render call
                            (SHADER_READ_ONLY_OPTIMAL or GENERAL) */
    uint32_t size;       /* face width = height in pixels (>= 1) */
    uint32_t mip_levels; /* mip levels of the image (>= 1); the library samples up to the last */
} CaelixSkyCubemap;

/* CaelixRenderOptions::upscaler (see "Upscalers" above). */
typedef enum CaelixUpscaler
{
    CAELIX_UPSCALER_NONE = 0,
    CAELIX_UPSCALER_DLSS_SR = 1, /* NVIDIA DLSS Super Resolution; NRD stays the denoiser */
    CAELIX_UPSCALER_DLSS_RR = 2  /* NVIDIA DLSS Ray Reconstruction; replaces NRD */
} CaelixUpscaler;

/* CaelixRenderOptions::upscale_quality: the render size per output size (DLSS execution modes). */
typedef enum CaelixUpscaleQuality
{
    CAELIX_UPSCALE_DLAA = 0,             /* render size = output size (anti-aliasing only) */
    CAELIX_UPSCALE_QUALITY = 1,          /* about 0.67 per axis */
    CAELIX_UPSCALE_BALANCED = 2,         /* about 0.58 per axis */
    CAELIX_UPSCALE_PERFORMANCE = 3,      /* 0.5 per axis */
    CAELIX_UPSCALE_ULTRA_PERFORMANCE = 4 /* about 0.33 per axis */
} CaelixUpscaleQuality;

/* CaelixRenderOptions::upscale_flags */
#define CAELIX_UPSCALE_FLAG_RESET_HISTORY (1u << 0) /* this frame: drop the upscaler's history (a cut) */
#define CAELIX_UPSCALE_FLAG_AUTO_EXPOSURE (1u << 1) /* DLSS SR: NGX auto exposure instead of a fixed
                                                       exposure of 1 (the colour is already exposed) */

/* Zero-initialized options are valid: CAELIX_MODE_PRIMARY, the albedo view, no flags. */
typedef struct CaelixRenderOptions
{
    uint32_t debug_view; /* CaelixDebugView; read in CAELIX_MODE_PRIMARY (views 0..5, 9) and
                            CAELIX_MODE_PATH_TRACE (views 6, 7, 8, 10), checked in every mode */
    uint32_t flags;      /* CAELIX_RENDER_FLAG_* */
    float sun_dir[3];    /* world-space direction TOWARD the sun; the library normalizes it ((0, 1, 0)
                            when zero). Every mode: the Lambert term of the primary mode and the sun
                            disk of the lighting modes. */
    float pad;
    float sky_color[3]; /* CAELIX_MODE_PRIMARY: colour of rays that hit nothing (every view) */
    float pad2;
    float miss_depth;  /* depth target value of a miss: 1.0 for a normal-Z host, 0.0 for reverse Z */
    float miss_view_z; /* view_z target value of a miss, for example 1e30 */

    /* Lighting (CAELIX_MODE_PATH_TRACE and CAELIX_MODE_REFERENCE). All radiance values are linear RGB. */
    uint32_t mode;                     /* CaelixRenderMode */
    uint32_t bounce_count;             /* stochastic bounces after the primary hit, 0 .. CAELIX_MAX_BOUNCES
                                          (Unity default 4). 0 = primary emission only. */
    uint32_t denoise;                  /* CAELIX_MODE_PATH_TRACE with a colour target: 1 = NRD denoises the
                                          frame (REBLUR_DIFFUSE_SPECULAR), 0 = raw 1 spp. Ignored in the other
                                          modes. */
    uint32_t reset_accumulation;       /* 1 = restart the CAELIX_MODE_REFERENCE accumulation, and drop the NRD
                                          history (accumulation mode RESTART), with this frame. Both are also
                                          restarted by a mode change, a target size change and the first
                                          render; the NRD history also after a frame without denoising.
                                          Reference mode: set it when the camera or the scene changed. Path
                                          trace mode: NRD follows camera and entity motion through the motion
                                          vectors; set it only for a cut (a teleport, a new scene). */
    float exposure;                    /* the composite multiplies the colour (sky included) by it before the
                                          sRGB encode; 0 is treated as 1 */
    float sun_radiance[3];             /* radiance of the sun disk (constant over the disk). The irradiance
                                          of a surface facing the sun is sun_radiance * 2 pi (1 - cos r),
                                          r = sun_angular_radius: about 2.39e-4 * sun_radiance at 0.5 degrees */
    float sun_angular_radius;          /* angular radius of the sun disk in radians; 0 = 0.00873 (0.5 degrees) */
    float sky_zenith[3];               /* sky radiance straight up */
    float sky_horizon[3];              /* sky radiance at the horizon */
    float sky_ground[3];               /* radiance straight down (below the horizon) */
    uint32_t frame_seed;               /* random seed of the frame: the per-pixel random stream is a function of
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
    float emission_scale;              /* every mode: multiplies the emission of every material (both encodings)
                                          at every hit, primary and bounce. 0 is treated as 1. */
    float fog_density;                 /* lighting modes: extinction of the air (the medium outside glass) per
                                          world unit. 0 = no fog. A segment of length t that ends on a surface
                                          has the transmission exp(-fog_density * t) and adds
                                          fog_color * (1 - that); the camera ray, the air gaps of the delta chain
                                          and every bounce segment. A segment that leaves the scene (a miss) is not
                                          affected: the sky stays sharp, so choose a fog colour near the horizon
                                          colour. No fog inside glass (its own extinction). Shadow rays ignore it. */
    float fog_color[3];                /* radiance of the fog, linear RGB (exposure applies) */
    float pad3;
    CaelixSkyCubemap sky; /* lighting modes: the sky cubemap; image 0 = the procedural sky */
    uint32_t upscaler;        /* CaelixUpscaler; CAELIX_MODE_PATH_TRACE with a colour target only, else
                                 ignored (an unknown value is an error in every mode) */
    uint32_t upscale_quality; /* CaelixUpscaleQuality (checked when upscaler != NONE) */
    uint32_t upscale_preset;  /* 0 = the NGX default; else the raw NGX preset enum value (SR:
                                 NVSDK_NGX_DLSS_Hint_Render_Preset_*, RR:
                                 NVSDK_NGX_RayReconstruction_Hint_Render_Preset_*), set as the hint
                                 of the chosen quality mode */
    uint32_t upscale_flags;   /* CAELIX_UPSCALE_FLAG_* */
} CaelixRenderOptions;

/* The images of one render call. A target is PRESENT when its `image` is nonzero; an absent target
   is not written. At least one target must be present, and every present target must have the same
   width and height and a different image.
   Every present image needs STORAGE usage (the kernel writes it as a storage image without a format
   qualifier, so its format must support VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT with
   optimal tiling). It arrives in CaelixTargetImage::layout and is left in VK_IMAGE_LAYOUT_GENERAL.
   Every pixel of a present target is written on every render call, a miss included. */
typedef struct CaelixRenderTargets
{
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
   CAELIX_MAX_BOUNCES, a bounce_count_transparent above CAELIX_MAX_BOUNCES_TRANSPARENT, a frame number
   not above safe_frame_number, a frame number not above this renderer's previous successful call's
   (one render per renderer per frame), a ninth frame in flight,
   a sky image (CaelixRenderOptions::sky) with format VK_FORMAT_UNDEFINED, size 0, mip_levels 0 or
   layout VK_IMAGE_LAYOUT_UNDEFINED, a negative or non-finite fog_density or emission_scale, an
   unknown upscaler, and with an active upscaler an unknown upscale_quality, an output below 64 x 32
   or a target that does not have its expected size (see "Upscalers");
   CAELIX_ERROR_OUT_OF_MEMORY when the internal targets of the lighting modes or the NRD instance
   cannot be allocated (nothing is recorded); CAELIX_ERROR_UNSUPPORTED when a target format
   cannot be written as a storage image without a format qualifier, or when the requested upscaler
   is not available.
   Upscaler (see "Upscalers"): after the composite (which then writes an internal render-size
   image instead of the colour target) the call records the upscaler's evaluate (DLSS records its
   own dispatches and barriers; the library rebinds its own state after it) and one compute dispatch
   (8 x 8 per workgroup, at the output size) that writes the colour target from the upscaler's
   output, both between two timestamps (upscale_ms). The first frame of a new upscaler also records
   the NGX feature creation.
   Sky cubemap (lighting modes with CaelixRenderOptions::sky.image != 0): before the trace the call
   records one small compute dispatch that projects the cubemap onto the irradiance SH and a
   barrier; both lie inside the trace timestamps (trace_ms).
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
   have completed (for example after vkDeviceWaitIdle). The cached cube views of sky cubemaps
   (CaelixRenderOptions::sky) are released too: a host that recreates its cubemap calls it the same
   way. NULL is ignored. */
CAELIX_API void caelix_renderer_release_target_views(CaelixRenderer* renderer);

/* GPU times of the newest frame whose timestamps were read back. The renderer reads them in a
   render call whose safe_frame_number has reached that frame, so the values trail the current
   frame by the number of frames in flight. */
typedef struct CaelixGpuTimings
{
    float trace_ms;        /* the trace dispatch of that frame (with a sky cubemap, the sky SH
                              dispatch before it too) */
    float flush_ms;        /* caelix_scene_flush of that frame; 0 when the flush recorded no work */
    uint64_t frame_number; /* the frame the values belong to; 0 = none yet */
    float composite_ms;    /* the composite dispatch of that frame; 0 when it had none (primary mode) */
    float denoise_ms;      /* NRD's dispatches of that frame; 0 when it was not denoised */
    float upscale_ms;      /* the upscaler of that frame: its evaluate (DLSS) and the dispatch that
                              encodes its output into the colour target (on the fallback after an
                              NGX error, that dispatch only); 0 when the frame had no upscaler */
} CaelixGpuTimings;

/* Fills *out. A NULL argument is ignored. All zero when the queue has no timestamp support. */
CAELIX_API void caelix_renderer_get_gpu_timings(const CaelixRenderer* renderer,
                                                CaelixGpuTimings* out);

/* The NRD library and the renderer's NRD instance. The instance fields are 0 until the first
   denoised render created the instance; they describe the newest instance (a target size change
   replaces it). */
typedef struct CaelixNrdInfo
{
    uint32_t version_major; /* NRD library version (nrd::GetLibraryDesc) */
    uint32_t version_minor;
    uint32_t version_build;
    uint32_t width, height;           /* size of the instance; 0 = no instance yet */
    uint32_t pipeline_count;          /* compute pipelines of the instance */
    uint32_t permanent_texture_count; /* NRD's permanent pool (history, kept between frames) */
    uint32_t transient_texture_count; /* NRD's transient pool (scratch within a frame) */
    uint64_t permanent_bytes;         /* GPU memory of the permanent pool images */
    uint64_t transient_bytes;         /* GPU memory of the transient pool images */
    uint32_t dispatch_count;          /* NRD dispatches of the newest denoised frame */
    uint32_t failed;                  /* 1 after an NRD error: the renderer no longer denoises */
} CaelixNrdInfo;

/* Fills *out. Returns CAELIX_ERROR_INVALID_ARGUMENT (logged) for a NULL argument. */
CAELIX_API CaelixResult caelix_renderer_get_nrd_info(const CaelixRenderer* renderer,
                                                     CaelixNrdInfo* out);

/* ---------------------------------------------------------------------------
 * Upscalers: availability, Vulkan extensions, jitter, state (see "Upscalers" in the renderer
 * section and docs/DLSS.md)
 * ------------------------------------------------------------------------- */

typedef struct CaelixUpscalerQueryDesc
{
    uint32_t upscaler;                  /* CaelixUpscaler */
    uint32_t quality;                   /* CaelixUpscaleQuality */
    uint32_t output_width, output_height; /* the colour target size */
} CaelixUpscalerQueryDesc;

typedef enum CaelixUpscalerUnsupportedReason
{
    CAELIX_UPSCALER_OK = 0,
    CAELIX_UPSCALER_NOT_BUILT = 1,           /* the library was built without the upscaler (CAELIX_DLSS=OFF) */
    CAELIX_UPSCALER_NO_NGX = 2,              /* NGX did not start: no NVIDIA driver, a DLL or a device
                                                extension is missing (see the log) */
    CAELIX_UPSCALER_FEATURE_UNAVAILABLE = 3, /* the GPU or the NGX runtime lacks the feature */
    CAELIX_UPSCALER_DRIVER_TOO_OLD = 4,      /* see min_driver_major / min_driver_minor */
    CAELIX_UPSCALER_QUALITY_UNAVAILABLE = 5, /* the feature has no render size for this quality */
    CAELIX_UPSCALER_BAD_ARGUMENT = 6         /* output below 64 x 32 */
} CaelixUpscalerUnsupportedReason;

typedef struct CaelixUpscalerQueryResult
{
    uint32_t supported; /* 1 = the device, driver and build support this upscaler at this quality */
    uint32_t reason;    /* CaelixUpscalerUnsupportedReason when supported = 0, else CAELIX_UPSCALER_OK */
    uint32_t render_width, render_height; /* optimal render size, the size the renderer requires
                                             (0 when unsupported) */
    uint32_t render_min_width, render_min_height, render_max_width, render_max_height; /* the
                                             feature's dynamic range (informative; the renderer uses
                                             the optimal size) */
    uint32_t min_driver_major, min_driver_minor; /* when the driver is too old, else 0 */
} CaelixUpscalerQueryResult;

/* Fills *out for `desc`. Returns CAELIX_OK also when the upscaler is unsupported (read
   `supported`); CAELIX_ERROR_INVALID_ARGUMENT for a null argument or an unknown upscaler or quality.
   CAELIX_UPSCALER_NONE is always supported with render size = output size. Creates the device's
   NGX runtime on first use (also done by the first upscaled render). */
CAELIX_API CaelixResult caelix_upscaler_query(CaelixDevice* device, const CaelixUpscalerQueryDesc* desc,
                                              CaelixUpscalerQueryResult* out);

/* Vulkan extensions the upscalers need, as NGX reports them (the union over DLSS SR and RR). The
   strings and the array are owned by the library; count 0 (and a null array) when the build has no
   upscaler. The instance list needs nothing: call it before vkCreateInstance and enable the names
   the instance supports. A null argument is ignored. */
CAELIX_API void caelix_upscaler_instance_extensions(uint32_t* out_count, const char* const** out_names);
/* The device list needs the VkInstance and the VkPhysicalDevice (NGX's own loader queries them):
   call it before vkCreateDevice and enable the names the physical device supports (without them
   NGX does not start: CAELIX_UPSCALER_NO_NGX). The array stays valid until the next call of this
   function. CAELIX_ERROR_INVALID_ARGUMENT for a null argument or a zero handle; CAELIX_ERROR_VULKAN
   when NGX's query fails (logged; count 0). */
CAELIX_API CaelixResult caelix_upscaler_device_extensions(uint64_t instance, uint64_t physical_device,
                                                          uint32_t* out_count,
                                                          const char* const** out_names);

/* Jitter helpers (pure functions, any thread, also without an upscaler in the build).
   caelix_upscaler_jitter_phases: the jitter sequence length the DLSS guide recommends,
   8 * ceil((output_width / render_width)^2), at least 32 for CAELIX_UPSCALER_DLSS_RR (the RR
   guide asks for at least 32). A zero width counts as a ratio of 1.
   caelix_jitter_halton: the Halton(2, 3) point number (index % phases) + 1 minus 0.5, so both
   values lie in [-0.5, 0.5): out_jitter[0] = Halton base 2, out_jitter[1] = Halton base 3 (pass
   them as CaelixCamera::jitter). phases 0 does not wrap. */
CAELIX_API uint32_t caelix_upscaler_jitter_phases(uint32_t upscaler, uint32_t render_width,
                                                  uint32_t output_width);
CAELIX_API void caelix_jitter_halton(uint32_t index, uint32_t phases, float out_jitter[2]);

/* The renderer's current upscaler: the options of its newest upscaled render call. All zero until
   the first one. */
typedef struct CaelixUpscalerInfo
{
    uint32_t upscaler; /* CaelixUpscaler */
    uint32_t quality;  /* CaelixUpscaleQuality */
    uint32_t preset;   /* the preset hint (0 = NGX default) */
    uint32_t render_width, render_height, output_width, output_height;
    uint32_t failed;         /* 1 after an NGX error: the renderer falls back to the nearest upscale
                                until the upscaler options change */
    uint32_t depth_inverted; /* 1 when the host projection is reverse Z (detected per call) */
} CaelixUpscalerInfo;

/* Fills *out. Returns CAELIX_ERROR_INVALID_ARGUMENT (logged) for a NULL argument. */
CAELIX_API CaelixResult caelix_renderer_get_upscaler_info(const CaelixRenderer* renderer,
                                                          CaelixUpscalerInfo* out);

/* Global illumination method of the lighting modes (docs/GI_EXPERIMENTS.md). Experimental. */
typedef enum CaelixGiMethod
{
    CAELIX_GI_PATH_TRACE = 0, /* the path tracer as before */
    CAELIX_GI_RESTIR = 1      /* screen-space ReSTIR GI on the first diffuse bounce */
} CaelixGiMethod;

#define CAELIX_GI_FLAG_TEMPORAL (1u << 0)     /* temporal reuse (previous frame's reservoir) */
#define CAELIX_GI_FLAG_SPATIAL (1u << 1)      /* spatial reuse */
#define CAELIX_GI_FLAG_VISIBILITY (1u << 2)   /* ReSTIR GI: a shadow ray per reused neighbour sample (occluded ->
                                                 weight 0). ReSTIR DI ignores it (its shade pass traces the ray) */
#define CAELIX_GI_FLAG_UNBIASED (1u << 3)     /* Z-count MIS instead of 1/M. ReSTIR GI: implies VISIBILITY rays
                                                 per neighbour. ReSTIR DI: Z by target support in the SPATIAL
                                                 pass only (the temporal pass is 1/M in every mode), no rays;
                                                 recommended for DI (1/M loses ~10 % where the supports differ:
                                                 horizon, near-field split) */
#define CAELIX_GI_FLAG_DIFFUSE_ONLY (1u << 4) /* force the Lambert lobe at every vertex */
#define CAELIX_GI_FLAG_LIGHT_NEE (1u << 5)    /* next-event estimation on the scene's light list at every
                                                 diffuse vertex (docs/LIGHT_LIST.md): world-space reservoirs
                                                 (light cells) of light samples, brick-level or (with
                                                 LIGHT_CELL_PUSH) voxel-face; both methods */
#define CAELIX_GI_FLAG_RESTIR_DI (1u << 6)    /* per-pixel ReSTIR DI on the light list at the primary
                                                 surface (docs/RESTIR_DI.md); needs LIGHT_NEE; both methods */
#define CAELIX_GI_FLAG_LIGHT_CELL_JITTER (1u << 7) /* light NEE: each vertex looks its light cell up at a point
                                                 jittered by up to half a cell, so that neighbouring pixels read
                                                 different cells (the per-cell flicker becomes per-pixel noise;
                                                 docs/LIGHT_LIST.md section 2.5). Unbiased. */
#define CAELIX_GI_FLAG_LIGHT_MIS (1u << 8)    /* with LIGHT_NEE: MIS (balance heuristic) between the light NEE and
                                                 the diffuse bounce (docs/LIGHT_MIS.md); without it a list-light
                                                 hit after a light NEE vertex is skipped. With RESTIR_DI the DI
                                                 at the primary surface takes part as the NEE strategy (not with
                                                 CAELIX_GI_RESTIR) */
#define CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL (1u << 9) /* with LIGHT_NEE: the cell fill merges each reservoir's
                                                 previous frame reservoir as temporal history (M-capped by
                                                 light_temporal_frames) and estimates the cell normalisation
                                                 from dedicated draws accumulated over frames
                                                 (docs/LIGHT_LIST.md section 2.2). Assumes a static scene (no
                                                 scene edits, no entity motion); the history is dropped when
                                                 the light table was rebuilt or reset is set */
#define CAELIX_GI_FLAG_LIGHT_CELL_PUSH (1u << 10) /* with LIGHT_NEE: every vertex that sampled a light cell
                                                 pushes its light sample and its shadow ray result into
                                                 the cell; the next fill merges the pushes (confidence
                                                 light_push_confidence, an occluded sample with a lower
                                                 target), so voxel-face samples enter the cells
                                                 (docs/LIGHT_LIST.md section 2.2). Unbiased. Needs
                                                 light_nee_candidates 1 */
#define CAELIX_GI_FLAG_LIGHT_PROXY_VERTEX (1u << 11) /* with LIGHT_MIS: the MIS proxy density of the light NEE
                                                 at a vertex with a light cell is the vertex form (the
                                                 voxel-face weight times the receiver's own geometry
                                                 term over the cell's normalisation) instead of the cell
                                                 form (docs/LIGHT_MIS.md "Vertex proxy"). Unbiased: both
                                                 strategies evaluate the same proxy */
#define CAELIX_GI_FLAG_LIGHT_CELL_ORIENTED (1u << 12) /* with LIGHT_NEE: the light cell key also holds the
                                                 receiver's face (the axis of its world normal, 6 values),
                                                 so a spatial cell splits into up to 6 oriented cells; the
                                                 fill's target and the MIS proxy's cell form get the
                                                 receiver orientation term (floor 0.1, never 0)
                                                 (docs/LIGHT_LIST.md section 2.7). Unbiased */
#define CAELIX_GI_FLAG_PAIRWISE_MIS (1u << 13) /* ReSTIR GI: pairwise MIS (balance heuristic between each spatial
                                                 neighbour and the pixel's own sample, with visibility in both
                                                 densities: two shadow rays per accepted neighbour) in the
                                                 spatial pass and the balance heuristic in the temporal pass,
                                                 instead of 1/M or the Z-count (VISIBILITY and UNBIASED are then
                                                 ignored by ReSTIR GI; ReSTIR DI ignores this flag). Bounds the
                                                 reconnection Jacobian's outliers (docs/GI_EXPERIMENTS.md,
                                                 "Pairwise MIS") */

/* The GI options of one renderer. ReSTIR GI (CAELIX_GI_RESTIR) runs in the lighting modes when
   bounce_count >= 1: the trace kernel stores one initial sample per pixel (the first bounce of the
   path, with the path's radiance from there on), then a temporal pass, up to two spatial passes and a
   shade pass write the diffuse radiance that the composite (or NRD) reads. It forces the Lambert lobe
   at every vertex, ignores sun NEE and the fog on the GI path, and writes no specular radiance.
   CAELIX_GI_FLAG_PAIRWISE_MIS weights its temporal and spatial reuse with the balance heuristic
   (pairwise MIS with two shadow rays per spatial neighbour) instead of 1/M or the Z-count.
   CAELIX_GI_FLAG_LIGHT_NEE (either method): every diffuse vertex that has a next segment samples one
   point on a list light (an emissive voxel face whose material is a list light of the scene, see
   CaelixLightOptions) through the renderer's light cells (world-space reservoirs keyed by entity,
   level and cell coordinate, refilled before every trace) and traces a shadow ray; a segment that
   leaves such a vertex no longer counts the emission of a list light it hits when the NEE can
   sample that point (a brick the table dropped keeps its emission). With CAELIX_GI_FLAG_LIGHT_MIS
   the NEE and such a hit are combined by multiple importance sampling instead (balance heuristic
   over a proxy of the NEE density and the cosine density of the diffuse bounce); the primary
   vertex keeps the exclusive rule with ReSTIR GI. The light_* fields size the cells.
   CAELIX_GI_FLAG_RESTIR_DI (with LIGHT_NEE, either method, bounce_count >= 1): the primary vertex's
   one-sample light NEE is replaced by per-pixel ReSTIR DI: di_candidates initial light points per
   pixel, then temporal and spatial reuse with the TEMPORAL, SPATIAL, VISIBILITY and UNBIASED flags
   and the reuse fields above, and one shadow ray per pixel; the result is added to the diffuse
   radiance (after ReSTIR GI's). The bounce vertices keep the one-sample light NEE. With
   CAELIX_GI_FLAG_LIGHT_MIS and CAELIX_GI_PATH_TRACE the DI takes part in the MIS: its target and its
   contribution carry the NEE's balance heuristic weight at the pixel's primary surface, and the
   first bounce segment counts a list-light hit with the bounce's weight. */
typedef struct CaelixGiOptions
{
    uint32_t method;          /* CaelixGiMethod */
    uint32_t flags;           /* CAELIX_GI_FLAG_* */
    uint32_t temporal_m_cap;  /* 0 = 20; the previous reservoir's M is clamped to this before merging */
    uint32_t spatial_samples; /* 0 = 5; 1 .. 8 neighbours per pass */
    uint32_t spatial_passes;  /* 0 = 1; 1 .. 2 */
    float spatial_radius;     /* pixels, 0 = 30 */
    float depth_threshold;    /* 0 = 0.1: |x_v(q) - x_v(r)| must be below threshold * view_z(r) */
    uint32_t reset;           /* 1 = drop the reservoir history with the next render */
    uint32_t light_cell_capacity; /* 0 = 65536 light cells; else a power of two in 1024 .. 1048576
                                     (64 + 24 x light_cell_reservoirs bytes per cell) */
    uint32_t light_candidates;    /* 0 = 8; 1 .. 64 candidates per cell reservoir and frame */
    float light_level_footprint;  /* 0 = 0.25: a vertex whose pixel footprint (world units per pixel at
                                     its camera distance) is below this uses 32-voxel cells, else
                                     128-voxel cells */
    uint32_t di_candidates;       /* 0 = 4; 1 .. 32 initial light candidates per pixel (RESTIR_DI) */
    uint32_t light_temporal_frames; /* 0 = 20; LIGHT_CELL_TEMPORAL: the history behind a cell reservoir is
                                       clamped to this many frames (x light_candidates candidates) before
                                       the merge; 1 .. 1024 */
    float light_push_confidence; /* 0 = 8; LIGHT_CELL_PUSH: the confidence (M) of a pushed sample in the
                                    cell fill; finite, 0 .. 1024 */
    uint32_t light_nee_candidates; /* 0 = 1; LIGHT_NEE: light candidates per diffuse vertex, resampled by
                                      the exact target at the vertex before the one shadow ray
                                      (docs/LIGHT_LIST.md section 2.6); 1 = no resampling (one
                                      sample, the earlier NEE); 1 .. 32. LIGHT_CELL_PUSH needs 1 */
    uint32_t light_cell_reservoirs; /* 0 = 8; LIGHT_NEE: reservoirs per light cell (the push ring has as many
                                       entries); a power of two in 1 .. 64. A cell is 64 + 24 x this bytes
                                       (256 at 8); a change recreates the cell table (history dropped) */
} CaelixGiOptions;

/* Sets the GI options of the renderer; they apply from the next render on and stay until the next
   call. Zero-initialised options = the path tracer. Lighting modes only (the primary mode ignores
   them). CAELIX_MODE_REFERENCE accumulates the chosen method's output, so a biased method converges
   to its biased image (that is how the bias is measured). A new method or new flags also drop the
   reservoir history. With CAELIX_GI_PATH_TRACE and without CAELIX_GI_FLAG_DIFFUSE_ONLY the image is
   the path tracer's, bit for bit. The reservoir buffers (4 x 48 bytes per pixel) are allocated at
   the first ReSTIR render and kept until the renderer is destroyed (recreated on a size change);
   the DI reservoir buffers (4 x 48 bytes per pixel) likewise at the first render with
   CAELIX_GI_FLAG_RESTIR_DI. The DI history is also dropped when the scene's light table was rebuilt
   since the renderer's previous render.
   The light cells (light_cell_capacity x (64 + 24 x light_cell_reservoirs) bytes + 32) are allocated
   at the first render with CAELIX_GI_FLAG_LIGHT_NEE and recreated when the capacity or the reservoir
   count changes.
   Validation: unknown method or flags, spatial_samples > 8, spatial_passes > 2, non-finite or negative
   floats, a light_cell_capacity that is not 0 or a power of two in 1024 .. 1048576,
   light_candidates > 64, CAELIX_GI_FLAG_RESTIR_DI without CAELIX_GI_FLAG_LIGHT_NEE, di_candidates > 32,
   light_temporal_frames > 1024, a light_push_confidence that is not finite or not in 0 .. 1024,
   light_nee_candidates > 32, CAELIX_GI_FLAG_LIGHT_CELL_PUSH with an effective light_nee_candidates
   >= 2, a light_cell_reservoirs that is not 0 or a power of two in 1 .. 64
   -> CAELIX_ERROR_INVALID_ARGUMENT (logged; the previous options stay). */
CAELIX_API CaelixResult caelix_renderer_set_gi_options(CaelixRenderer* renderer, const CaelixGiOptions* options);

typedef struct CaelixGiStats
{
    float gi_ms;              /* GPU time of the GI and DI reuse passes (temporal + spatial + shade) of the
                                 frame of CaelixGpuTimings (0 when that frame ran no reuse pass) */
    uint64_t reservoir_bytes; /* bytes of the GI and DI reservoir buffers currently allocated */
    uint64_t light_cell_bytes; /* bytes of the light cell table currently allocated */
    float light_fill_ms;       /* GPU time of the light cell fill of the frame of CaelixGpuTimings (0 when
                                  that frame ran no fill) */
    /* The light cell usage counters of the frame of CaelixGpuTimings (all 0 when that frame ran no
       light NEE). light_cell_bytes includes their 32 bytes at the tail of the cell table. */
    uint32_t light_cells_live;          /* cells with a key after this frame's eviction */
    uint32_t light_cells_active;        /* of those, looked up by the previous frame's trace */
    uint32_t light_cells_inserted;      /* cells inserted by this frame's lookups */
    uint32_t light_cell_lookups_failed; /* lookups that found neither the key nor a free slot in 8 probes
                                           (the table is full around there) */
    uint32_t light_cell_pushes;         /* LIGHT_CELL_PUSH: pushes of the previous frame's trace that got a
                                           ring entry (counted by this frame's fill) */
    uint32_t light_cell_pushes_dropped; /* LIGHT_CELL_PUSH: pushes of the previous frame's trace beyond the
                                           light_cell_reservoirs ring entries of their cell */
} CaelixGiStats;

/* Fills *out; a null argument is ignored. */
CAELIX_API void caelix_renderer_get_gi_stats(const CaelixRenderer* renderer, CaelixGiStats* out);

/* Reserved for later (not declared yet):
 *   caelix_scene_upload_brick_summaries - per-brick summaries for bricks that are known but not
 *                                         resident (slot state "summary-only").
 *   caelix_scene_poll_brick_requests    - bricks that the GPU asked for (slot state "requested").
 */

#ifdef __cplusplus
}
#endif
#endif
