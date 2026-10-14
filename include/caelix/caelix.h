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
 * Thread rule: calls on one object (a CaelixDevice, a CaelixScene) are
 * externally synchronized by the host. The library does not lock: the host must
 * not call two functions on the same scene (or on a scene and its device) at the
 * same time from different threads.
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
   the finalize step appended fields to CaelixSceneDesc (encoding, default 0), CaelixSceneStats and
   CaelixBrickInfo without a bump. A host built against an older version-2 header must be rebuilt
   (the library writes the larger output structs). */
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
  CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT     = 1u << 8  /* VkPhysicalDeviceVulkan12Features::scalarBlockLayout */
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
   caelix_get_device_requirements; a missing item is logged and returns CAELIX_ERROR_UNSUPPORTED.
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

typedef struct CaelixTargetImage {
  uint64_t image;   /* VkImage */
  uint32_t format;  /* VkFormat */
  uint32_t width;
  uint32_t height;
  uint32_t layout;  /* VkImageLayout of the image when the library starts recording */
} CaelixTargetImage;

/* TEMPORARY, removed when the renderer lands: records into `command_buffer` (a VkCommandBuffer in the
   recording state, owned by the host) a clear of `target` to `rgba`. On return the image is in
   VK_IMAGE_LAYOUT_GENERAL. This is the layout contract every later render call keeps: the library
   leaves the target in GENERAL; the host transitions it for present.
   The target image needs VK_IMAGE_USAGE_TRANSFER_DST_BIT. */
CAELIX_API CaelixResult caelix_clear_target(CaelixDevice* device, uint64_t command_buffer,
                                            const CaelixTargetImage* target, const float rgba[4]);

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
   neighbours, with barriers: one before the copies (all earlier commands -> transfer), transfer ->
   compute before the kernel, and one at the end (transfer and compute -> all later commands).
   Recycles staging spans whose frame <= safe. With nothing to do it records nothing. A frame_number that is not greater than
   the previous flush's, or a safe_frame_number >= frame_number, returns
   CAELIX_ERROR_INVALID_ARGUMENT. */
CAELIX_API CaelixResult caelix_scene_flush(CaelixScene* scene, const CaelixFrameDesc* desc);

/* Tells the scene that every frame <= safe_frame_number has completed on the GPU and releases the
   staging spans and old GPU buffers of those frames now. Optional: caelix_scene_flush does the same.
   Use it when the host waits for a frame and then uploads before the next flush, for example in a
   loop that streams more bricks than the staging ring holds. A value lower than an earlier one has
   no effect. */
CAELIX_API CaelixResult caelix_scene_recycle(CaelixScene* scene, uint64_t safe_frame_number);

/* Fields are only appended. The last four fields were added to ABI version 2 by the finalize step. */
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
} CaelixSceneStats;

/* Fills *out. A NULL argument is ignored. */
CAELIX_API void caelix_scene_get_stats(const CaelixScene* scene, CaelixSceneStats* out);

/* Inspection and debug aid: where the GPU record, the slot table row and the AABB row of a brick
   live. The GPU
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
} CaelixBrickInfo;

CAELIX_API CaelixResult caelix_scene_query_brick(const CaelixScene* scene, CaelixEntityId entity,
                                                 CaelixBrickKey key, CaelixBrickInfo* out);

/* Reserved for later (not declared yet):
 *   caelix_scene_upload_brick_summaries - per-brick summaries for bricks that are known but not
 *                                         resident (slot state "summary-only").
 *   caelix_scene_poll_brick_requests    - bricks that the GPU asked for (slot state "requested").
 *   caelix_renderer_render              - records the frame's rendering into the host's command
 *                                         buffer.
 */

#ifdef __cplusplus
}
#endif
#endif
