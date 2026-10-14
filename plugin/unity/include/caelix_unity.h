#ifndef CAELIX_UNITY_H
#define CAELIX_UNITY_H
/*
 * caelix_unity: the Unity native plugin of the Caelix renderer. C API for C# (P/Invoke).
 *
 * The plugin is preloaded by Unity ("Load on startup" on the plugin importer). At load it
 * intercepts Unity's Vulkan initialization (vkCreateInstance, vkCreateDevice) and adds the
 * renderer's requirements (caelix_get_device_requirements); at the graphics device
 * initialization it adopts Unity's device (caelix_device_adopt).
 *
 * Thread rule: every caelix_* call of the renderer library happens on Unity's render thread,
 * inside a plugin event or a graphics device event. The functions below never call the library:
 * they only read atomics, so C# may call them from any thread.
 *
 * Packets: C# owns the packet memory (unmanaged, for example Marshal.AllocHGlobal). It fills one
 * packet per render event and passes its address as the data of
 * CommandBuffer.IssuePluginEventAndData(caelix_unity_get_render_event_func(),
 * status.render_event_id, packet). The render thread reads the packet and its records IN PLACE
 * during the event (the renderer library copies brick data into its staging ring; the plugin
 * copies the bricks the ring cannot take into its own pending buffer) and publishes the packet's
 * sequence number as CaelixUnityStatus::last_consumed_sequence when it is done with the memory,
 * on success and on failure. C# may reuse or free a packet when last_consumed_sequence >= the
 * packet's sequence. The GPU never reads packet memory.
 *
 * Target views: the renderer caches one image view per target image (CaelixRenderTargets). When
 * C# destroys a target RenderTexture, a later RenderTexture can get the same VkImage handle, and
 * the cached view would then show a destroyed image. So whenever C# replaces a target, it issues
 * the release event (status.release_event_id) BEFORE the first render event that uses the new
 * target. That event runs with graphics queue access: it waits for the graphics queue to be idle
 * and calls caelix_renderer_release_target_views. Its data is NULL or the packet of the render
 * event that follows it: when that packet's encoding differs from the live scene's, the release
 * event also destroys the renderer and the scene, and the render event creates them again with
 * the new encoding. C# issues the release event for that reason too.
 */
#include <stdint.h>
#include <caelix/caelix.h>

#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32)
  #if defined(CAELIX_UNITY_BUILD)
    #define CAELIX_UNITY_API __declspec(dllexport)
  #else
    #define CAELIX_UNITY_API __declspec(dllimport)
  #endif
#else
  #define CAELIX_UNITY_API __attribute__((visibility("default")))
#endif

/* Bumped on every change of CaelixUnityPacket, its records or CaelixUnityStatus. */
#define CAELIX_UNITY_PACKET_VERSION 2u

/* A render target, given as Unity's native texture pointer. */
typedef struct CaelixUnityTarget {
  void* native_texture;        /* Texture.GetNativeTexturePtr(); NULL = the target is absent */
  void* native_render_buffer;  /* RenderTexture.colorBuffer.GetNativeRenderBufferPtr(); the
                                  fallback when AccessTexture does not accept native_texture.
                                  May be NULL. */
  uint32_t width;              /* must match the texture's size */
  uint32_t height;
} CaelixUnityTarget;

/* CaelixUnityEntity::flags. 0 = a reference only (a brick remove or upload names the entity). */
#define CAELIX_UNITY_ENTITY_ADD 1u        /* add the entity; an existing entity with the same guid
                                             is removed first, with all its bricks */
#define CAELIX_UNITY_ENTITY_REMOVE 2u     /* remove the entity and all its bricks */
#define CAELIX_UNITY_ENTITY_TRANSFORM 4u  /* set the entity transform */

/* One entity record. 48 bytes. The plugin maps guid -> CaelixEntityId. */
typedef struct CaelixUnityEntity {
  uint8_t guid[16];
  float position[3];  /* world position of the entity origin (ADD, TRANSFORM) */
  float rotation[4];  /* unit quaternion x, y, z, w (ADD, TRANSFORM) */
  uint32_t flags;     /* CAELIX_UNITY_ENTITY_* */
} CaelixUnityEntity;

/* One brick removal. 16 bytes. */
typedef struct CaelixUnityBrickRemove {
  uint32_t entity_index;  /* index into this packet's entity records */
  CaelixBrickKey key;
} CaelixUnityBrickRemove;

/* One upload batch of one entity. 16 bytes. Offsets are bytes from the packet start. */
typedef struct CaelixUnityUpload {
  uint32_t entity_index;   /* index into this packet's entity records */
  uint32_t brick_count;
  uint32_t keys_offset;    /* brick_count CaelixBrickKey, 4-byte aligned */
  uint32_t blocks_offset;  /* brick_count * 512 uint16 block values (raw storage values, voxel
                              index x + 8 y + 64 z), 16-byte aligned */
} CaelixUnityUpload;

/* One entry of a partial material table (CAELIX_UNITY_MATERIALS_PARTIAL). 52 bytes. */
typedef struct CaelixUnityMaterialEntry {
  uint32_t code;  /* 16-bit block code, 0 .. 0xFFFF */
  CaelixMaterial material;
} CaelixUnityMaterialEntry;

/* CaelixUnityPacket::material_mode */
#define CAELIX_UNITY_MATERIALS_NONE 0u     /* no material change */
#define CAELIX_UNITY_MATERIALS_FULL 1u     /* caelix_scene_set_materials(CaelixMaterial
                                              [material_count]); count 0 restores the built-in
                                              table */
#define CAELIX_UNITY_MATERIALS_PARTIAL 2u  /* the built-in table with CaelixUnityMaterialEntry
                                              [material_count] written over it */

/* CaelixUnityPacket::scene_flags */
#define CAELIX_UNITY_SCENE_RESET 1u  /* before this packet's records: remove every entity, drop the
                                        plugin's pending uploads and restore the built-in material
                                        table. C# sets it on its first scene packet (the plugin's
                                        scene outlives Play sessions) and when it sends the world
                                        again. */

/* One render event. The fixed part (this struct) is followed, in the same memory block, by the
   records it points at; total_bytes covers all of it. Offsets are bytes from the packet start.
   C# writes everything before the event is issued. Apply order on the render thread: reset, the
   entity records in list order, the plugin's pending uploads of earlier packets, the brick
   removes, the uploads (appended to the pending uploads while some are still waiting, so the
   order per entity is kept), the materials, then flush and render. */
typedef struct CaelixUnityPacket {
  uint32_t size;          /* sizeof(CaelixUnityPacket); the plugin rejects any other value */
  uint32_t version;       /* CAELIX_UNITY_PACKET_VERSION */
  uint64_t sequence;      /* > 0, strictly increasing per issued packet */
  uint32_t encoding;      /* scene block encoding: 0 packed scene colour, 1 block-ID. See the
                             release event for a change of encoding. */
  uint32_t target_generation; /* changes whenever C# replaced a target texture (see the release
                                 event); the plugin warns when it changes without a release */
  CaelixUnityTarget color;    /* RGBA16F with random write (STORAGE usage) */
  CaelixCamera camera;
  CaelixRenderOptions options;

  /* Scene section (version 2). */
  uint32_t scene_flags;       /* CAELIX_UNITY_SCENE_* */
  uint32_t total_bytes;       /* the whole block: this struct plus every record */
  uint32_t entity_count;      /* CaelixUnityEntity[entity_count] at entities_offset */
  uint32_t entities_offset;
  uint32_t remove_count;      /* CaelixUnityBrickRemove[remove_count] at removes_offset */
  uint32_t removes_offset;
  uint32_t upload_count;      /* CaelixUnityUpload[upload_count] at uploads_offset */
  uint32_t uploads_offset;
  uint32_t material_mode;     /* CAELIX_UNITY_MATERIALS_* */
  uint32_t material_count;    /* entries at materials_offset (layout per material_mode) */
  uint32_t materials_offset;
  uint32_t pad;
} CaelixUnityPacket;

/* Plugin state for C#. Counters are monotonic. */
typedef struct CaelixUnityStatus {
  uint32_t size;                  /* sizeof(CaelixUnityStatus) */
  uint32_t packet_size;           /* sizeof(CaelixUnityPacket) as compiled into the plugin */
  uint32_t packet_version;        /* CAELIX_UNITY_PACKET_VERSION */
  uint32_t unity_renderer;        /* UnityGfxRenderer at the last check (21 = Vulkan) */
  uint32_t intercept_registered;  /* 1: InterceptInitialization returned true */
  uint32_t intercept_called;      /* 1: Unity called the init callback (Vulkan initialization) */
  uint32_t instance_patched;      /* 1: vkCreateInstance succeeded with the plugin's api version */
  uint32_t device_patched;        /* 1: vkCreateDevice succeeded with the plugin's extensions and
                                     features */
  uint32_t device_adopted;        /* 1: caelix_device_adopt returned CAELIX_OK */
  int32_t last_error;             /* CaelixResult of the newest failure (0 = none) */
  int32_t render_event_id;        /* event id of the render event */
  int32_t release_event_id;       /* event id of the target-view release event */
  uint64_t last_consumed_sequence; /* sequence of the newest packet the render thread is done with */
  uint64_t last_unity_frame;      /* Unity's currentFrameNumber at the newest render event */
  uint64_t last_safe_frame;       /* Unity's safeFrameNumber at the newest render event */
  uint64_t last_plugin_frame;     /* the newest frame number the plugin passed to the library */
  uint64_t renders;               /* render events that recorded a render */
  uint64_t failed_events;         /* render events that failed (see the Unity log) */
  uint64_t view_releases;         /* release events that ran */
  /* Version 2: the scene, as of the newest flush. */
  uint32_t scene_encoding;        /* encoding of the live scene; 0xFFFFFFFF = no scene */
  uint32_t entities;              /* CaelixSceneStats::entities */
  uint32_t groups;                /* CaelixSceneStats::groups */
  uint32_t scene_recreations;     /* scenes destroyed for an encoding change */
  uint64_t bricks_resident;       /* CaelixSceneStats::bricks_resident */
  uint64_t bricks_pending_plugin; /* bricks the plugin holds because the staging ring was full */
  uint64_t as_builds_pending;     /* CaelixSceneStats::as_builds_pending */
  uint64_t uploaded_total;        /* bricks accepted by caelix_scene_upload_bricks so far */
  uint64_t removed_total;         /* brick removes passed to caelix_scene_remove_bricks so far */
  uint64_t scene_packets;         /* render events whose packet carried scene records */
  uint64_t record_bytes;          /* CaelixSceneStats::record_bytes */
  uint64_t blas_bytes;            /* CaelixSceneStats::blas_bytes */
  float flush_ms;                 /* CaelixGpuTimings of the newest frame read back */
  float trace_ms;
  float composite_ms;
  float denoise_ms;
  uint64_t timings_frame;         /* CaelixGpuTimings::frame_number */
} CaelixUnityStatus;

/* The UnityRenderingEventAndData callback for CommandBuffer.IssuePluginEventAndData. Never NULL. */
CAELIX_UNITY_API void* caelix_unity_get_render_event_func(void);
/* Fills *out (NULL is ignored). Any thread. */
CAELIX_UNITY_API void caelix_unity_status(CaelixUnityStatus* out);

#ifdef __cplusplus
}
#endif
#endif
