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
 * Three plugin events, each with its own packet kind (CaelixUnityPacketHeader::kind), all issued
 * through CommandBuffer.IssuePluginEventAndData(caelix_unity_get_render_event_func(), event id,
 * packet):
 *
 *   frame event (status.frame_event_id, CaelixUnityFramePacket): ONCE per Unity frame, from the
 *     main thread's per-frame tick through Graphics.ExecuteCommandBuffer, before any camera of the
 *     frame renders. The render thread applies the packet's scene records (entities, brick
 *     removes, brick uploads) and flushes the scene (caelix_scene_flush) with Unity's own frame
 *     numbers: frame_number = currentFrameNumber and safe_frame_number = safeFrameNumber of
 *     UnityVulkanRecordingState, no conversion. Every view of the frame then renders the same scene
 *     state, and entity transforms never lag the camera. A frame packet whose Unity frame was
 *     already flushed (a second frame event in one Unity frame) applies its records but does not
 *     flush (logged); they reach the GPU with the next frame's flush.
 *
 *   render event (status.render_event_id, CaelixUnityViewPacket): once per view per Unity frame,
 *     from the view's camera render pass. Renders the view's renderer (created on the view's first
 *     render event) into the packet's targets with the same Unity frame numbers. A renderer renders
 *     at most once per Unity frame (library rule): C# gives every camera its own view.
 *
 *   release event (status.release_event_id, CaelixUnityReleasePacket): runs with graphics queue
 *     access and waits for the queue to be idle, then does what its flags say: release the cached
 *     target image views of every renderer (before the first render event that uses a replaced
 *     target texture: a new RenderTexture can get the VkImage handle of a destroyed one), destroy
 *     one view's renderer (the camera is gone), or destroy every renderer and the scene (a session
 *     ends, or starts: see CAELIX_UNITY_FRAME_RESET).
 *
 * Sessions: the scene's block encoding, its material table and its staging ring size are fixed
 * for a session. A session starts with a frame packet that carries CAELIX_UNITY_FRAME_RESET and the
 * session config (encoding, staging size, material entries), preceded by a release event with
 * CAELIX_UNITY_RELEASE_SCENE; it ends with another such release event (C# issues it when its
 * component is disabled, so the plugin holds no scene between Play sessions). A material or
 * encoding change during a session is a C# error, not a reset.
 *
 * Packets: C# owns the packet memory (unmanaged, for example Marshal.AllocHGlobal) and writes
 * everything before the event is issued. The render thread reads a packet and its records IN
 * PLACE during the event and publishes the packet's sequence number as the matching
 * CaelixUnityStatus::last_consumed_*_sequence when it is done with the memory, on success and on
 * failure. Sequences are per packet kind, > 0 and strictly increasing. C# may reuse or free a
 * packet when the published sequence of its kind is >= the packet's sequence. The GPU never reads
 * packet memory.
 *
 * Upload sizing: there is no plugin-side queue for bricks the staging ring cannot take. C# sizes a
 * frame packet's uploads from CaelixUnityStatus::upload_bricks_free (published after every flush)
 * minus the bricks of frame packets it has issued since, capped at upload_bricks_cap (the ring's
 * sustained rate: frames in flight keep their staging space until they complete). Bricks the
 * library still refuses are dropped and counted in CaelixUnityStatus::bricks_dropped (logged); C#
 * treats a growing count as an error and sends the world again.
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

/* Bumped on every change of the packets, their records or CaelixUnityStatus.
   Version 5: three packet kinds with a common header (frame, view, release), sessions with a
   reset config, sky cubemap target, no material mode, no per-packet encoding, 8 views. */
#define CAELIX_UNITY_PACKET_VERSION 5u

/* Number of views (CaelixUnityViewPacket::view). Each view has its own renderer over the shared
   scene. C# gives every camera that renders through the plugin its own view. */
#define CAELIX_UNITY_MAX_VIEWS 8u

/* CaelixUnityPacketHeader::kind */
#define CAELIX_UNITY_PACKET_FRAME 1u   /* CaelixUnityFramePacket, frame event */
#define CAELIX_UNITY_PACKET_VIEW 2u    /* CaelixUnityViewPacket, render event */
#define CAELIX_UNITY_PACKET_RELEASE 3u /* CaelixUnityReleasePacket, release event */

/* The first 24 bytes of every packet. The plugin rejects (logged, counted in failed_events) a
   packet whose size or version differs from what it was compiled with, or whose kind is not the
   kind of the event it came with; such a packet's sequence is not published. */
typedef struct CaelixUnityPacketHeader
{
    uint32_t size;     /* sizeof the packet struct of `kind` */
    uint32_t version;  /* CAELIX_UNITY_PACKET_VERSION */
    uint32_t kind;     /* CAELIX_UNITY_PACKET_* */
    uint32_t pad;
    uint64_t sequence; /* > 0, strictly increasing per kind */
} CaelixUnityPacketHeader;

/* A texture, given as Unity's native texture pointer. */
typedef struct CaelixUnityTarget
{
    void* native_texture;       /* Texture.GetNativeTexturePtr(); NULL = the target is absent */
    void* native_render_buffer; /* RenderTexture.colorBuffer.GetNativeRenderBufferPtr(); the
                                   fallback when AccessTexture does not accept native_texture.
                                   May be NULL. */
    uint32_t width;             /* must match the texture's size (a cubemap: its face size) */
    uint32_t height;
} CaelixUnityTarget;

/* ---------------------------------------------------------------------------------------------
 * Frame packet
 * ------------------------------------------------------------------------------------------- */

/* CaelixUnityEntity::flags. 0 = a reference only (a brick remove or upload names the entity). */
#define CAELIX_UNITY_ENTITY_ADD 1u       /* add the entity; an existing entity with the same guid
                                            is removed first, with all its bricks */
#define CAELIX_UNITY_ENTITY_REMOVE 2u    /* remove the entity and all its bricks */
#define CAELIX_UNITY_ENTITY_TRANSFORM 4u /* set the entity transform */

/* One entity record. 48 bytes. The plugin maps guid -> CaelixEntityId. */
typedef struct CaelixUnityEntity
{
    uint8_t guid[16];
    float position[3]; /* world position of the entity origin (ADD, TRANSFORM) */
    float rotation[4]; /* unit quaternion x, y, z, w (ADD, TRANSFORM) */
    uint32_t flags;    /* CAELIX_UNITY_ENTITY_* */
} CaelixUnityEntity;

/* One brick removal. 16 bytes. */
typedef struct CaelixUnityBrickRemove
{
    uint32_t entity_index; /* index into this packet's entity records */
    CaelixBrickKey key;
} CaelixUnityBrickRemove;

/* One upload batch of one entity. 16 bytes. Offsets are bytes from the packet start. */
typedef struct CaelixUnityUpload
{
    uint32_t entity_index;  /* index into this packet's entity records */
    uint32_t brick_count;
    uint32_t keys_offset;   /* brick_count CaelixBrickKey, 4-byte aligned */
    uint32_t blocks_offset; /* brick_count * 512 uint16 block values (raw storage values, voxel
                               index x + 8 y + 64 z), 16-byte aligned */
} CaelixUnityUpload;

/* CaelixUnityFramePacket::flags */
#define CAELIX_UNITY_FRAME_RESET 1u /* the first frame packet of a session: the scene config fields
                                       are read and the scene is created (the release event with
                                       CAELIX_UNITY_RELEASE_SCENE that C# issues right before it
                                       destroyed the previous one; when the plugin still has a
                                       scene here it logs an error, removes every entity and
                                       restores the built-in materials instead), then the material
                                       entries are applied, then the records. */

/* One frame event. 80 bytes, followed in the same memory block by the records it points at;
   total_bytes covers all of it. Offsets are bytes from the packet start. Apply order on the render
   thread: reset (scene creation + material entries), the entity records in list order, the brick
   removes, the uploads, then the flush. */
typedef struct CaelixUnityFramePacket
{
    CaelixUnityPacketHeader header; /* kind CAELIX_UNITY_PACKET_FRAME */
    uint32_t flags;                 /* CAELIX_UNITY_FRAME_* */
    uint32_t total_bytes;           /* the whole block: this struct plus every record */

    /* Session config: read only with CAELIX_UNITY_FRAME_RESET. */
    uint32_t encoding;         /* scene block encoding: 0 packed scene colour, 1 block-ID */
    uint32_t staging_mib;      /* CaelixSceneDesc::staging_bytes in MiB; 0 = the library default */
    uint32_t material_count;   /* CaelixMaterialEntry[material_count] at materials_offset, applied
                                  over the built-in table of the new scene
                                  (caelix_scene_set_material_entries); 0 = the built-in table */
    uint32_t materials_offset; /* 4-byte aligned */

    /* Records. */
    uint32_t entity_count;    /* CaelixUnityEntity[entity_count] at entities_offset */
    uint32_t entities_offset;
    uint32_t remove_count;    /* CaelixUnityBrickRemove[remove_count] at removes_offset */
    uint32_t removes_offset;
    uint32_t upload_count;    /* CaelixUnityUpload[upload_count] at uploads_offset */
    uint32_t uploads_offset;
    uint32_t brick_count;     /* the bricks of every upload record together (C# accounting) */
    uint32_t pad;
} CaelixUnityFramePacket;

/* ---------------------------------------------------------------------------------------------
 * View packet
 * ------------------------------------------------------------------------------------------- */

/* CaelixUnityViewPacket::flags */
#define CAELIX_UNITY_VIEW_REPORT_TIMINGS 1u /* this view's GPU timings go to CaelixUnityStatus */

/* One render event. 504 bytes (488 before the DLSS step grew CaelixRenderOptions). The plugin
   accesses every present target (AccessTexture); a target that AccessTexture refuses, without
   STORAGE usage, or of another size than the colour target is dropped (logged a few times) and the
   frame renders without it. */
typedef struct CaelixUnityViewPacket
{
    CaelixUnityPacketHeader header; /* kind CAELIX_UNITY_PACKET_VIEW */
    uint32_t view;                  /* 0 .. CAELIX_UNITY_MAX_VIEWS - 1 */
    uint32_t flags;                 /* CAELIX_UNITY_VIEW_* */
    uint32_t target_generation;     /* changes whenever C# replaced a target texture of any view
                                       (see the release event); the plugin warns when it changes
                                       without a release */
    uint32_t pad;
    CaelixUnityTarget color;  /* RGBA16F with random write (STORAGE usage); required */
    CaelixUnityTarget depth;  /* R32F: device depth of camera.projection; optional */
    CaelixUnityTarget motion; /* RGBA16F: previous - current pixel, w = delta checkerboard flag; optional */
    CaelixUnityTarget albedo; /* RGBA8 UNORM: albedo, a = coverage; optional */
    /* The sky cubemap of the lighting modes (CaelixSkyCubemap, caelix.h): a Cubemap or a cube
       RenderTexture with mips, SAMPLED usage, no sun disk. NULL = the procedural sky. The plugin
       observes the image (AccessTexture, ObserveOnly) and fills options.sky from it (format, face
       size, mip count, current layout); an image that is not a 6-layer square cube, or whose layout
       is neither SHADER_READ_ONLY_OPTIMAL nor GENERAL, is dropped for the frame (logged a few
       times). native_render_buffer is ignored. */
    CaelixUnityTarget sky;
    CaelixCamera camera;
    CaelixRenderOptions options; /* options.sky is written by the plugin (see `sky`) */
} CaelixUnityViewPacket;

/* ---------------------------------------------------------------------------------------------
 * Release packet
 * ------------------------------------------------------------------------------------------- */

/* CaelixUnityReleasePacket::flags. Several may be combined; SCENE implies TARGET_VIEWS. */
#define CAELIX_UNITY_RELEASE_TARGET_VIEWS 1u /* caelix_renderer_release_target_views on every renderer */
#define CAELIX_UNITY_RELEASE_VIEW 2u         /* destroy the renderer of `view` (nothing when it has none) */
#define CAELIX_UNITY_RELEASE_SCENE 4u        /* destroy every renderer and the scene; forget every entity.
                                                The next frame packet must carry CAELIX_UNITY_FRAME_RESET. */

/* One release event. 32 bytes. */
typedef struct CaelixUnityReleasePacket
{
    CaelixUnityPacketHeader header; /* kind CAELIX_UNITY_PACKET_RELEASE */
    uint32_t flags;                 /* CAELIX_UNITY_RELEASE_* */
    uint32_t view;                  /* CAELIX_UNITY_RELEASE_VIEW: the view whose renderer goes */
} CaelixUnityReleasePacket;

/* ---------------------------------------------------------------------------------------------
 * Status
 * ------------------------------------------------------------------------------------------- */

/* Plugin state for C#. 280 bytes. Counters are monotonic. */
typedef struct CaelixUnityStatus
{
    uint32_t size;                 /* sizeof(CaelixUnityStatus) */
    uint32_t packet_version;       /* CAELIX_UNITY_PACKET_VERSION */
    uint32_t frame_packet_size;    /* sizeof(CaelixUnityFramePacket) as compiled into the plugin */
    uint32_t view_packet_size;     /* sizeof(CaelixUnityViewPacket) */
    uint32_t release_packet_size;  /* sizeof(CaelixUnityReleasePacket) */
    uint32_t unity_renderer;       /* UnityGfxRenderer at the last check (21 = Vulkan) */
    uint32_t intercept_registered; /* 1: InterceptInitialization returned true */
    uint32_t intercept_called;     /* 1: Unity called the init callback (Vulkan initialization) */
    uint32_t instance_patched;     /* 1: vkCreateInstance succeeded with the plugin's api version */
    uint32_t device_patched;       /* 1: vkCreateDevice succeeded with the plugin's extensions and
                                      features */
    uint32_t device_adopted;       /* 1: caelix_device_adopt returned CAELIX_OK */
    int32_t last_error;            /* CaelixResult of the newest failure (0 = none) */
    int32_t frame_event_id;        /* event ids (see the top of this header) */
    int32_t render_event_id;
    int32_t release_event_id;
    uint32_t pad0;

    /* Packets the render thread is done with, per kind. */
    uint64_t last_consumed_frame_sequence;
    uint64_t last_consumed_view_sequence;
    uint64_t last_consumed_release_sequence;

    /* Frames. */
    uint64_t last_unity_frame;  /* Unity's currentFrameNumber at the newest frame or render event */
    uint64_t last_safe_frame;   /* Unity's safeFrameNumber at that event */
    uint64_t last_flushed_frame; /* the newest Unity frame the scene was flushed in */
    uint64_t frames;            /* frame events that flushed */
    uint64_t renders;           /* render events that recorded a render */
    uint64_t releases;          /* release events that ran */
    uint64_t failed_events;     /* events that failed (see the Unity log) */

    /* The scene, as of the newest flush. */
    uint32_t scene_encoding;    /* encoding of the live scene; 0xFFFFFFFF = no scene */
    uint32_t entities;          /* CaelixSceneStats::entities */
    uint32_t groups;            /* CaelixSceneStats::groups */
    uint32_t scene_resets;      /* scenes created by CAELIX_UNITY_FRAME_RESET so far */
    uint32_t live_views;        /* bit v = view v has a renderer */
    uint32_t pad1;
    uint64_t bricks_resident;   /* CaelixSceneStats::bricks_resident */
    uint64_t as_builds_pending; /* CaelixSceneStats::as_builds_pending */
    uint64_t uploaded_total;    /* bricks accepted by caelix_scene_upload_bricks so far */
    uint64_t removed_total;     /* brick removes passed to caelix_scene_remove_bricks so far */
    uint64_t bricks_dropped;    /* bricks the staging ring refused (see "Upload sizing" above) */
    uint64_t upload_bricks_free; /* CaelixSceneStats::upload_bricks_free after the newest flush */
    uint64_t upload_bricks_cap;  /* a quarter of the empty ring's upload_bricks_free: the most
                                    bricks one frame packet should carry */
    uint64_t record_bytes;      /* CaelixSceneStats::record_bytes */
    uint64_t blas_bytes;        /* CaelixSceneStats::blas_bytes */
    uint64_t staging_bytes;     /* CaelixSceneStats::staging_bytes (the ring's capacity) */

    /* GPU timings of the view that set CAELIX_UNITY_VIEW_REPORT_TIMINGS (CaelixGpuTimings of the
       newest frame read back). */
    float flush_ms;
    float trace_ms;
    float composite_ms;
    float denoise_ms;
    uint64_t timings_frame; /* CaelixGpuTimings::frame_number */
    uint32_t timings_view;  /* the view the timings belong to */
    uint32_t pad2;
} CaelixUnityStatus;

/* The UnityRenderingEventAndData callback for CommandBuffer.IssuePluginEventAndData. Never NULL. */
CAELIX_UNITY_API void* caelix_unity_get_render_event_func(void);
/* Fills *out (NULL is ignored). Any thread. */
CAELIX_UNITY_API void caelix_unity_status(CaelixUnityStatus* out);

#ifdef __cplusplus
}
#endif
#endif
