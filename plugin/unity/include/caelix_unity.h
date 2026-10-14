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
 * status.render_event_id, packet). The render thread COPIES the packet at the start of the
 * event and then publishes its sequence number as CaelixUnityStatus::last_consumed_sequence. C#
 * may reuse or free a packet when last_consumed_sequence >= the packet's sequence. The GPU never
 * reads packet memory.
 *
 * Target views: the renderer caches one image view per target image (CaelixRenderTargets). When
 * C# destroys a target RenderTexture, a later RenderTexture can get the same VkImage handle, and
 * the cached view would then show a destroyed image. So whenever C# replaces a target, it issues
 * the release event (status.release_event_id, data NULL) BEFORE the first render event that uses
 * the new target. That event runs with graphics queue access: it waits for the graphics queue to
 * be idle and calls caelix_renderer_release_target_views.
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

/* Bumped on every change of CaelixUnityPacket or CaelixUnityStatus. */
#define CAELIX_UNITY_PACKET_VERSION 1u

/* A render target, given as Unity's native texture pointer. */
typedef struct CaelixUnityTarget {
  void* native_texture;        /* Texture.GetNativeTexturePtr(); NULL = the target is absent */
  void* native_render_buffer;  /* RenderTexture.colorBuffer.GetNativeRenderBufferPtr(); the
                                  fallback when AccessTexture does not accept native_texture.
                                  May be NULL. */
  uint32_t width;              /* must match the texture's size */
  uint32_t height;
} CaelixUnityTarget;

/* One render event. All fields are written by C# before the event is issued. 320 bytes. */
typedef struct CaelixUnityPacket {
  uint32_t size;          /* sizeof(CaelixUnityPacket); the plugin rejects any other value */
  uint32_t version;       /* CAELIX_UNITY_PACKET_VERSION */
  uint64_t sequence;      /* > 0, strictly increasing per issued packet */
  uint32_t encoding;      /* scene block encoding: 0 packed scene colour, 1 block-ID. Read at the
                             first render event (the scene is created then). */
  uint32_t target_generation; /* changes whenever C# replaced a target texture (see the release
                                 event); the plugin warns when it changes without a release */
  CaelixUnityTarget color;    /* RGBA16F with random write (STORAGE usage) */
  CaelixCamera camera;
  CaelixRenderOptions options;
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
  uint64_t last_consumed_sequence; /* sequence of the newest packet the render thread copied */
  uint64_t last_unity_frame;      /* Unity's currentFrameNumber at the newest render event */
  uint64_t last_safe_frame;       /* Unity's safeFrameNumber at the newest render event */
  uint64_t last_plugin_frame;     /* the newest frame number the plugin passed to the library */
  uint64_t renders;               /* render events that recorded a render */
  uint64_t failed_events;         /* render events that failed (see the Unity log) */
  uint64_t view_releases;         /* release events that ran */
} CaelixUnityStatus;

/* The UnityRenderingEventAndData callback for CommandBuffer.IssuePluginEventAndData. Never NULL. */
CAELIX_UNITY_API void* caelix_unity_get_render_event_func(void);
/* Fills *out (NULL is ignored). Any thread. */
CAELIX_UNITY_API void caelix_unity_status(CaelixUnityStatus* out);

#ifdef __cplusplus
}
#endif
#endif
