# Caelix-Renderer

Native Vulkan renderer for the Caelix voxel engine. The core is a C++20 shared
library (`caelix_renderer`) that exposes a small C ABI (`include/caelix/caelix.h`).
A Rust standalone host (`host-rs/`) links the library through bindgen-generated
bindings. A Unity native plugin that uses the same C ABI comes later.

## Architecture

The host owns the Vulkan instance, the device and the swapchain. The library
never creates a device: it publishes what it needs
(`caelix_get_device_requirements`: Vulkan 1.3, extensions, feature bits), the
host creates a device that meets those requirements, and the library adopts it
(`caelix_device_adopt`). The library loads Vulkan through the host's
`vkGetInstanceProcAddr` (volk), so it sees the same layers as the host. Each
frame, the library records its work into a command buffer that the host
provides; the host submits and presents. Vulkan handles cross the C ABI as
`uint64_t`, so the header needs no Vulkan headers.

In `host-rs/`:

- `caelix-sys`: raw bindings to the C ABI.
- `caelix-vk`: instance, device, swapchain and frame loop. It is the only Rust
  crate with Vulkan code (`ash`); its public API exposes raw `u64` handles only.
- `caelix-app`: the window (winit), the CLI and the glue between the two.

### ABI version 2: the scene

`CAELIX_ABI_VERSION` is 2. Version 2 adds the scene API (`caelix_scene_*`):
entities, raw brick upload and removal, a per-frame flush, stats and a brick
query for inspection. There are no kernels, acceleration structures or
rendering yet. All calls on one scene (or on a scene and its device) are
externally synchronized by the host.

Scene model:

- The host owns the raw voxel data. The library stores only GPU records plus a
  CPU index from brick keys to records. A brick is 8x8x8 blocks; its key is the
  entity-local block coordinate shifted right by 3.
- Record layout (1096 bytes for 16-bit blocks, mirrors Unity's
  `CaelixBrickTrace.hlsl`): 2 info words, 16 micro-occupancy words, then the raw
  blocks two per word (even x in the high 16 bits). Lossless rule: the payload
  always holds the raw uploaded values; later kernels only add derived bits in
  the header words.
- Record pool: device-local pages of `records_per_page` records. Pages are only
  added, and a record never moves while its brick lives.
- Render groups of `(1 << group_shift)^3` bricks (default 16^3). Each group has
  dense 16-bit brick ids and its own range in one shared slot table buffer
  (16-byte rows: position and state, summary, record index, last used). The
  slot table buffer grows by doubling; offsets never change.
- Staging ring: one host-visible buffer. `caelix_scene_upload_bricks` writes
  records into the ring and reports how many bricks it accepted in `consumed`.
  When `consumed < count`, flush and submit the frame, then upload the rest. The
  last 1 MiB of the ring is reserved for slot table rows.

Frame contract: call `caelix_scene_flush` once per frame with a command buffer
in the recording state, a strictly increasing `frame_number` (1 or higher), and
`safe_frame_number` = the newest frame that has completed on the GPU (0 = none).
The flush records the staging-to-pool copies between two barriers. Staging
space and replaced GPU buffers of a frame are released only when that frame is
safe. `caelix_scene_recycle(safe)` releases them without a flush, for a host
that waits for a frame and then uploads before the next flush. Destroy every
scene before `caelix_device_destroy`, with the device idle.

### Layout contract

A render call receives the target image in the layout that
`CaelixTargetImage::layout` names, and leaves it in `VK_IMAGE_LAYOUT_GENERAL`.
The host transitions it from `GENERAL` for present (or for its own use).

## Build

Requirements: Visual Studio 2022 (MSVC and its bundled LLVM/libclang),
CMake 3.28 or newer, the Vulkan SDK (`VULKAN_SDK` set), and a Rust MSVC toolchain.

```
cmake --preset vs2022
cmake --build --preset release
cd host-rs && cargo run -p caelix-app
```

The first CMake configure fetches the pinned dependencies (volk,
VulkanMemoryAllocator, unordered_dense, and for the tests Catch2 and
vk-bootstrap) into `build/_deps/`.

## Tests

The CMake option `CAELIX_BUILD_TESTS` (default `ON`) builds the tests and
fetches their test-only dependencies (Catch2, vk-bootstrap). The library never
links them. Configure with `-DCAELIX_BUILD_TESTS=OFF` to skip them.

- `caelix_scene_smoke`: headless. It creates a Vulkan 1.3 device with
  vk-bootstrap (validation layer and debug messenger when installed), adopts it,
  and uses the public C ABI to upload, replace and remove bricks, flush across
  several frames, grow the slot table, and read records and slot rows back from
  the GPU. Any validation error or warning fails the test. Needs a GPU with the
  library's device requirements (ray query).
- `caelix_abi_c99`: compiles `caelix.h` as C99 with `/W4` (a compile check, not
  run).

```
cmake --build --preset release
ctest --test-dir build -C Release --output-on-failure
ctest --test-dir build -C Debug --output-on-failure   # after cmake --build --preset debug
```

## Running the host

`cargo run -p caelix-app -- [flags]`

- `--frames <N>`: exit with code 0 after N presented frames. Default: run until
  the window closes.
- `--verbose`: log level `debug` (includes the Vulkan loader and validation
  layer messages). Default level: `info`. `RUST_LOG` overrides both.
- `--resize-at <N>` (hidden test aid): after N presented frames, request a
  960x540 window size once, so the swapchain recreation path runs without user
  input.
- `--world <PATH>`: a `.cxw` world to stream into the scene. Without it the
  scene stays empty.
- `--staging-mib <N>`: size of the scene's staging ring in MiB
  (`CaelixSceneDesc::staging_bytes`). Default 64, minimum 4.
- `--group-shift <N>`: render group edge in bricks = `1 << N`. Default 4
  (16^3). The library implements 1..5.

The host creates one `CaelixScene` after it adopts the device. With `--world`
it adds one scene entity per file entity, then streams the bricks:

- One background thread runs `CxwFile::decode_all` (parallel on the rayon pool)
  and sends each decoded sector through a bounded channel of 32 sectors. When
  the channel is full, the decoder blocks, so it cannot run far ahead of the
  uploads.
- Every frame, after `begin_frame`: `caelix_scene_recycle(safe)`, then upload
  sectors from the channel with `caelix_scene_upload_bricks` until the channel
  is empty or `consumed < count` (the staging ring is full; the rest of that
  sector waits for the next frame), then exactly one `caelix_scene_flush` into
  the frame's command buffer. Frame numbers start at 1; `safe_frame_number` is
  the frame that last used the frame slot whose fence `begin_frame` waited for.
- The window keeps showing the temporary clear colour. When the decoder has
  finished and every brick is uploaded, the host logs one summary (entities,
  groups, resident bricks, record pages, GPU memory, decode and upload time,
  upload frames, bricks/s, the full `CaelixSceneStats`) and queries the first
  key of the first 3 decoded sectors with `caelix_scene_query_brick`. When
  `bricks_resident` differs from the number of decoded bricks, it logs an error
  and exits with a non-zero code.
- `--frames N` exits after N frames even when the load is not complete (with a
  warning). At exit the host drops the channel receiver and joins the decoder
  thread. `decode_all` cannot be cancelled: it decodes the remaining sectors and
  drops them.

Expected numbers for `VoxelTestScenes/epic-citadel.8192.packed.cxw` (release,
RTX 5080, 64 MiB staging): 3837 sectors, 1 entity, 3837 groups, 1,095,051
resident bricks in 5 record pages (1370 MiB of record pages, 32 MiB of slot
tables). The upload takes about 0.3 s over 37 frames, 19 of which upload (the
ring holds one frame's records until that frame is safe), about 3.5 M bricks/s.
That is the decoder's own speed: decode and upload overlap completely. Peak
working set of the process is about 0.6 GB.

Debug builds enable `VK_LAYER_KHRONOS_validation` when it is installed; its
messages are logged with target `vulkan`. Library messages use target `caelix`.
Release builds run without validation.

Implicit layers that other software installs can add loader warnings that are
not from this project. Example: the Overwolf layers report an old API version;
set `DISABLE_VULKAN_OW_OVERLAY_LAYER=1` and `DISABLE_VULKAN_OW_OBS_CAPTURE=1`
to disable them for a run.

## Environment variables

- `CAELIX_LIB_DIR`: directory that contains `caelix_renderer.lib` and
  `caelix_renderer.dll`. Default: `<repo>/build/bin/Release`. The Rust host
  links the Release build only.
- `LIBCLANG_PATH`: directory that contains `libclang.dll`, used by bindgen.
  `host-rs/.cargo/config.toml` sets it to the 64-bit Visual Studio 2022
  Community LLVM directory (`VC/Tools/Llvm/x64/bin`); set your own value if your
  installation is different. Do not use `VC/Tools/Llvm/bin`: its
  `libclang.dll` is 32-bit.
