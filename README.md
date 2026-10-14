# Caelix-Renderer

Native Vulkan renderer for the Caelix voxel engine. The core is a C++20 shared
library (`caelix_renderer`) that exposes a small C ABI (`include/caelix/caelix.h`).
A Rust standalone host (`host-rs/`) links the library through bindgen-generated
bindings. A Unity native plugin that uses the same C ABI comes later.

The library contains NVIDIA Real-Time Denoisers (NRD, compiled in). Its license
obligations (credits line, NVIDIA marks, distribution) are summarized in
[`third_party/NOTICES.md`](third_party/NOTICES.md); read it before you ship a build.

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
query for inspection. The flush runs the finalize kernel and builds the
acceleration structures (see below). The renderer API (`caelix_renderer_*`,
see "Renderer") traces the scene into the host's image. All calls on one scene
(or on a scene, its renderer and its device) are externally synchronized by the
host.

Version 2 is still in development: the finalize step appended fields to
`CaelixSceneDesc` (`encoding`, default 0 = packed), `CaelixSceneStats`
(`finalized_bricks_total`, `reverse_table_bytes`, `aabb_bytes`, `grid_bytes`)
and `CaelixBrickInfo` (`aabb_buffer`, `aabb_row_offset_bytes`) without a
version bump. The acceleration structure step appended
`CaelixSceneDesc::scratch_bytes` (default 0 = 128 MiB), six `CaelixSceneStats`
fields (`blas_count` .. `as_builds_pending`), four `CaelixBrickInfo` fields
(`blas_address`, `blas_handle`, `entity_info_buffer`,
`entity_info_offset_bytes`) and the function `caelix_scene_get_tlas`, also
without a bump. The renderer step removed the temporary `caelix_clear_target`,
added the renderer API and the feature bit
`CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT` (bit 9,
`shaderStorageImageWriteWithoutFormat`), also without a bump. The lighting
step (6a) appended the path-tracing fields to `CaelixRenderOptions` (`mode`
.. `frame_seed`), `CAELIX_RENDER_FLAG_SKY_ON_MISS`, `CaelixRenderMode`,
`CAELIX_MAX_BOUNCES` and `CaelixGpuTimings::composite_ms`, also without a
bump. The NRD step (6b) added `CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT`
(bit 10), the debug views `CAELIX_VIEW_NRD_VALIDATION` / `CAELIX_VIEW_NRD_DIFF`,
`CaelixNrdInfo` and `caelix_renderer_get_nrd_info`, renamed
`CaelixGpuTimings::pad` to `denoise_ms` (same layout) and made
`CaelixRenderOptions::denoise` active, also without a bump. Rebuild hosts
against the current header.

Scene model:

- The host owns the raw voxel data. The library stores only GPU records plus a
  CPU index from brick keys to records. A brick is 8x8x8 blocks; its key is the
  entity-local block coordinate shifted right by 3.
- Record layout (1096 bytes for 16-bit blocks, mirrors Unity's
  `CaelixBrickTrace.hlsl`): 2 info words, 16 micro-occupancy words, then the
  blocks two per word (even x in the high 16 bits). The upload writes the raw
  blocks and the group-local position; the finalize kernel fills the rest of
  the header and adds face bits to transparent blocks (see "Finalize kernel").
- Record pool: device-local pages of `records_per_page` records. Pages are only
  added, and a record never moves while its brick lives.
- Render groups of `(1 << group_shift)^3` bricks (default 16^3). Each group has
  dense 16-bit brick ids and its own range in one shared slot table buffer
  (16-byte rows: position and state, summary, record index, last used). The
  slot table buffer grows by doubling; offsets never change.
- Staging ring: one host-visible buffer. `caelix_scene_upload_bricks` writes
  records into the ring and reports how many bricks it accepted in `consumed`.
  When `consumed < count`, flush and submit the frame, then upload the rest. The
  last `capacity / 16 + 1 MiB` of the ring is reserved for table rows (slot
  rows, reverse table words, info rows, grid cells, the page table and the
  finalize list); records cannot use it.

### Finalize kernel

`shaders/caelix_finalize.hlsl` turns raw records into renderer records. Each
flush dispatches it over every brick uploaded since the previous flush and
every resident face neighbour (same entity, across groups too) of an uploaded
or removed brick: one workgroup of 64 threads per brick, one thread per row of
8 voxels. A brick is finalized at the flush after its upload; the host must not
read records or AABB rows before that flush has completed on the GPU.

Per voxel: `alive` = not all six neighbours opaque (an absent brick is air).
Opaque blocks keep their value. A transparent block gets face bit `d` (order
+X, -X, +Y, -Y, +Z, -Z) when its neighbour in direction `d` is not opaque and
has a different id; air (id 0) next to glass gets face bits too. A voxel is
occupied when it is alive and its value is not 0. The kernel writes the micro
occupancy words, the coarse byte (word 0, bits 16..23), the tight bounds (word
1) and the brick's AABB row (entity-local block coordinates; six NaNs when no
voxel is occupied, which makes it an inactive primitive).

Block encodings (`CaelixSceneDesc::encoding`): 0 = packed scene colour (the
default; opaque when `value & 0xC000`, id `value & 0xFF`, faces at bit 8),
1 = block-ID (opaque when `value & 0x8000`, id `value & 0x1FF`, faces at bit 9).

Lossless rule: `raw(v) = opaque(v) ? v : (v & id_mask)` recovers the uploaded
value from a finalized one, and the kernel always works on `raw()` of its own
and its neighbours' blocks. So finalizing a brick again gives the same record,
a neighbour may be read while another workgroup rewrites it, and dead voxels
keep their value. Uploaded transparent values must have no bits above the id.

Lookup tables (all in device-local buffers, read through buffer device
addresses; the kernel has push constants only, no descriptor sets):

- Page table: `uint64_t` device address per record page.
- Group info (32 B per group: coordinate, entity, slot row, reverse table and
  AABB row bases, high water) and entity info (32 B per entity: group grid
  minimum, dimensions and base).
- Group grid per entity: a dense array of group indices over the entity's
  group bounding box. Dimensions are powers of two and grow geometrically; the
  grid is rebuilt and fully restaged when it grows. Grids above 2^24 cells are
  refused (the upload fails with `CAELIX_ERROR_OUT_OF_MEMORY`).
- Reverse table per group: one 16-bit brick id per group-local position.
- AABB rows per group: 24 B per dense id, same capacity as the slot rows. When
  the range grows, the rows are copied on the GPU and the new part is filled
  with NaN; a removed brick's row is set to NaN.

All pools grow by doubling with a copy at the next flush, like the slot table.
Small table updates are staged as runs that also cover short clean gaps,
because every `vkCmdCopyBuffer` region has a fixed GPU cost.

### Acceleration structures

Each flush builds the acceleration structures after the finalize kernel
(`Scene::PlanAccelerationStructures` and `RecordAccelerationStructureBuilds` in
`lib/src/scene.cpp`). The renderer binds the TLAS (see "Renderer").

- BLAS per render group: procedural AABB geometry
  (`VK_GEOMETRY_TYPE_AABBS_KHR`, `VK_GEOMETRY_OPAQUE_BIT_KHR`) over the
  group's AABB rows, stride 24, `primitiveCount` = the group's high water. Free
  ids and bricks without occupied voxels have NaN rows, which are inactive
  primitives. Build flag `PREFER_FAST_TRACE` only, mode BUILD; refit
  (`ALLOW_UPDATE`) and compaction come later.
- TLAS per scene: one instance per live group that has a BLAS.
  `instanceCustomIndex` = the scene-wide group index (24 bits, so a scene
  holds at most 2^24 groups), mask 0xFF, SBT record offset 0,
  `FORCE_OPAQUE`, transform = the entity transform (unit quaternion and
  position, scale 1) as a 3x4 row-major matrix: the AABBs are in
  entity-local block coordinates. The instance array is rebuilt on the CPU,
  staged through the ring and copied to a device-local buffer.
- Storage (`lib/src/as_pool.h`, `BlockAllocator`): an acceleration structure
  must never move (the TLAS refers to BLASes by device address), so it does
  not use `RowPool`. Storage comes in 64 MiB device-local chunks
  (`ACCELERATION_STRUCTURE_STORAGE | SHADER_DEVICE_ADDRESS`) that are only
  added. Blocks come from power-of-two size classes from 4 KiB, with one free
  list per class per chunk and a bump pointer per chunk; a block larger than a
  chunk gets a chunk of its own. A structure is rebuilt in place when its
  block is large enough (safe: one queue, and the flush starts with a barrier
  from all earlier commands). Otherwise it gets a new block and a new handle;
  the old block and handle are released when the safe frame reaches the
  flush, because a TLAS of a frame in flight may still refer to them. The
  BLAS of a freed group is released after the TLAS build that drops it.
- Scratch budget: one buffer of `CaelixSceneDesc::scratch_bytes` (default
  128 MiB), bump-allocated per flush with
  `minAccelerationStructureScratchOffsetAlignment` (128 on an RTX 5080). The
  TLAS takes its scratch first (sized for an instance per live group), then
  BLASes in first-come order until one does not fit; the rest wait for the
  next flush. The buffer grows when the TLAS and the largest single BLAS do
  not fit together. Builds are recorded in calls of up to 256 infos.
- Rebuild policy: a group's BLAS is out of date when one of its bricks is
  queued for finalization (its own upload, or the upload or removal of a
  face neighbour) or removed. It is built in the first flush in which its AABB
  rows are final: its table rows were staged and none of its finalize entries
  wait for a later flush. The TLAS is out of date when a BLAS was built, a
  group with a BLAS was freed, or an entity was moved or removed. Every BLAS
  build invalidates the TLAS, so BLASes are built only in a flush that also
  builds the TLAS: when the ring has no room for the instances, all builds
  wait.
- Barriers inside the flush: the first barrier now goes from all earlier
  commands to all commands (it also covers in-place rebuilds and the reused
  scratch); after the copies and the kernel, transfer and compute writes ->
  all commands; after the BLAS builds, `AS_BUILD` write -> `AS_BUILD` read;
  after the TLAS build, `AS_BUILD` write -> all commands, memory read.
- Entity transforms on the GPU: the entity info row is 128 bytes (grid fields,
  then the current and the previous object-to-world rows as `float4` x 3
  each). `prev` differs from `cur` for exactly one flush after
  `caelix_scene_set_entity_transform`; the trace kernels read both for motion
  vectors.

What the host can query: `CaelixSceneStats` (`blas_count`, `blas_bytes` and
`tlas_bytes` including blocks that wait for a safe frame, `scratch_bytes`,
`as_builds_total`, `as_builds_pending`), the BLAS of a brick's group and the
entity info row through `caelix_scene_query_brick`, and the TLAS handle and
address through `caelix_scene_get_tlas` (owned by the library; it can change
at any flush). A brick is traceable in the commands recorded after the flush
that finalizes it when the scratch and ring budgets allowed its group's build;
`as_builds_pending == 0` tells the host that every structure matches the
uploaded bricks.

### Shaders

HLSL in `shaders/` is compiled at build time by DXC (`$VULKAN_SDK/Bin/dxc.exe`,
`-T cs_6_6 -spirv -fspv-target-env=vulkan1.3 -HV 2021 -O3 -WX`) to
`build/shaders/<name>.spv`, then embedded into the library as `uint32_t` arrays
(`cmake/Shaders.cmake`, `cmake/EmbedShaders.cmake`). DXC writes a depfile, so
an edit of an included header rebuilds the kernels that include it. The SPIR-V
does not depend on the C++ configuration. `shaders/caelix_platform.hlsl` is the
only file with DXC-specific syntax (buffer device address loads and stores,
push constants, descriptor bindings, attributes); a Slang port replaces this
header. Storage images that are written without a format qualifier need
`CAELIX_FORMAT_UNKNOWN` (`[[vk::image_format("unknown")]]`): without it DXC 1.8
infers the format from the element type (`RWTexture2D<float4>` becomes
`Rgba32f`), and it has no command-line switch to stop that.

At scene creation (and renderer creation) the library loads each kernel by name. When the environment
variable `CAELIX_SHADER_DIR` names a directory that contains `<name>.spv`, that
file is used instead of the embedded array (logged at info level as
`shader <name> loaded from <path>`), so a shader can be changed without
rebuilding the library.

Frame contract: call `caelix_scene_flush` once per frame with a command buffer
in the recording state, a strictly increasing `frame_number` (1 or higher), and
`safe_frame_number` = the newest frame that has completed on the GPU (0 = none).
The flush records the staging-to-pool copies, the finalize dispatch and the
acceleration structure builds between barriers. Staging space, replaced GPU
buffers and replaced acceleration structures of a frame are released only
when that frame is safe. `caelix_scene_recycle(safe)` releases them without a flush, for a host
that waits for a frame and then uploads before the next flush. Destroy every
scene before `caelix_device_destroy`, with the device idle.

When a flush records work, it brackets that work with two timestamp queries
(`lib/src/gpu_timer.h`); the time is reported as `flush_ms` by
`caelix_renderer_get_gpu_timings` (0 for a flush that recorded nothing).

### Renderer

`caelix_renderer_create(device, scene)` creates a renderer for one scene
(`lib/src/renderer.cpp`). `caelix_renderer_render(renderer, frame, targets,
camera, options)` records the frame's rendering. Call it after
`caelix_scene_flush` of the same frame, with the same `CaelixFrameDesc`: the
flush builds the TLAS that the render binds.

- Modes (`CaelixRenderOptions::mode`): `CAELIX_MODE_PRIMARY` (debug views of
  the primary hit, one dispatch), `CAELIX_MODE_PATH_TRACE` (path-traced
  lighting, raw 1 sample per pixel) and `CAELIX_MODE_REFERENCE` (the same,
  accumulated over frames). See "Lighting" below.
- Trace kernel `shaders/caelix_path_trace.hlsl`, `[numthreads(8, 8, 1)]`:
  one primary ray per pixel, `RayQuery<RAY_FLAG_FORCE_OPAQUE>` through the
  TLAS (`TraceScene`, also used by the bounce rays). For every procedural
  candidate it reads the group info (`CandidateInstanceID()` = the group
  index) and the slot row of the dense brick id (`CandidatePrimitiveIndex()`),
  skips bricks that are not resident, and runs the brick DDA on the record. A
  hit inside `[TMin, CommittedRayT())` is committed. Then it writes every
  present target (see "Render targets" below). A miss writes the sky colour
  and the miss values. In the lighting modes it also shades the hit and
  traces the bounces into the internal targets.
- Composite kernel `shaders/caelix_composite.hlsl`, `[numthreads(8, 8, 1)]`,
  lighting modes only: internal targets -> the host's colour target.
- NRD (`lib/src/nrd_executor.*`, path trace mode with `denoise = 1`): NRD's
  own compute dispatches between the trace and the composite (see "Denoising
  (step 6b)" below).
- Shading helpers `shaders/caelix_shading.hlsl`: block colours, the material
  decode, the random stream, the opaque shading rule and the procedural sky.
- Brick DDA `shaders/caelix_brick_trace.hlsl`: a copy of Unity's
  `CaelixBrickTrace.hlsl` (2-level occupancy DDA on the 8^3 brick, tight
  bounds from record word 1, transparency by face bits). Every change is
  marked `// NATIVE:`: no Unity includes (`caelix_block.hlsl` decodes the
  block by the scene's encoding at run time); the record loads read through
  the record's device address (`g_recordBase`, set per candidate, `brickBase`
  0); `CaelixTraceBrickPrimitiveCore` takes the brick origin in entity-local
  block coordinates as a parameter instead of deriving it from word 0 and the
  render group macros; `half` = float, `UNITY_BRANCH` -> `CAELIX_BRANCH`.
- Inputs: one descriptor set (binding 0 the TLAS, bindings 1..6 the colour,
  albedo, normal, depth, motion and view_z targets as storage images without a
  format qualifier, bindings 7..13 the internal targets of the lighting modes
  with fixed formats) and one push constant, the device address of a 512-byte
  `CaelixFrameConstants` block (`shaders/caelix_frame.hlsl`,
  `lib/src/frame_constants.h`): camera, target size, the addresses of the page
  table, slot tables, group infos and entity infos, the record layout, group
  shift, encoding, view, flags, sun and sky, the rows of the host projection,
  the previous render call's camera, the present-target mask, the miss values,
  `has_prev_camera` / `has_tlas`, and the lighting parameters (mode, bounces,
  seed, accumulation count, exposure, sun and sky, sRGB output, `t_max`). The
  composite kernel has its own set (binding 0 the colour target, bindings 1..6
  internal targets) and the same push constant. Every binding is `PARTIALLY_BOUND`
  (`VkDescriptorSetLayoutBindingFlagsCreateInfo`): a render call writes only the
  TLAS (when the scene has one) and the present targets, and the kernel touches
  a binding only when its bit is set. Absent targets and a scene without a TLAS
  therefore need no dummy descriptors.
- Per-frame resources: a ring of slots (two descriptor sets + host-visible
  constants buffer), 3 at first, grown up to 8 when a slot's frame is not safe
  yet.
- Image views: one per target image, cached (16; the least recently used view
  is destroyed once its frame is safe). `caelix_renderer_release_target_views`
  destroys all of them; call it before the host destroys target images (a new
  image can reuse the handle of a destroyed one).
- Before the scene's first TLAS build, the kernel still runs but does not
  start a ray query (`has_tlas` = 0): every pixel gets the sky colour and the
  miss values. The targets need `STORAGE` usage only.
- Debug views (`CaelixRenderOptions::debug_view`): albedo (optionally Lambert
  shaded with `sun_dir`), world normal, depth (`near_z`..`far_z`), hash colour
  of the group, hash colour of the brick, and a heat map of the procedural
  candidates per ray (0..64). Albedo in the packed encoding: opaque rgb5,
  emissive rgb4, glass a fixed tint (0.6, 0.8, 1.0), air magenta (a primary
  ray should never report an air boundary). In the block-ID encoding the
  colour is a pastel hash of the id.
- GPU timings: two timestamps around each dispatch, read back in
  a later render call whose safe frame has reached that frame;
  `caelix_renderer_get_gpu_timings` returns `trace_ms`, `denoise_ms` (0 for
  a frame without NRD), `composite_ms` (0 in the primary mode), `flush_ms` and
  the frame they belong to.

Pixel mapping: pixel (px, py), row 0 = the top row,
`ndc = ((px + 0.5) / w * 2 - 1, 1 - (py + 0.5) / h * 2)`, direction
`normalize(right * ndc.x * tan_half_fov_y * aspect + up * ndc.y * tan_half_fov_y + forward)`.
The camera basis is Unity's (+X right, +Y up, +Z forward); the host's
`camera.rs` builds it from Unity's `eulerAngles` (see "Running the host").

### Lighting (step 6a)

The lighting modes port the opaque part of Unity's path tracer
(`CaelixPathTraceCommon.hlsl`: `CaelixShadeVoxelHit` and the bounce loop).
`CAELIX_MODE_PATH_TRACE` shows the 1 spp estimate (denoised by NRD with
`denoise = 1`, see "Denoising (step 6b)") and `CAELIX_MODE_REFERENCE` the running
mean of the raw estimate.

- Primary hit: the material (packed: opaque rgb5 / 31, emissive rgb4 / 15 with
  `emission = albedo * 2^(3 * strength)`; block-ID: the pastel hash colour, no
  emission) is shaded with Unity's rule: specular chance
  `lerp(metallic, 1, fresnel * smoothness)` (0 for packed materials, so they
  are diffuse only), cosine-weighted diffuse direction `normalize(N + unit)`,
  `k = 1`, origin pushed off 0.002 along N (Unity's `K_RAY_ORIGIN_PUSH_OFF`).
  Its emission goes to the deterministic target.
- Bounces: up to `bounce_count` (0..14, Unity default 4) segments of length
  1e4 (`K_T_MAX`). A miss adds `SkyRadiance(dir) * throughput` and stops; a hit
  adds `emission * throughput`, multiplies the throughput by the albedo, and
  Russian roulette on the largest throughput channel may stop the path. The
  incident radiance goes to the diffuse or the specular target by the lobe of
  the primary hit, packed for NRD REBLUR
  (`REBLUR_FrontEnd_PackRadianceAndNormHitDist`: YCoCg radiance, alpha = the
  hit distance of the first segment normalized by
  `REBLUR_FrontEnd_GetNormHitDist` with A, B, C = 3, 0.1, 20 at the primary
  view z, roughness 1 for the diffuse lobe; 1e4 before normalization when the
  segment reaches the sky, 0 when there is no segment). The lobe that was not
  sampled gets 0 radiance and 0 hit distance. NaN or Inf drops the sample;
  values are clamped to 65504 (FP16).
- Glass (packed transparent blocks) is shaded as an opaque diffuse surface with
  the tint (0.6, 0.8, 1.0) until step 6d ports the delta chain.
- Random numbers: `PcgHash(px ^ (py << 16) ^ (frame_seed * 0x9E3779B9))`,
  then xorshift32 (Unity samples a blue-noise texture). The host must change
  `frame_seed` every frame.
- Sky: above the horizon `lerp(horizon, zenith, sqrt(dir.y))`, below it
  `lerp(horizon, ground, sqrt(-dir.y))`, plus `sun_radiance` inside the sun
  disk (`sun_angular_radius` r, default 0.5 degrees). The disk subtends
  `omega = 2 pi (1 - cos r)` (2.39e-4 sr at 0.5 degrees), so a surface facing
  the sun receives the irradiance `sun_radiance * omega`.
- Sun next-event estimation (`CAELIX_RENDER_FLAG_SUN_NEE`, the host's
  default; skipped when `sun_radiance` is 0): every vertex that sampled the
  diffuse lobe and has a next segment (the primary hit when
  `bounce_count >= 1`, a bounce hit while segments remain) samples a direction
  `L` uniform in the sun cone (`cos theta = 1 - u1 (1 - cos r)`,
  `phi = 2 pi u2`) and traces a shadow ray (`TraceShadow`: the first DDA hit in
  range ends the ray query). Unoccluded, it adds
  `sun_radiance * omega * dot(N, L) / pi * throughput` (demodulated, like the
  bounces; before Russian roulette). The segment that leaves such a vertex sees
  the sky WITHOUT the disk, a segment that leaves a specular-lobe vertex (no
  NEE there) sees it, so the sun is counted once per path. A primary miss
  always shows the disk. NEE draws from its own random stream
  (`InitRandom(pixel, frame_seed ^ 0x68E31DA4)`), so the BSDF path consumes the
  same random numbers with NEE on or off. Without the flag the sun is found
  only by chance (6a behaviour, as in Unity): a small disk then needs a very
  high radiance, which does not converge.
- Composite: diffuse and specular are unpacked with
  `REBLUR_BackEnd_UnpackRadianceAndNormHitDist` (NRD's outputs when the frame
  was denoised, else the raw packed inputs); `raw = deterministic + albedo *
  (diffuse + specular)`; reference
  mode `accum = n == 0 ? raw : lerp(accum, raw, 1 / (n + 1))`; then `* exposure`;
  a miss is `SkyRadiance(primary direction) * exposure` with
  `CAELIX_RENDER_FLAG_SKY_ON_MISS`, else 0; alpha = coverage; linear -> sRGB
  for UNORM colour formats.
- Accumulation count `n`: 0 after `reset_accumulation`, a mode change, a size
  change (the internal targets are recreated) and on the first render; +1 per
  reference render with a colour target.
- Internal targets (owned by the renderer, `STORAGE | SAMPLED`, layout
  `GENERAL` after one transition, recreated when the host target size changes;
  the old ones are destroyed once their last frame is safe):

| Target | Format | Content |
|---|---|---|
| `view_z` | R32F | linear view depth; 1e30 on a miss (NRD `IN_VIEWZ`) |
| `normal_roughness` | RGB10A2 | `NRD_FrontEnd_PackNormalAndRoughness(N, 1 - smoothness, 0)` (`IN_NORMAL_ROUGHNESS`) |
| `motion` | RGBA16F | motion in pixels (xy), previous view z - view z (z) (`IN_MV`) |
| `albedo` | RGBA8 | material albedo; a = coverage |
| `deterministic` | RGBA16F | primary emission |
| `diff_in`, `spec_in` | RGBA16F | REBLUR-packed incident radiance and normalized hit distance (`IN_DIFF/SPEC_RADIANCE_HITDIST`) |
| `accum` | RGBA32F | reference accumulation |
| `diff_out`, `spec_out` | RGBA16F | NRD's outputs (`OUT_DIFF/SPEC_RADIANCE_HITDIST`) |
| `validation` | RGBA8 | NRD's validation layer (`OUT_VALIDATION`) |

### Denoising (step 6b)

`CaelixRenderOptions::denoise = 1` in `CAELIX_MODE_PATH_TRACE` (with a colour
target) runs NVIDIA NRD v4.18.0, denoiser `REBLUR_DIFFUSE_SPECULAR`, between the
trace and the composite. NRD is a static library with embedded SPIR-V; the
renderer drives it without NRI through `NrdExecutor` (`lib/src/nrd_executor.*`):

- From `nrd::GetInstanceDesc` / `GetLibraryDesc` (nothing hard-coded): one
  pipeline layout shared by NRD's 14 pipelines, set 0 = 18 sampled images from
  binding 20 and 7 storage images from binding 3 (all `PARTIALLY_BOUND`),
  set 1 = the two immutable samplers (bindings 0, 1: nearest and linear clamp)
  and one `UNIFORM_BUFFER_DYNAMIC` (binding 2). NRD's pools are one image each
  (13 permanent, 8 transient for this denoiser; 39 + 26 MiB at 1280x720, 158 +
  105 MiB at 2560x1440), `SAMPLED | STORAGE`, moved to `GENERAL` once and kept
  there (no layout tracking; free on NVIDIA).
- Constants: a host-visible ring of 960-byte blocks (912 rounded up to the
  uniform offset alignment), `setsMaxNum` (36) blocks per frame slot, written
  only when a dispatch's data differs from the previous dispatch's. Set 1 is
  written once and bound with a dynamic offset per dispatch.
- Per frame: one descriptor pool per renderer frame slot, reset and filled
  with one set 0 per dispatch; every dispatch follows one global
  compute-to-compute memory barrier (conservative: finer per-resource barriers
  would let independent passes overlap). 8 dispatches per frame (classify
  tiles, hit distance reconstruction, pre-pass, temporal accumulation, history
  fix, blur, post-blur, temporal stabilization), 9 with the validation layer,
  32 on the first frame of an instance (the pool clears of
  `CLEAR_AND_RESTART`).
- `CommonSettings`: world-to-view = rows right, up, forward of `CaelixCamera`
  (column-major, column vectors), view-to-clip = `CaelixCamera::projection`,
  the previous render call's pair, `motionVectorScale = (1/w, 1/h, 1)` (motion
  in pixels, z = view z delta), no jitter, `denoisingRange = 1e6` (the worlds
  are at most 8192 units across; a miss has view z 1e30),
  `disocclusionThreshold = 0.01`, `frameIndex` +1 per denoised frame,
  `accumulationMode` `CLEAR_AND_RESTART` on an instance's first frame,
  `RESTART` after `reset_accumulation` or a frame without denoising, else
  `CONTINUE`; `timeDeltaBetweenFrames` from a steady clock.
- `ReblurSettings`: NRD's defaults, plus `hitDistanceParameters` (3, 0.1, 20)
  (the same as the trace kernel), `enableAntiFirefly = true`,
  `hitDistanceReconstructionMode = AREA_3X3`, `maxAccumulatedFrameNum = 30`.
- NRD has no resize: a target size change creates a new instance (new pools,
  14 pipelines again, history lost); the old one is destroyed once its last
  frame is safe.
- An NRD error (`nrd::Result`) is logged; the frame is recorded without
  denoising and that render call returns `CAELIX_ERROR_VULKAN`; the renderer
  never denoises again (`CaelixNrdInfo::failed`).
- Debug views (path trace mode): `CAELIX_VIEW_NRD_VALIDATION` (NRD's
  validation layer blended over the image), `CAELIX_VIEW_NRD_DIFF` (the
  denoised diffuse radiance without the albedo).
- Device: NRD's shaders declare `StorageImageReadWithoutFormat` (feature bit
  10), `StorageImageWriteWithoutFormat`, `StorageImageExtendedFormats` and
  `GroupNonUniformQuad` (quad intrinsics in compute: `Device::Adopt` checks the
  subgroup properties; CMake option `CAELIX_NRD_QUAD_INTRINSICS=OFF` builds NRD
  without them). No 16-bit types, no compute derivatives (no
  `DerivativeGroupQuads` execution mode), so `VK_KHR_compute_shader_derivatives`
  is not required.

Measured 2026-10-14 (RTX 5080, release, citadel, sky light only): denoise
0.23 ms (overview) and 0.33 ms (detail) at 1280x720, 0.99 ms and 1.43 ms at
2560x1440; denoised images 40.3 dB (overview) and 38.4 dB (detail) PSNR against
the 1024-frame references, raw 1 spp 18.7 dB and 15.5 dB. Known limitation:
the sun's direct light (NEE) is part of the diffuse signal, and REBLUR blurs
hard sun shadows (its pre-pass and blur radii, 30 pixels, do not see the
shadow; NVIDIA's answer is a separate shadow denoiser, SIGMA). Images and
numbers: `Artifacts/native-renderer-step6b-20261014/`.

### Render targets and G-buffer mode

`CaelixRenderTargets` has six targets. Each is optional: a target is present
when its `image` is nonzero. At least one must be present; every present target
must have the same size and a different image. Every pixel of a present target
is written on every render call, a miss included.

| Target | Formats | Content | Miss |
|---|---|---|---|
| `color` | any storage format (RGBA8, BGRA8, RGBA16F, RGBA32F) | primary mode: the debug view (albedo, optionally Lambert shaded), alpha 1 everywhere; lighting modes: premultiplied colour, a = coverage (sRGB-encoded for UNORM formats) | primary: sky colour, alpha 1; lighting: the procedural sky (`CAELIX_RENDER_FLAG_SKY_ON_MISS`) or 0, alpha 0 |
| `albedo` | RGBA8, RGBA16F | material albedo as stored in the block (sRGB), no shading; a = coverage | (0, 0, 0, 0) |
| `normal` | RGBA16F, RGBA8_UNORM (stored as n * 0.5 + 0.5) | world-space face normal; w = coverage | (0, 0, 0, 0) |
| `depth` | R32F (storage image, not a depth attachment) | device depth clip.z / clip.w of `CaelixCamera::projection` | `options.miss_depth` |
| `motion` | RG16F, RG32F | previous pixel - current pixel, in pixels of the current size | (0, 0) |
| `view_z` | R32F | linear view depth dot(hit - position, forward) | `options.miss_view_z` |

A host with its own deferred pipeline passes only the geometry targets
("G-buffer mode"): the library then writes no lighting at all.

`CaelixCamera::projection` is the host's column-major projection for the
camera's view space (x = right, y = up, z = forward). It must describe the same
frustum as `tan_half_fov_y` and `aspect` (not checked) and is used only for the
depth target. A normal-Z host passes `miss_depth` 1, a reverse-Z host 0. The
Rust host passes a reverse-Z infinite projection (`camera.rs`,
`perspective_reverse_z`; depth = near / z).

Motion: the renderer keeps the camera of its previous successful render call.
The kernel moves the hit point (entity-local) through the entity's previous
transform (`EntityInfo` prev rows, which differ from the current rows for one
flush after `caelix_scene_set_entity_transform`) and projects it with the
previous camera, with the same pinhole mapping as the ray generation. The
current pixel is the pixel centre. Sign: `motion = previous - current`. A
camera that moves +1 unit along `right` sees a static point at depth z move
left on the screen, so `motion.x = +width / (2 * tan_half_fov_y * aspect * z)`
(+11.08 px for the test camera at z = 20, 256 px, 60 degrees). The first render
call of a renderer has no previous camera and writes 0; so does a point that
was behind the previous camera.

Mapping to NVIDIA NRD (the renderer's own NRD instance uses internal copies of
these targets; a host with its own NRD integration can use these): `view_z`
is `IN_VIEWZ`; `motion` is `IN_MV` with `motionVectorScale = (1 / width,
1 / height)` and `isMotionVectorInWorldSpace = false` (NRD computes
`uvPrev = uv + mv * scale`, the same convention, top-left origin); `normal`
becomes `IN_NORMAL_ROUGHNESS` after packing with a roughness
(`NRD_FrontEnd_PackNormalAndRoughness`); `albedo` is not an NRD input but
demodulates the radiance before the denoiser and modulates it after.

### Layout contract

A render call receives every present target in the layout that its
`CaelixTargetImage::layout` names, and leaves it in `VK_IMAGE_LAYOUT_GENERAL`:
one `vkCmdPipelineBarrier2` with one image barrier per present target before
the dispatches, one after them. The last barrier makes the kernels' writes
available to all later commands. The host transitions the images from
`GENERAL` for present (or for its own use).

## Build

Requirements: Visual Studio 2022 (MSVC and its bundled LLVM/libclang),
CMake 3.30 or newer (NRD's CMakeLists needs it), the Vulkan SDK (`VULKAN_SDK`
set; its DXC compiles our shaders and NRD's, tested with DXC 1.8 from SDK
1.4.304.1), and a Rust MSVC toolchain. Keep the build directory path short:
NRD's shader build (ShaderMake) fails on very long paths.

```
cmake --preset vs2022
cmake --build --preset release
cd host-rs && cargo run -p caelix-app
```

The first CMake configure fetches the pinned dependencies (volk,
VulkanMemoryAllocator, unordered_dense, NRD v4.18.0 with its ShaderMake and
MathLib, and for the tests Catch2 and vk-bootstrap) into `build/_deps/`. NRD
builds as a static library with its SPIR-V embedded (about 16 s on the first
build); our kernels include its `NRD.hlsli` (DXC `-I <nrd>/Shaders`, where
NRD's configure also writes `NRDConfig.hlsli` with the normal and roughness
encodings).

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
- `caelix_finalize_test`: the finalize kernel against a CPU reference in the
  test, by GPU readback of every record (1096 B) and AABB row (24 B): random
  opaque bricks in many groups, a solid cube across group boundaries, a glass
  slab in both encodings, removal and replacement (neighbours re-finalized),
  an identical re-upload (records stay bit-identical) and a neighbour that
  arrives later across a group boundary. The device setup is shared with the
  smoke test (`tests/test_device.h`); the synthetic scenes and the one-entity
  scene rig are shared with the acceleration structure test
  (`tests/test_scenes.h`).
- `caelix_as_test`: the acceleration structures. Every group of the random
  opaque scene and of the solid cube gets a BLAS and the scene a TLAS
  (`blas_count == groups`); `vkCmdWriteAccelerationStructuresPropertiesKHR`
  (serialization size) returns a nonzero size for BLASes and the TLAS; after
  removals, replacements and new groups every group is rebuilt; after an
  entity removal the BLAS storage is released only once the frame of the TLAS
  rebuild is safe; a 4 MiB scratch budget with 2000 one-brick groups spreads
  the builds over two flushes; a transform change rebuilds only the TLAS and
  the entity info row shows the old rows as `prev` for exactly one flush.
- `caelix_trace_test`: the trace kernel, by readback of 256x256 storage images
  (STORAGE and TRANSFER_SRC usage only). Before any brick the kernel writes the
  sky colour and the miss values. One pure-red opaque brick at key (0, 0, 0),
  camera at (4, 4, -20) looking +Z: the centre pixel is red, the corners are
  sky, the normal view shows (0.5, 0.5, 0) (the -Z face), the brick view and the
  step view are not sky at the centre, every other view runs without validation
  messages, `caelix_renderer_get_gpu_timings` reports a resolved frame with
  `trace_ms > 0`, and an unknown debug view is rejected. Geometry targets
  (colour RGBA8, albedo RGBA8, normal RGBA16F, depth R32F, motion RG32F, view_z
  R32F): at the centre albedo (255, 0, 0, 255), normal (0, 0, -1, 1),
  view_z 20 (the -Z face is the plane z = 0), depth = the projection of view_z
  within 1e-5 for a normal-Z and a reverse-Z infinite projection; at the corner
  the miss values; coverage agrees across the targets; motion is exactly 0 on
  the first render, below 1e-3 px with the same camera, +11.085 px after the
  camera moves +1 along `right`, -22.17 px after the entity moves +2 along X,
  and 0 again one frame later. A depth-only render works; no target, different
  sizes, one image twice and a wrong format are rejected. The random opaque
  scene: some pixels hit and none is magenta. Lighting, energy: a grey floor
  (albedo 15/31) under an emissive white ceiling (emission 1) 24 blocks above
  it, black sky, one bounce, camera looking straight down: the reference mode
  (2048 frames) over the centre 32x32 pixels is within 3 % of a CPU Monte Carlo
  estimate (albedo times the probability that a cosine direction reaches the
  ceiling, 200k samples), and one raw 1 spp frame over the centre 128x128
  pixels within 5 % (every raw pixel is exactly 0 or the albedo). Lighting,
  modes: with no bounce the red brick is black with alpha 1 and the corner is
  the CPU sky radiance with alpha 0; the sun disk adds its radiance; exposure
  scales; without the sky flag a miss is (0, 0, 0, 0); an RGBA8 target gets the
  sRGB-encoded sky; geometry targets work in a lighting mode without a colour
  target; an emissive red brick is exactly (1, 0, 0); an unknown mode and 15
  bounces are rejected. Lighting, accumulation: the first reference frame after
  a reset equals the raw frame of the same seed bit for bit, the next frames
  are the mean of the raw frames (relative error below 1e-5). Lighting, sun
  NEE: the grey floor without ceiling, black sky, sun straight up with
  radiance 1000 and radius 2 degrees: NEE (4 bounces, 256 reference frames)
  is within 1 % of `albedo * R * omega * (1 + cos r) / 2 / pi` on every centre
  pixel; NEE off (1 bounce, 4096 frames) agrees with NEE within 10 %; NEE with
  1 bounce equals NEE with 4 (no double counting through the bounce ray);
  0 bounces adds nothing; under a 16^3 block 8 blocks above the floor the
  floor is exactly 0 (1 bounce), far from it the value of the open floor; a
  primary miss still shows the sun disk with NEE on.
- `caelix_nrd_test`: NRD through the renderer. At 320x240 the first denoised
  render creates the instance (14 pipelines, 13 permanent and 8 transient pool
  textures, `denoise_ms > 0`); another size replaces it, and the first size once
  more. Quality, at 1280x720 (NRD's radii are in pixels): the floor with the
  16^3 block of the NEE test, sun radiance 1000 at 2 degrees from
  (0.4, 1, -0.3), the test sky, 4 bounces, a static camera in front of and
  above the block; the 64th denoised frame against a 1024-frame reference over
  the hit pixels: RMSE below a third of the raw 1 spp frame's (measured 0.0219
  vs 0.0862, 33.2 vs 21.3 dB PSNR at peak 1) and mean within 5 % (measured
  -1.7 %). Reset: the first frame after `reset_accumulation` changes 4x more
  against the converged frame than one converged frame against the next, and
  64 frames later the error is back within 10 %. The validation and diffuse
  debug views render. No validation message in any case.
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
- `--pose <X,Y,Z,PITCH,YAW>`: initial camera pose in Unity's terms: position,
  `eulerAngles.x` (pitch, positive looks down) and `eulerAngles.y` (yaw), in
  degrees. Default: the world's player pose (format version 7), else
  (0, 64, -128) with pitch 15 and yaw 0.
- `--fov <DEGREES>`: vertical field of view. Default 60.
- `--mode <primary|pt|ref>`: the rendering mode. Default `pt` with `--world`,
  else `primary`.
- `--view <albedo|normal|depth|group|brick|steps|nrd-validation|nrd-diff>`:
  the debug view; the first six in the primary mode, `nrd-validation` (NRD's
  validation layer) and `nrd-diff` (the denoised diffuse radiance) in `pt`
  mode. Default `albedo`.
- `--no-denoise`: `pt` mode without NRD (the raw 1 spp image). Default: NRD on.
- `--size <WxH>`: inner size of the window. Default 1280x720.
- `--warmup <N>`: `pt` mode, take `--screenshot` N frames after the world is
  complete (at least 3), so that NRD's history is warm (64 for the artifacts).
  In `pt` mode the NRD history restarts only while the world is loading;
  camera motion is followed through the motion vectors.
- `--no-shading`: primary mode, albedo without Lambert shading.
- `--bounces <N>`: lighting modes, stochastic bounces (0..14). Default 4.
- `--exposure <F>`: lighting modes, colour multiplier. Default 1.
- `--sun <X,Y,Z>`: direction toward the sun. Default 0.45,0.7,0.55 in the
  lighting modes, -0.45,0.8,-0.3 (the Lambert direction) in the primary mode.
- `--sun-radiance <R,G,B>` (default 0,0,0 = sun off, sky light only;
  suggested `2e4,1.9e4,1.8e4` for the default 0.5-degree disk: `omega` =
  2.3925e-4 sr, so the irradiance on a surface facing the sun is
  (4.79, 4.55, 4.31), luminance 4.58, about 3.1x the default sky's 1.48 on an
  up-facing surface, `pi * (0.2 * horizon + 0.8 * zenith)` =
  (0.94, 1.51, 2.86)), `--sun-angular-radius <DEGREES>` (default 0.5), `--sky-zenith <R,G,B>` (default 0.2,0.4,0.9),
  `--sky-horizon <R,G,B>` (default 0.7,0.8,0.95), `--sky-ground <R,G,B>`
  (default 0.25,0.22,0.2): the procedural sky, linear RGB radiance.
- `--no-sky-on-miss`: lighting modes, black instead of the sky where the
  primary ray misses.
- `--no-sun-nee`: lighting modes, no sun next-event estimation (the 6a
  behaviour: the sun is found only by the bounce rays).
- `--accumulate <N>`: reference mode; the accumulation restarts whenever the
  camera changes or the world is not complete yet, and `--screenshot` is taken
  after N accumulated frames. The frame seed is the frame number.
- `--screenshot <PATH.png>`: write a PNG of the swapchain image (alpha forced
  to 255). With `--world` it is taken 3 frames after the world is complete on
  the GPU (every brick uploaded and `as_builds_pending == 0`); without
  `--world` at frame 10; with `--accumulate` after N accumulated frames.
- `--exit-after-screenshot`, `--no-input` (ignore keyboard and mouse for deterministic automated runs). In the window, press `P` to log the current camera as a `--pose ... --fov ...` line: exit once the PNG is written.

Camera convention (`caelix-app/src/camera.rs`, Unity's): for yaw θ and pitch φ,
`forward = (sin θ cos φ, -sin φ, cos θ cos φ)`, `right = (cos θ, 0, -sin θ)`,
`up = (sin θ sin φ, cos φ, cos θ sin φ)`. Free camera: W/S/A/D move along
forward and right, Q/E move down and up along world Y, Shift multiplies the
speed by 8, the mouse looks around while the right button is held (0.1 degree
per pixel), and the scroll wheel scales the speed (default 100 units/s).

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
  Then `caelix_renderer_render` traces the scene into the swapchain image
  (the swapchain has `STORAGE`, `TRANSFER_DST` and `TRANSFER_SRC` usage), and
  `end_frame` presents. When the swapchain was recreated, the host calls
  `caelix_renderer_release_target_views` first.
- The window title shows the GPU times once per second
  (`Caelix Renderer | trace 0.53 ms | denoise 0.23 ms | composite 0.01 ms | flush 0.00 ms | 144 fps`); the same line
  is logged at info level every 120 frames and when a screenshot is written.
  At the second frame of the renderer's NRD instance the host logs one NRD line
  (`caelix_renderer_get_nrd_info`: version, size, pipelines, pool textures and
  MiB, dispatches per frame).
  Presentation uses FIFO, so the frame rate is limited by the display.
- When the decoder has
  finished and every brick is uploaded, the host logs one summary (entities,
  groups, resident bricks, record pages, GPU memory, decode and upload time,
  upload frames, bricks/s, the acceleration structure stats, the full
  `CaelixSceneStats`) and queries the first key of the first 3 decoded sectors
  with `caelix_scene_query_brick`. When `bricks_resident` differs from the
  number of decoded bricks, it logs an error and exits with a non-zero code.
- After the load, the first flush that leaves `as_builds_pending == 0` logs
  one more line: its frame number, the time since the loader opened the file,
  the frames since the first upload and since the load summary, and the BLAS,
  TLAS and scratch sizes. When `blas_count` differs from the group count, it
  logs an error and exits with a non-zero code.
- `--frames N` exits after N frames even when the load is not complete (with a
  warning). At exit the host drops the channel receiver and joins the decoder
  thread. `decode_all` cannot be cancelled: it decodes the remaining sectors and
  drops them.

Expected numbers for `VoxelTestScenes/epic-citadel.8192.packed.cxw` (release,
RTX 5080, 64 MiB staging): 3837 sectors, 1 entity, 3837 groups, 1,095,051
resident bricks in 5 record pages (1370 MiB of record pages, 32 MiB of slot
tables, 32 MiB of reverse tables, 48 MiB of AABB rows, 4 to 8 MiB of group
grid). The upload takes about 0.52 s over 39 frames, 20 of which upload (the
ring holds one frame's records until that frame is safe), about 2.1 M bricks/s;
before the acceleration structure step it took 0.38 s, and without the
finalize step 0.30 s. About 1.18 M bricks are finalized (bricks are finalized
again when a neighbour arrives in a later frame). The acceleration structures
are complete in the same flush that completes the load (frame 46, 0.56 s after
the file was opened): 3837 BLASes in 64 MiB, a 2 MiB TLAS, 128 MiB of scratch,
about 8250 builds (a group is rebuilt when a neighbour arrives later, and the
TLAS is rebuilt every upload frame). Peak working set of the process is about
0.6 GB.

Debug builds enable `VK_LAYER_KHRONOS_validation` when it is installed; its
messages are logged with target `vulkan`. Library messages use target `caelix`.
Release builds run without validation.

Implicit layers that other software installs can add loader warnings that are
not from this project. Example: the Overwolf layers report an old API version;
set `DISABLE_VULKAN_OW_OVERLAY_LAYER=1` and `DISABLE_VULKAN_OW_OBS_CAPTURE=1`
to disable them for a run.

## Environment variables

- `CAELIX_SHADER_DIR`: directory with `<name>.spv` files that replace the
  embedded kernels at scene creation (see "Shaders").
- `CAELIX_LIB_DIR`: directory that contains `caelix_renderer.lib` and
  `caelix_renderer.dll`. Default: `<repo>/build/bin/Release`. The Rust host
  links the Release build only.
- `LIBCLANG_PATH`: directory that contains `libclang.dll`, used by bindgen.
  `host-rs/.cargo/config.toml` sets it to the 64-bit Visual Studio 2022
  Community LLVM directory (`VC/Tools/Llvm/x64/bin`); set your own value if your
  installation is different. Do not use `VC/Tools/Llvm/bin`: its
  `libclang.dll` is 32-bit.
