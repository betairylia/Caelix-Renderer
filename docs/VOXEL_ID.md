# Voxel id target (as built, 2026-10-27)

A G-buffer target with one key per voxel: per pixel, a hash of the voxel that the camera ray hits
first. The ABI contract is `CaelixRenderTargets::voxel_id` in `include/caelix/caelix.h`.

## Key

- The voxel: the entity id and the entity-local integer voxel position of the camera ray's FIRST
  hit, transparent voxels included (the key is taken before the delta chain; glass does not pass
  the key on to the surface behind it).
- The position: the brick of the hit (group coordinate from the group info, brick position from
  the slot row, the same formula as `TraceCandidate`) times 8, plus the DDA cell of the hit.
- The hash (`VoxelIdHash` in `shaders/render/caelix_scene_trace.hlsl`, below the shader boundary):
  two independent words with the seeds `0x9E3779B9` and `0x85EBCA6B`, each
  `h = avalanche(seed ^ entity); h = avalanche(h ^ x); h = avalanche(h ^ y); h = avalanche(h ^ z)`
  (`CaelixHashAvalanche`). A word of `0xFFFFFFFF` becomes `0xFFFFFFFE`, so a hit never has the miss
  value.
- Two voxels share a word only on a hash collision (2^-32 per pair per word).

## Target

- Formats: `R32_UINT` (x = word 0) or `R32G32_UINT` (x = word 0, y = word 1).
- Miss value: `0xFFFFFFFF` in every channel for a miss, for a hit farther than
  `CaelixRenderOptions::voxel_id_tmax` along the camera ray (0 = no limit; finite and >= 0, else
  `CAELIX_ERROR_INVALID_ARGUMENT`), and for every pixel before the scene's first TLAS build.
- `CAELIX_RENDER_FLAG_FLIP_Y` applies. Every mode writes it.

## Two write paths

- Without an upscaler (also the primary and the reference mode, which ignore the upscaler): the
  trace kernel writes it at the common size from the primary ray, with the camera jitter
  (`caelix_path_trace.hlsl`, `VoxelIdOrMiss`, next to the view_z write).
- With an upscaler: the target has the OUTPUT size (the colour target's; another size is
  `CAELIX_ERROR_INVALID_ARGUMENT`). The trace skips it (`fc.upscale = 1`), and the kernel
  `caelix_voxel_id` (`shaders/render/caelix_voxel_id.hlsl`, the trace set, 8 x 8, output size) runs
  right after the trace: one UNJITTERED ray through the centre of each output pixel, so that the key
  image pairs with the upscaled colour. It reads only the TLAS and the scene tables, so it needs no
  barrier after the trace; the final barrier of the render call covers its writes. Its time is part
  of `CaelixGpuTimings::trace_ms`.
- `caelix_dlss_test` checks that the dedicated pass after three jittered SR (and RR) frames is
  byte-identical to the trace kernel's image of a render without an upscaler and with jitter 0 at
  the output size (same pinhole expression).

## Debug view

`CAELIX_VIEW_VOXEL_ID` (11), primary mode: `Pastel(word 0)` of the primary hit, no shading; a
miss and a hit beyond `voxel_id_tmax` show `sky_color`. The lighting modes ignore it.

## Attribute word

The brick DDA now stores the hit cell in `AttributeData::matID_faceNormal`:
`(31) [matID:16] [reserved:1] [cell:9] [faceNormalFlags:6] (0)`, cell = x | y << 3 | z << 6
(`CaelixBrickCellIndex`). `TraceResult::cell` holds it (0 on a miss). Every reader of the face flags
masks bits 0..5 (`UnpackObjectNormal` masks with `0x3F`; `firstbitlow(attrib & 0x3F)` in the path
tracer); the block code is `attrib >> 16`.

## Host

`caelix-app --voxel-id` allocates an `R32G32_UINT` target at the window size (`caelix_vk::Context::
create_storage_image`, recreated on a resize after `caelix_renderer_release_target_views`) and
passes it every frame in layout `UNDEFINED`; `--voxel-id-tmax <F>` sets the limit; `--view voxel-id`
(primary mode) shows the debug view. The F1 Render window has a "Voxel id target" checkbox and a
"Voxel id t max" value.

## Cost

Registers (`CAELIX_PIPELINE_STATS=1`, RTX 5080): `caelix_path_trace` 128 + 16 B spill and
`caelix_path_trace_no_light_nee` 96 + 48 B spill (both unchanged by this step), `caelix_voxel_id`
56 + 16 B spill.
