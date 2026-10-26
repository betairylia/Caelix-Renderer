# Per-voxel colour post-process (experiment)

Branch `sss-voxel-color-expr` (from `sss-expr` `281224e`), 2026-10-27. Experiment, not for merge: a
visual test of a "one colour per voxel" look.

## Goal

Every pixel whose camera ray first hits voxel V gets the MEAN of the final linear colour of all pixels of
the same frame (screen space) whose camera ray first hits V. All faces of a voxel share that colour.
Pixels that miss keep their own colour. The pass runs on the final colour: after NRD in the normal path,
on the raw composite without the denoiser, on the reference accumulation in `CAELIX_MODE_REFERENCE`, and
on the DLSS (RR or SR) output at the OUTPUT size when an upscaler runs.

Switch: `CAELIX_RENDER_FLAG_VOXEL_COLOR` (`1u << 7`, `include/caelix/caelix.h`; no struct change, no ABI
bump). Host: `--voxel-color`, key `V`, F1 checkbox "voxel colour (V)".

## Key rule

The key is the FIRST hit of the camera ray (`TraceScene`, the committed closest hit), transparent voxels
included: all pixels that see one water or glass surface voxel share one colour, although each pixel's
colour comes from the surface that the delta chain found behind it (primary surface replacement).

- Ray: without an upscaler `CaelixPrimaryDirection(fc, pixel)`, the trace kernel's own jittered primary
  ray (same origin, TMin 0, TMax 1e30), so the key is exactly the first hit that the frame traced. With
  an upscaler `CaelixOutputCentreDirection(fc, pixel)` (`shaders/common/caelix_frame.hlsl`): the
  unjittered ray through the centre of the OUTPUT pixel, the view that the upscaler's output shows.
- Voxel: `floor(object_hit - 0.5 * object face normal)`, clamped into the brick that the hit's candidate
  traced: the start voxel rule of the SSS walk (`MediumCursorFromHit`, `shaders/render/caelix_subsurface.hlsl`).
- Key words: (entity of the hit's group, voxel x, y, z) in entity-local block coordinates.

## Hash table

`shaders/common/caelix_voxel_color.hlsl` mirrors `lib/src/render/voxel_color_layout.h`
(`caelix_frame_layout_test` compares the constants).

- One renderer-owned storage buffer of `max(65536, bit_ceil(pass pixels) / 2)` slots (a power of two;
  frame constant `voxel_slot_mask` = slots - 1). 1280x720: 524288 slots, 16 MiB; 1920x1080: 32 MiB;
  3840x2160: 256 MiB. Pass size = the output size (= the render size without an upscaler). Recreated (the
  old buffer retired until its frame is safe) when the size changes.
- Slot = 8 uint words (32 B): `[0]` fingerprint (0 = empty), `[1]` pixel count, `[2..4]` sum low words
  r, g, b, `[5..7]` sum high words r, g, b.
- Cleared by `vkCmdFillBuffer` (0) in every frame that runs the pass, before the trace (barriers: compute
  -> transfer before the fill, transfer -> compute after it).
- Hashes: `VoxelKeyHash(key, seed)` = one lowbias32 finaliser round per key word (the finaliser of
  `HashU32`), two seeds: start slot and fingerprint (forced non-zero).
- Insert: linear probe `(start + i) & mask`, i < 64: `InterlockedCompareExchange(word0, 0, fp)`; the slot
  is ours when the word was 0 or already `fp`. No spinning, no pending state. 64 failed probes: the pixel
  keeps its colour.
- Add: `InterlockedAdd(count, 1)`; per channel `v = round(clamp(c, 0, 65504) * 65536)` (fits 32 bits),
  `InterlockedAdd(lo, v, before)` and `InterlockedAdd(hi, 1)` when `before + v < before` (exact: every
  low add is atomic and returns the value it added to). Mean = `(hi * 2^32 + lo) / 65536 / count` in
  float; no float <-> uint64_t conversion (DXC 1.8 drops the high word of `float(uint64_t)`).

## Passes

Kernels: `shaders/render/caelix_voxel_accumulate.hlsl` (new; voxel set: 0 TLAS, 1 source, 2 internal
albedo, 3 slot image R32_UINT at the output size, 4 hash buffer) and `caelix_upscale_encode.hlsl`
(encode set grows by 3 slot image, 4 hash buffer; it writes the slot mean instead of the source colour
where the slot is valid, everything else unchanged: sRGB for UNORM targets, flip-Y, coverage alpha).

`voxel_color = 1` in the frame constants only when the flag is set, the frame has a composite (lighting
mode with a colour target) and the view is not `CAELIX_VIEW_MOTION` or `CAELIX_VIEW_NRD_VALIDATION` of the
path trace mode (those frames render as without the flag).

Without an upscaler (time inside `composite_ms`):

1. composite (and the cross resolve when it runs) writes the linear colour to the upscale input
   (`kInternalUpscaleIn`) instead of the colour target, as with an upscaler;
2. barrier, voxel accumulate over the upscale input (`output_size = size`, `upscale_src_size = size`:
   the source and coverage mappings are the identity);
3. barrier, encode kernel from the upscale input into the colour target.

With an upscaler (time inside `upscale_ms`): composite -> upscale input -> evaluate (or the fallback
barrier) -> voxel accumulate over the image the encode reads (the upscaler's output, or the upscale input
on the fallback) -> barrier -> encode. Pass A binds its pipeline and sets after the NGX evaluate, the
encode binds again after it.

Cost (sss scene, 1920x1080, RTX 5080, one frame): composite 0.04 -> 0.24 ms without an upscaler (accumulate
+ encode). The fill (32 MiB) is outside every timer.

## Limits

- Upscaled output: the key is one centre ray per output pixel, so voxel edges in the output are aliased
  (one voxel per pixel, no sub-pixel coverage); the colour comes from the upscaler's anti-aliased output.
- A voxel covered by one pixel shows that pixel's colour (the mean of one); voxels much smaller than a
  pixel are unchanged, and when the table is full (more distinct voxels than slots) the extra pixels keep
  their colour.
- Two voxels with the same 32-bit fingerprint in one probe chain merge (expected well below 0.01 merges
  per frame).
- The NRD validation overlay is unavailable with the flag (the frame renders without the flag in that
  view); the colour of every flagged frame goes through the RGBA16F upscale input (FP16 rounding, clamp to
  65504).
- `CAELIX_MODE_PRIMARY` ignores the flag (no composite).

## Validation (2026-10-27)

- `ctest`: same as before the change (10/11; `caelix_finalize_test` fails on this line of branches);
  `caelix_frame_layout_test` covers `caelix_voxel_accumulate` and the new layout constants.
- Host smoke runs on `ScenesPlayground/Artifacts/sss-20261026/out/sss.cxw` (pt, pt + `--no-denoise`,
  `--upscaler dlss-rr`, `--upscaler dlss-sr`, `--mode ref`) and a scratch glass/water scene: flat colour
  per voxel, all faces of a voxel alike; the glass pane and the water pool show one mean per surface voxel
  of what is seen through them. The motion and NRD validation views are bit-identical with and without
  the flag. The debug host (Khronos validation layer) reports no message for the four paths.
