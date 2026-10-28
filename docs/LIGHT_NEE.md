# Light NEE (as built, 2026-10-24)

Status: as built on `main`. This file is the truth for the code. The first version kept light cells
(world-space caches of light tree draws); the owner removed them on 2026-10-24 after a visual check, and
the light tree draw per vertex, until then an experimental variant, is now the only light NEE (section
11). Base: the light tree (`docs/LIGHT_TREE.md`). Owner goals (`docs/human_wrote/Caelix-Renderer Towards
A Million Stars.md`): a raw signal with low spatial and temporal correlation, the denoiser / upscaler
carries the rest; the trace kernel stays at or below 128 registers. Owner rule: visuals and stability
come before strict unbiasedness when the difference is within what the eye accepts.

## 1. Purpose and shape

Next-event estimation of the scene's light voxels (the light faces of the light tree,
`caelix_scene_set_light_options`) in the lighting modes (path trace and reference), next to the sun
NEE. Every diffuse path vertex with a next segment draws ONE lamp point from the light tree for its
own receiver point and normal, chooses between that lamp candidate and the sun candidate, and traces
ONE shadow ray. The next BSDF hit on a lamp that the NEE can sample is weighted by the balance
heuristic with the exact pdf of the same draw. The light NEE keeps no memory: nothing is kept from
frame to frame or shared between pixels, so its error is white per ray (no cell-sized blobs, no
temporal lag).

Code: `shaders/render/caelix_path_trace.hlsl` (section "Light NEE": `LampCandidate`, `SunCandidate`,
`NextEventAtVertex`, `LampHitWeight`; the gate in `TracePath`), `shaders/render/caelix_light_nee.hlsl`
(`kLightNeeNearField`, `LightNeeMisWeight`), the light tree device API
(`shaders/light/caelix_light_tree.hlsl`: `LightTreeView`, `LightTreeSample`, `LightTreeFindLightFace`,
`LightTreeBrickPdf`), the renderer side `lib/src/render/renderer_light.cpp` (options, the choice per
render call, stats) and `renderer_render.cpp` (flags, kernel binding).

## 2. The NEE at a vertex

Where: at the vertices where the sun NEE runs without light NEE: the primary surface (when
`bounce_count >= 1`) and every bounce hit while `bounce_index <= bounce_count`, if the shading chose
the diffuse lobe. The receiver is `x_r` = the shade's pushed-off `bounce_origin`, the normal `n_r` =
the hit's world normal.

Draw (`LampCandidate`): `LightTreeSample(tree, x_r, n_r, rng, p)` with the PCG state `point_seed`: the
tree by the importance of the two root summaries, then down the 4-wide nodes by the point form of the
importance at every level, to a leaf (one light brick), then the face family, the face entry and a
point on the face, all for the receiver `x_r` and `n_r`. `p.pdf` is the area pdf of the point:
`P(brick | x_r, n_r) P(face | brick, x_r) / face_count / face area` (a voxel face has area 1).

Rejections (no lamp candidate, contribution 0, the vertex stays on the MIS path):
- the draw fails or `p.pdf` is not positive (a tree without lights: no tree is chosen);
- the point lies inside the near field, `distance < kLightNeeNearField` (section 5);
- the point is back-facing: `cos_r = n_r . w <= 0` or `cos_l = -n_l . w <= 0`.

Contribution (demodulated; the caller multiplies the throughput, which holds the vertex's albedo):
`L_e (cos_r / pi) (cos_l / d^2) / p.pdf`, times `w_nee` with MIS (section 4). `L_e` = the material
table's emission of the point's block times `emission_scale` (`LampEmission`). The shadow ray of a
lamp candidate stops two push-offs short of the point (`t_max = d - 2 K_RAY_ORIGIN_PUSH_OFF`).

Random streams: the lamp stream `light_rng` (its own seed, `frame_seed ^ 0x7F4A7C15`) draws one number
per vertex (the point seed, `RandomU32`); the sun stream `nee_rng` draws the sun direction's two
numbers and one number for the choice. The BSDF path consumes the same numbers with light NEE on or
off, and a frame without light NEE consumes every stream exactly as before.

## 3. The sun / lamp choice and the shadow ray

`NextEventAtVertex` evaluates both candidates without their shadow rays: the lamp candidate (section
2) and, when the sun NEE runs (`CAELIX_RENDER_FLAG_SUN_NEE` with a nonzero sun radiance), the sun
candidate (`SunCandidate`: a direction uniform in the sun cone, `sun_radiance omega cos_l / pi`). It
keeps one of them by one-sample RIS over the disjoint domains with the weights `w = luminance` of the
unoccluded contributions (no visibility hint): `pick_sun = u w_sum < w_sun`, and the kept candidate's
contribution is multiplied by `w_sum / w_kept`. So `E = C_s V_s + C_l V_l` stays unbiased. The lamp
candidate stays live through the sun's draw (its evaluation is the expensive one); a kept sun candidate
is drawn again from its saved random state. One shadow ray toward the kept candidate (glass occludes,
as for the sun NEE; at the primary vertex with `CAELIX_RENDER_FLAG_SHADOW_TRANSMISSION` the ray passes
glass and SSS voxels, attenuated, docs/SHADOW_TRANSMISSION.md). The segment that leaves the vertex sees the sky without the sun disk when the sun
NEE runs (as without light NEE).

## 4. MIS with the BSDF path

The segment that leaves a vertex that ran the light NEE carries one state word: `kSegmentLampNee`, plus
`kSegmentLampMis` with MIS. When that segment hits an emitting voxel, `LampHitWeight` weights the
emission:
- 1 (the full emission) when the vertex ran no light NEE, inside the near field (`hit.T <
  kLightNeeNearField`), or for a point the NEE cannot sample (not a light face of the tree:
  `LightTreeFindLightFace` fails, or the brick has no leaf);
- 0 without MIS (`CAELIX_LIGHT_NEE_FLAG_NO_MIS`: the exclusive rule, the NEE alone counts the lamps it
  can sample);
- else the BSDF's balance heuristic weight `p_bsdf / (p_sa + p_bsdf)`.

Both sides use the exact pdf of the same draw: on the NEE side `p.pdf` of `LightTreeSample`, on the
BSDF side `LightTreeBrickPdf(tree, x, n, brick) x in_brick` (the pdf walk from the brick's leaf up the
parent links, times the face's in-brick density from `LightTreeFindLightFace`), with `x` = the
segment's origin and `n` = `segment_normal`. These are bit for bit the `x_r` and `n_r` that
`NextEventAtVertex` got: `TracePath` passes `shade.bounce_origin` and the hit's normal to the NEE and
sets the next segment's origin and normal from the same values. `p_sa = p_area d^2 / cos_l`, `p_bsdf =
cos_r / pi` (the cosine-weighted diffuse bounce). The two weights of a point sum to 1 up to the
rounding of the two pdf walks (the draw multiplies its choices top-down, the evaluation bottom-up;
measured: relative pdf difference at most 2.8e-7, weight sum within 1.4e-7 of 1). `LightNeeMisWeight`
handles the degenerate inputs the same way on both sides: `cos_l <= 0` or a non-finite `p_sa` counts as
`+inf` (NEE 1, BSDF 0), both densities 0 as NEE 0, BSDF 1. Segments that leave a specular or
transparent vertex carry no state and count every emission, except after a transmissive primary NEE
(the claim of docs/SHADOW_TRANSMISSION.md section 1: through transparent vertices the state stays
`kSegmentLampNee` without the MIS bit).

## 5. The near field

Lamp points closer than `kLightNeeNearField` = 8 world units to the receiver belong to the BSDF path on
both sides of the MIS: the NEE rejects them (section 2) and the BSDF hit keeps their full emission
(section 4). Area sampling with a `1 / d^2` term is extremely heavy-tailed near a lamp; the BSDF path
finds near emitters with a bounded weight. The value is unchanged since the first light NEE (the old
branch's rule).

## 6. Kernels, frame constants and registers

Two trace kernels:
- `caelix_path_trace`: the trace with the light NEE. The renderer binds it for every frame that runs
  light NEE.
- `caelix_path_trace_no_light_nee` (`shaders/render/caelix_path_trace_no_light_nee.hlsl`, a
  `CAELIX_KERNEL_VARIANT`): `CAELIX_SPEC_NO_LIGHT_NEE` makes `light_nee_flags` 0 at compile time, so the
  compiler removes the light NEE. The renderer binds it for every other frame (light NEE off, the
  primary mode, a scene without a light tree): with the light NEE code in, a frame without light NEE
  traced 20 % slower (rootfall 1440p 9.9 vs 8.2 ms).

Light NEE runs when the options enable it, the mode is a lighting mode and the scene has a light tree.
Enabled without a light tree, it logs one warning per renderer and does not run.

Frame constants (`lib/src/render/frame_constants.h`, `shaders/common/caelix_frame.hlsl`; 560 bytes, the
size before the light NEE): `light_tree` (offset 544, the light tree descriptor) and `light_nee_flags`
(offset 552): `kLightNeeOn` (bit 0) = the light NEE runs, `kLightNeeMis` (bit 1) = with MIS; offset 556
is a pad. `caelix_frame_layout_test` checks the uniform block of both trace kernels and the four other
frame-constant kernels.

Registers (`CAELIX_PIPELINE_STATS=1`, RTX 5080, driver 610.88): `caelix_path_trace` 128 registers and
16 bytes of spill (driver binary 174976 bytes, shared memory 2560 bytes), `caelix_path_trace_no_light_nee`
96 and 48 bytes of spill (since the shadow transmission step both are 128 registers + 16 bytes:
docs/SHADOW_TRANSMISSION.md section 8). Both equal the kernels of `18282cd`: the SPIR-V function bodies are the same
instructions modulo id numbering, except the two flag bit values (`kLightNeeContribute` 2 -> `kLightNeeOn`
1, `kLightNeeMis` 8 -> 2); `Artifacts/light-nee-variants-20261024/cleanup/spirv/compare.txt`.

## 7. ABI and host

```c
#define CAELIX_LIGHT_NEE_FLAG_NO_MIS (1u << 0)  /* the exclusive rule instead of the balance heuristic */

typedef struct CaelixLightNeeOptions
{
    uint32_t enabled;     /* 0 = off (the default), 1 = on */
    uint32_t flags;       /* CAELIX_LIGHT_NEE_FLAG_* */
    uint32_t reserved[2]; /* must be 0 */
} CaelixLightNeeOptions;  /* 16 bytes */

typedef struct CaelixLightNeeStats
{
    uint32_t enabled;     /* 1 while the options enable light NEE */
    uint32_t running;     /* 1 when the newest render call ran it (a lighting mode, a light tree) */
    uint32_t reserved[2]; /* 0 */
} CaelixLightNeeStats;    /* 16 bytes */

CaelixResult caelix_renderer_set_light_nee_options(CaelixRenderer*, const CaelixLightNeeOptions*);
void caelix_renderer_get_light_nee_stats(const CaelixRenderer*, CaelixLightNeeStats*);
```

Per renderer; the options apply from the next render call. `enabled` above 1, unknown flag bits or a
nonzero `reserved` word return `CAELIX_ERROR_INVALID_ARGUMENT` (logged) and change nothing. The library
default is off. ABI v2 is in development: the structs shrank from 32 / 96 bytes without a version bump
(the Rust host is the only user). The light NEE uses no GPU memory of its own.

Rust host (`caelix-app`): light NEE on by default (it also enables the light tree); `--no-light-nee`
turns it off, `--light-no-mis` uses the exclusive rule. `--screenshot-frame N` takes the screenshot at
frame number N (the frame seed), so runs with different options or binaries capture the same random
numbers. The Stats window (F1) has a "Light NEE" section (enabled, running, MIS); `--timing-frames`
adds `light_nee: running=<0|1>` to the `timings_avg:` line.

## 8. Tests

- `caelix_trace_test`, "light NEE: unbiased against the path tracer (lamps, sun, glass)": a closed grey
  box with single-voxel lamps (some inside the near field of a pillar), a sun opening half covered by
  clear glass; reference mode, 4 bounces, 4 transparent bounces; light NEE (1024 frames) against the
  plain path tracer (4096 frames), lamps alone and with the sun, for the defaults (MIS) and for the
  exclusive rule. Tolerances: mean ratio 0.97 .. 1.03, 16 x 16 block PSNR above 30 dB. Measured
  (2026-10-24): ratio 0.9983 (sun 0) and 0.9984 (sun 1000) for both, block PSNR 40.1 / 40.2 and
  39.6 / 39.6 dB, against the path tracer's own floor (other seeds) 0.9982 / 0.9986, 37.6 / 37.7 dB.
- `caelix_trace_test`, "light NEE: lifecycle (entity and group reuse, a paused renderer, modes, no
  tree, options)": option checks (`enabled` 2, unknown flags, nonzero `reserved`), `running` in both
  lighting modes and not in the primary mode, an entity removed and its id reused at once, groups
  released and re-created at the same keys, a renderer paused over 24 flushes with moves and edits, the
  scene's light tree disabled (one warning, not running) and enabled again, disable and enable without
  MIS; every image finite, no validation or library error.
- `caelix_light_nee_test` (probe kernel `tests/shaders/caelix_light_nee_probe.hlsl`, a scene with lamps
  on a plain and a rotated entity): the box sampler's support (the light tree's region receiver: every
  light face that can reach a receiver point of a box has a positive box pdf and target); both MIS
  sides see the same exact pdf for 25 600 drawn lamp points (same brick, relative pdf difference below
  1e-4, weights sum to 1 within 1e-4); `LightNeeMisWeight` at degenerate inputs (cos_l <= 0, densities
  0, +inf, NaN, overflow, a receiver facing away).
- `caelix_frame_layout_test`: the 560-byte uniform block of the six frame-constant kernels.
- `caelix_abi_c99`: the 16-byte structs from C99.

Not built: dynamic response curves (a lamp switched on, removed, moved; a receiver entity rotated). The
light NEE has no history, so it responds in the next frame by construction.

## 9. Measured cost

RTX 5080, driver 610.88, Rust host, 100-frame means of the trace dispatch after 64 warm-up frames, the
host's 4 bounces, no sun, dark sky, NRD on, one GPU job at a time
(`Artifacts/light-nee-variants-20261024/README.md` and `cleanup/README.md`):

| scene | 2560x1440 | 1920x1080 |
|---|---|---|
| bistro 8192 | 28.07 .. 28.12 ms | 15.14 / 15.21 ms (2026-10-24, after the removal of the cells) |
| rootfall | 27.55 .. 27.60 ms | - |
| without light NEE (section 11 table) | bistro 8.09, rootfall 8.27 ms | bistro 4.75, rootfall 4.86 ms |

On 2026-10-24 the cell variants ran at about 16.5 or 22.5 ms at 1440p on both scenes (two machine
states); the light tree draw per vertex never showed the faster state and cost about 5 ms (+22 %) more
than the slower state of the cells, without their fill (1.5 ms) and their 290 MiB per renderer. The
extra cost is a light tree descent with the point importance at every NEE vertex instead of a reservoir
load and an in-brick draw.

Bias on a real scene (bistro, reference mode, 512 frames, against a 1024-frame reference with light NEE
in the cell mode "rotate"): linear mean luminance ratio 0.9991 (quadrants 0.9985 .. 1.0004), block PSNR
56.85 dB.

## 10. Open

- A per-pixel sun hint at the primary vertex is the planned replacement for the lost cell sun hint
  (owner, 2026-10-24): the sun / lamp choice now has no visibility information.
- Shadow rays are about 37 % of the trace with light NEE (rootfall 1440p, measured with the cells:
  about 7.5 of 20.3 ms); 85 .. 88 % of the lamp shadow rays are occluded on bistro, shroom and
  meridian-deep (`Artifacts/light-visibility-20261023/`). A visibility-aware draw is not designed yet.
- The NRD output was about 10 % darker than the reference in the measurements of the first version
  (denoiser bias, not the NEE); not measured again without the cells.
- The light tree's box receiver functions (`LightTreeSampleBrickBox`, `LightTreeBrickPdfBox`,
  `LightTreeLeafImportanceBox`) and the reusable hash table (`docs/HASH_TABLE.md`) have no user in the
  trace now; they are kept for region receivers (for example volumes) and other caches.

## 11. History: light cells (removed 2026-10-24)

What they were (`650c457`, spec approved after two external reviews): every renderer kept a GPU hash
table of world-space light cells keyed by (entity, entity life, level, cell coordinate, receiver face),
32- or 128-voxel cells chosen by the pixel footprint, with a cell-size controller that could only
coarsen. Each cell held 64 reservoirs (1088 bytes per cell; 128 and 256 as an experimental option),
filled once per frame for the ACTIVE cells (looked up by the previous frame) from the light tree's box
draw for the cell's receiver box (`LightTreeSampleBrickBox`), M = 1 or 2 candidates per refreshed
reservoir. Reservoir modes: "off" (every reservoir redrawn per fill) and "rotate" (a rotating quarter
per fill, each at most 4 fills old). The trace looked the vertex's jittered cell up (inserting a missing
one), drew a uniform reservoir and a point on its brick, chose sun or lamp with the cell's sun
visibility hint (`max(0.05, visible / tests)`), and weighted BSDF hits with a proxy MIS (the leaf's box
importance over the cell's normalisation C). Memory: 262144 cells x 1088 bytes + the key table, about
290 MiB per renderer (1.03 GiB with 256 reservoirs).

Key measurements (2560x1440 unless stated; `Artifacts/light-nee-main-20261024/`):
- fill on rootfall (42 808 active cells): mode off 1.39 ms (M 1) / 2.29 ms (M 2), rotate 0.61 / 1.03 ms;
  maintenance 0.06 ms. The bench rule chose rotate, M 1; the owner then chose off (`d5670aa`).
- trace with / without light NEE: rootfall 20.46 / 8.27 ms, bistro 21.55 / 8.09 ms (1080p 11.73 / 4.86
  and 11.84 / 4.75 ms); lookups alone 10.9 ms on rootfall; registers 128 + 16 bytes, fill 80.
- bias: 1024-frame references off vs on, ratio 1.005 rootfall, 0.999 bistro, 1.000 emberdeep; 512-frame
  bistro 0.9994 (off, 64), 0.9998 (rotate, 256).
- temporal (crimson-crown wall, 256 frames): raw lag-1 autocorrelation rotate 0.69 .. 0.80, off -0.08 ..
  -0.04 (white); after NRD std(m_t) / ref rotate 0.009 .. 0.020, off 0.005 .. 0.012. The older branch's
  128-frame cell history gave 0.94 .. 0.98 raw.

Why they were removed (owner, 2026-10-24, after a visual check of `18282cd`): heavy flicker with many
lights; spatially correlated error in cell-sized blobs (shared reservoirs) that hurts DLRR; the memory
footprint; and the cost saving is small with 2 bounces and shrinks as the cell set grows.

Commits: `650c457` (light cells, the hash table, the NEE, the MIS), `d5670aa` (default mode off),
`18282cd` (experimental variants: up to 256 reservoirs per cell, and the no-cell light tree draw per
vertex in a third trace kernel, which became this light NEE). Artifacts:
`Artifacts/light-nee-main-20261024/` (fill bench, measurements, temporal and bias runs, validation) and
`Artifacts/light-nee-variants-20261024/` (the variants; `cleanup/` = the validation of the removal).

## 12. Visibility hint and bounce NEE budget (SPEC 2026-10-28; the implementer makes it "as built")

Two additions, owner request 2026-10-28 (open items of section 10).

### 12.1 Per-pixel-patch visibility hint at the primary vertex

Purpose: the sun / lamp choice (section 3) gets visibility information again. Two internal images
`light_hint_a` / `light_hint_b` (R8G8_UNORM, size `ceil(size / patch)`, patch = 1, 2 or 4 pixels,
`hint_patch_shift` in the frame constants) hold per patch an estimate of the SUN visibility (R) and
the LAMP visibility (G) of the primary vertices of that patch; one is "prev" (trace binding 20),
the other "cur" (binding 21); the renderer swaps the two descriptor writes every lighting render
(a renderer-side parity bit, not the frame number). They live in `Renderer::Internal` after
`kInternalRrHitDistance` (`kTraceInternalCount` 12 -> 14, the sky bindings move to 22 / 23), are
created with the internal targets (recreated when the size OR the patch changes) and are never
cleared: the frame after a creation carries `kLightNeeHintReset` in `light_nee_flags` and the kernel
reads 0.5 instead of "prev".

Read (light NEE kernel only, at the primary vertex, flag `kLightNeeHint`): `hint = reset ? 0.5 :
prev[prev_patch]`, `prev_patch` = the previous pixel of the primary hit (`PreviousPixel`, the same
reprojection as the motion target, nearest) >> shift, clamped to the image; the current patch when
the point was behind the previous camera or there is no previous camera. `NextEventAtVertex` takes
the hint: `w_sun *= max(kHintFloor, hint.r)`, `w_lamp *= max(kHintFloor, hint.g)`, `kHintFloor =
0.05`. The estimator stays unbiased (one-sample RIS with any positive weights; the floor keeps every
candidate with a nonzero contribution selectable). Bounce vertices do not use the hint.

Write: every lane that traced a candidate at the primary vertex contributes its result (sun: visible
0 / 1 when the sun was the kept candidate; lamp likewise). Wave reduction without barriers (the
kernel returns early for out-of-range pixels, so no group barrier): the patch's lanes form a mask
computed arithmetically from the lane layout (`numthreads(8, 8, 1)`, lane = x + 8 (y mod rows), rows
= WaveGetLaneCount() / 8; patch shift <= 2 keeps a patch inside one 32- or 64-lane wave; a 16-lane
wave splits a 4-pixel patch over two waves, last writer wins, acceptable); `WaveActiveBallot` of
(traced sun), (sun visible), (traced lamp), (lamp visible); counts = countbits(mask & ballot); the
lane with `WaveMultiPrefixCountBits(true, mask) == 0` writes `cur[patch] = (sun_count > 0 ?
lerp(prev.r, sun_visible / sun_count, kHintAlpha) : prev.r, same for G)` with `kHintAlpha = 0.2`
and `prev` = the value that lane read. DXC: cs_6_6, `WaveMultiPrefixCountBits` is SM 6.5. Do not
use `WaveMatch` (it needs SPV_NV_shader_subgroup_partitioned).

Debug view `CAELIX_VIEW_LIGHT_HINT` (12, path trace mode): the trace writes `deterministic = (hint.r,
hint.g, 0)` and zero radiance; the composite then shows the hint (kDebugViewCount 13; the host's
View enum gets it).

ABI: `CAELIX_LIGHT_NEE_FLAG_NO_VISIBILITY_HINT (1u << 1)` in CaelixLightNeeOptions::flags; the
`reserved[0]` of CaelixLightNeeOptions becomes `uint32_t hint_patch` (0 = the library default, else
1, 2 or 4; other values are INVALID_ARGUMENT); `CaelixLightNeeStats::reserved[0]` becomes
`hint_patch` (the patch in use, 0 when the hint is off). The library default patch is chosen by the
measurement (section 12.3); until then 2. Host: `--light-no-hint`, `--light-hint-patch N`, the
Stats window shows the patch.

Frame constants: `light_nee_flags` gains `kLightNeeHint (1u << 2)` and `kLightNeeHintReset (1u <<
3)`; `hint_patch_shift` uint @588.

### 12.2 Bounce NEE budget

`CaelixRenderOptions::bounce_nee_probability` (float, the former `reserved_voxel_id[0]`; 0 = 1 =
every vertex, else in (0, 1]; other values INVALID_ARGUMENT): at every bounce vertex that would run
NEE (sun and lamp alike), the NEE runs with probability q (one number of `nee_rng`, drawn only when
q < 1) and its radiance is multiplied by 1 / q; when it does not run, the vertex still behaves as if
it had: the next segment sees no sun disk (sun NEE on) and carries the lamp state
(`kSegmentLampNee` with the MIS bit as usual, light NEE on). Unbiased (E = q (NEE / q)), bounded 1 /
q spikes instead of the fireflies of a BSDF-found sun. The primary vertex always runs its NEE. Frame
constant `bounce_nee_probability` @584. Host: `--bounce-nee-probability` default 0.5 (library default:
every vertex), a slider in the lighting UI.

### 12.3 Measurements (main agent)

Patch size: bistro and emberdeep (and one more scene with sharp sun shadows) at 1080p, path trace
mode without the denoiser, 100-frame means: 1-spp PSNR and 16 x 16 block PSNR against a 1024-frame
reference, for hint off / patch 1 / 2 / 4, plus trace ms. Pick the coarsest patch whose block PSNR is
within 0.1 dB of patch 1 on every scene. Bounce budget: the same metrics for q = 1, 0.5, 0.25 and
the trace ms. Unbiasedness: the ctest "light NEE: unbiased" gains the variants hint on (default),
hint off, and q = 0.25.
