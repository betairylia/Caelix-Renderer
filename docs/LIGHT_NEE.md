# Light NEE (as built, 2026-10-24; visibility hint and bounce NEE budget 2026-10-28)

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
heuristic with the exact pdf of the same draw. The light NEE keeps only one small memory, the
visibility hint of the primary vertex (section 12.1: two R8G8 images per renderer that weight the sun /
lamp choice, never the estimate itself); the samples stay independent per ray (no cell-sized blobs,
no shared reservoirs), so the error is white per ray. Bounce vertices may skip their NEE with the
bounce NEE budget (section 12.2).

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
unoccluded contributions: `pick_sun = u w_sum < w_sun`, and the kept candidate's contribution is
multiplied by `w_sum / w_kept`. So `E = C_s V_s + C_l V_l` stays unbiased. At the primary vertex the
two weights are multiplied by the visibility hint (`w_sun *= max(0.05, hint.r)`, `w_lamp *= max(0.05,
hint.g)`, section 12.1); any positive weights keep the estimator unbiased. The lamp
candidate stays live through the sun's draw (its evaluation is the expensive one); a kept sun candidate
is drawn again from its saved random state. One shadow ray toward the kept candidate (glass occludes,
as for the sun NEE; at the primary vertex with `CAELIX_RENDER_FLAG_SHADOW_TRANSMISSION` the ray passes
glass, attenuated, docs/SHADOW_TRANSMISSION.md). The segment that leaves the vertex sees the sky without the sun disk when the sun
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
(offset 552): `kLightNeeOn` (bit 0) = the light NEE runs, `kLightNeeMis` (bit 1) = with MIS,
`kLightNeeHint` (bit 2) = the visibility hint runs, `kLightNeeHintReset` (bit 3) = the previous hint
image has no history (section 12.1). The frame constants are now 608 bytes; the hint and the budget use
`bounce_nee_probability` (584) and `hint_patch_shift` (588), which existed before (section 12). `caelix_frame_layout_test` checks the uniform block of both trace kernels and the four other
frame-constant kernels.

Registers (`CAELIX_PIPELINE_STATS=1`, RTX 5080, driver 610.88): `caelix_path_trace` 128 registers and
16 bytes of spill (driver binary 174976 bytes, shared memory 2560 bytes), `caelix_path_trace_no_light_nee`
96 and 48 bytes of spill (the first shadow transmission step lifted both to 128 registers + 16 bytes,
docs/SHADOW_TRANSMISSION.md "History"; since the subsurface step, which moved every SSS code path into the
variants `caelix_path_trace_sss` and `caelix_path_trace_sss_no_light_nee`, they are again 128 + 16 bytes
and 96 + 48 bytes: docs/SUBSURFACE.md section 6). Both equal the kernels of `18282cd`: the SPIR-V function bodies are the same
instructions modulo id numbering, except the two flag bit values (`kLightNeeContribute` 2 -> `kLightNeeOn`
1, `kLightNeeMis` 8 -> 2); `Artifacts/light-nee-variants-20261024/cleanup/spirv/compare.txt`. Since the visibility
hint and the bounce NEE budget (2026-10-28) the numbers are in section 12.4 (no register boundary
crossed; `_no_light_nee` 96 + 64 bytes).

## 7. ABI and host

```c
#define CAELIX_LIGHT_NEE_FLAG_NO_MIS (1u << 0)             /* the exclusive rule instead of the balance heuristic */
#define CAELIX_LIGHT_NEE_FLAG_NO_VISIBILITY_HINT (1u << 1) /* the primary vertex's choice without the hint */

typedef struct CaelixLightNeeOptions
{
    uint32_t enabled;    /* 0 = off (the default), 1 = on */
    uint32_t flags;      /* CAELIX_LIGHT_NEE_FLAG_* */
    uint32_t hint_patch; /* the visibility hint's patch in pixels: 0 = the library default (4), 1, 2 or 4 */
    uint32_t reserved;   /* must be 0 */
} CaelixLightNeeOptions; /* 16 bytes */

typedef struct CaelixLightNeeStats
{
    uint32_t enabled;    /* 1 while the options enable light NEE */
    uint32_t running;    /* 1 when the newest render call ran it (a lighting mode, a light tree) */
    uint32_t hint_patch; /* the hint's patch of the newest render call (1, 2, 4); 0 = the hint did not run */
    uint32_t reserved;   /* 0 */
} CaelixLightNeeStats;   /* 16 bytes */

/* CaelixRenderOptions: */
float bounce_nee_probability; /* 0 = 1 = every bounce vertex runs its NEE; else in (0, 1] */
#define CAELIX_VIEW_LIGHT_HINT 12 /* CaelixDebugView, path trace mode */

CaelixResult caelix_renderer_set_light_nee_options(CaelixRenderer*, const CaelixLightNeeOptions*);
void caelix_renderer_get_light_nee_stats(const CaelixRenderer*, CaelixLightNeeStats*);
```

Per renderer; the options apply from the next render call. `enabled` above 1, unknown flag bits, a
`hint_patch` other than 0, 1, 2 or 4, or a nonzero `reserved` word return
`CAELIX_ERROR_INVALID_ARGUMENT` (logged) and change nothing. The library default is off (and with it on:
MIS, the hint at patch 2, every bounce vertex runs its NEE). ABI v2 is in development: the structs
shrank from 32 / 96 bytes without a version bump (the Rust host is the only user), and the hint step
turned `reserved[0]` of both into `hint_patch` (still 16 bytes). The light NEE's only GPU memory is the
pair of hint images (section 12.1).

Rust host (`caelix-app`): light NEE on by default (it also enables the light tree); `--no-light-nee`
turns it off, `--light-no-mis` uses the exclusive rule, `--light-no-hint` turns the visibility hint off,
`--light-hint-patch N` sets its patch (0 = the library default), `--bounce-nee-probability P` sets the
bounce NEE budget (host default 0.5; the library default is 1), `--view light-hint` shows the hint.
`--screenshot-frame N` takes the screenshot at frame number N (the frame seed), so runs with different
options or binaries capture the same random numbers. The Stats window (F1) has a "Light NEE" section
(enabled, running, MIS, hint patch), the Render window a "bounce NEE p" slider; `--timing-frames` adds
`light_nee: running=<0|1>` to the `timings_avg:` line.

## 8. Tests

- `caelix_trace_test`, "light NEE: unbiased against the path tracer (lamps, sun, glass)": a closed grey
  box with single-voxel lamps (some inside the near field of a pillar), a sun opening half covered by
  clear glass; reference mode, 4 bounces, 4 transparent bounces; light NEE (1024 frames) against the
  plain path tracer (4096 frames), lamps alone and with the sun, for the defaults (MIS) and for the
  exclusive rule. Tolerances: mean ratio 0.97 .. 1.03, 16 x 16 block PSNR above 30 dB. Measured
  (2026-10-24): ratio 0.9983 (sun 0) and 0.9984 (sun 1000) for both, block PSNR 40.1 / 40.2 and
  39.6 / 39.6 dB, against the path tracer's own floor (other seeds) 0.9982 / 0.9986, 37.6 / 37.7 dB.
  Since 2026-10-28 five variants (the hint and the budget run in the reference mode too): defaults
  (MIS, hint at patch 2), no MIS, no hint, hint patch 4, bounce NEE probability 0.25; the numbers are
  in section 12.4.
- `caelix_trace_test`, "light NEE: visibility hint converges and reprojects": section 12.4.
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
- `caelix_frame_layout_test`: the uniform block (608 bytes since the subsurface steps) of the frame-constant
  kernels.
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

- Built 2026-10-28: the per-pixel-patch visibility hint at the primary vertex (sun and lamps) and the
  bounce NEE budget (section 12); the patch default and the budget's effect wait for the measurements
  of section 12.3.
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

## 12. Visibility hint and bounce NEE budget (as built 2026-10-28)

Two additions, owner request 2026-10-28 (open items of section 10). The spec of the same day is
implemented with the deviations listed in 12.1 and 12.2.

### 12.1 Per-pixel-patch visibility hint at the primary vertex

Purpose: the sun / lamp choice (section 3) gets visibility information again, without the cells'
correlated error: the hint only changes the probabilities of the choice, never the estimate.

Images: `Renderer::Internal` has `kInternalLightHintA` and `kInternalLightHintB` right after
`kInternalRrHitDistance` (`kInternalSpecs` "light_hint_a" / "light_hint_b", R8G8_UNORM): R = the SUN
visibility, G = the LAMP visibility of the primary vertices of one patch of `patch x patch` pixels. Size
`((width + patch - 1) >> shift, (height + patch - 1) >> shift)`, every other internal image keeps the
render size. `kTraceInternalCount` 12 -> 14: trace binding 20 = the previous image (read), 21 = the
current one (written); the sky moved to 22 / 23. `EnsureInternalTargets(width, height, hint_shift,
frame)` recreates all internal targets when the size or the shift changes (member
`internal_hint_shift_`). The shift comes from the options' patch whether the hint runs or not
(`Renderer::LightNeeHintShift`), so switching the hint (or the light NEE) on and off keeps the targets;
only a new patch recreates them (and restarts the reference accumulation like a size change). The
renderer member `hint_parity_` chooses which image is bound as the previous one; it flips after every
render that ran the hint (not on other frames, so the history survives frames without the hint). The
images are never cleared: `hint_history_valid_` is false after a creation (and before any render ran the
hint), and such a frame, or a render without a previous camera, carries `kLightNeeHintReset`.

Frame constants: `hint_patch_shift` @588 = the shift in use (0 when the hint does not run);
`light_nee_flags` gains `kLightNeeHint (1u << 2)` (light NEE runs and the options lack
`CAELIX_LIGHT_NEE_FLAG_NO_VISIBILITY_HINT`) and `kLightNeeHintReset (1u << 3)`. The kernels without light
NEE fold all of it away (`light_nee_flags` is 0 there).

Read (`ReadLightHint`, in `main` for every in-range lane): with the reset flag about (0.5, 0.5); else the
previous image at `prev_patch`: the previous pixel of the PRIMARY SURFACE (`chain.surface`, the hit that
the delta chain ended on, moved by `EntityPrevWorld` and projected by `PreviousPixel`, nearest pixel)
`>> shift` when that pixel lies inside the image; the lane's own patch when there is no surface, no
previous camera, or the point was behind the previous camera or outside its image. `TracePath` passes
the hint to the primary vertex's `NextEventAtVertex`, which multiplies `w_sun` by `max(kHintFloor,
hint.r)` and `w_lamp` by `max(kHintFloor, hint.g)`, `kHintFloor = 0.05`. Bounce vertices pass (1, 1)
(the weights unchanged, the same random numbers). `NextEventAtVertex` also returns which shadow ray it
traced (`kNeeTracedNone`, `kNeeTracedSun`, `kNeeTracedLamp`) and whether it reached its candidate
(`TraceShadow` found nothing, or the transmissive ray's transmittance has a channel above 0).

Write (`WriteLightHint`, in `main` after the `if (surface)` block, where every active lane of the wave
arrives): `PatchWaveMask` builds the ballot mask of the lanes of this pixel's patch arithmetically
(`numthreads(8, 8, 1)`, lane = x + 8 (y mod rows), rows = `WaveGetLaneCount() / 8`, the wave's first
group row = `(y & 7) - lane / 8`; one byte mask per row, an unrolled loop over at most 8 rows).
`WaveActiveBallot` of (traced sun), (sun visible), (traced lamp), (lamp visible); counts =
`countbits(mask & ballot)`. The lowest active lane of the patch writes `cur[patch] = (sun_count > 0 ?
lerp(prev.r, sun_visible / sun_count, kHintAlpha) : prev.r, the same for G)`, `kHintAlpha = 0.2`, `prev`
= the value that lane read. The write also happens in a reset frame (it seeds the images). A wrong
lane layout only weakens the hint (the estimator stays unbiased); a 16-lane wave splits a 4-pixel patch
over two waves (last writer wins).

Debug view `CAELIX_VIEW_LIGHT_HINT` (12, path trace mode; `kDebugViewCount` 13): surface pixels get
`deterministic = (hint.r, hint.g, 0)` and zero radiance in both NRD targets, so the composite shows the
hint the primary vertex read (exposure applied; 0.5 where the hint does not run). Host `--view
light-hint` (View::LightHint, "light hint (pt)").

Deviations from the spec:
- The "first lane of the patch" test is `countbits(mask & WaveActiveBallot(true) & lanes_below) == 0`
  instead of `WaveMultiPrefixCountBits(true, mask) == 0`: DXC's SPIR-V backend lowers the
  WaveMultiPrefix* intrinsics to the partitioned group operations of SPV_NV_shader_subgroup_partitioned,
  the same extension that rules out `WaveMatch`. The ballot form gives the same lane.
- The reduction lives in `main`, not in `TracePath`: `main` calls `TracePath` inside the divergent `if
  (surface)`. `TracePath` returns the primary NEE's shadow ray in one word (`nee_result` = `kNeeTraced*
  | kNeeResultVisible`) and `main` reads the hint for every lane (a pixel without a surface reads its
  own patch: it may be the lane that writes the patch).
- The hint travels through the path packed into one word (`PackHint` / `UnpackHint`: the two 8-bit
  values of the RG8 image, exact because the image stores k / 255), and the reset value is 128 / 255
  (0.502) instead of 0.5: two words instead of four registers live across the bounce loop (the
  driver's shared memory of `caelix_path_trace` 4352 -> 3840 bytes). Without the hint the word is
  (1, 1), so the choice weights are bit for bit the ones before the hint; the debug view then shows
  0.5.
- The images are swapped only after renders that ran the hint (the spec: every lighting render), and
  the reset flag also covers "the hint has not run since the creation" (`hint_history_valid_`).
- `CaelixLightNeeStats::hint_patch` reports the patch the internal targets were sized for in the
  newest render that ran the hint (`1 << internal_hint_shift_`), 0 when that render did not run it.

### 12.2 Bounce NEE budget

`CaelixRenderOptions::bounce_nee_probability` q (0 = 1 = every vertex; else in (0, 1]; other values
INVALID_ARGUMENT; the renderer writes 1 for 0 into the frame constant @584). In the bounce loop of
`TracePath`, a vertex that would run its NEE (`!shade.specular_lobe && bounce_index <= bounce_count`,
and sun NEE or light NEE on) first sets the state of a vertex that ran it (`segment_sees_sun = !nee`; with
light NEE `segment_light = kSegmentLampNee | kSegmentLampMis` with MIS), then runs the NEE when `q >= 1`
or `RandomFloat01(nee_rng) < q` (one number of the sun stream, drawn only when q < 1) and multiplies its
radiance by `1 / q` (exactly 1 for q = 1). Unbiased: E = q (NEE / q); the MIS of a lamp hit after a
skipped vertex keeps the BSDF weight of a vertex that ran the NEE, whose expected NEE weight is the
complement. The primary vertex always runs its NEE. Host `--bounce-nee-probability` (default 0.5),
`RenderSettings::bounce_nee_probability`, a "bounce NEE p" slider (0.05 .. 1) in the Render window.
Deviation: none (the "as if it ran" state is set before the draw instead of in an else branch; same
result, fewer branches).

### 12.3 Measurements (2026-10-28)

RTX 5080, driver 610.88, Rust host, 1920x1080, path trace mode, no denoiser, 2 bounces, sun 2e4,1.9e4,1.8e4,
light NEE on; per config two runs with `--warmup 64 --timing-frames 100` (100-frame trace means) at the
screenshot frames 400 / 401, PSNR and 16 x 16 block PSNR (sRGB, peak 255) of that 1-spp frame against a
1024-frame reference (hint off, q 1), and the linear mean ratio of a 512-frame accumulation against the
reference (bias). `Artifacts/light-hint-budget-20261028/` (`summary.md`, `results.json`, the run logs and
images, the script `measure.py`). The reference's own noise sets the bias floor (emberdeep 0.998 in every
config).

| scene | config | trace ms (2 runs) | PSNR dB | block PSNR dB | ratio 512 / ref |
|---|---|---|---|---|---|
| bistro | hint off | 7.94, 8.01 | 12.04 | 16.88 | 1.0001 |
| bistro | patch 1 | 8.11, 8.03 | 12.30 | 17.44 | 1.0000 |
| bistro | patch 2 | 7.94, 7.98 | 12.32 | 17.48 | 1.0000 |
| bistro | patch 4 | 7.88, 7.93 | 12.33 | 17.52 | 1.0000 |
| bistro | patch 2, q 0.5 | 7.50, 7.49 | 12.11 | 17.03 | 1.0000 |
| bistro | patch 2, q 0.25 | 7.15, 7.16 | 12.01 | 16.69 | 1.0000 |
| emberdeep | hint off | 7.05, 6.99 | 16.45 | 20.49 | 0.9978 |
| emberdeep | patch 1 | 7.06, 7.03 | 16.60 | 20.75 | 0.9978 |
| emberdeep | patch 2 | 7.04, 7.03 | 16.60 | 20.75 | 0.9979 |
| emberdeep | patch 4 | 7.02, 7.03 | 16.60 | 20.75 | 0.9979 |
| emberdeep | patch 2, q 0.5 | 6.67, 6.69 | 16.47 | 20.52 | 0.9978 |
| emberdeep | patch 2, q 0.25 | 6.37, 6.37 | 16.39 | 20.36 | 0.9977 |
| rootfall | hint off | 8.13, 8.18 | 13.07 | 19.53 | 1.0000 |
| rootfall | patch 1 | 8.19, 8.19 | 13.08 | 19.55 | 1.0000 |
| rootfall | patch 2 | 8.19, 8.16 | 13.08 | 19.55 | 1.0000 |
| rootfall | patch 4 | 8.17, 8.17 | 13.08 | 19.54 | 1.0000 |
| rootfall | patch 2, q 0.5 | 8.51, 8.45 | 12.91 | 19.15 | 1.0000 |
| rootfall | patch 2, q 0.25 | 8.19, 8.17 | 12.84 | 18.85 | 1.0000 |

Hint: +0.56 dB block PSNR on bistro (sun shadows and many lamps), +0.26 dB on emberdeep (the cave: the
lamp channel), +0.02 dB on rootfall; no measurable cost; no bias. Patch 4 is within 0.1 dB of patch 1 on
every scene (on bistro it is 0.08 dB better: the smoother hint), so the library default is 4
(`kDefaultHintPatch`); patch 1 costs about 0.1 ms more on bistro (the write of 16x more texels).

Bounce budget: q 0.5 saves 0.45 ms on bistro and 0.35 ms on emberdeep for -0.45 / -0.23 dB block PSNR;
q 0.25 saves 0.8 / 0.65 ms for -0.8 / -0.4 dB; no bias. On rootfall q 0.5 is SLOWER (+0.3 ms) and q 0.25
equal: a per-vertex decision leaves the lanes that skip their NEE idle while the other lanes of the wave
trace (the wave's cost is its slowest lane), so the saving appears only where the skipped work is the
lamp tree descent and the shadow ray of many lanes at once. A wave-coherent decision (the measurement hook
`CAELIX_SPEC_BOUNCE_NEE_COHERENT`: one decision per 8 x 8 workgroup, frame and bounce) is measured below;
it trades the per-ray white error for a per-frame blocky pattern of the bounce NEE's variance. Host
default stays q 0.5; the owner decides on the coherent form.

Coherent decision (`CAELIX_SPEC_BOUNCE_NEE_COHERENT`, the embedded kernel replaced through
`CAELIX_SHADER_DIR`, patch 2, q 0.5, the same protocol): bistro 8.00 / 7.97 ms median (per vertex 7.50,
q 1 7.96), emberdeep 6.89 / 6.90 (per vertex 6.68, q 1 7.03), rootfall 8.19 / 8.19 (per vertex 8.48, q 1
8.18); PSNR / block PSNR within 0.03 dB of the per-vertex form. So the coherent decision removes the
rootfall loss but also the bistro gain: the idle-lane hypothesis does not explain the numbers (the trace
cost of a wave is not the shadow rays of its slowest lane alone). The per-vertex form stays (white error
per ray); the hook stays for later experiments. Net: q 0.5 is worth 0.35 .. 0.45 ms where the bounce NEE
is expensive (many lamps or long shadow rays) and nothing elsewhere; the owner may lower the host
default to 1 if the -0.2 .. -0.45 dB matter more than the time.

### 12.4 Registers and tests

Registers (`CAELIX_PIPELINE_STATS=1`, RTX 5080, driver 610.88, `--test-entity`; registers + spill bytes,
driver shared memory):

| kernel | before (docs/SUBSURFACE.md section 6) | budget only (hint compiled out) | as built |
|---|---|---|---|
| `caelix_path_trace` | 128 + 16 B, 2560 B | 128 + 16 B, 3328 B | 128 + 16 B, 3840 B |
| `caelix_path_trace_no_light_nee` | 96 + 48 B, 4864 B | 96 + 64 B, 4864 B | 96 + 64 B, 4864 B |
| `caelix_path_trace_sss` | 128 + 32 B, 6144 B | 128 + 32 B, 6144 B | 128 + 64 B, 6144 B |
| `caelix_path_trace_sss_no_light_nee` | 128 + 16 B, 4608 B | 128 + 16 B, 5120 B | 128 + 16 B, 5120 B |

No kernel crosses a register boundary. The budget costs 16 bytes of spill in `_no_light_nee` and shared
memory in two kernels; the hint (packed, 12.1) adds 512 bytes of shared memory to `caelix_path_trace`
and 32 bytes of spill to `caelix_path_trace_sss`. Before packing, the hint's unpacked state gave 4352
bytes of shared memory in `caelix_path_trace`. Whether the shared memory costs occupancy at 128
registers is for the timing of 12.3.

Tests (`caelix_trace_test`, 2026-10-28):
- "light NEE: unbiased against the path tracer": five light NEE variants against the path tracer's
  4096-frame reference (path tracer floor with other seeds: ratio 0.9982 / 0.9986, block PSNR 37.62 /
  37.70 dB for sun 0 / sun 1000):

  | variant | sun 0: ratio, block PSNR | sun 1000: ratio, block PSNR |
  |---|---|---|
  | defaults (MIS, hint patch 2) | 0.9981, 40.11 dB | 0.9982, 39.48 dB |
  | no MIS | 0.9981, 40.13 dB | 0.9982, 39.50 dB |
  | no hint | 0.9981, 40.11 dB | 0.9983, 39.55 dB |
  | hint patch 4 | 0.9981, 40.11 dB | 0.9982, 39.48 dB |
  | bounce NEE 0.25 | 0.9982, 40.22 dB | 0.9984, 39.54 dB |

  Without a sun the hint cannot change the choice (one candidate), so "no hint" equals the defaults; in
  the closed box most receivers have the sun hint at the floor for every patch size, so patch 4 and 2
  pick the same candidates almost everywhere.
- "light NEE: visibility hint converges and reprojects": the lamp box with a 0.25-degree sun of the
  same energy as the 2-degree test sun. The direct sun of every primary vertex = (reference, light NEE
  off, one bounce, 64 frames) with the sun minus without it, same seeds (the BSDF path's lamps consume the
  same random numbers, so the difference is exactly the primary NEE's sun). A lit pixel (5 x 5
  neighbourhood above 0.9 of the brightest non-lamp value, upward-facing) and a shadowed one (5 x 5
  exactly 0, upward-facing); then 64 path trace frames without the denoiser in the hint view, exposure
  1: lit sun hint 0.992 (> 0.8), shadowed 0.008 (< 0.2), G finite in [0, 1] everywhere (lamp hints 0.69
  and 0.57 there). The camera then turns by one pixel per frame for 8 frames (yaw about the camera
  position, so a world direction maps to a pixel exactly): the points moved 6.4 px, their sun hints
  stay 0.992 and 0.008 (within 0.2). Patches 1, 2 and 4 run (stats report them), the hint switched off
  reports 0 and the view shows 0.5, `hint_patch` 3 is rejected.
- "light NEE: lifecycle": the option checks use the new layout (unknown flag bit 1 << 2, `hint_patch`
  3, `reserved` 7).
