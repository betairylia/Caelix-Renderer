# Shadow rays through transparent and subsurface voxels (primary vertex)

Status: AS BUILT, 2026-10-28 (spec of the same day; the deviations and measurements are in section 8,
"As built"). Since the subsurface step of the same day (docs/SUBSURFACE.md) the SSS parts of this ray
(section 1's SSS rules, the SSS-aware probe, the straight SSS walk) are compiled only into the trace
kernel variants `caelix_path_trace_sss` and `caelix_path_trace_sss_no_light_nee` (section 9). Owner request: shadow rays (sun and lamp) at the PRIMARY hit vertex pass through
transparent voxels and through subsurface-scattering (SSS) opaque voxels in a straight line (no
reflection, no refraction, as if one medium), attenuated by the glass extinction (tinted glass leaves
coloured shadows) and by the SSS albedo, and die when they travel farther in SSS voxels than a rough
estimate of a random walk's reach. Bounce vertices keep the hard any-hit shadow ray.

## 1. Model

`T(ray) in [0, 1]^3` replaces the boolean occlusion of the shadow ray at the primary vertex:

- An opaque voxel that is not an SSS voxel occludes (`T = 0`).
- A transparent medium of id `g` (a glass palette / block-ID transparent code; 0 = air) attenuates a
  segment of length `t` inside it by `exp(-(1 - albedo) * extinction * t)` (the same formula as
  `MediumExtinction`; the air's fog is ignored: shadow rays ignore fog, as documented in caelix.h).
- An SSS voxel (opaque code with `sss_model == CAELIX_SSS_MODEL_HG` and `sss_probability > 0`,
  docs/SUBSURFACE.md) attenuates a segment of length `l` (voxel units = world units, rigid entities)
  by `sss_albedo^l` (the walk multiplies by the albedo once per voxel step) and adds `l / L` to the
  ray's SSS budget, `L` = the material's shadow length (section 2). The budget starts at 0; at 1 the
  ray dies (`T = 0`). Entering an SSS run from a non-SSS voxel (air, glass, the ray's start)
  multiplies by `sss_probability` once (the fraction of the light that enters the medium in the
  walk model; the rest is reflected by the surface BSDF).
- A ray whose transmittance falls below `kShadowTransmitCutoff` (max channel < 1e-3) is occluded.

The straight line is an approximation (no refraction). Its interaction with the BSDF path:

- Sun: after a transmissive NEE at the primary vertex that included the sun candidate (sun NEE on),
  the path CLAIMS the sun through every transparent (delta) vertex that follows: the segments see the
  sky without the sun disk until the first opaque vertex, where the usual rule resumes (NEE vertex:
  no disk; specular-lobe vertex: the disk). Without the claim the sun through glass would be counted
  twice (straight NEE + refracted BSDF path). SSS is not claimed: a walk exit (docs/SUBSURFACE.md)
  runs its own NEE, so thin SSS objects can count the sun up to twice in their shadow (the budget
  kills the straight ray through thick objects). Owner decision 2026-10-28.
- Lamps: through the same claimed chain the segment state stays `kSegmentLampNee` WITHOUT the MIS
  bit (the exclusive rule): a lamp that the light tree can sample, hit after a delta vertex, counts
  0; a lamp the tree cannot sample, or a hit inside the near field (`hit.T < kLightNeeNearField`),
  counts in full. The MIS pdf of the receiver is not carried across interfaces (the receiver point
  would differ). Known double count: a lamp farther than the near field from the receiver whose last
  segment after the glass is shorter than the near field (rare; documented, accepted). Known loss: the
  BSDF share of the balance heuristic for a lamp seen through glass (the NEE keeps only its own MIS
  weight; measured 0.1 % in test 3, section 5).
- As built, the claim starts whenever the primary vertex ran a transmissive NEE, sun or lamp (section 8,
  deviation 1): the sun part needs the sun NEE, the lamp part the light NEE.

## 2. Material data

The subsurface GPU entry (docs/MATERIALS.md) gains two values; the ABI struct does not change:

| Word | Offset | Content |
| --- | --- | --- |
| w2 | 8 | `sss_steps & 0xFF` \| `h(shadow_length) << 16` |
| w3 | 12 | `h(sss_probability)` \| `sss_model << 16` (a mirror of hot w5, so one 16-byte load gives a voxel's SSS state) |

`shadow_length = sqrt(sss_steps * (1 + g) / (1 - g))` voxels with `g = clamp(sss_anisotropy, -0.95,
0.95)` (the RMS end-to-end distance of a persistent random walk of `sss_steps` steps; owner
decision: derived, no new field); 0 when `sss_steps == 0`. Computed by `SubsurfaceShadowLength` and
packed by `PackMaterialSubsurface` (lib/src/scene/material_pack.h; C++ constants
`kSubsurfaceProbability` 12, `kSubsurfaceShadowLengthShift`, `kSubsurfaceModelShift`,
`kSubsurfaceShadowMaxAnisotropy`). `CaelixSubsurface` (shaders/common/caelix_material.hlsl) gained
`shadow_length`, `probability`, `model`; offset macro `kSubsurfaceProbability 12`; model macros
`kSssModelNone 0u`, `kSssModelHg 1u` (mirrors of `CAELIX_SSS_MODEL_NONE` / `CAELIX_SSS_MODEL_HG` in
include/caelix/caelix.h; the layout test compares them). `MaterialTable` counts the SSS entries
(`subsurface_count_`, kept by `Store`: an opaque code with model HG, probability > 0 and steps > 0);
`has_subsurface()` is count > 0; `Scene::has_subsurface()` exposes it; the frame constants carry it
(`subsurface_present`).

## 3. Code layout (the boundary of caelix_scene_trace.hlsl is kept)

- `shaders/scene/caelix_scene_lookup.hlsl`: `SceneBrickAt(s, inout group, entity, int3 brick)` (the
  record of the brick at an entity-local brick coordinate: same group -> reverse table; else the
  entity's group grid; 0 = absent), taken unchanged from branch `sss-voxel-color-expr`.
- `shaders/render/caelix_voxel_cursor.hlsl` (new, data layer, included by caelix_scene_trace.hlsl
  after TraceResult): `VoxelCursor {uint group; uint entity; uint64_t record; int3 voxel;}`,
  `VoxelCursorFromHit(fc, TraceResult)` = the brick origin as TraceCandidate derives it, times 8, plus
  `hit.cell` (exact, no floor of object_hit), `VoxelCursorStep(fc, inout c, int3 delta)` (a brick
  crossing goes through SceneBrickAt; false = absent brick; once absent the cursor stays absent),
  `VoxelCursorBlock(c)` (the payload word read; 0 in an absent brick; the payload holds interior
  voxels too), `FrameSceneLookup(fc)`, and the entity transform helpers by entity id
  (`EntityWorldDirectionById`, `EntityObjectDirectionById`, `EntityWorldPointById`). The frame
  constants gained `reverse_tables` and `group_grids` (section 6; `Scene::reverse_tables_address`,
  `Scene::group_grids_address`).
- `shaders/render/caelix_shadow_medium.hlsl` (new, the material rules of section 1, reads the
  material tables through `g_materialTable` / `g_subsurfaceTable`):
  `struct ShadowMedium {float3 transmittance; float sss_budget;}`, `ShadowMediumInit()`,
  `ShadowGlassSegment(inout m, uint medium_id, float t)` (id 0: nothing),
  `ShadowSssSegment(inout m, uint block, float l)` (one 16-byte load of w0..w3: `T *=
  exp2(log2(max(albedo, 1e-4)) * l)`, `budget += L > 0 ? l / L : 1`),
  `ShadowSssEnter(inout m, uint block)` (times probability), `ShadowVoxelIsSubsurface(uint block)`
  (opaque, model HG, probability > 0: ONE 4-byte load of subsurface w3), `ShadowMediumDead(m)`
  (budget >= 1 or max channel < `kShadowTransmitCutoff`).
- `shaders/render/caelix_ray.hlsl` (new): `K_RAY_ORIGIN_PUSH_OFF`, moved out of caelix_shading.hlsl so
  that caelix_scene_trace.hlsl can push new ray origins off voxel faces (section 8, deviation 3).
- `shaders/render/caelix_brick_trace.hlsl` (NATIVE change 5): `CaelixShouldTerminateBrickRay(index,
  blockID, normalFlags, entryT, uint mode, inout bool medium_seen)`; `CaelixTraceBrickRay` and
  `CaelixTraceBrickPrimitiveCore` take `uint mode, inout bool medium_seen` before `attrib`. Modes:
  `kBrickDdaClosest` (0, Unity's rule), `kBrickDdaShadowOpaque` (1: only opaque voxels terminate; a
  visited transparent voxel with id != 0 sets `medium_seen`; an air cell with face bits does not),
  `kBrickDdaShadowSssAware` (2: as 1, and an opaque voxel for which `CaelixShadowPassesOpaque(block)` is
  true does not terminate and sets `medium_seen`). `CaelixShadowPassesOpaque` is a PROTOTYPE in this
  file, defined in caelix_scene_trace.hlsl as `ShadowVoxelIsSubsurface`: DXC 1.8 accepts the prototype
  with the later definition. Every caller passes a literal mode, so the closest-hit DDA compiles to the
  same code as before (checked: the function bodies of the caelix_voxel_id kernel, which uses only
  TraceScene, are identical to `b34452d` instruction for instruction modulo id numbering).
- `shaders/render/caelix_scene_trace.hlsl`:
  - `TraceCandidate(fc, group, id, origin, direction, t, uint mode, inout bool medium_seen, out attrib)`
    plus the old seven-parameter overload (mode 0, a dummy flag), so `TraceScene` and `TraceShadow` are
    unchanged in source; `TraceShadow` stays the bounce vertices' shadow ray.
  - `TraceShadowProbe(fc, ray, bool sss_aware, out bool medium_seen) -> bool` = the any-hit query of
    TraceShadow with DDA mode 1 or 2 (a run-time branch per candidate between the two literal modes).
    `medium_seen` is conservative: a candidate's DDA also visits voxels beyond TMax.
  - `float3 TraceShadowTransmissive(fc, RayDesc ray, uint medium)`: `medium` = the transparent id the
    ray starts in (0 air). Step 1: the probe (`sss_aware = fc.subsurface_present != 0`); hard occluded
    -> 0; not occluded and no medium seen and `medium == 0` -> 1 (the common case costs one any-hit
    query, as today). Step 2 (the chain): `ShadowMedium m`; `cur = medium`; `remaining = ray.TMax`; up
    to `kShadowMaxHops` (8) closest-hit queries (`TraceScene` with TMax = remaining): a miss -> if `cur
    != 0` and `remaining < kShadowUnboundedT` (1e29) apply `ShadowGlassSegment(cur, remaining)`; return
    `m.transmittance` (0 when dead). A hit at `h.T`: `ShadowGlassSegment(cur, h.T)` (dead -> 0);
    `remaining -= h.T`; block = `h.attrib >> 16`:
    - transparent (an interface voxel: glass entry, glass exit = an air cell with face bits, glass to
      glass): `cur = CaelixTransparentId(block)`, origin = hit point `- K_RAY_ORIGIN_PUSH_OFF *
      h.normal` (into the entered voxel; `h.normal` faces the ray), `remaining -= dot(offset,
      direction)` (section 8, deviation 4), continue;
    - opaque, not SSS: return 0;
    - opaque SSS: the straight voxel walk: `VoxelCursorFromHit`, `p = h.object_hit` (on the entry
      face), `d = normalize(EntityObjectDirectionById(entity, ray.Direction))`, `ShadowSssEnter`; then
      up to `kShadowMaxVoxelSteps` (128) voxels: block_v = `VoxelCursorBlock`; not opaque -> EXIT (`cur
      = CaelixTransparentId(block_v)`, origin = `EntityWorldPointById(entity, p) + K_RAY_ORIGIN_PUSH_OFF
      * EntityWorldDirectionById(entity, exit face normal)` (the axis of the last face crossed, its side
      the sign of d on it), `remaining -= dot(offset, direction)`, continue the hop loop); opaque and
      not SSS -> return 0; SSS: `l` = the distance along `d` to the voxel's exit face (`bound`, `t3`,
      `axis` as the SGGX walk of `sss-voxel-color-expr`), `l = min(max(l, 0), remaining)`;
      `ShadowSssSegment(block_v, l)`; dead -> return 0; `remaining -= l`; `remaining <= 0` -> return
      `m.transmittance`; three steps in a row with `l < 1e-6` -> return 0 (`kShadowMaxZeroSteps`); `p +=
      l d`, the crossed coordinate snapped onto the face plane, `VoxelCursorStep` across that face (an
      absent brick reads block 0 next = air = exit). After 128 voxels: 0.
    - after the hop cap: 0.
  - The comment block at the top of the file names the new exports (`TraceShadowTransmissive`, the
    voxel cursor and the entity helpers, used by docs/SUBSURFACE.md's walk).
- `shaders/render/caelix_path_trace.hlsl`: `SunNextEvent(fc, origin, normal, bool transmissive, uint
  medium, inout rng)` and `NextEventAtVertex(fc, x_r, n_r, sun, bool transmissive, uint medium, ...)`:
  with `transmissive` the shadow ray is `TraceShadowTransmissive` and the kept candidate's
  contribution is multiplied by it (the sun / lamp choice still uses the unoccluded contributions);
  else `TraceShadow` as before. The primary vertex passes `transmissive = (fc.flags &
  kRenderFlagShadowTransmission) != 0` and TracePath's `medium` (after the primary shade; an opaque
  primary leaves it unchanged); bounce vertices pass the literal `false, 0u`, so their code is the code
  of before. The claim: `bool claim` set at the primary when transmissive and the primary NEE ran (sun
  or light NEE); in the bounce loop, after the reset of the segment state, at a hit whose block is
  transparent while `claim`: `segment_sees_sun = !nee` and `segment_light = light_nee ?
  kSegmentLampNee : 0` (no MIS bit), the claim stays; at an opaque hit `claim = false` before the
  vertex's own rule. The kernel variant without light NEE compiles the same code (`segment_light`
  folds to 0 there). Every SSS part above (`ShadowVoxelIsSubsurface`, the `kBrickDdaShadowSssAware` branch
  of the probe, the straight SSS walk of the chain) is inside `#if CAELIX_SUBSURFACE`; the main kernels
  compile it out (section 9).

## 4. ABI and host

- `CAELIX_RENDER_FLAG_SHADOW_TRANSMISSION (1u << 6)` in CaelixRenderOptions::flags: lighting modes;
  the primary vertex's NEE shadow rays (sun and lamp) use the transmissive trace. HLSL mirror
  `kRenderFlagShadowTransmission`. The renderer keeps the bit only in the lighting modes
  (`renderer_render.cpp`).
- `CAELIX_SSS_MODEL_NONE 0u`, `CAELIX_SSS_MODEL_HG 1u` next to `CaelixMaterial` (other values act as
  none). The "Materials" comment and the `sss_model` field comment say that the shadow ray reads the
  subsurface entry and that the walk comes later (since the subsurface step they describe the walk,
  docs/SUBSURFACE.md).
- `CaelixRenderOptions::bounce_nee_probability` (float, the former `reserved_voxel_id[0]`; `reserved[2]`
  follows; still 208 bytes): 0 = 1, else in (0, 1]; other values (NaN included) return
  CAELIX_ERROR_INVALID_ARGUMENT from `caelix_renderer_render`. No kernel reads it yet
  (docs/LIGHT_NEE.md section 12); the frame constant gets `p == 0 ? 1 : p`.
- Rust host (caelix-app): on by default; `--no-shadow-transmission` turns it off; the checkbox "shadow
  transmission" next to "sun NEE" in the render settings UI (enabled in the lighting modes);
  `settings.rs` field `shadow_transmission`; the startup line prints `shadow transmission true|false`.
  The host passes `bounce_nee_probability` 0.
- Unity plugin: no packet change (a flag bit; `bounce_nee_probability` arrives as the 0 of the former
  reserved word); the C# side may set the flag later.

## 5. Tests (tests/trace_test.cpp)

1. "shadow transmission: tinted glass slab": a white floor (rgb5 31, albedo 1), the sun straight up (2
   degrees, radiance 1000, sun NEE on, no light tree), a smooth glass slab 4 voxels thick floating
   over half of the floor (host entry: albedo (1, 0.5, 0.25), extinction 0.2, ior 1.5, smoothness 1),
   a black wall under the slab's edge (so a bounce ray that the slab reflects cannot reach the lit
   floor), black sky, camera between floor and slab. Checks (reference mode): (a) flag on, 1 bounce,
   256 frames: under / beside = exp(-(1 - albedo) 0.2 4) per channel within 5 %; (b) flag off, 3
   bounces, 4096 frames: the refracted BSDF path alone gives T (1 - F)^2 with F = ((1.5 - 1) / 2.5)^2
   (Schlick at normal incidence) within 5 %; (c) flag on, 3 bounces: still T within 5 % (the claim);
   (d) flag on with the sun NEE off: the image equals the flag off bit for bit (4 path trace frames);
   (e) `bounce_nee_probability` 0.5 is accepted, -0.25, 1.5 and NaN are rejected.
2. "shadow transmission: SSS slab budget": the same floor; an opaque SSS slab (albedo 0.5, model HG,
   probability 1, sss_albedo 0.8, steps 16 -> shadow length 4) 2 voxels thick over one quarter, 6 over
   another, a plain opaque block over the third; flag on, 1 bounce: 0.64 within 5 % (every channel),
   below 0.01 and a centre maximum of 0 through 6 voxels, exactly 0 under the plain block;
   probability 0.5: 0.32 within 5 %; flag off: 0.
3. "shadow transmission: hard occluder and lamp candidate": a 2 x 2 lamp 30 units above a white floor,
   the box between them tinted glass, a plain opaque block or air; light tree on, light NEE on (MIS),
   no sun radiance, flag on, 1 bounce: the plain block gives exactly 0; glass / air = the slab's
   transmittance per channel within 10 %.
4. `caelix_frame_layout_test` (the 608-byte uniform block; `kSubsurfaceProbability`, `kSssModelNone`,
   `kSssModelHg` in the constant table) and `caelix_material_test` (w2 = steps | h(shadow length), w3 =
   the mirror of hot w5, `SubsurfaceShadowLength` at g = 0, the clamps and 0 steps) updated;
   `caelix_abi_c99` touches the new constants and `bounce_nee_probability` at offset 196.

## 6. Frame constants (lib/src/render/frame_constants.h, shaders/common/caelix_frame.hlsl)

576 -> 608 bytes. New fields: `reverse_tables` u64 @568 (replaces pad_568), `group_grids` u64 @576,
`bounce_nee_probability` float @584 (docs/LIGHT_NEE.md section 12; written, no consumer yet),
`hint_patch_shift` uint @588 (same section; written 0, no consumer yet), `subsurface_present` uint
@592, `pad_596` uint, `pad_600` u64. The static_asserts, the size comments (frame_constants.h,
caelix_frame.hlsl, renderer.h, README.md) and the layout test follow.

## 7. Measurements to report (after the implementation, by the main agent)

Registers of both trace kernels (`CAELIX_PIPELINE_STATS=1`), trace ms on bistro and emberdeep at
1440p with the host defaults before / after (`Claude/regstat.py --time`), and one capture of a
tinted glass shadow and an SSS object's shadow (`Artifacts/shadow-transmission-20261028/`). The
register lines of the implementation are in section 8.

## 8. As built (2026-10-28)

Deviations from the spec, and why:

1. The claim starts after any transmissive primary NEE, sun or lamp (`claim = transmissive` in both NEE
   branches), and sets `segment_sees_sun = !nee`. The spec set it only when the sun NEE ran; with the
   light NEE alone a lamp seen through glass would then be counted twice (the straight NEE and the
   refracted BSDF path, whose lamp hit after the glass has the state 0 = full weight). With the sun NEE
   on this is the spec's rule.
2. In the claim, `segment_light = light_nee ? kSegmentLampNee : 0u`: the kernel without light NEE must
   not call LampHitWeight (no light tree), and the expression folds there.
3. `K_RAY_ORIGIN_PUSH_OFF` moved to the new header render/caelix_ray.hlsl, included by
   caelix_shading.hlsl and caelix_scene_trace.hlsl. caelix_scene_trace.hlsl could not include
   caelix_shading.hlsl: the trace kernel declares the sky bindings by macro before it includes the
   shading file, after the scene trace.
4. Each push-off of the chain also shortens `remaining` by its component along the ray (`remaining -=
   dot(offset, direction)`, at interfaces and at SSS exits). Without it the last hop of a lamp ray
   that crossed glass ended on the lamp's face instead of two push-offs short of it and hit the lamp:
   14 % of the lamp rays of test 3 came back occluded (ratio 0.862 for every channel).
5. `TraceCandidate` is overloaded (the old signature calls the new one with mode 0 and a dummy flag), so
   `TraceScene` and `TraceShadow` keep their source.
6. The SSS walk keeps the exit face as the axis of the last crossing (its side is the sign of d) and
   ends on the step counter, instead of an int3 and an `exited` flag (two registers fewer in the
   walk's state; the kernel's register count did not change, see below).
7. `ShadowVoxelIsSubsurface` does not test `sss_steps` (the specified single load of w3): an HG entry
   with probability > 0 and steps 0 passes the probe and dies in the chain (shadow length 0 -> budget
   1), the same result as an occluder; `has_subsurface` does not count such entries.

Registers (`CAELIX_PIPELINE_STATS=1`, RTX 5080, driver 610.88, the Rust host with `--test-entity`):

    pipeline stats caelix_path_trace [CS]: Register Count = 128, Binary Size = 257024, Stack Size = 0, Local Memory Size = 68719476752, Shared Memory Size = 5376
    pipeline stats caelix_path_trace_no_light_nee [CS]: Register Count = 128, Binary Size = 133504, Stack Size = 0, Local Memory Size = 68719476752, Shared Memory Size = 3072

Local Memory Size is printed as 2^36 + the spill bytes: 16 bytes of spill in both kernels. Before
(`b34452d`): 128 registers + 16 bytes of spill and 96 registers + 48 bytes of spill (binary 174976
bytes for caelix_path_trace). So the light NEE kernel keeps its register count (binary +47 %, shared
memory 2560 -> 5376 bytes); the kernel without light NEE went from 96 to 128 registers (occupancy).
Ablation on the same host before deviation 6 (DXC -O3 -WX, the two kernels loaded through
CAELIX_SHADER_DIR; deviation 6 did not change the counts):

| variant | caelix_path_trace | caelix_path_trace_no_light_nee |
| --- | --- | --- |
| as built | 128 + 16 B | 128 + 16 B |
| `CAELIX_SPEC_FLAGS_MASK` without bit 6 (the transmissive path folds away) | 128 + 16 B | 96 + 64 B |
| probe only (no chain) | 128 + 16 B | 96 + 64 B |
| probe and glass chain, no SSS walk | 128 + 16 B | 96 + 64 B |
| SSS probe mode off, walk in | 128 + 16 B | 128 + 16 B |

The straight SSS walk (the voxel cursor and its brick crossings through SceneBrickAt, inside the hop
loop inside the primary NEE) is what lifts the kernel without light NEE to 128. The rule of
docs/SUBSURFACE.md section 4 (variant kernels when the main kernels lose registers) applies to this
walk too; trace ms on bistro / emberdeep are not measured here (owner rule: small scenes only in this
step's validation).

Test results (caelix_trace_test, RTX 5080):

- tinted slab, flag on, 1 bounce: ratios 1.00011 / 0.67046 / 0.54895 against 1 / 0.67032 / 0.54881
  (relative 1.0001 .. 1.0003); flag off, refracted path only: 0.92112 / 0.61759 / 0.50571 against T (1
  - F)^2 = 0.92160 / 0.61777 / 0.50578 (relative 0.9995 .. 0.9999); flag on, 3 bounces: 1.00016 /
  0.67052 / 0.54901; sun NEE off: bit-identical images.
- SSS slab: 2 voxels 0.64000 (largest pixel 0.64004), 6 voxels 0 (largest pixel 0), plain block 0;
  probability 0.5: 0.32000.
- lamp: plain block 0 (largest pixel 0); glass / air 0.99875 / 0.66955 / 0.54821 against 1 / 0.67032 /
  0.54881 (relative 0.9988 .. 0.9989: the BSDF share of the MIS that the claim drops).

Open: thin SSS objects can count the sun twice now that the walk exists (section 1, accepted); the
probe's `medium_seen` can run the chain for a medium behind the end of a lamp ray (cost only). The
register count of the kernel without light NEE is solved by section 9.

## 9. Kernel variants (subsurface step, 2026-10-28)

The SSS parts of this ray moved into the variants `caelix_path_trace_sss` and
`caelix_path_trace_sss_no_light_nee` (`CAELIX_SUBSURFACE 1`, docs/SUBSURFACE.md section 4), which the
renderer binds for the lighting frames of a scene whose material table has an SSS entry
(`Scene::has_subsurface()`). The main kernels `caelix_path_trace` and `caelix_path_trace_no_light_nee`
compile with `CAELIX_SUBSURFACE 0`: `ShadowVoxelIsSubsurface` (and so `CaelixShadowPassesOpaque`)
returns false without a load, `TraceShadowProbe` compiles only `kBrickDdaShadowOpaque`, and
`TraceShadowTransmissive` returns 0 at every opaque hit (no straight SSS walk). The glass part (probe,
chain, claim) stays in all four kernels. A scene without SSS materials keeps its images (the SSS rules
never applied to it). Registers (`CAELIX_PIPELINE_STATS=1`, RTX 5080, driver 610.88):

| kernel | registers + spill |
| --- | --- |
| `caelix_path_trace` | 128 + 16 B |
| `caelix_path_trace_no_light_nee` | 96 + 48 B (was 128 + 16 B in section 8) |
| `caelix_path_trace_sss` (with the vertex walk) | 128 + 32 B |
| `caelix_path_trace_sss_no_light_nee` (with the vertex walk) | 128 + 16 B |

The three tests of section 5 pass with the same numbers; test 2 (the SSS slab) now runs on
`caelix_path_trace_sss`, tests 1 and 3 on the main kernels.
