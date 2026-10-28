# Shadow rays through transparent and subsurface voxels (primary vertex)

Status: SPEC, 2026-10-28. The implementer turns this file into "as built" (keep the structure, add an
"As built" section with every deviation and measurement). Owner request: shadow rays (sun and lamp)
at the PRIMARY hit vertex pass through transparent voxels and through subsurface-scattering (SSS)
opaque voxels in a straight line (no reflection, no refraction, as if one medium), attenuated by the
glass extinction (tinted glass leaves coloured shadows) and by the SSS albedo, and die when they
travel farther in SSS voxels than a rough estimate of a random walk's reach. Bounce vertices keep the
hard any-hit shadow ray.

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
  segment after the glass is shorter than the near field (rare; documented, accepted).

## 2. Material data

The subsurface GPU entry (docs/MATERIALS.md) gains two values; the ABI struct does not change:

| Word | Offset | Content |
| --- | --- | --- |
| w2 | 8 | `sss_steps & 0xFF` \| `h(shadow_length) << 16` |
| w3 | 12 | `h(sss_probability)` \| `sss_model << 16` (a mirror of hot w5, so one 16-byte load gives a voxel's SSS state) |

`shadow_length = sqrt(sss_steps * (1 + g) / (1 - g))` voxels with `g = clamp(sss_anisotropy, -0.95,
0.95)` (the RMS end-to-end distance of a persistent random walk of `sss_steps` steps; owner
decision: derived, no new field); 0 when `sss_steps == 0`. Computed in `PackMaterialSubsurface`
(lib/src/scene/material_pack.h). `CaelixSubsurface` (shaders/common/caelix_material.hlsl) gains
`probability`, `model`, `shadow_length`; new offset macro `kSubsurfaceProbability 12`. Add
`CAELIX_SSS_MODEL_NONE 0` and `CAELIX_SSS_MODEL_HG 1` to include/caelix/caelix.h (see
docs/SUBSURFACE.md). `MaterialTable` counts the SSS entries (`has_subsurface()`: any entry with
model HG, probability > 0, opaque code); `Scene::has_subsurface()` exposes it; the frame constants
carry it (`subsurface_present`).

## 3. Code layout (keep the boundary of caelix_scene_trace.hlsl)

- `shaders/scene/caelix_scene_lookup.hlsl`: `SceneBrickAt(s, inout group, entity, int3 brick)` (the
  record of the brick at an entity-local brick coordinate: same group -> reverse table; else the
  entity's group grid; 0 = absent). Take it from branch `sss-voxel-color-expr`
  (`git show sss-voxel-color-expr:shaders/scene/caelix_scene_lookup.hlsl`), unchanged.
- `shaders/render/caelix_voxel_cursor.hlsl` (new, data layer, included by caelix_scene_trace.hlsl):
  `VoxelCursor {uint group; uint entity; uint64_t record; int3 voxel;}`,
  `VoxelCursorFromHit(fc, TraceResult)` = the hit voxel from the brick origin (as TraceCandidate
  derives it) plus `hit.cell` (exact, no floor of object_hit), `VoxelCursorStep(fc, inout c, int3
  delta)` (brick crossing through SceneBrickAt; false = absent brick), `VoxelCursorBlock(c)` (the
  payload word read; 0 in an absent brick), `FrameSceneLookup(fc)`, and the entity transform helpers
  by entity id (`EntityWorldDirectionById`, `EntityObjectDirectionById`, `EntityWorldPointById`,
  from the same branch's caelix_scene_trace.hlsl). The frame constants gain `reverse_tables` and
  `group_grids` (section 6).
- `shaders/render/caelix_shadow_medium.hlsl` (new, the material rules of section 1, reads the
  material tables through `g_materialTable` / `g_subsurfaceTable`):
  `struct ShadowMedium {float3 transmittance; float sss_budget;}`, `ShadowMediumInit()`,
  `ShadowGlassSegment(inout m, uint medium_id, float t)` (id 0: nothing),
  `ShadowSssSegment(inout m, uint block, float l)` (albedo^l, budget += l / L; L = 0 -> budget 1),
  `ShadowSssEnter(inout m, uint block)` (times probability), `ShadowVoxelIsSubsurface(uint block)`
  (opaque, model HG, probability > 0: ONE 4-byte load of subsurface w3), `ShadowMediumDead(m)`.
- `shaders/render/caelix_brick_trace.hlsl`: `CaelixShouldTerminateBrickRay` and the DDA get a
  compile-time `mode` parameter (NATIVE-marked): `kBrickDdaClosest` (0, today's rule),
  `kBrickDdaShadowOpaque` (1: only opaque voxels terminate; a visited transparent voxel with id != 0
  sets the out flag `medium_seen`), `kBrickDdaShadowSssAware` (2: as 1, and an opaque voxel for which
  `CaelixShadowPassesOpaque(block)` is true does not terminate and sets `medium_seen`).
  `CaelixShadowPassesOpaque` is a PROTOTYPE in this file (`bool CaelixShadowPassesOpaque(uint
  block);`), defined in caelix_scene_trace.hlsl as `ShadowVoxelIsSubsurface` (DXC accepts a
  prototype with a later definition; if it does not, include caelix_shadow_medium.hlsl from the brick
  trace instead and say so). The existing callers pass mode 0: the closest-hit DDA must compile to
  the same code (the mode folds at compile time).
- `shaders/render/caelix_scene_trace.hlsl`:
  - `TraceShadow(fc, ray)` unchanged (bounce vertices).
  - `TraceShadowProbe(fc, ray, bool sss_aware, out bool medium_seen) -> bool` = the any-hit query of
    TraceShadow with DDA mode 1 or 2.
  - `float3 TraceShadowTransmissive(fc, RayDesc ray, uint medium)`: `medium` = the transparent id the
    ray starts in (0 air). Step 1: the probe (`sss_aware = fc.subsurface_present != 0`); hard
    occluded -> 0; not occluded and no medium seen and `medium == 0` -> 1 (the common case costs one
    any-hit query, as today). Step 2 (the chain): `ShadowMedium m`; `cur = medium`; `remaining =
    ray.TMax`; up to `kShadowMaxHops` (8) closest-hit queries (`TraceScene`): a miss -> if `cur != 0`
    and `remaining < 1e29` apply `ShadowGlassSegment(cur, remaining)`; return `m.transmittance`. A hit
    at `h.T`: `ShadowGlassSegment(cur, h.T)`; `remaining -= h.T`; block = `h.attrib >> 16`:
    - transparent (an interface voxel: glass entry, glass exit = an air cell with face bits, glass to
      glass): `cur = CaelixTransparentId(block)`, origin = hit point `- K_RAY_ORIGIN_PUSH_OFF *
      h.normal` (into the entered voxel; `h.normal` faces the ray), continue;
    - opaque, not SSS: return 0;
    - opaque SSS: the straight voxel walk: `VoxelCursorFromHit`, `p = h.object_hit` (on the entry
      face), `d = normalize(EntityObjectDirectionById(entity, ray.Direction))`, `ShadowSssEnter`;
      then up to `kShadowMaxVoxelSteps` (128) voxels: block_v = `VoxelCursorBlock`; not opaque ->
      EXIT (`cur = CaelixTransparentId(block_v)`, origin = `EntityWorldPointById(entity, p) +
      K_RAY_ORIGIN_PUSH_OFF * EntityWorldDirectionById(entity, float3(exit face axis sign))`, i.e.
      pushed through the face; continue the hop loop); opaque and not SSS -> return 0; SSS: `l` = the
      distance along `d` to the voxel's exit face (the 3D DDA boundary math of the SGGX walk on
      branch `sss-voxel-color-expr`, `shaders/render/caelix_subsurface.hlsl` SggxWalk: `bound`,
      `t3`, `axis`), `l = min(l, remaining)`; `ShadowSssSegment(block_v, l)`; dead or below the
      cutoff -> return 0; `remaining -= l`; `remaining <= 0` -> return `m.transmittance`; `p += l d`,
      snap the crossed coordinate to the face, `VoxelCursorStep` across that face (an absent brick
      reads block 0 next = air = exit).
    - after the hop cap: 0.
  - Comment block at the top of the file: the boundary now also exports `TraceShadowTransmissive`
    and the voxel cursor (used by docs/SUBSURFACE.md's walk).
- `shaders/render/caelix_path_trace.hlsl`: `SunNextEvent` and `NextEventAtVertex` take
  `bool transmissive, uint medium`; with `transmissive` the shadow ray's result is
  `TraceShadowTransmissive` and the contribution is multiplied by it; else `TraceShadow` as today.
  The primary vertex passes `transmissive = (fc.flags & kRenderFlagShadowTransmission) != 0` and the
  chain's medium (`medium` of TracePath; 0 after an SSS walk exit, docs/SUBSURFACE.md); bounce
  vertices pass false. The claim (section 1): `bool sun_claim` set at the primary when transmissive
  and the sun NEE ran; in the bounce loop, at a hit whose block is transparent while `sun_claim`:
  `segment_sees_sun = false` and `segment_light = kSegmentLampNee` (no MIS bit), the claim stays;
  at an opaque hit `sun_claim = false` before the vertex's own rule. The kernel variant without light
  NEE compiles the same code (`segment_light` is unused there after the fold).

## 4. ABI and host

- `CAELIX_RENDER_FLAG_SHADOW_TRANSMISSION (1u << 6)` in CaelixRenderOptions::flags: lighting modes;
  the primary vertex's NEE shadow rays (sun and lamp) use the transmissive trace. HLSL mirror
  `kRenderFlagShadowTransmission`. The renderer keeps the bit only in the lighting modes.
- Rust host (caelix-app): on by default; `--no-shadow-transmission` turns it off; a checkbox in the
  render settings UI next to "sun NEE"; `settings.rs` field `shadow_transmission`.
- Unity plugin: no packet change (a flag bit); the C# side may set it later.

## 5. Tests (tests/trace_test.cpp, new TEST_CASEs; use the existing rig helpers)

1. "shadow transmission: tinted glass slab": a white floor (albedo 1, opaque), the sun straight up
   (sun_dir (0, 1, 0), a 2-degree disk, sun NEE on, no light tree), a smooth glass slab of thickness 4
   voxels floating above half of the floor (host material table: albedo (1, 0.5, 0.25), extinction
   0.2, ior 1.5, smoothness 1), path trace mode, bounce_count 1, bounce_count_transparent 4, no
   denoise, reference mode 256 frames. Expected per channel: floor under the slab / floor beside it =
   `exp(-(1 - albedo) * 0.2 * 4)` (= 1, 0.67, 0.55) within 5 % (the refracted BSDF path contributes
   nothing: the floor's diffuse bounce rays that hit the slab from below and refract see the sky
   without the disk because of the claim; the sky is black). The same with the flag OFF: the ratio is
   the refracted-only value, which for a flat slab and a vertical sun equals the same transmittance
   (unbiased reference), so check both; and with the flag ON and the sun NEE OFF nothing changes
   (no claim without NEE).
2. "shadow transmission: SSS slab budget": the same floor, an opaque SSS slab (albedo 0.5, model HG,
   probability 1, sss_albedo 0.8, steps 16 -> shadow length 4) of thickness 2 over one part of the
   floor and thickness 6 over another; flag on, SSS walk flag OFF (the walk is not under test):
   expected ratios 0.8^2 = 0.64 (within 5 %) and 0 (below 0.01); with probability 0.5: 0.32.
3. "shadow transmission: hard occluder and lamp candidate": an opaque non-SSS block gives 0; a lamp
   behind a tinted slab lights a floor patch by the slab's transmittance (light tree on, light NEE
   on, lamps only, no sun) within 10 %.
4. `caelix_frame_layout_test` and `caelix_material_test` updated for the new entry words.

## 6. Frame constants (lib/src/render/frame_constants.h, shaders/common/caelix_frame.hlsl)

576 -> 608 bytes. New fields: `reverse_tables` u64 @568 (replaces pad_568), `group_grids` u64 @576,
`bounce_nee_probability` float @584 (docs/LIGHT_NEE.md section 12), `hint_patch_shift` uint @588
(same section), `subsurface_present` uint @592, `pad_596` uint, `pad_600` u64. Update the
static_asserts, the test, and every comment that says 576.

## 7. Measurements to report (after the implementation, by the main agent)

Registers of both trace kernels (`CAELIX_PIPELINE_STATS=1`), trace ms on bistro and emberdeep at
1440p with the host defaults before / after (`Claude/regstat.py --time`), and one capture of a
tinted glass shadow and an SSS object's shadow (`Artifacts/shadow-transmission-20261028/`).
