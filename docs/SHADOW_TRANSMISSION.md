# Shadow rays through glass (primary vertex)

Status: AS BUILT, 2026-10-28 (second version of the same day; the first version is in "History" at
the end). Owner request: shadow rays (sun and lamp) at the PRIMARY hit vertex pass through transparent
voxels in a straight line (no reflection, no refraction, as if one medium), attenuated by the glass
extinction, so tinted glass leaves coloured shadows. Every opaque voxel occludes, SSS voxels
(docs/SUBSURFACE.md) too. Bounce vertices keep the hard any-hit shadow ray (`TraceShadow`, glass
occludes).

## 1. Model

`T(ray) in [0, 1]^3` replaces the boolean occlusion of the shadow ray at the primary vertex:

- An opaque voxel occludes (`T = 0`), an SSS voxel included.
- A transparent medium of id `g` (a glass palette / block-ID transparent code; 0 = air) attenuates a
  segment of length `t` inside it by `exp(-(1 - albedo) * extinction * t)` (the same formula as
  `MediumExtinction`; the air's fog is ignored: shadow rays ignore fog, as documented in caelix.h).
- A ray whose transmittance falls below `kShadowTransmitCutoff` (max channel < 1e-3) is occluded.

The straight line is an approximation (no refraction). Its interaction with the BSDF path:

- Sun: after a transmissive NEE at the primary vertex, the path CLAIMS the sun through every
  transparent (delta) vertex that follows: the segments see the sky without the sun disk until the
  first opaque vertex, where the usual rule resumes (NEE vertex: no disk; specular-lobe vertex: the
  disk). Without the claim the sun through glass would be counted twice (straight NEE + refracted BSDF
  path).
- Lamps: through the same claimed chain the segment state stays `kSegmentLampNee` WITHOUT the MIS
  bit (the exclusive rule): a lamp that the light tree can sample, hit after a delta vertex, counts
  0; a lamp the tree cannot sample, or a hit inside the near field (`hit.T < kLightNeeNearField`),
  counts in full. The MIS pdf of the receiver is not carried across interfaces (the receiver point
  would differ). Known double count: a lamp farther than the near field from the receiver whose last
  segment after the glass is shorter than the near field (rare; accepted). Known loss: the BSDF share
  of the balance heuristic for a lamp seen through glass (the NEE keeps only its own MIS weight;
  measured 0.1 % in test 3, section 5).
- The claim starts whenever the primary vertex ran a transmissive NEE, sun or lamp (`claim =
  transmissive` in both NEE branches), and sets `segment_sees_sun = !nee`. With the light NEE alone a
  lamp seen through glass would otherwise be counted twice. In the claim, `segment_light = light_nee ?
  kSegmentLampNee : 0u`: the kernel without light NEE must not call LampHitWeight (no light tree), and
  the expression folds there.

## 2. Code layout (the boundary of caelix_scene_trace.hlsl is kept)

- `shaders/render/caelix_scene_trace.hlsl`:
  - `TraceScene` and `TraceShadow` (the bounce vertices' shadow ray) are unchanged; `TraceCandidate`
    runs the closest-hit brick DDA (Unity's rule; `caelix_brick_trace.hlsl` has no shadow DDA modes).
  - `float3 TraceShadowTransmissive(fc, RayDesc ray, uint medium)`: `medium` = the transparent id the
    ray starts in (0 = air). A chain of up to `kShadowMaxHops` (8) closest-hit queries (`TraceScene`
    with TMax = `remaining`), interface to interface; no any-hit probe. `cur = medium`, `remaining =
    ray.TMax`. A miss: if `cur != 0` and `remaining < kShadowUnboundedT` (1e29, so the sun's unbounded
    ray adds no segment) the glass segment `remaining`; return `T` (0 below the cutoff). A hit at
    `h.T`: the glass segment `h.T` (below the cutoff -> 0); `remaining -= h.T`; block = `h.attrib >>
    16`:
    - opaque (an SSS voxel included): return 0;
    - transparent (an interface voxel: glass entry, glass exit = an air cell with face bits, glass to
      glass): `cur = CaelixTransparentId(block)`, origin = hit point `- K_RAY_ORIGIN_PUSH_OFF *
      h.normal` (into the entered voxel; `h.normal` faces the ray), `remaining -= dot(offset,
      direction)`, next hop.
    - after the hop cap: 0.
    The common case (air, nothing or an opaque voxel on the ray) costs one closest-hit query.
  - `ShadowGlassTransmittance(medium_id, t)` (id 0: 1) and `ShadowOccluded(T)` (max channel below
    `kShadowTransmitCutoff`) are the two material rules; `GET_MATERIAL` reads the hot entry.
  - The push-off shortens `remaining` by its component along the ray. Without it the last hop of a
    lamp ray that crossed glass ended on the lamp's face instead of short of it and hit the lamp: 14 %
    of the lamp rays of test 3 came back occluded in the first version (ratio 0.862 for every channel).
- `shaders/render/caelix_ray.hlsl`: `K_RAY_ORIGIN_PUSH_OFF`, shared by caelix_shading.hlsl and
  caelix_scene_trace.hlsl (caelix_scene_trace.hlsl cannot include caelix_shading.hlsl: the trace
  kernel declares the sky bindings by macro before it includes the shading file).
- `shaders/render/caelix_voxel_cursor.hlsl` (included by caelix_scene_trace.hlsl after TraceResult)
  and `SceneBrickAt` in `shaders/scene/caelix_scene_lookup.hlsl` came with the first version of this
  ray; since the second version only the subsurface walk (docs/SUBSURFACE.md) uses them.
- `shaders/render/caelix_path_trace.hlsl`: `SunNextEvent(fc, origin, normal, bool transmissive, uint
  medium, inout rng)` and `NextEventAtVertex(fc, x_r, n_r, sun, bool transmissive, uint medium, ...)`:
  with `transmissive` the shadow ray is `TraceShadowTransmissive` and the kept candidate's
  contribution is multiplied by it (the sun / lamp choice still uses the unoccluded contributions);
  else `TraceShadow`. The primary vertex passes `transmissive = (fc.flags &
  kRenderFlagShadowTransmission) != 0` and TracePath's `medium` (after the primary shade; an opaque
  primary leaves it unchanged; a subsurface walk exit sets it to 0); bounce vertices pass the literal
  `false, 0u`. The claim (section 1) lives in the bounce loop: at a transparent hit while `claim`
  the segment state above, at an opaque hit `claim = false` before the vertex's own rule.
- The same code is in all four trace kernels (`caelix_path_trace`, `_no_light_nee`, `_sss`,
  `_sss_no_light_nee`); the ray has no SSS part.

## 3. ABI and host

- `CAELIX_RENDER_FLAG_SHADOW_TRANSMISSION (1u << 6)` in CaelixRenderOptions::flags: lighting modes;
  the primary vertex's NEE shadow rays (sun and lamp) use the transmissive trace. HLSL mirror
  `kRenderFlagShadowTransmission`. The renderer keeps the bit only in the lighting modes
  (`renderer_render.cpp`).
- The shadow ray reads no subsurface entry. `CAELIX_SSS_MODEL_NONE` / `CAELIX_SSS_MODEL_HG` and the
  subsurface entry words 2 and 3 (steps; probability + model) serve the walk only (docs/MATERIALS.md).
- `CaelixRenderOptions::bounce_nee_probability` (float, the former `reserved_voxel_id[0]`) came with
  the first version of this step; docs/LIGHT_NEE.md section 12.2 describes it.
- Rust host (caelix-app): on by default; `--no-shadow-transmission` turns it off; the checkbox "shadow
  transmission" next to "sun NEE" in the render settings UI (enabled in the lighting modes);
  `settings.rs` field `shadow_transmission`; the startup line prints `shadow transmission true|false`.
- Unity plugin: no packet change (a flag bit); the C# side may set the flag later.

## 4. Tests (tests/trace_test.cpp)

1. "shadow transmission: tinted glass slab": a white floor (rgb5 31, albedo 1), the sun straight up (2
   degrees, radiance 1000, sun NEE on, no light tree), a smooth glass slab 4 voxels thick floating
   over half of the floor (host entry: albedo (1, 0.5, 0.25), extinction 0.2, ior 1.5, smoothness 1),
   a black wall under the slab's edge, black sky, camera between floor and slab. Checks (reference
   mode): (a) flag on, 1 bounce, 256 frames: under / beside = exp(-(1 - albedo) 0.2 4) per channel
   within 5 %; (b) flag off, 3 bounces, 4096 frames: the refracted BSDF path alone gives T (1 - F)^2
   within 5 %; (c) flag on, 3 bounces: still T within 5 % (the claim); (d) flag on with the sun NEE
   off: the image equals the flag off bit for bit; (e) `bounce_nee_probability` 0.5 is accepted,
   -0.25, 1.5 and NaN are rejected.
2. "shadow transmission: SSS slab occludes": the same floor; an opaque SSS slab (albedo 0.5, model
   HG, probability 1, sss_albedo 0.8, steps 16) 2 voxels thick over one quarter, 6 over another, a
   plain opaque block over the third; flag on, 1 bounce, 64 frames each: exactly 0 (mean and centre
   maximum, every channel) under both SSS slabs and under the plain block, the open floor > 0.1; flag
   off: 0. The scene binds `caelix_path_trace_sss`.
3. "shadow transmission: hard occluder and lamp candidate": a 2 x 2 lamp 30 units above a white floor,
   the box between them tinted glass, a plain opaque block or air; light tree on, light NEE on (MIS),
   no sun radiance, flag on, 1 bounce: the plain block gives exactly 0; glass / air = the slab's
   transmittance per channel within 10 %.

Results of the second version (RTX 5080, driver 610.88), the same numbers as the first version for
tests 1 and 3:

- tinted slab, flag on, 1 bounce: ratios 1.00011 / 0.67046 / 0.54895 against 1 / 0.67032 / 0.54881;
  flag off, refracted path only: 0.92112 / 0.61759 / 0.50571 against T (1 - F)^2 = 0.92160 / 0.61777 /
  0.50578; flag on, 3 bounces: 1.00016 / 0.67052 / 0.54901; sun NEE off: bit-identical images.
- SSS slab: 0 under 2 voxels, 6 voxels and the plain block (largest pixel 0 in each); open floor
  1.21755.
- lamp: plain block 0 (largest pixel 0); glass / air 0.99875 / 0.66955 / 0.54821 against 1 / 0.67032 /
  0.54881 (relative 0.9988 .. 0.9989: the BSDF share of the MIS that the claim drops).

## 5. Frame constants (lib/src/render/frame_constants.h, shaders/common/caelix_frame.hlsl)

608 bytes. The first version of this step grew the struct from 576 to 608 bytes: `reverse_tables`
u64 @568 and `group_grids` u64 @576 (the voxel cursor), `bounce_nee_probability` float @584 and
`hint_patch_shift` uint @588 (docs/LIGHT_NEE.md section 12). The second version removed
`subsurface_present` (uint @592, the probe's SSS mode): @592 is now `pad_592` (u64), followed by
`pad_600` (u64). `caelix_frame_layout_test` checks the uniform block of every frame-constant kernel.

## 6. History

The first version (commit `6e503ce`, 2026-10-28) let the primary vertex's shadow rays pass SSS voxels
too: an any-hit probe with SSS-aware shadow DDA modes in `caelix_brick_trace.hlsl` ("change 5":
`kBrickDdaClosest`, `kBrickDdaShadowOpaque`, `kBrickDdaShadowSssAware`, a `medium_seen` flag), then the
closest-hit chain, with a straight voxel walk (the voxel cursor) through SSS runs: times
`sss_probability` on entry, `sss_albedo` per voxel unit, and death after a derived budget `L =
sqrt(sss_steps (1 + g) / (1 - g))` voxels (g clamped to [-0.95, 0.95]; stored as a half in subsurface
entry w2, `SubsurfaceShadowLength`). The material rules lived in `shaders/render/caelix_shadow_medium.hlsl`;
`FrameConstants::subsurface_present` chose the probe mode. The subsurface step moved the SSS parts into
the `caelix_path_trace_sss*` kernel variants.

Dropped the same day by owner decision after the doverley-sss measurement
(`F:/RP_Games/Caelix-family/Artifacts/doverley-sss-cost-20261028/README.md`): on foliage scenes every
tree crown is thousands of thin SSS voxels; the SSS-aware any-hit probe could not stop at them and
walked every leaf of every crown brick on the ray (doverley-sss at 1440p: +5 ms; the vertex walk itself
costs about 1 ms there and stays). Since then SSS voxels occlude, the ray is a chain of closest-hit
queries only, `caelix_brick_trace.hlsl` is back to its state before `6e503ce`, and
`caelix_shadow_medium.hlsl`, `subsurface_present` and the shadow length are gone.

Measured before / after the chain-only ray (1440p, 100-frame trace means, two or more runs each, noise
about +-1.5 ms; `Artifacts/doverley-sss-cost-20261028/ab_chain.py`, `ab-chain/`): bistro 15.6, 14.9 ->
16.1, 15.3 ms and emberdeep 13.2, 13.1 -> 13.3, 13.4 ms (sun + lamp NEE, no denoiser: one closest-hit
query per primary shadow ray costs 0.2 .. 0.5 ms more than the any-hit probe); doverley-sss 24.0, 24.3,
23.2, 21.9 -> 19.5, 22.1, 21.1, 21.0 ms (the crown walk is gone; the HG walk keeps about 1 ms over
doverley's 19.5, 19.3, 18.7 -> 19.9, 18.5, 20.6 ms).
