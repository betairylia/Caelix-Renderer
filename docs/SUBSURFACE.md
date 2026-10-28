# Subsurface scattering: the Henyey-Greenstein lattice walk (production)

Status: AS BUILT, 2026-10-28 (spec of the same day; the deviations and measurements are in sections 7
and 6). Source of the model: the experiment on branch `sss-expr` / `sss-voxel-color-expr` (`git show
sss-expr:docs/SSS_EXPERIMENT.md`, `git show sss-voxel-color-expr:shaders/render/caelix_subsurface.hlsl`),
owner decisions 2026-10-26..28: cheap approximation, opaque SSS codes, 6 face directions, dead at the step
cap, exit = diffuse vertex, entry jitter always on, one phase function for now (HG, model index 1).

## 1. Material

`CaelixMaterial` (112 B, unchanged): `sss_model` = `CAELIX_SSS_MODEL_NONE` (0) or `CAELIX_SSS_MODEL_HG`
(1, include/caelix/caelix.h); `sss_probability` p in [0, 1]; `sss_albedo` per step; `sss_anisotropy` g in
[-1, 1] (clamped to [-0.99, 0.99] on the GPU for the HG inversion, `kSubsurfaceMaxAnisotropy`);
`sss_steps` 1..255 (0 = never enters: treated as a non-SSS material); `sss_params` unused by HG. An SSS
voxel = an OPAQUE code with `sss_model == HG`, `sss_probability > 0` and `sss_steps > 0`. Other model values
are stored but ignored by the kernels (not SSS). The GPU subsurface entry layout is in
docs/MATERIALS.md (w0 / w1 one minus sss_albedo and g, w2 steps, w3 probability + model). In the SSS
kernels the primary vertex's transmissive shadow ray reads it too: it enters an SSS voxel with its
sss_probability and walks straight through the SSS voxels (sss_albedo per voxel, occluded beyond sss_steps
voxels); every other shadow ray stops at an SSS voxel like at any opaque voxel (docs/SHADOW_TRANSMISSION.md).
`CaelixMaterialSssWord(table, code)`
(shaders/common/caelix_material.hlsl) is one 4-byte load of the hot entry's w5 (the vertex test; a cache
hit after ShadeVoxelHit, whose CaelixGetMaterial loads the same 16-byte row).

## 2. The walk (`shaders/render/caelix_subsurface.hlsl`, above the scene-trace boundary)

Everything in the file is inside `#if CAELIX_SUBSURFACE` (section 4). It uses the voxel cursor of
`shaders/render/caelix_voxel_cursor.hlsl` (`VoxelCursor*`, the entity helpers; included by
caelix_scene_trace.hlsl, which came with the first version of docs/SHADOW_TRANSMISSION.md), all in the hit
entity's voxel grid (object space; rigid transforms, scale 1).

- `SubsurfaceVoxel(block, out albedo)`: opaque, then ONE 16-byte load of the subsurface entry (w0..w3):
  model HG, probability > 0 and steps > 0; `albedo` = `1 - one_minus_albedo`. Defined, with
  `SubsurfaceEntry` (also probability and steps), in caelix_scene_trace.hlsl under `#if
  CAELIX_SUBSURFACE`: the transmissive shadow ray uses the same reads.
- `SampleHenyeyGreenstein(w, g, rng)`: inversion of the HG cosine (|g| < 1e-3: uniform sphere), the
  azimuth around `w` with `OrthonormalBasis(w, t, b)` (shaders/render/caelix_shading.hlsl: the Duff et al.
  construction of SampleSunDirection, which keeps its own copy). Two numbers.
- `SubsurfaceWalk(fc, hit, ray_direction, g, steps, entry_albedo, rng, out origin, out normal, out
  direction, out weight) -> bool`:
  1. Start voxel: `VoxelCursorFromHit(hit)` (exact: `hit.cell`). Entry jitter (always): the voxel of
     `hit.object_hit - 0.5 n + offset`, offset uniform in [-0.5, 0.5] along the two face-tangent axes (two
     numbers), reached by one `VoxelCursorStep` of at most one voxel per tangent axis; if that voxel is not
     an SSS voxel of the entity, the hit voxel. `a` = the start voxel's sss_albedo.
  2. State: cursor, continuous direction `d` (start: the ray direction in object space), weight `w = 1`.
     `g` and the step cap `N` come from the ENTRY (hit) material.
  3. Per step (at most N): `w *= a`; `d = SampleHenyeyGreenstein(d, g)`; `delta` = the signed unit axis of
     the largest |d| component (ties x, y, z); `cursor + delta` (a cursor copy, `VoxelCursorBlock`; an
     absent brick = air): an SSS voxel -> move, `a` = its albedo; opaque non-SSS -> a wall, the cursor
     stays; anything else (air, glass) -> EXIT.
  4. Exit through the cursor voxel's face `delta`: point = face centre + uniform offsets in [-0.5, 0.5] on
     the two other axes; normal = delta; to world space with the entity's current transform; origin =
     point + `K_RAY_ORIGIN_PUSH_OFF` normal; direction = `SafeNormalize(normal + RandomUnitVector(rng))`
     (the diffuse lobe of ShadeVoxelHit; the normal when degenerate); weight = `w`. Returns true.
  5. Step cap reached: false (dead: absorbed; the vertex's emission was already counted).
- `SubsurfaceVertex(fc, hit, ray_direction, rng, inout shade, inout vertex_normal, out guide_albedo, bool
  last_vertex) -> {kSubsurfaceNone, kSubsurfaceBsdf, kSubsurfaceExit, kSubsurfaceDead}`, in this order:
  `last_vertex` or a transparent block -> None; the hot w5 test (model HG, p > 0) fails -> None; the
  entry's 16-byte subsurface load; steps 0 -> None (all without a random number); `guide_albedo =
  lerp(shade.albedo, sss_albedo, p)`; one number u: `u >= p` -> Bsdf (the shade stands); else the walk:
  Exit replaces the vertex (`shade.albedo = w`, bounce origin / direction = the exit's, `specular_lobe =
  false`, `roughness = 1`, `vertex_normal` = the exit normal); Dead sets `shade.albedo = 0`.

Random numbers: the BSDF stream `rng`: one for the choice, two for the jitter, two per step, four at the
exit. A material without SSS draws nothing extra: scenes without SSS keep their images bit for bit (test 2,
section 5: even the SSS kernel renders a scene without SSS voxels bit for bit like the main kernel).

## 3. Integration (`shaders/render/caelix_path_trace.hlsl`, TracePath)

`caelix_subsurface.hlsl` is included after `caelix_light_nee.hlsl`. Only with `CAELIX_SUBSURFACE` and
`fc.flags & kRenderFlagSubsurface`, `SubsurfaceVertex` runs after `ShadeVoxelHit` at the primary surface
(`last_vertex = false`) and at every bounce hit (`last_vertex = bounce_index + 1 > bounce_count`, before
the increment: no next segment, only the emission counts).

- Primary vertex, a state other than None: the throughput is `shade.albedo / max(guide, 0.001)` (the
  branch weight: the albedo for Bsdf, the walk's weight for Exit) and the returned albedo is the guide
  `lerp(albedo, sss_albedo, p)` (G-buffer albedo, NRD demodulation, the composite multiplies by it).
  Non-SSS materials: unchanged (1 / k). Dead: `stochastic = 0` and return (no NEE, no segment; the primary
  emission is deterministic and stays). The G-buffer normal, depth and motion stay the entry face (the NRD
  primary surface).
- A vertex normal `primary_normal` / `vertex_normal` (the hit normal, or the exit normal after a walk)
  replaces `hit.normal` in the NEE (sun and lamps) and in `segment_normal`; the NEE's receiver point is
  `shade.bounce_origin` (the exit origin). So LampHitWeight gets the same x_r, n_r as the NEE.
- The walk uses the vertex's bounce: no extra bounce count.
- Bounce vertex: the emission counts first; Dead then ends the path (`break`); Exit and Bsdf multiply the
  throughput by `shade.albedo` as any vertex (the mixture of the two branches with probabilities p and 1 -
  p needs no division).
- An Exit sets `medium = 0` (the exit vertex lies in the air), at the primary (so the transmissive shadow
  ray of docs/SHADOW_TRANSMISSION.md starts in the air) and at the bounces.
- DLSS RR diffuse albedo guide (main): `transmission * shade.albedo * (1 - metallic)`, `shade` being the
  result TracePath returned (the guide albedo after an SSS primary vertex). For non-SSS opaque materials
  `shade.albedo` is `material.albedo`, the same half-precision value as before (bit-identical guide).
- Known limits (as in the experiment): an exit into glass starts the next ray in the air (medium 0); the
  walk stays inside the hit entity (other entities count as air); after an Exit the NRD roughness of the
  pixel is 1 (the exit's diffuse lobe), after Bsdf the material's (as on the experiment branch).

## 4. Kernels and flags

- `CAELIX_RENDER_FLAG_SUBSURFACE (1u << 5)` (caelix.h, HLSL `kRenderFlagSubsurface`): lighting modes.
- Every subsurface code path lives in two trace kernel variants: `shaders/render/caelix_path_trace_sss.hlsl`
  (`#define CAELIX_SUBSURFACE 1`) and `caelix_path_trace_sss_no_light_nee.hlsl` (`CAELIX_SUBSURFACE 1` and
  `CAELIX_SPEC_NO_LIGHT_NEE 1`), both `CAELIX_KERNEL_VARIANT`s that include caelix_path_trace.hlsl (cmake
  compiles them as kernels; lib/CMakeLists.txt lists them). `CAELIX_SUBSURFACE` defaults to 0
  (caelix_subsurface.hlsl). With 0 the main kernels `caelix_path_trace` and
  `caelix_path_trace_no_light_nee` contain no walk: TracePath has no `SubsurfaceVertex`. The SSS variants
  hold the vertex walk and the straight SSS walk of the primary vertex's transmissive shadow ray; in the
  main kernels every shadow ray stops at SSS voxels (docs/SHADOW_TRANSMISSION.md).
- `RenderKernels` (lib/src/render/render_kernels.*): four trace pipelines, the same trace set layout:
  `trace()`, `trace_no_light_nee()`, `trace_sss()`, `trace_sss_no_light_nee()`.
- Renderer (renderer_render.cpp): `sss_kernel = lighting && scene_.has_subsurface()` picks the SSS pair
  (then light NEE on / off as before); `CAELIX_RENDER_FLAG_SUBSURFACE` stays in `flags` only when
  `sss_kernel` (the main kernels never see it). The SSS kernel is bound with or without the subsurface
  flag. The primary mode binds the main kernels. (`FrameConstants::subsurface_present`, which chose the
  first shadow ray's SSS probe mode, no longer exists: @592 is padding.)
- `CAELIX_SPEC_NO_SUBSURFACE` (SpecializeFrameConstants: clears `kRenderFlagSubsurface`) is the
  measurement hook: compiled into an SSS variant it removes the vertex walk.
- Rust host: `--no-sss` (default on), the checkbox "subsurface scattering" next to "shadow transmission",
  `RenderSettings::subsurface`, the startup line prints `sss true|false`; the JSON materials loader parses
  `sss_model` (default 0). The ScenesPlayground scene `Artifacts/sss-20261026/out/sss-materials.json` got
  `"sss_model": 1` on its 9 entries with `sss_probability > 0` (ids 10, 21..24, 31 and the former SGGX ids
  12, 13, 25, which now walk as HG).

## 5. Tests (tests/trace_test.cpp)

1. "subsurface: furnace": a 4^3 cube of an SSS material (albedo 0.5, sss_albedo 1, p 1, g 0, steps 64,
   model HG) in a white procedural sky (all three colours 1, no sun radiance), the camera 6 units above
   its top face (the face fills the image), reference mode, 4 bounces, 512 frames: mean 0.999998 per
   channel (checked in [0.95, 1.0 + 1e-4]); the dead fraction is below the float noise (2.4e-6; the
   survival of a lattice walk in 4 cells per axis is about 0.81^steps, 1e-6 at 64 steps, so the spec's 0.95
   bound needed no change). Steps 2: mean 0.333 (dead fraction 0.667; the cap works). The flag off (the
   SSS kernel, no walk) and `sss_model 0` (no SSS entry: the main kernel): 0.498 = the plain albedo 0.5
   (127 / 255 through the RGBA8 albedo target), checked within 0.01 of 0.5.
2. "subsurface: lifecycle": (a) a white floor and a plain block without SSS materials, path trace mode, 3
   frames: the flag on equals the flag off bit for bit; (b) the same scene with an SSS entry on a code no
   voxel uses (the SSS kernel is bound): 0 of 6 images (flag off and on) differ from the main kernel's; (c)
   the block becomes SSS (p 0.8, sss_albedo 0.9, g 0.3, steps 32), the camera sees only the floor beside it:
   reference 64 frames, flag off 0.898, flag on 0.890, the images differ (walks at bounce vertices), finite;
   (d) dead walks: a slab 16 voxels thick, steps 1, g 0.9, p 1, sss_albedo 1, white sky: the lit top face
   is 0.0054 (checked < 0.05; a walk survives only when its single step turns back out of the face).
3. Passing: the three "shadow transmission" tests (their SSS slab scene binds the SSS kernel without the
   subsurface flag; since the straight SSS walk of the shadow ray experiment the slab transmits
   sss_albedo per voxel, docs/SHADOW_TRANSMISSION.md section 4) and the light NEE tests. `caelix_frame_layout_test` checks the uniform block of both new variants;
   `caelix_abi_c99` touches `CAELIX_RENDER_FLAG_SUBSURFACE`.

## 6. Measurements

Registers (`CAELIX_PIPELINE_STATS=1`, RTX 5080, driver 610.88, the Rust host; RenderKernels creates all
four trace pipelines, so `--test-entity` prints them all; Local Memory Size = 2^36 + the spill bytes):

    pipeline stats caelix_path_trace [CS]: Register Count = 128, Binary Size = 228096, Stack Size = 0, Local Memory Size = 68719476752, Shared Memory Size = 2560
    pipeline stats caelix_path_trace_no_light_nee [CS]: Register Count = 96, Binary Size = 121344, Stack Size = 0, Local Memory Size = 68719476784, Shared Memory Size = 4864
    pipeline stats caelix_path_trace_sss [CS]: Register Count = 128, Binary Size = 286336, Stack Size = 0, Local Memory Size = 68719476768, Shared Memory Size = 6144
    pipeline stats caelix_path_trace_sss_no_light_nee [CS]: Register Count = 128, Binary Size = 160896, Stack Size = 0, Local Memory Size = 68719476752, Shared Memory Size = 4608

| kernel | registers + spill |
| --- | --- |
| `caelix_path_trace` | 128 + 16 B (as before the shadow transmission step) |
| `caelix_path_trace_no_light_nee` | 96 + 48 B (back from 128 + 16 B of `6e503ce`) |
| `caelix_path_trace_sss` | 128 + 32 B |
| `caelix_path_trace_sss_no_light_nee` | 128 + 16 B |
| `caelix_path_trace_sss` with `CAELIX_SPEC_NO_SUBSURFACE` (SSS shadow ray only) | 128 + 16 B |
| `caelix_path_trace_sss_no_light_nee` with `CAELIX_SPEC_NO_SUBSURFACE` | 128 + 16 B |

So the straight SSS walk of the shadow ray alone lifts the kernel without light NEE to 128 (as measured in
the shadow transmission step); the vertex walk adds 16 B of spill to the light NEE variant. Scenes without
SSS materials bind the main kernels, which are back at their registers.

The SSS part of the shadow ray (the SSS-aware probe and the straight SSS walk) was removed later the same
day (docs/SHADOW_TRANSMISSION.md, "History"); the numbers above were measured with it. Registers after
the removal (`CAELIX_PIPELINE_STATS=1`, the Rust host on doverley-sss): `caelix_path_trace` 128 + 16 B,
`caelix_path_trace_no_light_nee` 96 + 64 B, `caelix_path_trace_sss` 128 + 48 B,
`caelix_path_trace_sss_no_light_nee` 128 + 16 B.

SSS scene (`ScenesPlayground/Artifacts/sss-20261026/out/sss.cxw` + its materials, block-ID, 1280 x 720, pt
mode, host defaults (4 bounces, NRD), the clouds pose of its doc/design.md `--pose 0,45,-120,-8,0 --sun
0.4,0.6,0.5 --sun-radiance 2e4,1.9e4,1.8e4`, `--warmup 64 --timing-frames 100`, two runs each, both bind
`caelix_path_trace_sss` (light NEE on, empty light tree)):

| | trace_ms (mean / median) |
| --- | --- |
| default (walk on) | 0.915 / 0.914, 0.915 / 0.915 |
| `--no-sss` (walk off, SSS shadow ray on) | 0.571 / 0.570, 0.571 / 0.570 |

The walk costs 0.34 ms at 1280 x 720 on this scene (many SSS pixels: clouds, leaves, wax). Bistro /
emberdeep (no SSS materials, so the main kernels) are measured by the main agent.

Visual check (the clouds pose and the backlit leaf walls pose `--pose 40,12,-100,-2,0 --sun 0.1,0.35,0.9`):
the SSS leaves (id 10) turn from the dark diffuse green to a bright translucent green; the 1-voxel leaf
wall glows against the sun behind it, the 3-voxel wall stays much darker; the wax (id 31) is lighter and
softer than the diffuse wax; the clouds with g 0.7 / 0.9 are darker than the diffuse cloud in sky light
(dead walks, known from the experiment) and the g 0.9 cloud gets a bright rim against the sun.

## 7. Deviations from the spec

1. `SubsurfaceVertex` returns the guide albedo (`out float3 guide_albedo`) instead of `out float3
   sss_albedo`: the guide needs p as well, and the bounce vertices drop the output (the compiler removes it).
2. Steps 0: the spec says "treated as non-SSS"; the design note said "return Bsdf after the one u". As
   built, the entry's subsurface entry is loaded right after the w5 test (the guide needs sss_albedo
   anyway) and steps 0 returns None before u is drawn: such a material is exactly a non-SSS material (no
   number, no guide albedo).
3. The walk tests every voxel it looks at with `SubsurfaceVoxel` (model, probability AND steps from the
   one 16-byte load); the spec lists the 16-byte load without naming the steps test.
4. The variant split (section 4) is the main agent's decision of this stage (the spec's rule "variant
   kernels when the main kernels lose registers" applied to the shadow ray's SSS walk too; that walk was
   removed later the same day, docs/SHADOW_TRANSMISSION.md "History").
