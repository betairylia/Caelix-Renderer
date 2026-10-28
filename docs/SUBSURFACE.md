# Subsurface scattering: the Henyey-Greenstein lattice walk (production)

Status: SPEC, 2026-10-28. The implementer turns this file into "as built". Source of the model: the
experiment on branch `sss-expr` / `sss-voxel-color-expr` (`git show sss-expr:docs/SSS_EXPERIMENT.md`,
`git show sss-voxel-color-expr:shaders/render/caelix_subsurface.hlsl`), owner decisions 2026-10-26..28:
cheap approximation, opaque SSS codes, 6 face directions, dead at the step cap, exit = diffuse
vertex, entry jitter always on, one phase function for now (HG, model index 1).

## 1. Material

`CaelixMaterial` (112 B, unchanged): `sss_model` = `CAELIX_SSS_MODEL_NONE` (0) or
`CAELIX_SSS_MODEL_HG` (1, new constants in include/caelix/caelix.h with a comment per field);
`sss_probability` p in [0, 1]; `sss_albedo` per step; `sss_anisotropy` g in [-1, 1] (clamped to
[-0.99, 0.99] on the GPU for the HG inversion); `sss_steps` 1..255 (0 = never enters: treated as a
non-SSS material); `sss_params` unused by HG. An SSS voxel = an OPAQUE code with `sss_model == HG`,
`sss_probability > 0` and `sss_steps > 0`. Other model values are stored but ignored by the kernels
(not SSS). The GPU subsurface entry layout is in docs/SHADOW_TRANSMISSION.md section 2 (w2 steps +
shadow length, w3 probability + model); `CaelixGetSubsurface` returns all of it; add
`CaelixMaterialSssWord(table, code)` (one 4-byte load of hot w5) for the vertex test.

## 2. The walk (`shaders/render/caelix_subsurface.hlsl`, above the scene-trace boundary)

Uses the voxel cursor of docs/SHADOW_TRANSMISSION.md section 3 (`VoxelCursor*`, entity helpers).
All in the hit entity's voxel grid (object space; rigid transforms, scale 1).

1. Start voxel: `VoxelCursorFromHit(hit)` = the hit voxel (exact: `hit.cell`). Entry jitter (always):
   the voxel of `hit.object_hit - 0.5 n + offset`, offset uniform in [-0.5, 0.5] along the two
   face-tangent axes (two numbers of `rng`), reached by one `VoxelCursorStep` of at most one voxel per
   tangent axis; if that voxel is not an SSS voxel of the entity, the hit voxel. `a` = the start
   voxel's sss_albedo (`1 - one_minus_albedo`).
2. State: cursor, continuous unit direction `d` (start: the ray direction in object space), weight
   `w = 1`. `g` and the step cap `N` come from the ENTRY material's subsurface entry.
3. Per step (at most N): `w *= a`; `d = SampleHenyeyGreenstein(d, g, rng)` (|g| < 1e-3: uniform
   sphere); `delta` = the signed unit axis of the largest |d| component (ties x, y, z); look at
   `cursor + delta` (`VoxelCursorStep` on a copy + `VoxelCursorBlock`; an absent brick = air):
   opaque SSS voxel (ONE 16-byte load of its subsurface entry: w0..w3) -> move, `a` = its albedo;
   opaque non-SSS -> a wall, the cursor stays; anything else (air, glass) -> EXIT.
4. Exit through the cursor voxel's face in direction `delta`: point = face centre + uniform offsets in
   [-0.5, 0.5] on the two other axes; normal = delta; to world space with the entity's current
   transform; origin = point + `K_RAY_ORIGIN_PUSH_OFF` normal; direction = the cosine lobe around the
   normal (`SafeNormalize(normal + RandomUnitVector(rng))`, the diffuse lobe of ShadeVoxelHit);
   weight = `w`.
5. Step cap reached: dead (absorbed; the vertex's emission was already counted).

`SampleHenyeyGreenstein(w, g, rng)` and a shared `OrthonormalBasis(w, out t, out b)` (Duff et al.,
the same construction as SampleSunDirection; add the helper to caelix_shading.hlsl, do NOT change
SampleSunDirection) go into caelix_subsurface.hlsl / caelix_shading.hlsl.

Random numbers: the BSDF stream `rng`. A material without SSS draws nothing extra: scenes without
SSS keep their images bit for bit (frame constants growth aside).

## 3. Integration (`caelix_path_trace.hlsl`, TracePath)

`SubsurfaceVertex(fc, hit, ray_direction, rng, inout shade, inout vertex_normal, out float3
sss_albedo, bool last_vertex) -> {kSubsurfaceNone, kSubsurfaceBsdf, kSubsurfaceExit, kSubsurfaceDead}`
after `ShadeVoxelHit` at the primary surface and at every bounce hit, only when `fc.flags &
kRenderFlagSubsurface`: a non-SSS material returns None without drawing; `last_vertex` (no bounce
left) returns None (only the emission counts there; same estimate, cheaper); else one number `u` of
`rng`: `u >= p` -> Bsdf (the shade stands); else the walk: Exit replaces the vertex (`shade.albedo =
w`, bounce origin / direction = the exit's, `specular_lobe = false`, `roughness = 1`,
`vertex_normal` = the exit normal); Dead sets `shade.albedo = 0` and ends the path after the hit's
emission (at the primary: `stochastic = 0`, no NEE, no segment).

- NEE (sun and lamps) and `segment_normal` use the VERTEX normal and `shade.bounce_origin` (the exit
  origin): LampHitWeight then gets the same x_r, n_r as the NEE.
- The walk uses the vertex's bounce: no extra bounce count.
- Primary vertex demodulation: guide albedo `lerp(albedo, sss_albedo, p)` written as the primary
  albedo (G-buffer, NRD demodulation); the primary throughput is `branch weight / max(guide, 0.001)`
  with branch weight = `albedo` (Bsdf) or `w` (Exit). Non-SSS materials: unchanged (1 / k). The
  G-buffer normal, depth and motion stay the entry face (the NRD primary surface). DLSS RR diffuse
  albedo guide: `transmission * primary_shade.albedo * (1 - metallic)` (the guide albedo; for non-SSS
  materials this equals today's value).
- A walk exit's medium is the air (0): the primary vertex after an exit passes medium 0 to the
  transmissive shadow ray (docs/SHADOW_TRANSMISSION.md).
- Known limits (as in the experiment): an exit into glass starts the next ray with the outside
  medium; the walk stays inside the hit entity (other entities count as air).

## 4. Kernels and flags

- `CAELIX_RENDER_FLAG_SUBSURFACE (1u << 5)`: lighting modes; the renderer keeps the bit only when
  `scene_.has_subsurface()`; HLSL `kRenderFlagSubsurface`.
- The walk is compiled into the two main trace kernels and gated by the flag at run time. Macro
  `CAELIX_SPEC_NO_SUBSURFACE` (SpecializeFrameConstants: clears the flag) for the register
  measurement. Decision rule (main agent, after the measurement): if the main kernels lose registers
  or speed on a scene without SSS materials (bistro, emberdeep) the walk moves to variant kernels
  `caelix_path_trace_sss(_no_light_nee)` as on the experiment branch.
- Rust host: `--no-sss` (default on), a checkbox; the JSON materials loader already parses the keys.
  The ScenesPlayground scene `Artifacts/sss-20261026/out/sss-materials.json` needs `"sss_model": 1`
  on its SSS entries (the loader default stays 0).

## 5. Tests (tests/trace_test.cpp)

1. "subsurface: furnace": a closed grey box? No: an open scene with a white sky (procedural, all
   three colours 1), a cube of an SSS material (albedo 0.5 surface, sss_albedo 1, p 1, g 0, steps 64,
   model HG), reference mode 512 frames, 4 bounces: the cube's pixels converge to the sky radiance
   times (1 - dead fraction); with steps 255 and a 4^3 cube the dead fraction is below 2 % (the walk
   leaves a small cube fast): check the cube's mean within 0.95 .. 1.0 of the sky. With `sss_model 0`
   the same material renders as plain opaque (albedo 0.5): mean about 0.5 (the test shows the gate).
2. "subsurface: lifecycle": the flag without SSS materials changes nothing (bit-identical image to
   the flag off, same seed); the walk at a bounce vertex (a white diffuse floor next to the cube)
   produces finite images; dead at steps 1 with g 0.9 inside a thick slab (probability 1,
   sss_albedo 1): the slab's lit face stays close to black except the emission.

## 6. Measurements to report

Registers, trace ms on bistro / emberdeep (no SSS materials) with the walk compiled in versus
`CAELIX_SPEC_NO_SUBSURFACE`; the SSS scene (`ScenesPlayground/Artifacts/sss-20261026/out/sss.cxw`
+ its materials, block-ID) trace ms and a capture in `Artifacts/subsurface-20261028/`.
