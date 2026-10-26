# Cheap subsurface scattering by voxel lattice walks (experiment)

Branch `sss-expr` (from `main` `72937ae`), 2026-10-26. Experiment, not for merge: the findings are the
deliverable. The expensive volumetric model of branch `scattering-expr` (`06a278f`) stays there as the
reference (`Artifacts/scattering-20261026/README.md`).

## Owner model

```
Material: scatterProbability, scatterSteps (+ sss albedo, HG anisotropy)

if hit is an SSS face:
    if RNG() < p:  bounce = LatticeWalkScatter(hit)   # the exit is a normal bounce vertex
    else:          bounce = BSDF(hit)
    then shade, NEE, ...

LatticeWalkScatter(hit):
    cursor = voxel behind the entry face
    loop scatterSteps:
        cursor = random walk step
        if cursor voxel is not medium: return ray from the leaving face
    return dead
```

Owner decisions (2026-10-26): SSS voxels are OPAQUE codes (the ray query stops at the surface, shadow
rays stay any-hit, hard shadows); 6 face directions; a walk that reaches the step cap is dead (absorbed);
exit point uniform on the leaving face, exit direction cosine around its normal; the surface albedo and
the SSS albedo are separate fields; an opaque non-SSS voxel is a wall (the walk stays and draws a new
direction); glass and air are exits; HG anisotropy wanted; soft shadows through SSS later (not part of
this step).

## Material (C struct 64 bytes, was 48; GPU: a 48-byte main table + a 16-byte subsurface table)

| offset | field | meaning |
| --- | --- | --- |
| 0 | `albedo[3]`, `smoothness` | surface BSDF (the branch with probability 1 - p) |
| 16 | `emission[3]`, `metallic` | unchanged |
| 32 | `ior`, `extinction` | unchanged |
| 40 | `sss_anisotropy` | HG g of every walk step, [-0.99, 0.99] |
| 44 | `sss_steps` (uint32) | step cap of a walk, 1 .. 4096 when p > 0 |
| 48 | `sss_albedo[3]` | per-step survival colour, each in [0, 1] |
| 60 | `sss_probability` | p, [0, 1]; 0 = no SSS. Opaque codes only (ignored on transparent codes) |

The public `CaelixMaterial` is 64 bytes. The GPU keeps it as two tables (revised 2026-10-26, see "As
built"): the MAIN table, 48 bytes per code = offsets 0 .. 47 above (`kMaterialBytes` 48, so the main
kernels keep their stride; `kMaterialSssAnisotropy` 40, `kMaterialSssSteps` 44), and the SUBSURFACE
table, 16 bytes per code = (`sss_albedo`, `sss_probability`) (`kSubsurfaceRowBytes`, 65536 rows = 1 MiB,
device address `FrameConstants::subsurface_table` @576). `MaterialTable::WriteGpuChunk` splits every
1024-entry chunk into both tables at upload (one staging span, two copies); both buffers are zero-filled
in the first flush. The subsurface row is the one row that a walk step loads per voxel.

## Walk (`shaders/render/caelix_subsurface.hlsl`, compiled only with `CAELIX_SUBSURFACE 1`)

All in the entity's local voxel grid of the hit (object space, block units).

1. Cursor = the hit voxel (`MediumCursorFromHit` of `scattering-expr`: object_hit - 0.5 n). With
   `CAELIX_RENDER_FLAG_SSS_ENTRY_JITTER (1u << 6)` (added 2026-10-26, host default on,
   `--sss-no-entry-jitter`): the voxel of object_hit - 0.5 n + a uniform offset in [-0.5, 0.5] along the
   two face-tangent axes (two numbers of `rng`, only when the walk runs), reached from the hit voxel by
   one `MediumStep` (a neighbour brick is resolved); if it is not an SSS voxel of the entity (absent
   brick, air, opaque without SSS) the hit voxel. `a` starts as the start voxel's `sss_albedo`.
2. State: cursor, continuous unit direction `d` (start: the ray direction in object space), weight
   `w = 1`, `a` = the current voxel's `sss_albedo` (start: the entry material's). `g` and the step cap
   come from the ENTRY material only.
3. Per step (at most `sss_steps`):
   - `w *= a` (every step is one scattering event in the current voxel);
   - `d = SampleHenyeyGreenstein(d, g)` (g = 0: uniform sphere);
   - `delta` = the signed unit axis of the largest |d| component (ties: x, then y, then z);
   - look at the voxel `cursor + delta` (`MediumStep` + `MediumBlock`; absent brick = air):
     - opaque with `sss_probability > 0`: move there, `a` = its `sss_albedo`;
     - opaque without SSS: a wall, the cursor stays (the step is still used);
     - anything else (air, glass, other transparent codes): EXIT.
4. Exit through the face of the cursor voxel in direction `delta`: point = face centre + uniform
   offsets in [-0.5, 0.5] on the two other axes; normal = `delta`; both go to world space with the
   entity's current transform; origin = point + `K_RAY_ORIGIN_PUSH_OFF` x normal; direction =
   `SafeNormalize(normal + RandomUnitVector(rng))` (cosine lobe, as the diffuse lobe of
   `ShadeVoxelHit`). Result weight = `w`.
5. Step cap reached: dead (the path ends; emission of the entry hit already counted).

Random numbers: the path stream `rng` (the BSDF stream). A material without SSS draws nothing extra, so
scenes without SSS keep their images bit for bit.

## Integration (`caelix_path_trace.hlsl`, `TracePath`)

The exit replaces the vertex: it is a diffuse vertex at the exit point with the exit normal and albedo
`w`. The branch probability cancels: the estimate is the mixture p x SSS + (1 - p) x BSDF.

- After `ShadeVoxelHit` of a hit (primary and every bounce hit) with the frame flag set: if the block is
  opaque and its `sss_probability > 0`, draw `u`; `u < p` runs the walk and overrides the shade:
  `albedo = w`, `bounce_origin`, `bounce_direction`, `specular_lobe = false`, `roughness = 1`, and the
  vertex normal becomes the exit normal. A dead walk sets `albedo = 0` and ends the path after the
  emission of the hit (no NEE, no further ray).
- NEE (sun and lamps) and `segment_normal` use the VERTEX normal (the exit normal after a walk) and
  `shade.bounce_origin` (the exit origin). `LampHitWeight` then gets the same x_r and n_r as the NEE.
- The walk uses the vertex's bounce: no extra bounce count.
- Primary vertex (demodulation): for an SSS material the guide albedo is
  `guide = lerp(albedo, sss_albedo, p)` (written as the primary albedo: G-buffer / NRD demodulation /
  RR diffuse albedo guide); the primary throughput is `branch weight / max(guide, 0.001)` with branch
  weight = `albedo` (BSDF branch) or `w` (walk). Non-SSS materials: unchanged (throughput 1 / k).
  The G-buffer normal, depth and motion stay the entry face (the NRD primary surface).

## Library and host

- `CAELIX_RENDER_FLAG_SUBSURFACE (1u << 5)`: lighting modes run the walk on SSS materials. The renderer
  binds the SSS kernels (`caelix_path_trace_sss`, `caelix_path_trace_sss_no_light_nee`, the main sources
  with `CAELIX_SUBSURFACE 1`) only when the flag is set AND the scene's material table has an SSS entry;
  otherwise it clears the bit and binds the main kernels (their SPIR-V stays identical to `main`).
- Frame constants grow 560 -> 592 B: `reverse_tables` @560, `group_grids` @568 (as on `scattering-expr`),
  `subsurface_table` @576, `pad_584`, `pad_588`.
- Host: SSS on by default, `--no-sss`, a UI checkbox; entry jitter on by default, `--sss-no-entry-jitter`,
  a UI checkbox; JSON fields `sss_albedo` (default [1, 1, 1]),
  `sss_probability` (0), `sss_steps` (16), `sss_anisotropy` (0).
- Not updated: the Unity plugin C# mirror (not used on this branch).

## Known limitations (accepted for the experiment)

- An exit into a glass voxel starts the next ray with the outside medium (the `medium` tracking of the
  glass chain does not know it).
- The walk stays inside the hit entity (another entity's voxels count as air).
- No soft shadows: SSS voxels are opaque for every shadow ray.

## As built: deviations and findings (2026-10-26)

- DEVIATION (RR guide only): the DLSS RR diffuse albedo guide of an SSS material is
  `lerp(albedo * (1 - metallic), sss_albedo, p)` (the existing guide keeps the BSDF's metallic term).
  The G-buffer / NRD demodulation albedo is `lerp(albedo, sss_albedo, p)` as specified. The RR specular
  albedo guide is unchanged (not scaled by 1 - p).
- DEVIATION (cost only, same estimate): a bounce vertex that has no next segment (the last vertex of the
  path) skips the walk; only its emission counts there, so the result does not change.
- REVISED (first version): a 64-byte GPU material table moved `caelix_path_trace_no_light_nee` from 96 to
  128 driver registers by the stride change alone (emberdeep 1080p, no light NEE 3.17 -> 3.27 .. 3.41 ms).
  Now the GPU main table stays 48 bytes and the subsurface row has its own table (see "Material"): the
  main kernels' SPIR-V equals `main` plus the frame constants growth only, and the registers are back at
  128 / 96.
- FINDING -> FIX: a walk that always starts at the hit voxel gives every hit on one voxel face the same
  walk distribution: converged SSS surfaces showed per-voxel brightness steps (large flat faces). The
  entry jitter (Walk step 1) removes them (`Artifacts/sss-20261026/impl/`).
