# Per-face light masks (step 4)

Design (Fable, 2026-10-21). Follows `LIGHT_LIST.md` (step 1), `RESTIR_DI.md` (step 2) and
`LIGHT_MIS.md` (step 3). Replaces the light table entry layout of step 1. The "As built" section at
the end is the truth once the step is implemented.

## Why

Step 1's entry has one 512-bit mask (voxel is a list light) and eight slice weights; the sampler
picks a voxel, then one of the 1..3 axis faces that point to the receiver, with equal probability,
without knowing whether that face is covered by a neighbour voxel. A light voxel set into a ceiling
has one open face out of three candidates, so two of three NEE samples hit the ceiling from inside
and return 0. Measured: 60 % of the pixels lit at 1 spp in the Cornell test box, 88 % of the lit
bistro pixels black per frame (`RESTIR_DI.md`, "Open points"), and it is the reason ReSTIR DI gains
only 0.2..0.5 dB over one-sample NEE (most reservoir candidates are occluded samples).

The fix: the entry stores, per face direction, the set of list-light voxels whose face in that
direction is open, with the luminance sum of that set. The sampler picks a face direction among the
candidates by its weight and a voxel uniformly among the voxels open in that direction. A covered
face is never sampled. Open is still decided inside the brick (a face on the brick border counts as
open; a neighbour brick is not consulted), as in step 1's mean normal.

## 1. Entry layout (448 B, was 128 B)

```
0     uint2 face_mask[6][8]   bit v of slice z of face f: voxel v (x | y << 3 | z << 6) is a list light
                              and its face f (+X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5) is open; 64 B per face
384   float face_weight[6]    luminance sum over the set bits of face_mask[f]
408   float total_weight      luminance sum over the brick's list-light voxels (the CDF weight; the
                              value of the light word)
412   uint group
416   uint id
420   uint entity
424   int3 origin
436   uint pad[3]             -> 448
```

`mean_normal` is gone: the cell fill's orientation term derives it from the face weights,
`normalize(w0 - w1, w2 - w3, w4 - w5)` (0 = none; length 0 when every direction weighs the same).
That is the luminance-weighted version of step 1's open-face direction sum.

HLSL: `LightEntryHeader` (bytes 384..447: the six weights, total, group, id, entity, origin) is what
the sampler, the fill, the DI resolve and the BSDF-side MIS load (`LoadLightEntryHeader`, 4 x 16 B);
a face's mask is loaded on demand (`LoadLightFaceMask(address, face)` -> `uint2[8]`, 4 x 16 B;
`LightFaceMaskBit(address, face, voxel)` -> one 8 B load). The step 1 `LightEntry` with the full
masks disappears from the shaders; the C++ `LightEntry` mirror is the 448 B struct (static_assert).
`kLightEntryBytes = 448`; the CDF starts at `64 + capacity * 448`. The slice-word addressing of the
mask (`uint2` per slice, low half y 0..3, high half y 4..7, bit x + 8 (y & 3)) stays.

Table size at the default `max_bricks` 65536: 29.6 MB (was 8.6 MB); at 1048576: 474 MB
(`maxStorageBufferRange` note of `LIGHT_LIST.md` applies: NVIDIA allows it, the Vulkan minimum does
not; the trace reads the table through addresses, only the compact kernel binds it).

## 2. Compact kernel

Per occupied list-light voxel (as today): for each of the six faces, open = on the brick border or
the neighbour voxel inside the brick is not occupied (the `MaskBit` test that step 1 used for the
normal sum); an open face sets bit v of `face_mask[f][z]` and adds the luminance to
`face_weight[f]`. `total_weight` adds the luminance whatever the faces (it must equal the light
word's rule; a voxel with no open face cannot exist after the finalize kernel's interior removal, and
if one did it would only weigh the CDF). Store: 24 x 16 B of masks, 2 x 16 B of weights (the last
two lanes of the second store carry `total_weight` and `group`), then `id, entity, origin` and the
pad, then the CDF word. Everything else (index allocation, dropped count, the entry map of step 3) is
unchanged.

## 3. Sampling within a brick (`SampleLightEntryWith`)

Given the header and the receiver `x_r`:

1. Candidate faces, per axis `a`, from the receiver's coordinate `t_a = dot(axis_a, x_r - brick
   origin)` in voxel units (the brick spans `0..8`): `+a` is a candidate when `t_a > 0`, `-a` when
   `t_a < 8`; both when the receiver lies within the brick's slab on that axis. This is the smallest
   per-brick set that holds every face a point of the brick can show to the receiver: a `+a` face
   plane lies at `t >= 1`, so `cos_l > 0` needs `t_a > 1`; a `-a` plane at `t <= 7` needs `t_a < 7`.
   A brick-centre sign test (one face per axis) would give pdf 0 to visible faces of a receiver
   inside the slab (the top faces of the lower rows seen from a floor at the brick's height), a
   bias, so it is not used; step 1's per-voxel sign test is replaced, not kept. Inside the slab a
   voxel whose chosen face points away from the receiver gets `cos_l <= 0`: a wasted sample, not a
   bias. The BSDF-side proxy uses the SAME rule (`LightCandidateWeight(d, x_r)`), and 1..6 faces are
   candidates.
2. Face `f` among the candidates with probability `face_weight[f] / W_cand`, `W_cand` = the sum
   over the candidate faces. `W_cand = 0` -> no sample (every candidate face is covered: the NEE
   cannot reach this brick from there; the BSDF side gives such a point `w_bsdf = 1` through the
   mask-bit test of `LightListCanSample`, which now tests the HIT face's bit).
3. Voxel uniformly among the set bits of `face_mask[f]`: `n_f` = the sum of the 16 `countbits`,
   the k-th set bit found slice by slice then `NthSetBit` in the half word.
4. Point uniform on the face (unchanged), `pdf_area = face_weight[f] / W_cand / n_f`.

Random numbers: face, voxel, two for the point (one fewer than step 1's region + voxel + face;
`SampleLightEntryWith` draws the same count in every branch, as today).

## 4. The MIS proxy (`LightMisDesc`)

`pdf_base` becomes receiver-dependent: `LightMisDesc` carries `face_weight[6]`, the sample's
`face` and `inv_n_face = 1 / n_f`, the brick origin (world) and the axes; `LightNeeProxyPdfArea(ctx, d, x_r)`
uses `face_weight[face] * inv_n_face / LightCandidateWeight(d, x_r)` in place of `pdf_base / n_faces`
(`LightCandidateFaces` goes away; the voxel frame is no longer needed in the description). `LightMisDescOf(fc, header, voxel, face)` (BSDF side) counts
`n_f` from one face mask load. `LightListCanSample` returns the hit face and tests
`face_mask[face]` bit v instead of the step 1 mask. `DiResolve` builds the description for the
reservoir's face. The cell fill reads the header and derives the mean normal from the weights.

## 5. Tests (`tests/trace_test.cpp`)

- All step 1..3 cases keep passing (the table bytes in "light table counts one emissive brick" and
  the cell test change to the new entry size).
- "closed faces are never sampled": the step 1 light box (one ceiling voxel, only -Y open) at
  1 spp with light NEE: the lit fraction of the floor pixels rises from ~60 % to > 95 % (every
  floor pixel sees the open face; the box is empty). Print both the old expectation and the
  measured value.
- "faces weigh the brick": a brick with a bright -Y face set (an 8 x 8 panel of strength 3 in the
  ceiling) and the panel's +Y faces covered by a grey slab on top: the NEE mean against the path
  tracer stays at ratio 1.00 +- 0.03 from the floor camera.
- "light NEE with MIS is unbiased" and "ReSTIR DI unbiased" run unchanged (they are the
  proxy's regression tests).

## 6. Bench

`pt_nee`, `pt_nee_nomis`, `di`, `di_unbiased` (+ `_conv`) on the four scenes against the step 3
references (`Artifacts/gi-experiments-20261020/light-mis/`), lit-pixel fraction and PSNR at 1 spp;
the expected gain is in the lit fraction and in the DI's 1-spp PSNR.

## As built (sections 1-5; 2026-10-21)

The bench of section 6 was not run in this step.

### Sizes

| item | size |
| --- | --- |
| light table entry | 448 B (was 128 B), layout as in section 1; C++ `LightEntry` with `static_assert`s on the size and on the offsets 384 / 408 / 424 |
| light table | 64 B header + 452 B per `max_bricks` entry (448 B entry + 4 B CDF): 28.3 MiB at 65536 (`CaelixLightStats::table_bytes` 30146624 with the 2 x 256 KiB of light words and entry map), 452 MiB at 2^20 |
| `LightMisDesc` | 27 values (was 20): brick origin, 3 axes, 6 face weights, face, `inv_n_face`, `brick_total`, brick centre, brick normal |
| ABI | none; comments in `caelix.h` (entry size, `table_bytes`) updated |

### Kernels and code touched

- `common/caelix_light.hlsl`: `LightEntryHeader`, `LoadLightEntryHeader` (4 x 16 B from byte 384),
  `LoadLightFaceMask`, `LightFaceMaskBit` (one 8 B load), `LightBrickNormalLocal` (the mean normal
  from the weights), offset constants; `LightEntry`, `LoadLightEntry` and the octahedral code are
  gone (nothing else used them).
- `scene/caelix_light_compact.hlsl`: `OpenWord(occupied, face, z)` computes the open bits of one
  slice for one face with word operations (x: one bit, masked at the row ends; y: one byte, carried
  between the two halves; z: the next slice). Pass 1 adds each list-light voxel's luminance to the
  total and to the weight of every face whose open bit is set; after the index is allocated, pass 2
  stores `mask[z] & OpenWord(...)` per face (24 x 16 B), then 4 x 16 B of weights, total, group, id,
  entity, origin and pad. `MaskBit` and the normal sum are gone.
- `render/caelix_light_cells.hlsl`: the fill loads the header; `orient` uses
  `LightBrickNormalLocal` rotated by the entity rows (1 when it is 0).
- `render/caelix_light_sample.hlsl`: `LightBrickFrame` (+ `LightBrickFrameOf`),
  `LightCandidateFaceMask`, `LightCandidateWeightSum`, `LightVoxelFrameIn`, the new `LightMisDesc`
  (+ `LightMisDescWith`, `LightMisDescOf(fc, header, address, voxel, face)`), `LightFaceCount`,
  `LightCandidateWeight`; `SampleLightEntryWith(fc, header, address, x_r, quantize_uv, rng)`,
  `LightPointAt(fc, header, ...)`, `LightNeeProxyPdfArea(ctx, d, x_r)`; `LightListCanSample` returns
  the header and the entry address and tests the hit face's mask bit. `LightCandidateFaces` is gone.
- `render/caelix_restir_di.hlsl`: `DiInitialReservoir` and `DiResolve` load the header and pass the
  entry address; `DiResolve` builds the description for the reservoir's face.
- `render/caelix_path_trace.hlsl`: the BSDF-side call site and two comments.
- `lib/src/scene/gpu_structs.h`: the 448 B `LightEntry`, the `LightTableHeader` comment.

### Differences from the design, and why

1. **The proxy keeps the entry's weight.** Section 4 says `face_weight[face] * inv_n_face /
   W_cand` replaces `pdf_base / n_faces`. These two are not the same quantity: step 1's
   `pdf_base / n_faces = region_weight[z] / (n_z n_faces)` is the entry's total weight times the
   in-brick density (the total cancelled against `p_brick`), while the new expression is the
   in-brick density alone. `LightNeeProxyPdfArea` therefore uses `brick_total * face_weight[face] *
   inv_n_face / W_cand` in that place: with a cell `G * brick_total * in_brick / C_c`, without one
   `brick_total * in_brick / total`. Without the factor the proxy would be off by the brick's
   luminance (any scale keeps MIS unbiased, but the weights would be wrong).
2. **One candidate function for both sides.** `LightCandidateFaceMask(origin, axes, x_r)` and
   `LightCandidateWeightSum(weights, mask)` (face order) are called by the sampler and by
   `LightCandidateWeight` of the proxy, so `W_cand` is the same number on both sides.
3. **The brick frame.** The sampler needs the brick's world origin before it knows the voxel, so
   the frame functions start from the brick (`LightBrickFrameOf`: origin corner + axes from the
   entity rows) and derive the voxel frame from it (`LightVoxelFrameIn`); the description's brick
   centre is `origin + 4 (a0 + a1 + a2)`. Step 3 went the other way (voxel frame first). The NEE
   side, the BSDF side and `LightPointAt` all use the same functions, so the numbers agree.
4. **The mean normal threshold.** `LightBrickNormalLocal` returns 0 when the length of
   `(w0 - w1, w2 - w3, w4 - w5)` is at most `1e-6 * total_weight` (a relative test: the weights are
   luminance sums, and two equal sums in different order need not subtract to exactly 0).
5. **`LightMisDescOf` keeps a `voxel` argument** (as the brief's signature), but the description
   does not depend on it: every point of one face set has the same in-brick density.
6. **Candidate t in voxel units assumes unit axes.** `t_a = dot(axis_a, x_r - origin)` is in voxel
   units only for an entity transform without scale, as the rest of the light sampling assumes (the
   face area is taken as 1).
7. **Test "faces weigh the brick"**: the panel lights the box far above 1, so `CompareLight`'s
   block PSNR (block means clamped to [0, 1]) reads 99 dB for every panel case (also in "MIS on a
   large emitter"); the new case computes the block PSNR on both images scaled to a reference mean
   of 0.5. Its slab covers the panel's +Y faces, which the floor camera's receivers never have as a
   candidate (they are below the brick); the case checks the proxy's face weights and the side faces
   on the brick border (open, covered by the neighbour bricks' ceiling: wasted samples, no bias).

### Measurements (`tests/trace_test.cpp`, RTX 5080, 128 x 128, 4 bounces)

All 29 trace test cases pass.

| test | step 3 | step 4 |
| --- | --- | --- |
| closed faces are never sampled (floor and wall pixels lit at 1 spp, light NEE) | about 60 % of all pixels | 15880 of 15884 (100.0 %) |
| cells fill and sample (all pixels lit at 1 spp) | 60 % | 16380 of 16384 |
| faces weigh the brick (covered panel, light NEE + MIS, 1024 vs 4096-frame PT) | - | ratio 1.0009, scaled block PSNR 51.40 dB |
| light NEE is unbiased (1 voxel) | ratio 1.0095, block PSNR 45.78 dB | ratio 1.0062, 45.92 dB (PT vs PT other seeds: 1.0060, 42.04 dB) |
| light NEE with MIS is unbiased | 1.0095, 45.78 dB | 1.0062, 45.92 dB |
| MIS on a large emitter, 64-frame RMSE: light NEE / + MIS | 1.4757 / 1.4578 | 0.96447 / 0.97848 (ratio 1.015) |
| same, ReSTIR DI / + MIS | 2.5797 / 2.3871 | 1.19243 / 1.16880 (ratio 0.980) |
| same, converged ratios (NEE, NEE + MIS, DI, DI + MIS) | 1.0035, 1.0032, 1.0076, 1.0074 | 1.0009, 1.0009, 0.9993, 1.0001 |
| dropped bricks keep their emission (NEE, NEE + MIS) | 1.0032, 39.55 dB | 1.0023, 39.53 dB |
| ReSTIR DI unbiased vs light NEE (without / with MIS) | 0.9943 / 0.9998, pixel PSNR 38.78 / 38.71 dB | 0.9998 / 0.9999, pixel PSNR 54.98 / 54.98 dB (NEE vs NEE other seeds 55.04 dB) |
| single frame RMSE after 64 warm-up frames (NEE, DI `TS`, DI `TSU`, DI `TSU` + MIS) | 0.15678, 0.15367, 0.15484, 0.15430 | 0.04017, 0.04002, 0.04002, 0.04012 |

The single-voxel box: the NEE now samples only the open face, so the 1-spp error of the light NEE
drops by 3.9x, and the pixel PSNR between two 1024-frame accumulations (DI vs NEE) rises from
38.8 to 55.0 dB. ReSTIR DI gains almost nothing over the NEE in this box any more: with one
light voxel and no wasted candidates there is nothing for the reuse to choose between. The panel:
the side faces of the inner panel voxels are covered inside the brick and no longer sampled, so the
64-frame RMSE drops by 1.5x (NEE) and 2.2x (DI). MIS on the panel is now neutral to slightly worse
for the NEE (1.015x) and still helps the DI (0.980x).
