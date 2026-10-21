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
