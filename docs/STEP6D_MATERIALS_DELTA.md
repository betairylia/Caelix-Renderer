# Step 6d: material table, specular, glass delta chain

Status: design, 2026-10-14. Reads after the moving-entity test. Owner order: UI, moving entities,
then this.

## Where we are

The shading port already has the specular lobe logic of Unity's `CaelixShadeVoxelHit` and writes
the NRD specular channel with roughness `1 - smoothness`. It never runs: packed-scene materials
decode to smoothness 0 and metallic 0, and glass ids are shaded as a tinted opaque. So three
things are missing, in dependency order:

1. a material table the kernels read, instead of the decode-only `MaterialOf(code)`;
2. the glass palette for packed worlds and a host-supplied table for block-ID worlds;
3. the delta chain: reflection and refraction through smooth interfaces with primary surface
   replacement (PSR), the checkerboard split at the first interface and its cross resolve, and
   medium extinction.

Specular on opaque materials then works by itself: the lobe code is live as soon as a material
has smoothness or metalness.

## 1. Material table

One GPU buffer of 65536 `CaelixMaterial` entries, indexed by the raw 16-bit block code with the
face bits masked off, exactly like Unity's `g_Materials` (`CaelixMaterialTable.hlsl`). 48 bytes
per entry, 3 MiB, device local, device address in the frame constants.

```c
typedef struct CaelixMaterial {
  float albedo[3];     /* linear */
  float smoothness;    /* 0..1; NRD roughness = 1 - smoothness */
  float emission[3];   /* linear radiance */
  float metallic;      /* 0..1 */
  float ior;           /* transparent entries */
  float extinction;    /* per voxel unit; transmittance = exp(-(1 - albedo) * t * extinction) */
  float pad[2];
} CaelixMaterial;      /* 48 B */

/* Replaces the whole table. count <= 65536; entries past count keep the built-in default. */
CAELIX_API CaelixResult caelix_scene_set_materials(CaelixScene*, const CaelixMaterial* entries, uint32_t count);
/* Packed worlds: replaces the glass palette (ids 1..255) used to build the table. */
CAELIX_API CaelixResult caelix_scene_set_glass_palette(CaelixScene*, const CaelixMaterial* entries, uint32_t count);
```

Default content per encoding, built by the library at scene creation:

- Packed: codes with bits 15..14 set decode as today (opaque rgb5, emissive rgb4 with
  `exp2(3 * strength)`), smoothness 0, metallic 0, ior 1.5. Codes 1..255 come from the glass
  palette. The built-in palette is the 132 entries of `Caelix/Tools~/RawColor/glass-palette.json`,
  checked in as a generated C array (`lib/src/scene/glass_palette_default.inc`, with the
  generator script next to it). Entries 133..255 are clear glass with ior 1, as Unity does.
- Block ID: every entry is the pastel hash, smoothness 0, until the host sets the table. Titania
  will pass its `VoxelMaterials` table through the Unity bridge.

The kernels replace `MaterialOf(code)` with `GET_MATERIAL(code)` reading the buffer. The decode
rules move out of the shader into the table builder. `IsOpaque` stays a code test, as in Unity.

## 2. Delta chain (port of `AdvanceDeltaRay` and the primary loop)

New option `bounce_count_transparent` (Unity default 5). The primary stage becomes a loop:

```
trace primary; T_virtual = 0
for up to bounce_count_transparent:
  if miss: deterministic += throughput * sky(dir); break
  extinction of the medium we were in: throughput *= exp(-(1 - albedo_medium) * t * ext_medium)
  T_virtual += t
  material = GET_MATERIAL(code)
  if smoothness < 0.99 or (opaque and metallic < 0.5): stop, this is the surface   (Unity rule)
  if opaque: stop (Unity has reflection on smooth metals as TODO; see below)
  transparent interface: eta = ior_from / ior_to, reflect dir, refract dir, Fresnel
  first interface: checkerboard split by pixel parity (reflect on odd, refract on even),
     throughput *= 2 * (branch weight), mark PSR and the lobe (reflect = specular)
  later interfaces: choose by Fresnel with the RNG
  medium = refracted ? destination id : source id
  ray = (hit + push-off along +-interface normal, chosen dir)
```

After the loop, the surface that stopped the chain is "the primary surface" for everything that
follows: the G-buffer writes its albedo and normal, the depth target gets `T_virtual` along the
camera ray (PSR depth, as Unity), the motion vector comes from that surface's previous position,
and the stochastic shading runs there with `throughput` carried in. Radiance that went through a
transparent interface lands in NRD's specular channel with roughness 0 and the first-segment hit
distance, the lobe rule of `CaelixShadeVoxelHit` already does this.

Smooth opaque metals: Unity returns without reflecting. We keep parity in the first version and
add the one-line mirror reflection as an option afterwards, because Titania materials do have
metallic values.

Face bits: the DDA already reports a transparent hit only at a medium crossing, and the finalize
kernel writes the face bits, so the medium-tracking state is the same as Unity's
`previousTransparentMaterial`.

## 3. Cross resolve

The split leaves even and odd pixels with different branches. Unity resolves on the composited
colour, unconditionally, keyed by a per-pixel flag: `0.5 * centre + 0.125 * each orthogonal
neighbour` restricted to flagged pixels, renormalised. We do the same: the composite writes to an
internal colour image and sets the flag in the motion target's w, then a `caelix_cross_resolve`
kernel writes the host target. Reference mode skips it (accumulation averages both parities).

## 4. Tests

- Table: packed table matches the decode of 6a bit for bit for every opaque and emissive code;
  glass entries match the JSON; block-ID table round trip through `caelix_scene_set_materials`.
- Glass slab: a 1-voxel clear glass pane in front of a red wall, black sky, camera straight on.
  Reference-mode result at the centre = red times the Fresnel transmission squared at normal
  incidence for ior 1.55 (about 0.906 after two interfaces) within 2 percent. At 70 degrees the
  reflection branch dominates; check the ratio against the Fresnel formula.
- Extinction: an 8-voxel thick amber-7 block (extinction 0.032): transmittance per channel
  `exp(-(1 - tint) * 8 * 0.032)` within 2 percent.
- Total internal reflection: a ray inside glass at a grazing exit has no refracted branch and the
  pixel stays unsplit (flag 0).
- Specular: a block-ID scene with a host table where the floor is smooth metal (smoothness 1,
  metallic 1): the reflection of an emissive brick appears at the mirrored position; NRD specular
  channel receives it (nrd test variant).
- Fixture world `packed-materials.cxw` (glass and emissive): reference and NRD images for the
  artifact folder; no magenta pixels; validation silent.

## 5. Host

`--bounces-transparent N`, a `materials.json` loader for block-ID worlds (`--materials path`),
UI controls for both. The fixture pose from the step-5 artifact README frames the glass strip.

## Owner decisions (2026-10-14)

1. Smooth metal mirrors: Unity parity. The delta chain stops at an opaque surface; mirrors
   reflect through the stochastic specular lobe and NRD's specular channel, as Unity does. A
   deterministic mirror path is a later quality option.
2. First-interface choice: start with the plain Fresnel-weighted stochastic choice at every
   interface (no checkerboard, no cross resolve, less code). Measure on the fixture under NRD. The
   known risk is that a glass pixel's G-buffer flips between the reflected and the refracted
   surface each frame and REBLUR's history resets on glass (boiling, flashing, validation view
   shows disocclusion). If that happens, port the checkerboard split and cross resolve as the fix.
   Section 3 therefore applies only in that case.

## As built (2026-10-14, uncommitted after d5925da)

Implemented as designed in sections 1, 2, 4 and 5 with the owner decisions above (no
checkerboard, no cross resolve). README "Materials and glass (step 6d)" is the user
documentation. Differences and decisions taken during the implementation:

1. **Air is smooth.** The packed table's air entry (id 0) and the reserved ids 133..255 are
   Unity's `GetSharedGlass` default: albedo 1, smoothness 1, ior 1, extinction 0. Smoothness
   0 for air (as first proposed) would end the chain at every glass-to-air exit, because the
   DDA reports that exit as a hit on an air cell. The block-ID default table uses the same
   vacuum entry for id 0. (Titania's block-ID table has smoothness 0 for air; the chain then
   ends at an exit as in Unity, and the transparent shading branch continues the path.)
2. **Face bits** are removed in the shader before the lookup (`CaelixMaterialCode`), not by
   filling every face-bit variant in a host table. The built-in tables fill the variants
   anyway, so a raw dump is consistent; `caelix_scene_get_material` masks them too.
3. **set_materials semantics**: replaces the whole table; entries past `count` get the
   built-in default, whose transparent codes use the CURRENT glass palette. A later
   `set_glass_palette` rewrites the transparent codes.
4. **Budget**: up to `bounce_count_transparent` interfaces are crossed; the hit after them is
   the surface (shaded by the transparent branch with the correct current medium). Unity
   crosses one fewer and shades the last interface with the medium already advanced (the bug
   noted in the 2026-09-04 denoiser review); the native version has neither difference.
5. **Chain into the sky**: deterministic = transmission x sky (with the sun disk), coverage 1,
   internal view z 1e30 (outside NRD's range, so NRD neither uses nor writes the pixel), host
   depth / view_z from the chain length to the last interface, G-buffer normal and albedo of
   the last interface.
6. **NRD lobe**: build switch `CAELIX_PSR_LOBE` in `caelix_path_trace.hlsl`; 0 (default, as the
   brief): any interface on the camera ray sends the path to the specular target with
   roughness 0; 1: only a first-interface reflection does, a refraction keeps the surface's
   lobe and roughness (Unity's `BOUNCEIDX_IS_SPECULAR`, the usual PSR practice). Both were
   measured (below); the difference is small because the main noise source is elsewhere.
7. **Motion under PSR**: Unity's rule, the virtual point on the camera ray moved by the
   replaced surface's own motion (its previous minus current world position). Without an
   interface the 6a formula runs unchanged.
8. **Host**: `--encoding packed|block-id` was added so that `--materials` is usable (the host
   created packed scenes only). JSON entries take an optional `"transparent": true` to map to
   code `id` instead of `0x8000 | id`.
9. The GPU table upload is chunked (64 x 48 KiB, dirty mask), staged after everything else in
   the flush; the first flush zero-fills the buffer.

### Results

- Tests: see README "Tests" (`caelix_material_test`, the step 6d cases of `caelix_trace_test`).
  Slab normal incidence +0.16 %, 60 degrees +0.64 %, amber-7 extinction within +0.24 %, total
  internal reflection +0.06 % with 0 leakage, mirror 3.592 / 3.608.
- Regression: citadel primary overview byte-identical to step 5; denoised detail 38.42 dB
  against the 6a reference (unchanged).
- Glass under NRD (`Artifacts/native-renderer-step6d-20261014/README.md`): the known risk
  happened. On the fixture's glass the denoised image changes from frame to frame by
  19.7 dB PSNR (temporal std 7.1 levels) against 59.8 dB (0.11 levels) on sky-lit opaque
  faces. With 0 stochastic bounces it is still 21.1 dB: the main source is the stochastic
  reflect/refract flip itself, which changes the albedo and deterministic targets (emission
  seen through the glass, sky in the reflection) every frame, and those targets are not
  denoised. `CAELIX_PSR_LOBE` 1 changes the NRD part only slightly (21.8 instead of 20.9 dB
  against the reference). NRD's validation view shows the glass pixels with a near-zero
  specular history (frequent resets). Conclusion: the plain stochastic first-interface choice
  is not good enough under NRD; section 3 (checkerboard split + cross resolve, which keeps both
  branches every frame and removes the flip from the G-buffer of a pixel) is the next step.

## Follow-up as built: checkerboard split + cross resolve (2026-10-14, uncommitted after 5debe32)

The owner's decision 2 condition was met, so section 3 is implemented (Unity parity):

- `CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD` (bit 3; host default on, `--no-checkerboard` and a
  Render-window checkbox to compare). Path trace mode only. At the first interface with a
  refracted direction: even `px + py` refracts, odd reflects, weight `2 (1 - F)` / `2 F`, no
  random number; total internal reflection unsplit. Later interfaces stay random.
- Flag in the internal motion target's w; also in the host motion target's w, which now
  accepts RGBA16F / RGBA32F (RG formats drop it).
- **Albedo headroom**: the RGBA8 albedo target cannot hold a throughput of up to 2 (Unity's
  8-bit albedo target clamps it; a known Unity defect from the 2026-09-04 review). The trace
  stores half of a split pixel's albedo and the composite doubles flagged pixels.
- **Cross resolve** (`shaders/render/caelix_cross_resolve.hlsl`): the composite writes its
  linear output (after the exposure) to a new internal RGBA16F `resolve` image; the cross
  resolve blends flagged pixels exactly as Unity's `CaelixCrossResolve` (0.5 centre + 0.125
  per flagged orthogonal neighbour, renormalized, edge-clamped), then does the sRGB encode and
  the NRD validation overlay (moved out of the composite into the shared
  `render/caelix_composite_set.hlsl`). One descriptor set serves both kernels (binding 8 =
  resolve image); its GPU time is part of `composite_ms`.
- **Deviation from the follow-up brief**: the reference mode ignores the flag. With a fixed
  branch per pixel the accumulation converges to a checkerboard (each pixel to 2 (1 - F) or
  2 F times one branch), not to the image; only a region average would be right. The reference
  therefore stays the stochastic, per-pixel unbiased estimate, and the test compares the
  checkerboard path trace mode against it.

### Results (fixture close pose, static camera, NRD, screenshots 5 frames apart)

| Glass region | Frame-to-frame PSNR | Temporal std | NRD vs reference |
|---|---|---|---|
| stochastic (6d) | 19.7 dB | 7.1 levels | 20.9 dB |
| checkerboard + cross resolve | 39.6 dB | 1.1 levels | 27.7 dB |
| stochastic, 0 bounces | 21.1 dB | 3.8 levels | 21.3 dB |
| checkerboard, 0 bounces | 41.1 dB | 0.13 levels | 35.3 dB |
| opaque diffuse (front pose), either | 59.8 / 60.0 dB | 0.11 levels | 40.5 / 40.4 dB |

The 0-bounce checkerboard case is not bit-stable on the fixture (41 dB, not infinite): its glass
voxels are single voxels with different ids side by side, so many camera rays cross a second and
third interface, which are still random. On a single-interface scene (the test) it is bit-exact.
Residuals in the image: isolated dark samples from those later interfaces are spread by the cross
resolve into small "+" shapes; edges of glass faces are softened by one pixel; no parity pattern
is left (mean even-odd difference 0.22 levels against 0.02 in the reference). Citadel: primary
byte-identical, detail 38.41 dB (run-to-run NRD variation of the same command is about 57 dB).
