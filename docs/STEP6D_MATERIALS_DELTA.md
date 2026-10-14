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

## Open questions for the owner

1. Smooth metal mirror reflection in the delta chain: Unity parity first (no reflection), then
   add it as an option? Or add it from the start?
2. The checkerboard split halves the effective sample rate on glass pixels and the cross resolve
   softens them by one pixel. Unity accepted this. The alternative is Fresnel-weighted stochastic
   choice everywhere, which is noisier but sharper under NRD. Proposal: port the split as Unity
   has it, measure on the fixture, decide after.
