# Material guide for artists

Status: 2026-10-28, for the materials JSON of the block-ID encoding (`caelix-app --materials <file>`,
Unity `MaterialsJson`). The exact rules are in `docs/MATERIALS.md` and `include/caelix/caelix.h`
("Materials"); this guide explains what each value does to the picture.

## 1. One entry per block id

A material file is a JSON array. One entry per block id. Every key except `id` is optional.

```json
[
  {"id":1,"transparent":false,"albedo":[0.54,0.32,0.06],"smoothness":0.0,"metallic":0.0,"emission":[0,0,0],"ior":1.5,"extinction":0.0},
  {"id":3,"transparent":true,"albedo":[0.96,0.985,1.0],"smoothness":1.0,"metallic":0.0,"emission":[0,0,0],"ior":1.33,"extinction":0.05},
  {"id":10,"transparent":false,"albedo":[0.06,0.14,0.04],"smoothness":0.2,"metallic":0.0,"emission":[0,0,0],"ior":1.5,"extinction":0.0,
   "sss_model":1,"sss_probability":0.6,"sss_albedo":[0.8,0.95,0.4],"sss_anisotropy":0.3,"sss_steps":8}
]
```

- `transparent: false` (opaque) ids go from 0 to 32767. `transparent: true` (glass, water, air-like
  media) ids go from 1 to 511. Opaque and transparent ids are two separate lists: opaque 3 and
  transparent 3 are different blocks. Transparent id 0 is air; do not define it.
- All colours are linear, not sRGB. A mid grey of sRGB 128 is 0.22 here, not 0.5. Values of 0..1 are
  normal for `albedo` and `sss_albedo`; `emission` has no upper limit below 65504.
- One wrong value rejects the whole file. The log names the entry and the reason.

## 2. The keys

| Key | Range | What it does |
|---|---|---|
| `albedo` | 0..1 per channel | Opaque: the diffuse colour. Transparent: the tint of the medium (the colour that passes). |
| `smoothness` | 0..1 | Opaque: 0 = matte, 1 = mirror sharp. Transparent: 1.0 for glass and water; see section 4. |
| `metallic` | 0..1 | Opaque only. 0 = dielectric (paint, stone, wood). 1 = metal: every bounce is a reflection tinted by `albedo`. |
| `ior` | > 0 | Index of refraction. Transparent: bends and reflects light at the surface (water 1.33, glass 1.5, diamond 2.4). Opaque: only the strength of the grazing-angle reflection; keep 1.5. |
| `extinction` | >= 0 | Transparent only. How fast the medium absorbs light per voxel; see section 4. |
| `emission` | >= 0 | Emitted light (radiance). See section 5. |
| `sss_model` | 0 or 1 | 1 switches on subsurface scattering for an opaque block. See section 6. |
| `sss_probability` | 0..1 | Fraction of the light that goes inside the block. |
| `sss_albedo` | 0..1 per channel | Colour kept at every step inside the block. |
| `sss_anisotropy` | -1..1 | Direction of the scattering: 0 = any direction, towards 1 = forward (deep), towards -1 = back (shallow). |
| `sss_steps` | 0..255 | How far a path may travel inside before it is absorbed. |
| `sss_params`, `texture_set`, `connect_group` | | Reserved. Stored, not read. Leave them out. |

The renderer stores every value as a 16-bit float. Small differences in the third decimal are lost.

## 3. Opaque surfaces

At every hit the renderer picks one lobe: diffuse or specular. The specular chance is
`lerp(metallic, 1, fresnel * smoothness)`, and the specular direction is blurred towards the diffuse
direction by `1 - smoothness`.

- Matte (stone, cloth, dirt): `smoothness` 0, `metallic` 0.
- Satin paint, wet stone: `smoothness` 0.3..0.6, `metallic` 0. The reflection is white and strongest at
  grazing angles.
- Polished floor, lacquer: `smoothness` 0.8..1.0, `metallic` 0.
- Metal: `metallic` 1. `smoothness` 1 is a mirror, 0.5 is brushed, 0 is a rough metal that looks like a
  coloured matte surface. Metals take their colour from `albedo` (gold 0.9, 0.7, 0.3; iron 0.55, 0.55,
  0.55).
- Avoid `metallic` between 0.1 and 0.9 unless you want a half-metal look.

## 4. Glass and water (transparent)

A transparent block is a volume. Light reflects or refracts at its faces (`ior`) and loses energy while
it travels inside (`extinction`, `albedo`).

- Keep `smoothness` at 1.0 for glass and water. The renderer then follows the light through up to 5
  interfaces in one step and the denoiser keeps the picture sharp. A value below 0.99 does not blur the
  reflection; it only makes the block behave like a rough bounce (softer, noisier, one bounce used up).
- The light that passes a thickness of `T` voxels is `exp(-(1 - albedo) * extinction * T)` per colour
  channel. A channel with `albedo` 1 is never absorbed. The tint is the `albedo` colour; `extinction`
  sets how thick the glass must be before you see the tint.

| `albedo` | `extinction` | after 4 voxels | after 32 voxels |
|---|---|---|---|
| 1, 1, 1 | any | clear | clear |
| 0.65, 0.95, 0.72 | 0.001 | clear | clear (0.99, 1.0, 0.99) |
| 0.65, 0.95, 0.72 | 0.1 | 0.87, 0.98, 0.89 | 0.33, 0.85, 0.41 (green) |
| 0.8, 0.94, 0.97 | 0.12 | 0.91, 0.97, 0.99 | 0.46, 0.79, 0.89 (blue water) |
| 1.0, 0.5, 0.25 | 0.2 | 1.0, 0.67, 0.55 | 1.0, 0.04, 0.01 (deep red) |

- A rule of thumb: the thickness where a channel is halved is `0.69 / ((1 - albedo) * extinction)`
  voxels.
- Recipes: window glass `albedo` 1, `ior` 1.5; green bottle `albedo` 0.65, 0.95, 0.72, `extinction`
  0.1; sea water `albedo` 0.8, 0.94, 0.97, `ior` 1.33, `extinction` 0.05..0.12; ice `albedo` 0.92,
  0.98, 1.0, `ior` 1.31.
- Shadows: the sun and the lamps shine through glass at the first surface the camera sees, tinted by
  the glass. A red window casts a red shadow. Light that arrives after a bounce treats glass as opaque,
  so indirect light under glass stays a little dark.

## 5. Lamps (emission)

`emission` is the light the block sends out, in the same units as the sun. The test scenes use a sun
radiance of about 20000 (`--sun-radiance 2e4,1.9e4,1.8e4`); a visible indoor lamp is 50..2000, a glowing
marker 1..10.

- A block whose emission luminance (0.21 R + 0.72 G + 0.07 B) is 20 or more is a lamp: the renderer
  aims shadow rays at it, and it lights its surroundings cleanly. Below 20 it only glows and lights the
  surroundings by chance (dim, noisy). The limit is a scene setting (`lightMinEmission`), 20 by default.
- Many small lamps cost more than a few bright ones: every lit brick (8x8x8 voxels) is one entry of the
  light tree. Thousands are fine. A million (every window of a city) slows the scene update.
- `emission_scale` (a render setting) multiplies every emission at draw time; use it to balance a whole
  scene against the sun, not per material.
- A lamp can be any colour. Light tint comes from the emission, not from `albedo`.

## 6. Subsurface scattering (SSS)

For leaves, wax, snow, skin, marble, clouds. An opaque block with `sss_model` 1, `sss_probability` > 0
and `sss_steps` > 0 lets light travel inside it from voxel to voxel and come out somewhere else.

How it works, in short: when light hits the block, with probability `sss_probability` it goes inside.
Inside it moves one voxel per step, keeps the colour `sss_albedo` at every step, and turns by
`sss_anisotropy`. When it leaves the block it continues as diffuse light. When it has used `sss_steps`
steps, it is absorbed (black).

- `sss_probability`: 1 = everything goes inside (soft, translucent, no hard surface shading). 0.3..0.6
  = the surface keeps its `albedo` shading and gets a soft glow on top. Leaves 0.6..1, wax 0.8, snow
  0.5, clouds 1.
- `sss_albedo`: the exit light is `sss_albedo` multiplied once per step, so keep it high. With 8 steps,
  0.8 leaves 17 % and 0.95 leaves 66 %. Use the hue you want to see in the glow (leaves: more green,
  skin: more red). The surface `albedo` can be darker than `sss_albedo`.

| `sss_albedo` per channel | after 4 steps | after 8 steps | after 16 steps |
|---|---|---|---|
| 0.7 | 0.24 | 0.06 | 0.003 |
| 0.8 | 0.41 | 0.17 | 0.03 |
| 0.9 | 0.66 | 0.43 | 0.19 |
| 0.95 | 0.81 | 0.66 | 0.44 |

- `sss_anisotropy` (g): 0 = scatter in any direction. The light stays near the entry, exits soon, and
  the block looks bright and soft. 0.7..0.9 = the light keeps its direction: it goes deep and passes
  through thin objects (sun behind a leaf), but in a thick object it reaches the step cap more often and
  the block looks darker than a matte block of the same colour. Negative values keep the light at the
  surface.
- `sss_steps`: the step cap. The typical depth the light reaches is about
  `sqrt(sss_steps * (1 + g) / (1 - g))` voxels; this is also how far a shadow ray can pass through the
  material. Examples: 8 steps, g 0: 2.8 voxels; 8 steps, g 0.7: 6.7; 6 steps, g 0.8: 7.3; 16 steps,
  g 0.9: 17. Each step costs time; 6..16 is the normal range.
- A thick object needs a low g (0..0.3) and a high `sss_albedo` (0.9+) to stay bright. A thin object
  (leaves, petals, a lampshade) can take g 0.7..0.8 and 6..8 steps.
- Shadows: the sun and the lamps pass through SSS blocks up to the depth above, tinted by `sss_albedo`
  and scaled by `sss_probability`, so a leaf canopy casts a soft green shadow and a thin wax wall glows
  from behind. A thin SSS object may look slightly too bright on its shadow side (known, accepted).
- Cost: a scene with any SSS material runs the SSS kernels (small, fixed cost) plus the steps that
  actually walk. Leave `sss_model` at 0 (or leave the sss keys out) for every material that does not
  need it.

Recipes:

| Material | `albedo` | `sss_probability` | `sss_albedo` | g | `sss_steps` |
|---|---|---|---|---|---|
| Leaf (bright green) | 0.06, 0.14, 0.04 | 0.6 | 0.8, 0.95, 0.4 | 0.3 | 8 |
| Dense foliage | 0.19, 0.38, 0.065 | 1.0 | 0.5, 0.78, 0.2 | 0.7 | 8 |
| Autumn leaf | 0.54, 0.32, 0.055 | 1.0 | 0.8, 0.7, 0.2 | 0.8 | 8 |
| Wax candle | 0.9, 0.85, 0.7 | 0.8 | 0.95, 0.9, 0.8 | 0.2 | 12 |
| Snow | 0.9, 0.92, 0.95 | 0.5 | 0.97, 0.97, 0.98 | 0.0 | 8 |
| Skin | 0.6, 0.4, 0.3 | 0.4 | 0.95, 0.7, 0.6 | 0.5 | 6 |
| Cloud | 0.9, 0.9, 0.92 | 1.0 | 0.98, 0.98, 0.98 | 0.3 | 16 |

## 7. Check your material in the viewer

```
caelix-app --world scene.cxw --encoding block-id --materials scene-materials.json --mode pt
```

- `--sun-radiance 2e4,1.9e4,1.8e4` gives a daylight sun; without it only the sky lights the scene.
- `--no-sss` and `--no-shadow-transmission` switch the two SSS / glass-shadow features off for a
  comparison; `--emission-scale 2` doubles every lamp; `--no-denoise` shows the raw samples; `--mode ref`
  accumulates a clean reference over many frames.
- The F1 panel has the same switches live.
- In Unity, `CaelixNativeRenderer.MaterialsJson` loads the same file at the start of a session.
