# Materials (as built)

Status: material compaction step, 2026-10-27. The truth for the C side is `include/caelix/caelix.h`
("Materials"); this page describes the layouts behind it. History of the table: `STEP6D_MATERIALS_DELTA.md`.

## Rule

The per-bounce read is one 32-byte hot entry. Subsurface (SSS) and texture data live in side
tables, read only by the code that needs them. The GPU stores floats as IEEE halves.

No kernel reads the subsurface table, `sss_model`, `sss_probability`, `texture_set` or
`connect_group` yet. They are validated, stored, uploaded and readable on the GPU
(`CaelixGetSubsurface`, `CaelixMaterial::sss_probability` / `sss_model` in HLSL).

## C ABI: `CaelixMaterial` (112 bytes)

The host's view: every float linear, and `caelix_scene_get_material` returns exactly what was set
(the library keeps a CPU copy of the ABI structs). `CaelixMaterialEntry` (code + material) is 116 bytes.

| Offset | Field | Range |
| --- | --- | --- |
| 0 | `float albedo[3]` | finite |
| 12 | `float smoothness` | finite |
| 16 | `float emission[3]` | finite |
| 28 | `float metallic` | finite |
| 32 | `float ior` | > 0 |
| 36 | `float extinction` | >= 0 |
| 40 | `uint32_t sss_model` | 0..255, 0 = none |
| 44 | `float sss_probability` | 0..1 |
| 48 | `float sss_albedo[3]` | 0..1 each |
| 60 | `float sss_anisotropy` | -1..1 |
| 64 | `uint32_t sss_steps` | 0..255 |
| 68 | `float sss_params[8]` | per model (none defined yet) |
| 100 | `uint32_t texture_set` | 0..0xFFFF, reserved |
| 104 | `uint32_t connect_group` | 0..0xFFFF, reserved |
| 108 | `uint32_t reserved` | ignored, stored as 0 |

Validation (`MaterialTable::Valid`, every setter): every float finite with |v| <= 65504 (the half
range), plus the ranges in the table. One bad entry rejects the whole call
(`CAELIX_ERROR_INVALID_ARGUMENT`, logged with the reason) and changes nothing.

## GPU tables

Two device-local buffers of 65536 entries by block code (`MaterialTable::kHotTableBytes`,
`kSubsurfaceTableBytes`; 2 MiB each, `CaelixSceneStats::material_table_bytes` = 4 MiB). The CPU
packs them on every write (`lib/src/scene/material_pack.h`, round to nearest even); the flush
stages changed chunks of 1024 entries of both tables together. A transparent code is read without
its face bits (`CaelixMaterialCode`).

`h(x)` = the half bits of x; a word holds two halves, low 16 bits first.

Hot entry, 32 bytes (`FrameConstants::materials`, offset 440; `CaelixGetMaterial`: two 16-byte loads):

| Word | Offset | Content |
| --- | --- | --- |
| w0 | 0 (`kMaterialAlbedo`) | h(albedo.r) \| h(albedo.g) << 16 |
| w1 | 4 | h(albedo.b) \| h(smoothness) << 16 |
| w2 | 8 (`kMaterialEmission`) | h(emission.r) \| h(emission.g) << 16 |
| w3 | 12 | h(emission.b) \| h(metallic) << 16 |
| w4 | 16 (`kMaterialIor`) | h(ior) \| h(extinction) << 16 |
| w5 | 20 (`kMaterialSss`) | h(sss_probability) \| sss_model << 16 (8 bits) \| flags << 24 (0) |
| w6 | 24 (`kMaterialTexture`) | texture_set \| connect_group << 16 |
| w7 | 28 | 0 |

Subsurface entry, 32 bytes (`FrameConstants::subsurface_table`, offset 560; `CaelixGetSubsurface`):

| Word | Offset | Content |
| --- | --- | --- |
| w0 | 0 (`kSubsurfaceAlbedo`) | h(1 - sss_albedo.r) \| h(1 - sss_albedo.g) << 16 |
| w1 | 4 | h(1 - sss_albedo.b) \| h(sss_anisotropy) << 16 |
| w2 | 8 (`kSubsurfaceSteps`) | sss_steps (8 bits; bits 8..31 reserved, 0) |
| w3 | 12 | 0 |
| w4..w7 | 16 (`kSubsurfaceParams`) | h(sss_params[2i]) \| h(sss_params[2i + 1]) << 16 |

The albedo is stored as one minus albedo: an albedo near 1 keeps more precision as a half.

`CaelixMaterialEmission(table, code)` reads only the emission (one 8-byte load); the light NEE
(`LampEmission`) and the light tree build use it. The layout constants are mirrored in C++
(`material_pack.h`) and HLSL (`shaders/common/caelix_material.hlsl`); `caelix_frame_layout_test`
checks that both sides agree.

## Host JSON (`caelix-app --materials`)

A JSON array of entries (`host-rs/caelix-app/src/materials.rs`): `id`, `transparent`, `albedo`,
`smoothness`, `metallic`, `emission`, `ior`, `extinction`, and the optional keys `sss_model`,
`sss_probability`, `sss_albedo`, `sss_anisotropy`, `sss_steps`, `sss_params` (0..8 numbers, the
missing ones 0), `texture_set`, `connect_group` (all default 0).
