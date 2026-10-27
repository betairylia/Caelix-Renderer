# Background input (as built, 2026-10-28)

`CaelixRenderTargets::background` is the host's own image behind the scene. It is an INPUT, not a
target: the library reads it and never writes or transitions it. The ABI contract is in
`include/caelix/caelix.h`.

## Why

A host engine composites the renderer's output with its own sky and objects. Without a background a
camera-ray miss is black (or the procedural sky), so an upscaler receives silhouettes against black
and blends that black into the edge pixels of the upscaled image. With the background the upscaler
input is complete: the upscaled silhouettes blend the scene with the host's image.

## Contract

- Format: any colour format that supports sampled images (RGBA16F, RGBA8, RGBA32F, ...), SAMPLED
  usage. The library creates and caches a 2D view with SAMPLED usage (`Renderer::BackgroundViewFor`;
  the view cache key includes the usage, so a storage view and a sampled view of one image are two
  entries).
- Layout: `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` or `VK_IMAGE_LAYOUT_GENERAL`, kept for the whole
  render call, with the host's writes visible to compute-shader reads (the rule of the sky cubemap).
  No barrier in the render call touches it.
- Size: the RENDER size (with an upscaler the render size of `caelix_upscaler_query`, else the
  common size of the targets).
- `CAELIX_ERROR_INVALID_ARGUMENT` (checked in every mode): format `VK_FORMAT_UNDEFINED`, a size
  other than the render size, a layout other than the two above, or the image of a present target.
- Lighting modes with a colour target only; the primary mode and a render without a colour target
  ignore it.

## Meaning

- Where the camera ray misses the scene (coverage 0, no delta chain), the colour is
  `background.rgb` of the pixel instead of the procedural sky or black; alpha stays 0 (coverage).
  `CAELIX_RENDER_FLAG_SKY_ON_MISS` is ignored for those pixels.
- A delta chain that ends in the sky is a hit (coverage 1) and keeps the sky of its direction
  (cubemap or procedural): a refracted or reflected direction is not the pixel's direction.
- The background is final linear radiance in the host's units: the exposure does NOT apply; the
  sRGB encode of a UNORM colour target does (as to everything); the fog does not (misses are never
  fogged).
- Reference mode: the accumulation holds the scene only (0 at a miss, as before); the background
  replaces the output at the misses after the accumulation, so a changing background does not enter
  the mean.
- `CAELIX_RENDER_FLAG_FLIP_Y`: pixel (px, py) reads the background at the stored row
  `height - 1 - py` (`CaelixOutputPixel`), the convention of every host image. So a host whose images
  are bottom row first passes its background in that same convention.
- With an upscaler the composite writes the background into the upscale input at the misses; the
  upscaler output and the encode follow as before.

## Composite path

- `FrameConstants::background` (offset 556, the former `pad_556`): 1 = the composite reads binding
  12 at misses.
- Composite set binding 12 (`kCompositeBackgroundBinding`, `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE`,
  PARTIALLY_BOUND; written only when the background is used), `Texture2D<float4> g_background` in
  `shaders/render/caelix_composite_set.hlsl`, read with `Load` (no sampler). The descriptor pool of a
  frame slot has 4 sampled images (was 3).
- `caelix_composite.hlsl`: the miss branch skips the sky when `fc.background = 1`; after
  `color *= fc.exposure` a miss takes `g_background.Load(int3(CaelixOutputPixel(fc, pixel), 0)).rgb`.
  The cross resolve passes misses through (they are never flagged).
- Registers: `caelix_composite` 22, no spill.

## Tests

- `caelix_trace_test` "background: ...": a floor seen from above along +Z (38192 miss and 27344 hit
  pixels at 256 x 256), path trace mode, exposure 2. Every miss equals the RGBA32F background
  exactly, with and without `SKY_ON_MISS` and with `FLIP_Y`; the hits equal the render without a
  background bit for bit; reference mode over 4 frames with two alternating backgrounds: the hits
  equal the plain 4-frame mean exactly, the misses show the last background; the four argument
  checks; the primary mode output is unchanged.
- `caelix_dlss_test` "background and output-size depth with an upscaler": a background at the
  output size is refused, at the render size accepted. SR at 640 x 360 Quality, a constant
  background (0.2, 0.5, 0.8), 3 jittered frames: over the 86340 output pixels whose 5 x 5
  neighbourhood of voxel ids is all misses the largest relative deviation is 0.27 %. With a 3 x 3
  neighbourhood it is 6.4 %: at 1.5x DLSS still mixes the scene into pixels 2 output pixels from a
  hit (printed, not checked).
