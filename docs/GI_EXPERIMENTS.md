# GI experiments (branch `gi-experiments`, started 2026-10-20)

Goal: pick a global-illumination direction for the native renderer by measuring candidates against
the 1 spp path tracer, in the Rust host, diffuse only, no sun NEE, no denoiser. The question the
owner wants answered: can resampling give reliable renderings of scenes lit by **emissive voxels**
without storing an emissive voxel list (no light list, no NEE on emitters). Emitters are found by
BSDF-sampled paths and kept alive by reuse.

Owner decisions (2026-10-20):
- Emitters in scope: emissive voxels (packed encoding, light = albedo * 8^s * 0.1). Sky is black in
  the emissive tests. No light list of any kind.
- World-space cells (candidate 3): two levels chosen by projected footprint, voxel face near the
  camera, brick face far away; hash key (entity, level, quantized position, 3-bit normal).
- Build order: screen-space ReSTIR GI first (this document's first phase), then the brick-keyed
  world-space variant, then per-brick light reservoirs (ReGIR shape) only if needed.
- Test scenes: `F:/RP_Games/Titania/saves/bistro-combined.8192.packed.cxw` (many emitters; pose
  from the owner's free cam `--pose 1761.5,239.9,2456.0,-1.119,-127.260 --fov 60`),
  `F:/RP_Games/Caelix-family/VoxelTestScenes/aigc/ashwick-r2.packed.cxw`
  (`--pose 3358.5,56.7,1907.9,1.900,44.400 --fov 60.0`, black sky, no sun), and a Cornell-box
  scene with one small bright emissive voxel as the only light (ScenesPlayground project
  `Artifacts/gi-cornell-20261020`).

## Why these candidates (literature summary)

- **ReSTIR GI** (Ouyang et al. 2021; RTXDI 2.x `GI/` headers; kajiya): screen-space. One
  reservoir per pixel holds the visible point, the sample point, the outgoing radiance at the sample
  point, W and M. Temporal reuse through motion vectors, spatial reuse with the solid-angle
  Jacobian of the reconnection shift, geometric rejection by normal and depth. Only the first bounce
  gets many samples. In our setting the sample point of a path that hits an emissive voxel IS a
  light sample, so ReSTIR GI doubles as light discovery without a list. The unbiased candidate
  technique is the BSDF sample, which can reach every emitter, so unbiased MIS is possible without
  a light list (unlike ReGIR, whose defensive samples need power sampling over a list).
- **ReSTIR DI** needs points on emitters. Without a list its only form is the sky as an
  environment light, which is the "first segment misses" part of ReSTIR GI. Not a separate
  experiment.
- **World-space reservoirs**: Boissé 2021 / AMD GI-1.0 (light reservoirs per hash cell, temporal
  only, decay, darkening bias accepted), ReGIR (Boksansky 2021, Clabault 2025: normal in the hash
  key, cells only where rays hit, jittered lookup, 64 + 12 reservoirs per cell, canonical
  reservoirs for unbiasedness, spatial correlation when cells hold few reservoirs), Zhang & Wang
  2023 (ReSTIR GI samples in a world grid with a normal-aware key, reuse at non-primary vertices,
  17 to 42 % lower MSE than ReSTIR GI at 4 to 8 % extra time). kajiya combines screen-space ReSTIR
  GI for the first bounce with a sparse world cache (65k entries, 16 directional reservoirs each)
  for the multi-bounce tail. That split is the target shape for phase 2.
- Brick keys are entity-local: a moving entity carries its cache, no cell-size tuning, and the dirty
  brick list can age exactly the cells an edit touched (fits the "no scene version counter" rule).
- Patents: NVIDIA holds patents on ReSTIR, AMD on spatial hashing of world-space reservoirs.
  Irrelevant for the experiment; the owner decides for a shipped game.

Sources: [RTXDI ReSTIR GI](https://github.com/NVIDIAGameWorks/RTXDI/blob/main/doc/RestirGI.md),
[kajiya GI overview](https://github.com/EmbarkStudios/kajiya/blob/main/docs/gi-overview.md),
[Boissé 2021](https://gpuopen.com/download/SA2021_WorldSpace_ReSTIR.pdf),
[GI-1.0](https://arxiv.org/abs/2310.19855), [Clabault ReGIR](https://tomclabault.github.io/blog/2025/regir/),
[Zhang & Wang 2023](https://wangningbei.github.io/2023/ReSTIR.html) (code
[Otaku-hy/WorldSpaceReSTIRGI](https://github.com/Otaku-hy/WorldSpaceReSTIRGI)),
[Wyman et al. 2023 course](https://intro-to-restir.cwyman.org/).

## Phase 1: screen-space ReSTIR GI (this phase)

### Scope

- Diffuse only: `CAELIX_GI_FLAG_DIFFUSE_ONLY` forces the Lambert lobe at every vertex (packed
  scenes have smoothness 0 anyway). Glass chain off in the experiments (`bounce_count_transparent`
  0); PSR surfaces are not handled by the reservoir logic (documented, not guarded).
- Sun NEE off, NRD off, fog ignored by the GI path (fog on the GI segments is not traversed).
- Sky: a sample whose first segment misses is a "sky sample" (direction stored, Jacobian 1, radiance
  = the same value the path tracer adds on a miss). Black sky in the emissive tests.

### ABI (addition, no struct of the existing ABI changes)

The Unity C# bridge compares struct sizes with the plugin, so `CaelixRenderOptions` stays as it is.
New in `include/caelix/caelix.h`:

```c
typedef enum CaelixGiMethod
{
    CAELIX_GI_PATH_TRACE = 0, /* the path tracer as before */
    CAELIX_GI_RESTIR = 1      /* screen-space ReSTIR GI on the first diffuse bounce */
} CaelixGiMethod;

#define CAELIX_GI_FLAG_TEMPORAL (1u << 0)     /* temporal reuse (previous frame's reservoir) */
#define CAELIX_GI_FLAG_SPATIAL (1u << 1)      /* spatial reuse */
#define CAELIX_GI_FLAG_VISIBILITY (1u << 2)   /* a shadow ray per reused neighbour sample (occluded -> weight 0) */
#define CAELIX_GI_FLAG_UNBIASED (1u << 3)     /* Z-count MIS instead of 1/M (implies VISIBILITY rays per neighbour) */
#define CAELIX_GI_FLAG_DIFFUSE_ONLY (1u << 4) /* force the Lambert lobe at every vertex */

typedef struct CaelixGiOptions
{
    uint32_t method;          /* CaelixGiMethod */
    uint32_t flags;           /* CAELIX_GI_FLAG_* */
    uint32_t temporal_m_cap;  /* 0 = 20; the previous reservoir's M is clamped to this before merging */
    uint32_t spatial_samples; /* 0 = 5; 1 .. 8 neighbours per pass */
    uint32_t spatial_passes;  /* 0 = 1; 1 .. 2 */
    float spatial_radius;     /* pixels, 0 = 30 */
    float depth_threshold;    /* 0 = 0.1: |x_v(q) - x_v(r)| must be below threshold * view_z(r) */
    uint32_t reset;           /* 1 = drop the reservoir history with the next render */
} CaelixGiOptions;

/* Sets the GI options of the renderer; they apply from the next render on and stay until the next
   call. Zero-initialised options = the path tracer. Lighting modes only (the primary mode ignores
   them). CAELIX_MODE_REFERENCE accumulates the chosen method's output, so a biased method converges
   to its biased image (that is how the bias is measured). Validation: unknown method or flags,
   spatial_samples > 8, spatial_passes > 2, non-finite or negative floats -> INVALID_ARGUMENT. */
CAELIX_API CaelixResult caelix_renderer_set_gi_options(CaelixRenderer* renderer, const CaelixGiOptions* options);

typedef struct CaelixGiStats
{
    float gi_ms;              /* GPU time of the GI passes (temporal + spatial + shade) of the last resolved frame */
    uint64_t reservoir_bytes; /* bytes of the reservoir buffers currently allocated */
} CaelixGiStats;
CAELIX_API void caelix_renderer_get_gi_stats(const CaelixRenderer* renderer, CaelixGiStats* out);
```

`CAELIX_ABI_VERSION` is not bumped (additions only). `caelix-sys` picks the new declarations up
through bindgen.

### Reservoir (48 bytes, `shaders/render/caelix_restir_gi.hlsl`, mirrored in C++ only as a size)

```hlsl
struct GiReservoir
{
    float3 sample_pos;     // x_s world position; for a sky sample the unit direction
    uint sample_flags;     // bits 0..2 face index of n_s (0..5), bit 3 sky sample, bit 4 valid
    float3 radiance;       // L_o(x_s -> x_v), linear RGB (emission + continuation bounces)
    float W;               // unbiased contribution weight (0 when invalid)
    float3 visible_pos;    // x_v world position (the pushed-off bounce origin of the primary surface)
    uint visible_flags_m;  // bits 0..2 face index of n_v, bits 8..31 M
};
```

Face index <-> normal: the six axis directions (+X, -X, +Y, -Y, +Z, -Z). Voxel normals are
axis-aligned, so the geometric similarity test is "same face index".

Target function at a visible point r for a sample S: `p_hat_r(S) = luminance(L_o) * max(0, dot(n_v_r, w)) / pi`,
`w = normalize(x_s - x_v_r)` (sky: the stored direction). Albedo is not in the target (demodulated:
the composite multiplies by the albedo, as for the path tracer).

Jacobian of moving a sample from visible point q to r (reconnection shift):
`J = (cos_phi_r / cos_phi_q) * (d_q^2 / d_r^2)`, `cos_phi = max(0, dot(n_s, normalize(x_v - x_s)))`,
`d = |x_v - x_s|`. Sky samples: J = 1. A sample with `cos_phi_r <= 0` (the new visible point is
behind the sample surface) gets weight 0.

Merge (GRIS with 1/M MIS weights): for candidates i with target `p_hat_i` (after the shift to r),
weight `w_i = p_hat_i * W_i * M_i * J_i`; weighted reservoir selection; `M = sum M_i`;
`W = w_sum / (M * p_hat_r(Y))` (0 when `p_hat_r(Y) = 0`). This is the "biased" variant (no
visibility, 1/M). `VISIBILITY`: before merging, every reused neighbour sample gets a shadow ray
from `x_v_r` to `x_s` (sky: toward the direction, `t_max`); occluded -> `w_i = 0` (still counted
in M). `UNBIASED`: after the selection, `Z = sum of M_i over the candidates i for which p_hat_i(Y) > 0
and Y is visible from x_v_i`, `W = w_sum / (Z * p_hat_r(Y))` (the DI paper's unbiased combination;
one shadow ray per candidate for the selected sample). The initial sample needs no visibility test
(it was traced).

Initial sample (in the trace kernel, `TracePath`): the first bounce direction is cosine sampled
(pdf `cos / pi`); its hit gives `x_s`, `n_s`, `L_o = emission(x_s) + continuation`, where the
continuation is the existing bounce loop restarted with throughput 1 at `x_s` (bounces 2..N, RR,
sun NEE off). `W = pi / cos_theta_v`, `M = 1`. A miss gives a sky sample with `L_o` = the value the
path tracer adds on a miss. In expectation `f * W = L_o`, so a run with temporal and spatial off
must match the path tracer (self-check: the two reference accumulations agree to within noise).

### Passes and buffers

Four reservoir buffers of `width * height * 48` bytes each (device local, device address), owned
by the renderer, created with the internal targets and recreated with them (size change):
`initial` (written by the trace kernel), `prev` (last frame's final reservoirs), `temp`, `out`.
At 1920x1080 that is 398 MB, at 2560x1440 708 MB (within the owner's 1 GiB budget; shrinking the
reservoir to f16 radiance comes later if needed).

Dispatch order inside `Renderer::Render` when `method == RESTIR` and the mode is a lighting mode
with `bounce_count >= 1`:

1. trace kernel (as today; writes the G-buffer, `deterministic`, and the initial reservoir;
   `g_int_diff` is NOT final yet).
2. `caelix_restir_gi_temporal` (TEMPORAL flag; else a copy initial -> temp is done by running the
   same kernel with the flag off): reads `initial`, `prev` (when `gi_history_valid`), the internal
   motion target (previous pixel = `CaelixSamplePosition + motion.xy`, rounded); rejects the previous
   reservoir when invalid, face index differs, or `|prev.visible_pos - x_v| > depth_threshold * view_z`;
   clamps `prev.M` to `temporal_m_cap`; merges; writes `temp`.
3. `caelix_restir_gi_spatial` x `spatial_passes` (SPATIAL flag): pass p reads `temp` (p even) or
   `out` (p odd) and writes the other; neighbours uniform in a disk of `spatial_radius` pixels,
   `spatial_samples` of them, rejected like the temporal candidate; Jacobian applied; VISIBILITY /
   UNBIASED as above.
4. `caelix_restir_gi_shade`: reads the final buffer (`temp` after 0 or 2 passes, `out` after 1) and
   writes `g_int_diff` (REBLUR packing like the path tracer, radiance `L_o(Y) * cos / pi * W`, hit
   distance `|x_s - x_v|`, sky `t_max`) so the composite and the reference accumulation stay
   unchanged. Then the renderer rotates the buffers so that the final one is next frame's `prev`.

Every pass is a compute kernel over the render size with the TRACE descriptor set (TLAS for shadow
rays, internal targets), the frame constants as the push constant, and `TracePush` grown to 16 bytes
(`uint32_t gi_pass` = the spatial pass index, `uint32_t pad`); all kernels declare the same push
struct. Barriers: a compute -> compute memory barrier (storage write -> storage read) between every
pass. GPU timer `gi_timer_` around passes 2..4 -> `CaelixGiStats::gi_ms`.

Frame constants (internal, both mirrors) grow from 544 to 608 bytes:

```
uint64_t gi_initial, gi_prev, gi_temp, gi_out;   // 544, 552, 560, 568  reservoir buffer addresses
uint32_t gi_method, gi_flags, gi_temporal_m_cap, gi_spatial_samples;  // 576 .. 588
float gi_spatial_radius; uint32_t gi_spatial_passes; uint32_t gi_history_valid; float gi_depth_threshold; // 592 .. 604
```

`gi_history_valid` is 0 on the first GI frame, after `reset`, after a size change, a mode change,
`reset_accumulation`, or a frame rendered with another method. Random streams: `InitRandom(pixel,
frame_seed ^ <per-kernel constant>)`.

When `method == PATH_TRACE` nothing of this runs (the trace kernel shades as before); the reservoir
buffers stay allocated until the renderer is destroyed.

### Host (`host-rs/caelix-app`)

- `settings.rs`: `GiSettings { method, temporal, spatial, visibility, unbiased, diffuse_only,
  temporal_m_cap, spatial_samples, spatial_passes, spatial_radius, depth_threshold }` ->
  `CaelixGiOptions`; `caelix_renderer_set_gi_options` is called before the render when the settings
  changed (with `reset = 1` on a method or flag change).
- CLI: `--gi <pt|restir>` (default pt), `--gi-no-temporal`, `--gi-no-spatial`, `--gi-visibility`,
  `--gi-unbiased`, `--gi-m-cap N`, `--gi-spatial-samples N`, `--gi-spatial-passes N`,
  `--gi-radius F`, `--gi-depth-threshold F`. Diffuse only is always on in the host for now.
- F1 UI: a "GI" collapsing header with the same controls, plus `gi_ms` and the reservoir MB in the
  status line / timings.
- The timings log (P key and the screenshot flow) prints `gi_ms`.

### Measurement protocol

Tool `tools/image_metrics.py ref.png test.png [--crop x0 y0 x1 y1]`: PSNR (8-bit RGB), RMSE,
mean brightness ratio test/ref per channel (bias), and the fraction of pixels brighter than 2x the
reference's 99th percentile (fireflies). Runner `tools/gi_bench.py` (reads a JSON list of
scene/pose/size entries; runs the host per method; writes PNGs and a markdown table to
`F:/RP_Games/Caelix-family/Artifacts/gi-experiments-20261020/` with a README).

Per scene and pose, 1920x1080, bounces 4, black sky unless the scene needs it:

| run | host flags |
| --- | --- |
| reference | `--mode ref --accumulate 2048 --gi pt` |
| ReSTIR converged (bias) | `--mode ref --accumulate 512 --gi restir` |
| PT 1 spp | `--mode pt --no-denoise --warmup 64 --gi pt` |
| ReSTIR biased | `--mode pt --no-denoise --warmup 64 --gi restir` |
| ReSTIR + visibility | `... --gi-visibility` |
| ReSTIR unbiased | `... --gi-unbiased` |
| ReSTIR temporal only | `... --gi-no-spatial` |
| ReSTIR spatial only | `... --gi-no-temporal` |

Report per run: PSNR vs reference, mean ratio, firefly fraction, `trace_ms`, `gi_ms`. Equal-time
reading: PT at 1 spp vs ReSTIR at 1 initial sample + overhead; the ms columns make the trade visible.
Moving camera: the sequencer export of a short path with `--gi restir` to see disocclusion noise
(qualitative, two or three frames in the README).

Known limitations of phase 1 (accepted): moving entities lose temporal history when they move
more than `depth_threshold * view_z` per frame (the visible point is compared in world space);
glass/PSR pixels are treated like opaque ones; ReSTIR mode writes no specular radiance; the
visible normal is the axis of the face index, exact for unrotated entities only; no boiling
filter. The host sets DIFFUSE_ONLY for `--gi restir` (the library forces it anyway) and for
`--gi pt` only with `--gi-diffuse-only`; the packed test scenes have no specular materials, so
the bench compares one BSDF either way.

Implementation notes (2026-10-20, as built): `GiMergeCandidate` returns whether the candidate was
selected (the Z count needs it). In the unbiased Z count, the receiver's own term needs no extra
ray (a neighbour's Y was selected only after the visibility ray from the receiver passed) and the
neighbour that owns Y counts without a ray. Fixture results (`caelix_trace_test`, 256 frames):
ReSTIR without reuse is within the noise floor of the path tracer (0.78 % mean error vs 0.60 %
between two path tracer seeds); temporal + spatial with 1/M weights is 3.2 % darker, with Z
weights 1.0 % darker. On ashwick at 1024 reference frames: 1/M mean ratio 0.90 / 0.93 / 0.98 per
channel, Z weights 0.996 / 1.000 / 1.001; GI cost 0.10 ms without reuse, 0.43 ms with 1/M,
3.37 ms with Z weights at 1280x720.

### Phase 1 results (2026-10-20, commit 622cb1e, RTX 5080, 1920x1080, 4 bounces)

Full tables, comparison sheets and logs: `F:/RP_Games/Caelix-family/Artifacts/gi-experiments-20261020/`
(`bench/` for cornell and ashwick, `bench-bistro-pose2x/` for both bistro scenes: the bench config
had the 4096-world pose, fixed to `3523.0,479.8,4912.0,-1.119,-127.260` afterwards).

Bias of the converged runs in linear light (mean of a 512-frame accumulation over the 2048-frame
path tracer reference; the sRGB ratio in the bench tables is misleading because the references
are still noisy and single frames clip at 255):

| scene | 1/M (`restir`) | Z-count (`restir_unbiased`) |
| --- | --- | --- |
| cornell | 1.009 | 1.012 |
| ashwick | 0.921 | 0.990 |
| bistro black sky | 0.818 | 0.973 |
| bistro with sky | 0.885 | 0.981 |

Single frames after 64 warm-up frames, PSNR against the reference and the GI pass cost:

| scene | pt 1 spp | restir 1/M | restir Z-count | gi ms 1/M | gi ms Z | gi ms visibility |
| --- | --- | --- | --- | --- | --- | --- |
| cornell | 5.5 dB (black) | 11.5 dB (reference noise bound) | 11.5 dB | 0.97 | 2.3 | 1.5 |
| ashwick | 19.1 dB | 29.2 dB | 29.1 dB | 0.90 | 6.9 | 3.4 |
| bistro black sky | 13.8 dB | 18.1 dB | 17.9 dB | 0.94 | 7.5 | 4.7 |
| bistro with sky | 11.3 dB | 17.5 dB | 17.0 dB | 0.80 | 9.1 | 4.0 |

Findings:
- Emitter discovery without a light list works: the Cornell room is fully lit by the single voxel
  at 64 frames (the path tracer at 1 spp is black); bistro shows the whole street.
- The 1/M weights lose 8 to 18 % of the light in the complex scenes, concentrated in the light
  pools near the emitters (lamp glows, the bounce from lamp heads). The Z-count weights remove the
  bias but cost 7 to 9 ms at 1080p (one shadow ray per accepted neighbour for the selected sample).
- `VISIBILITY` as built (occluded neighbours keep their M in the 1/M normalisation) darkens the
  image to 0.5 to 0.7 linear. Wrong design, not a bug: the DI paper's visibility reuse tests the
  SELECTED sample at the receiver instead. Change for the next iteration.
- Temporal only and spatial only are both far below the combination (5 to 10 dB).
- Warm-up: 4 frames after a history start the image has large coloured blotches (reservoirs hold
  one bright sample each); 64 frames are clean. This is the disocclusion cost a moving camera pays,
  and the main argument for a persistent world-space cache (phase 2).
- Converged ReSTIR images keep sparkles along concave edges (cornell) and on near facades (bistro):
  temporal correlation of bright reservoirs; a boiling filter or M cap tuning is untested.
- No blocky patches and no depth-edge halos at 64 frames. Voxel face normals make the similarity
  test exact.
- Unrelated library bug seen in 4 of about 56 loads of ashwick and bistro: `entity group grid of
  64 x 128 x 4096 cells exceeds the limit of 16777216 cells` -> `caelix_scene_upload_bricks: out
  of memory`. `Scene::GrownGridShape` doubles the grid on an axis every time a group lands outside
  it, so the final size depends on the sector arrival order (1 to 64 MiB for the same world).
  Logs in `Artifacts/gi-experiments-20261020/failed-loads/`. Fix: grow to the actual span.
- A path tracer artefact: a dotted dark line on the short block's top face along the brick seam
  x = 32 in cornell (also in the reference, weaker): ray origin push-off at brick seams.

Cheap next steps inside phase 1 (before phase 2): (a) Z-count WITHOUT visibility rays (count
neighbours with a nonzero target function only; no rays, removes the geometric part of the 1/M
bias) or pairwise MIS; (b) visibility reuse on the selected sample; (c) raise or remove the
Jacobian clamp near emitters and measure the light pools again; (d) a converged run of the
unbiased mode is now part of the protocol (`restir_unbiased_converged`).

## Phase 2 (planned): brick-keyed world-space reservoirs

Sketch, to be designed after phase 1 numbers exist: hash cells keyed by (entity id, level,
quantized local position, face index), level chosen by projected footprint (voxel face when the
footprint is above ~8 px, brick face otherwise), 8 reservoirs per cell of the same `GiReservoir`
layout with `visible_pos` = the cell's jittered representative point, LRU decay, fill from the
screen-space initial samples and from secondary vertices, reuse at primary vertices (as extra
spatial candidates) and at secondary vertices (the tail, kajiya-style). Budget check: 32 B..48 B
per reservoir, 4M cells x 8 fit in 1 GiB.
