//! The render options of the host: filled from the command line, edited by the debug UI, and
//! converted to `CaelixRenderOptions` every frame.

use caelix_sys::*;
use caelix_vk::SkyCubemap;
use clap::ValueEnum;

/// Primary mode: direction toward the sun of the Lambert term (normalized by the library).
pub const PRIMARY_SUN_DIR: [f32; 3] = [-0.45, 0.8, -0.3];
/// Primary mode: colour of rays that hit nothing.
pub const SKY_COLOR: [f32; 3] = [0.55, 0.7, 0.9];
/// Lighting modes: default direction toward the sun.
pub const LIGHTING_SUN_DIR: [f32; 3] = [0.45, 0.7, 0.55];

/// Debug view of the renderer (`CaelixDebugView`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum View {
    Albedo,
    Normal,
    Depth,
    Group,
    Brick,
    Steps,
    /// Path trace mode: NRD's validation layer over the image.
    NrdValidation,
    /// Path trace mode: the denoised diffuse radiance without the albedo.
    NrdDiff,
    /// Path trace mode: the motion vector of the primary hit (grey = no motion, +-32 px saturate).
    Motion,
    /// Primary mode: the material table's albedo (transparent codes tinted blue, emitters white).
    Material,
    /// Path trace mode: the denoised specular radiance without the albedo.
    NrdSpec,
}

impl View {
    pub const ALL: [View; 11] = [
        View::Albedo,
        View::Normal,
        View::Depth,
        View::Group,
        View::Brick,
        View::Steps,
        View::NrdValidation,
        View::NrdDiff,
        View::Motion,
        View::Material,
        View::NrdSpec,
    ];

    pub fn raw(self) -> u32 {
        match self {
            View::Albedo => CaelixDebugView_CAELIX_VIEW_ALBEDO as u32,
            View::Normal => CaelixDebugView_CAELIX_VIEW_NORMAL as u32,
            View::Depth => CaelixDebugView_CAELIX_VIEW_DEPTH as u32,
            View::Group => CaelixDebugView_CAELIX_VIEW_GROUP as u32,
            View::Brick => CaelixDebugView_CAELIX_VIEW_BRICK as u32,
            View::Steps => CaelixDebugView_CAELIX_VIEW_STEPS as u32,
            View::NrdValidation => CaelixDebugView_CAELIX_VIEW_NRD_VALIDATION as u32,
            View::NrdDiff => CaelixDebugView_CAELIX_VIEW_NRD_DIFF as u32,
            View::Motion => CaelixDebugView_CAELIX_VIEW_MOTION as u32,
            View::Material => CaelixDebugView_CAELIX_VIEW_MATERIAL as u32,
            View::NrdSpec => CaelixDebugView_CAELIX_VIEW_NRD_SPEC as u32,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            View::Albedo => "albedo",
            View::Normal => "normal",
            View::Depth => "depth",
            View::Group => "group",
            View::Brick => "brick",
            View::Steps => "steps",
            View::NrdValidation => "nrd validation (pt)",
            View::NrdDiff => "nrd diff (pt)",
            View::Motion => "motion (pt)",
            View::Material => "material",
            View::NrdSpec => "nrd spec (pt)",
        }
    }

    /// Views that only the path trace mode shows.
    pub fn needs_pt(self) -> bool {
        matches!(self, View::NrdValidation | View::NrdDiff | View::Motion | View::NrdSpec)
    }
}

/// Block encoding of the scene (`CaelixSceneDesc::encoding`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Encoding {
    /// Packed scene colour (Caelix worlds): opaque rgb5, emissive rgb4, glass palette ids.
    Packed,
    /// Block IDs: opaque when bit 15 is set, transparent ids 0..511; materials from --materials.
    BlockId,
}

impl Encoding {
    pub fn raw(self) -> u32 {
        match self {
            Encoding::Packed => 0,
            Encoding::BlockId => 1,
        }
    }
}

/// Rendering mode (`CaelixRenderMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Debug views of the primary hit (`--view`).
    Primary,
    /// Path-traced lighting, raw 1 sample per pixel.
    Pt,
    /// Path-traced lighting accumulated over frames (reference image).
    Ref,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Primary, Mode::Pt, Mode::Ref];

    pub fn raw(self) -> u32 {
        match self {
            Mode::Primary => CaelixRenderMode_CAELIX_MODE_PRIMARY as u32,
            Mode::Pt => CaelixRenderMode_CAELIX_MODE_PATH_TRACE as u32,
            Mode::Ref => CaelixRenderMode_CAELIX_MODE_REFERENCE as u32,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Mode::Primary => "primary",
            Mode::Pt => "path trace",
            Mode::Ref => "reference",
        }
    }
}

/// Upscaler of the path trace mode (`CaelixUpscaler`, docs/DLSS.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Upscaler {
    /// No upscaler: the renderer traces at the window size.
    None,
    /// NVIDIA DLSS Super Resolution: NRD stays the denoiser, DLSS upscales the composited colour.
    DlssSr,
    /// NVIDIA DLSS Ray Reconstruction: replaces NRD, denoises and upscales the raw colour.
    DlssRr,
}

impl Upscaler {
    pub const ALL: [Upscaler; 3] = [Upscaler::None, Upscaler::DlssSr, Upscaler::DlssRr];

    pub fn raw(self) -> u32 {
        match self {
            Upscaler::None => CaelixUpscaler_CAELIX_UPSCALER_NONE as u32,
            Upscaler::DlssSr => CaelixUpscaler_CAELIX_UPSCALER_DLSS_SR as u32,
            Upscaler::DlssRr => CaelixUpscaler_CAELIX_UPSCALER_DLSS_RR as u32,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Upscaler::None => "Off",
            Upscaler::DlssSr => "DLSS SR",
            Upscaler::DlssRr => "DLSS RR",
        }
    }
}

/// Render size per output size of the upscaler (`CaelixUpscaleQuality`, DLSS execution modes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum UpscaleQuality {
    /// Render size = output size (anti-aliasing only).
    Dlaa,
    /// About 0.67 per axis.
    Quality,
    /// About 0.58 per axis.
    Balanced,
    /// 0.5 per axis.
    Performance,
    /// About 0.33 per axis.
    UltraPerformance,
}

impl UpscaleQuality {
    pub const ALL: [UpscaleQuality; 5] = [
        UpscaleQuality::Dlaa,
        UpscaleQuality::Quality,
        UpscaleQuality::Balanced,
        UpscaleQuality::Performance,
        UpscaleQuality::UltraPerformance,
    ];

    pub fn raw(self) -> u32 {
        match self {
            UpscaleQuality::Dlaa => CaelixUpscaleQuality_CAELIX_UPSCALE_DLAA as u32,
            UpscaleQuality::Quality => CaelixUpscaleQuality_CAELIX_UPSCALE_QUALITY as u32,
            UpscaleQuality::Balanced => CaelixUpscaleQuality_CAELIX_UPSCALE_BALANCED as u32,
            UpscaleQuality::Performance => CaelixUpscaleQuality_CAELIX_UPSCALE_PERFORMANCE as u32,
            UpscaleQuality::UltraPerformance => CaelixUpscaleQuality_CAELIX_UPSCALE_ULTRA_PERFORMANCE as u32,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            UpscaleQuality::Dlaa => "DLAA",
            UpscaleQuality::Quality => "Quality",
            UpscaleQuality::Balanced => "Balanced",
            UpscaleQuality::Performance => "Performance",
            UpscaleQuality::UltraPerformance => "Ultra Performance",
        }
    }
}

/// Largest `--upscale-preset` / preset value of the debug UI.
pub const MAX_UPSCALE_PRESET: u32 = 15;

/// Global illumination method of the lighting modes (`CaelixGiMethod`, docs/GI_EXPERIMENTS.md).
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
pub enum GiMethod {
    /// The path tracer.
    Pt,
    /// Screen-space ReSTIR GI on the first diffuse bounce.
    Restir,
}

impl GiMethod {
    pub const ALL: [GiMethod; 2] = [GiMethod::Pt, GiMethod::Restir];

    pub fn raw(self) -> u32 {
        match self {
            GiMethod::Pt => CaelixGiMethod_CAELIX_GI_PATH_TRACE as u32,
            GiMethod::Restir => CaelixGiMethod_CAELIX_GI_RESTIR as u32,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            GiMethod::Pt => "path trace",
            GiMethod::Restir => "ReSTIR GI",
        }
    }
}

/// The GI experiment options (`CaelixGiOptions`). ReSTIR always forces the Lambert lobe
/// (`CAELIX_GI_FLAG_DIFFUSE_ONLY`, the library does the same); the path tracer keeps its specular
/// lobes unless `diffuse_only` is set, so that a comparison can use the same BSDF.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GiSettings {
    pub method: GiMethod,
    /// Path tracer: force the Lambert lobe at every vertex (ReSTIR always does).
    pub diffuse_only: bool,
    /// ReSTIR: reuse of the previous frame's reservoir.
    pub temporal: bool,
    /// ReSTIR: reuse of neighbour reservoirs.
    pub spatial: bool,
    /// ReSTIR: a shadow ray per spatial neighbour.
    pub visibility: bool,
    /// ReSTIR: Z-count MIS weights (with the shadow rays) instead of 1/M.
    pub unbiased: bool,
    /// ReSTIR GI: pairwise MIS weights with visibility in the densities (`CAELIX_GI_FLAG_PAIRWISE_MIS`,
    /// two shadow rays per spatial neighbour) instead of 1/M or the Z-count; bounds the reconnection
    /// Jacobian's fireflies.
    pub pairwise_mis: bool,
    pub temporal_m_cap: u32,
    pub spatial_samples: u32,
    pub spatial_passes: u32,
    /// Pixels.
    pub spatial_radius: f32,
    /// Fraction of the view depth.
    pub depth_threshold: f32,
    /// Both methods: next-event estimation on the scene's light list (`CAELIX_GI_FLAG_LIGHT_NEE`,
    /// docs/LIGHT_NEE.md).
    pub light_nee: bool,
    /// Light cells (a power of two, 1024..1048576).
    pub light_cells: u32,
    /// Light NEE: reservoirs per light cell (a power of two, 1..64; a cell is 64 + 8 x this bytes).
    pub light_cell_reservoirs: u32,
    /// Light NEE: look the cell up at a point jittered by up to half a cell per vertex
    /// (`CAELIX_GI_FLAG_LIGHT_CELL_JITTER`): the per-cell flicker becomes per-pixel noise.
    pub light_cell_jitter: bool,
    /// Light NEE: the light cell key also holds the receiver's face (the axis of its world normal),
    /// and the fill's target and the MIS proxy get the receiver orientation term
    /// (`CAELIX_GI_FLAG_LIGHT_CELL_ORIENTED`, docs/LIGHT_NEE.md section 4.2).
    pub light_cell_oriented: bool,
    /// Light NEE: MIS between the light NEE and the diffuse bounce (`CAELIX_GI_FLAG_LIGHT_MIS`,
    /// docs/LIGHT_NEE.md section 6); off: a list-light hit after a light NEE vertex is skipped.
    pub light_mis: bool,
    /// Light NEE: temporal reuse in the light cell fill (`CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL`,
    /// docs/LIGHT_NEE.md section 4.3): each cell reservoir merges its previous frame's reservoir;
    /// assumes a static scene.
    pub light_cell_temporal: bool,
    /// Light NEE with `light_cell_temporal`: frames of history behind a cell reservoir (1..1024).
    pub light_temporal_frames: u32,
    /// Light NEE: light candidates per diffuse vertex, resampled by the exact target before the one
    /// shadow ray (docs/LIGHT_NEE.md section 5.3); 1 = no resampling (1..32).
    pub light_nee_candidates: u32,
    /// Candidates per cell reservoir and frame (1..64).
    pub light_candidates: u32,
    /// World units per pixel below which a vertex uses 32-voxel cells (else 128-voxel cells).
    pub light_footprint: f32,
    /// TIMING ONLY (biased): the sun NEE and the light NEE skip their shadow rays
    /// (`CAELIX_GI_FLAG_DEBUG_NEE_NO_SHADOW`).
    pub debug_nee_no_shadow: bool,
    /// NEE combine (`CaelixGiOptions::nee_combine`, docs/LIGHT_NEE.md section 10): 0 = one shadow ray
    /// per strategy (sun NEE and light NEE); 1 = one shadow ray per NEE vertex toward one of the two
    /// candidates, chosen in proportion to their unoccluded contributions; 2 = as 0 at the primary
    /// vertex, as 1 at the bounce vertices.
    pub nee_combine: u32,
    /// Light NEE with `nee_combine` 1 or 2: the per-light-cell sun visibility hint of the choice
    /// (`CAELIX_GI_FLAG_NEE_CELL_VISIBILITY`).
    pub nee_cell_visibility: bool,
}

impl Default for GiSettings {
    /// The command-line defaults (docs/LIGHT_NEE.md section 7: light NEE with every kept option on).
    fn default() -> Self {
        Self {
            method: GiMethod::Pt,
            diffuse_only: false,
            temporal: true,
            spatial: true,
            visibility: false,
            unbiased: false,
            pairwise_mis: true,
            temporal_m_cap: 20,
            spatial_samples: 5,
            spatial_passes: 1,
            spatial_radius: 30.0,
            depth_threshold: 0.1,
            light_nee: true,
            light_cells: 262144,
            light_cell_reservoirs: 64,
            light_cell_jitter: true,
            light_cell_oriented: true,
            light_mis: true,
            light_cell_temporal: true,
            light_temporal_frames: 128,
            light_nee_candidates: 2,
            light_candidates: 16,
            light_footprint: 0.25,
            debug_nee_no_shadow: false,
            nee_combine: 0,
            nee_cell_visibility: false,
        }
    }
}

impl GiSettings {
    /// The flags that the library receives (`CAELIX_GI_FLAG_*`).
    pub fn flags(&self) -> u32 {
        let mut flags = 0;
        if self.diffuse_only || self.method == GiMethod::Restir {
            flags |= CAELIX_GI_FLAG_DIFFUSE_ONLY;
        }
        if self.temporal {
            flags |= CAELIX_GI_FLAG_TEMPORAL;
        }
        if self.spatial {
            flags |= CAELIX_GI_FLAG_SPATIAL;
        }
        if self.visibility {
            flags |= CAELIX_GI_FLAG_VISIBILITY;
        }
        if self.unbiased {
            flags |= CAELIX_GI_FLAG_UNBIASED;
        }
        if self.pairwise_mis && self.method == GiMethod::Restir {
            flags |= CAELIX_GI_FLAG_PAIRWISE_MIS;
        }
        if self.light_nee {
            flags |= CAELIX_GI_FLAG_LIGHT_NEE;
            if self.light_cell_jitter {
                flags |= CAELIX_GI_FLAG_LIGHT_CELL_JITTER;
            }
            if self.light_cell_oriented {
                flags |= CAELIX_GI_FLAG_LIGHT_CELL_ORIENTED;
            }
            if self.light_mis {
                flags |= CAELIX_GI_FLAG_LIGHT_MIS;
            }
            if self.light_cell_temporal {
                flags |= CAELIX_GI_FLAG_LIGHT_CELL_TEMPORAL;
            }
            if self.nee_cell_visibility {
                flags |= CAELIX_GI_FLAG_NEE_CELL_VISIBILITY;
            }
        }
        if self.debug_nee_no_shadow {
            flags |= CAELIX_GI_FLAG_DEBUG_NEE_NO_SHADOW;
        }
        flags
    }

    /// `reset`: drop the reservoir history with the next render.
    pub fn options(&self, reset: bool) -> CaelixGiOptions {
        CaelixGiOptions {
            method: self.method.raw(),
            flags: self.flags(),
            temporal_m_cap: self.temporal_m_cap,
            spatial_samples: self.spatial_samples,
            spatial_passes: self.spatial_passes,
            spatial_radius: self.spatial_radius,
            depth_threshold: self.depth_threshold,
            reset: u32::from(reset),
            light_cell_capacity: self.light_cells,
            light_candidates: self.light_candidates,
            light_level_footprint: self.light_footprint,
            light_temporal_frames: self.light_temporal_frames,
            light_nee_candidates: self.light_nee_candidates,
            light_cell_reservoirs: self.light_cell_reservoirs,
            nee_combine: self.nee_combine,
        }
    }
}

/// Everything that `CaelixRenderOptions` holds, except the per-frame values (seed, reset).
#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub mode: Mode,
    pub view: View,
    pub bounces: u32,
    /// Lighting modes: smooth transparent interfaces of the primary ray's delta chain (0..15).
    pub bounces_transparent: u32,
    /// Pt mode: NRD denoising.
    pub denoise: bool,
    /// Lighting modes: sun next-event estimation.
    pub sun_nee: bool,
    /// Pt mode: the delta checkerboard at the first glass interface, with the cross resolve.
    pub checkerboard: bool,
    /// Lighting modes: the sky where the primary ray misses.
    pub sky_on_miss: bool,
    /// Primary mode: Lambert shading of the albedo view.
    pub shading: bool,
    pub exposure: f32,
    /// Direction toward the sun. `None`: the mode's default (`LIGHTING_SUN_DIR` or `PRIMARY_SUN_DIR`).
    pub sun_dir: Option<[f32; 3]>,
    /// Linear RGB radiance of the sun disk: the value that the library receives. The UI edits it
    /// as `sun_color * sun_intensity` and writes it only when one of the two changes, so the
    /// command-line value goes to the library unchanged.
    pub sun_radiance: [f32; 3],
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    pub sun_angular_radius_deg: f32,
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub sky_ground: [f32; 3],
    /// The uploaded `--sky` cubemap; `None` without `--sky` (and before the upload).
    pub sky_cubemap: Option<SkyCubemap>,
    /// Lighting modes: the cubemap instead of the procedural sky (when one is loaded).
    pub use_sky_cubemap: bool,
    /// Lighting modes: extinction of the air per world unit (0 = no fog).
    pub fog_density: f32,
    /// Lighting modes: linear RGB radiance of the fog.
    pub fog_color: [f32; 3],
    /// Every mode: multiplies the emission of every material.
    pub emission_scale: f32,
    /// Pt mode: the upscaler (the other modes always render without one).
    pub upscaler: Upscaler,
    pub upscale_quality: UpscaleQuality,
    /// Raw NGX preset value of the chosen quality mode; 0 = the NGX default.
    pub upscale_preset: u32,
    /// DLSS SR: NGX auto exposure instead of the fixed exposure 1.0.
    pub upscale_auto_exposure: bool,
    /// Lighting modes: the GI method and its options (`caelix_renderer_set_gi_options`).
    pub gi: GiSettings,
}

impl RenderSettings {
    /// Splits the sun radiance into a colour (largest component 1) and an intensity.
    pub fn split_radiance(radiance: [f32; 3]) -> ([f32; 3], f32) {
        let max = radiance.iter().copied().fold(0.0f32, f32::max);
        if max > 0.0 {
            (radiance.map(|c| c / max), max)
        } else {
            ([1.0; 3], 0.0)
        }
    }

    pub fn lighting(&self) -> bool {
        self.mode != Mode::Primary
    }

    /// The upscaler that the library receives: `None` outside the pt mode.
    pub fn effective_upscaler(&self) -> Upscaler {
        if self.mode == Mode::Pt {
            self.upscaler
        } else {
            Upscaler::None
        }
    }

    /// The direction toward the sun that the library receives.
    pub fn effective_sun_dir(&self) -> [f32; 3] {
        self.sun_dir.unwrap_or(if self.lighting() { LIGHTING_SUN_DIR } else { PRIMARY_SUN_DIR })
    }

    pub fn options(&self, reset_accumulation: bool, frame_seed: u32) -> CaelixRenderOptions {
        let lighting = self.lighting();
        let mut flags = 0;
        if self.mode == Mode::Primary && self.shading {
            flags |= CAELIX_RENDER_FLAG_SHADING;
        }
        if lighting && self.sky_on_miss {
            flags |= CAELIX_RENDER_FLAG_SKY_ON_MISS;
        }
        if lighting && self.sun_nee {
            flags |= CAELIX_RENDER_FLAG_SUN_NEE;
        }
        if self.mode == Mode::Pt && self.checkerboard {
            flags |= CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD;
        }
        // Without an upscaler the other upscaler fields are zero, so they do not restart the
        // reference accumulation (`same_options`). RESET_HISTORY is never needed from the host:
        // the library drops the upscaler history with reset_accumulation.
        let upscaler = self.effective_upscaler();
        let upscaling = upscaler != Upscaler::None;
        let upscale_flags = if upscaling && upscaler == Upscaler::DlssSr && self.upscale_auto_exposure {
            CAELIX_UPSCALE_FLAG_AUTO_EXPOSURE
        } else {
            0
        };
        CaelixRenderOptions {
            debug_view: self.view.raw(),
            flags,
            sun_dir: self.effective_sun_dir(),
            pad: 0.0,
            sky_color: SKY_COLOR,
            pad2: 0.0,
            // The camera's projection is reverse-Z infinite.
            miss_depth: 0.0,
            miss_view_z: 1e30,
            mode: self.mode.raw(),
            bounce_count: self.bounces,
            denoise: u32::from(self.mode == Mode::Pt && self.denoise),
            reset_accumulation: u32::from(reset_accumulation),
            exposure: self.exposure,
            sun_radiance: self.sun_radiance,
            sun_angular_radius: self.sun_angular_radius_deg.to_radians(),
            sky_zenith: self.sky_zenith,
            sky_horizon: self.sky_horizon,
            sky_ground: self.sky_ground,
            frame_seed,
            bounce_count_transparent: self.bounces_transparent,
            emission_scale: self.emission_scale,
            fog_density: self.fog_density,
            fog_color: self.fog_color,
            pad3: 0.0,
            sky: self.sky(),
            upscaler: upscaler.raw(),
            upscale_quality: if upscaling { self.upscale_quality.raw() } else { 0 },
            upscale_preset: if upscaling { self.upscale_preset } else { 0 },
            upscale_flags,
        }
    }

    /// The cubemap that the library receives: zero (the procedural sky) in the primary mode, without
    /// a loaded cubemap or with the cubemap switched off.
    fn sky(&self) -> CaelixSkyCubemap {
        match self.sky_cubemap.filter(|_| self.lighting() && self.use_sky_cubemap) {
            Some(c) => CaelixSkyCubemap {
                image: c.image,
                format: c.format,
                layout: c.layout,
                size: c.size,
                mip_levels: c.mip_levels,
            },
            None => CaelixSkyCubemap { image: 0, format: 0, layout: 0, size: 0, mip_levels: 0 },
        }
    }
}

/// Whether two option sets give the same image, ignoring the frame seed and the reset flag
/// (reference accumulation restarts when they differ).
pub fn same_options(a: &CaelixRenderOptions, b: &CaelixRenderOptions) -> bool {
    a.debug_view == b.debug_view
        && a.flags == b.flags
        && a.sun_dir == b.sun_dir
        && a.mode == b.mode
        && a.bounce_count == b.bounce_count
        && a.bounce_count_transparent == b.bounce_count_transparent
        && a.denoise == b.denoise
        && a.exposure == b.exposure
        && a.sun_radiance == b.sun_radiance
        && a.sun_angular_radius == b.sun_angular_radius
        && a.sky_zenith == b.sky_zenith
        && a.sky_horizon == b.sky_horizon
        && a.sky_ground == b.sky_ground
        && a.emission_scale == b.emission_scale
        && a.fog_density == b.fog_density
        && a.fog_color == b.fog_color
        && a.sky.image == b.sky.image
        && a.upscaler == b.upscaler
        && a.upscale_quality == b.upscale_quality
        && a.upscale_preset == b.upscale_preset
        && a.upscale_flags == b.upscale_flags
}
