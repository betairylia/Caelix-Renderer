//! The render options of the host: filled from the command line, edited by the debug UI, and
//! converted to `CaelixRenderOptions` every frame.

use caelix_sys::*;
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
}
