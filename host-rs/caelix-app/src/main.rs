//! Standalone host for the Caelix renderer.
//!
//! The host owns the window, the Vulkan device and the swapchain (`caelix-vk`).
//! The library adopts the device and records into the host's command buffer
//! through the C ABI (`caelix-sys`). This crate contains no Vulkan code.
//!
//! With `--world`, a `.cxw` file is decoded on a background thread and streamed
//! into a `CaelixScene` (see `world.rs`). Every frame: recycle, upload, flush the
//! scene, then `caelix_renderer_render` traces the scene into the swapchain image:
//! a debug view (`--mode primary`), the path-traced 1 spp image denoised by NRD
//! (`--mode pt`, `--no-denoise` for the raw image) or the accumulated reference
//! (`--mode ref`). Debug windows (egui, `ui.rs`, F1) are drawn over the image by
//! `caelix-vk` after the render; a screenshot is copied before them.
//!
//! `--test-entity` and `--test-edits` add a synthetic moving box and timed brick edits
//! (`testscene.rs`, Test window) to judge the motion vectors and NRD.

mod camera;
mod materials;
mod settings;
mod testscene;
mod ui;
mod world;

use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::Instant;

use anyhow::{anyhow, bail, Context as _};
use caelix_sys::*;
use clap::Parser;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use camera::Camera;
use settings::{same_options, Encoding, Mode, RenderSettings, View};
use testscene::{EditPattern, TestScene};

/// Free camera: default speed in world units (blocks) per second.
const DEFAULT_SPEED: f32 = 100.0;
/// Shift multiplies the speed by this.
const FAST_FACTOR: f32 = 8.0;
/// Mouse look: degrees per pixel of mouse motion.
const MOUSE_DEGREES_PER_PIXEL: f32 = 0.1;
/// One scroll wheel notch scales the speed by this.
const SCROLL_FACTOR: f32 = 1.25;
/// Without --world, the screenshot is taken at this frame.
const SCREENSHOT_FRAME_WITHOUT_WORLD: u64 = 10;
/// The screenshot is taken this many frames after the world is complete on the GPU.
const SCREENSHOT_EXTRA_FRAMES: u64 = 3;
/// The timing line is logged every this many frames.
const LOG_EVERY_FRAMES: u64 = 120;
/// Distance range of the depth view.
const DEPTH_NEAR: f32 = 1.0;
const DEPTH_FAR: f32 = 8192.0;
/// `--ui-demo`: entity 0 is turned at this presented-frame count.
const UI_DEMO_FRAME: u64 = 60;
/// `--ui-demo`: the yaw of entity 0 in degrees.
const UI_DEMO_YAW: f32 = 15.0;
/// An absent render target (image 0): the library does not write it.
const NO_TARGET: CaelixTargetImage = CaelixTargetImage { image: 0, format: 0, width: 0, height: 0, layout: 0 };

/// Parses "x,y,z" into three floats.
fn parse_vec3(text: &str) -> Result<[f32; 3], String> {
    let values: Vec<f32> = text
        .split(',')
        .map(|v| v.trim().parse::<f32>().map_err(|e| format!("'{v}': {e}")))
        .collect::<Result<_, _>>()?;
    match values[..] {
        [x, y, z] => Ok([x, y, z]),
        _ => Err(format!("expected 3 comma-separated numbers x,y,z, got {}", values.len())),
    }
}

/// Parses "WxH" into a width and a height (both nonzero).
fn parse_size(text: &str) -> Result<(u32, u32), String> {
    let (w, h) = text
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected WIDTHxHEIGHT, got '{text}'"))?;
    let w: u32 = w.trim().parse().map_err(|e| format!("width '{w}': {e}"))?;
    let h: u32 = h.trim().parse().map_err(|e| format!("height '{h}': {e}"))?;
    if w == 0 || h == 0 {
        return Err(format!("the size must be nonzero, got {w}x{h}"));
    }
    Ok((w, h))
}

/// A camera pose in Unity's terms.
#[derive(Clone, Copy, Debug)]
struct Pose {
    position: [f32; 3],
    /// Unity eulerAngles.x, degrees.
    pitch: f32,
    /// Unity eulerAngles.y, degrees.
    yaw: f32,
}

fn parse_pose(text: &str) -> Result<Pose, String> {
    let values: Vec<f32> = text
        .split(',')
        .map(|v| v.trim().parse::<f32>().map_err(|e| format!("'{v}': {e}")))
        .collect::<Result<_, _>>()?;
    match values[..] {
        [x, y, z, pitch, yaw] => Ok(Pose { position: [x, y, z], pitch, yaw }),
        _ => Err(format!("expected 5 comma-separated numbers x,y,z,pitch,yaw, got {}", values.len())),
    }
}

#[derive(Parser, Debug)]
#[command(about = "Standalone host for the Caelix native renderer")]
struct Args {
    /// Exit with code 0 after N presented frames. Default: run until the window closes.
    #[arg(long, value_name = "N")]
    frames: Option<u64>,
    /// Debug log level (includes Vulkan validation layer and loader messages).
    /// RUST_LOG overrides this.
    #[arg(long)]
    verbose: bool,
    /// Test aid: after N presented frames, request a 960x540 window size once,
    /// so the swapchain recreation path runs without user input.
    #[arg(long, value_name = "N", hide = true)]
    resize_at: Option<u64>,
    /// A .cxw world to decode and stream into the scene.
    #[arg(long, value_name = "PATH")]
    world: Option<PathBuf>,
    /// Size of the scene's staging ring in MiB (at least 4).
    #[arg(long, value_name = "N", default_value_t = 64)]
    staging_mib: u64,
    /// Render group edge in bricks = 1 << N (1..5).
    #[arg(long, value_name = "N", default_value_t = 4)]
    group_shift: u32,
    /// Initial camera pose: position, then Unity eulerAngles.x (pitch, positive looks down)
    /// and eulerAngles.y (yaw), in degrees. Default: the world's player pose, else a fixed pose.
    #[arg(long, value_name = "X,Y,Z,PITCH,YAW", value_parser = parse_pose, allow_hyphen_values = true)]
    pose: Option<Pose>,
    /// Vertical field of view in degrees.
    #[arg(long, value_name = "DEGREES", default_value_t = 60.0)]
    fov: f32,
    /// Rendering mode. Default: `pt` with --world, else `primary`.
    #[arg(long, value_enum)]
    mode: Option<Mode>,
    /// Primary mode: the debug view (albedo .. steps, material); pt mode: nrd-validation, nrd-diff,
    /// nrd-spec, motion.
    #[arg(long, value_enum, default_value_t = View::Albedo)]
    view: View,
    /// Pt mode: no NRD denoising (the raw 1 sample per pixel image).
    #[arg(long)]
    no_denoise: bool,
    /// Inner size of the window in pixels, WIDTHxHEIGHT.
    #[arg(long, value_name = "WxH", value_parser = parse_size, default_value = "1280x720")]
    size: (u32, u32),
    /// Pt mode: take the --screenshot N frames after the world is complete (at least 3), so the
    /// denoiser history is warm. The NRD history restarts while the world is loading.
    #[arg(long, value_name = "N", default_value_t = 0)]
    warmup: u64,
    /// Primary mode: albedo without Lambert shading.
    #[arg(long)]
    no_shading: bool,
    /// Lighting modes: stochastic bounces after the primary hit (0..14).
    #[arg(long, value_name = "N", default_value_t = 4)]
    bounces: u32,
    /// Lighting modes: smooth transparent interfaces (glass) that the primary ray crosses by
    /// reflection or refraction before the surface that is shaded (0..15, 0 = off).
    #[arg(long, value_name = "N", default_value_t = 5)]
    bounces_transparent: u32,
    /// Pt mode: no delta checkerboard at the first glass interface (the reflect/refract choice is
    /// random per frame, no cross resolve). Default: checkerboard on.
    #[arg(long)]
    no_checkerboard: bool,
    /// Block encoding of the scene: packed (Caelix worlds) or block-id.
    #[arg(long, value_enum, default_value_t = Encoding::Packed)]
    encoding: Encoding,
    /// A JSON material table (an array of {"id", "albedo", "smoothness", "metallic", "emission",
    /// "ior", "extinction", "transparent"}), applied with caelix_scene_set_materials over the
    /// built-in table. For block-ID worlds: an entry goes to code 0x8000 | id, or to code id with
    /// "transparent": true.
    #[arg(long, value_name = "PATH")]
    materials: Option<PathBuf>,
    /// Lighting modes: colour multiplier before the sRGB encode.
    #[arg(long, value_name = "F", default_value_t = 1.0)]
    exposure: f32,
    /// Direction toward the sun (normalized by the library). Default: 0.45,0.7,0.55 in the
    /// lighting modes; in the primary mode the Lambert direction -0.45,0.8,-0.3.
    #[arg(long, value_name = "X,Y,Z", value_parser = parse_vec3, allow_hyphen_values = true)]
    sun: Option<[f32; 3]>,
    /// Lighting modes: linear RGB radiance of the sun disk. Default 0 = sun off (sky light only).
    /// Suggested 2e4,1.9e4,1.8e4 for a 0.5-degree disk: an irradiance of about 4.8,4.5,4.3 on a
    /// surface facing the sun, about 3x the default sky's (by luminance).
    #[arg(long, value_name = "R,G,B", value_parser = parse_vec3, default_value = "0,0,0")]
    sun_radiance: [f32; 3],
    /// Lighting modes: angular radius of the sun disk in degrees.
    #[arg(long, value_name = "DEGREES", default_value_t = 0.5)]
    sun_angular_radius: f32,
    /// Lighting modes: sky radiance straight up.
    #[arg(long, value_name = "R,G,B", value_parser = parse_vec3, default_value = "0.2,0.4,0.9")]
    sky_zenith: [f32; 3],
    /// Lighting modes: sky radiance at the horizon.
    #[arg(long, value_name = "R,G,B", value_parser = parse_vec3, default_value = "0.7,0.8,0.95")]
    sky_horizon: [f32; 3],
    /// Lighting modes: radiance below the horizon.
    #[arg(long, value_name = "R,G,B", value_parser = parse_vec3, default_value = "0.25,0.22,0.2")]
    sky_ground: [f32; 3],
    /// Lighting modes: the window shows black instead of the sky where the primary ray misses.
    #[arg(long)]
    no_sky_on_miss: bool,
    /// Lighting modes: no next-event estimation of the sun (the sun is then found only by the
    /// bounce rays, as in step 6a).
    #[arg(long)]
    no_sun_nee: bool,
    /// Reference mode: take the --screenshot after N accumulated frames. The accumulation
    /// restarts whenever the camera changes or the world is not complete yet.
    #[arg(long, value_name = "N")]
    accumulate: Option<u64>,
    /// Write a PNG of the window: 3 frames after the world is complete on the GPU
    /// (every brick uploaded, no acceleration structure build pending); without
    /// --world at frame 10; with --accumulate after N accumulated frames.
    #[arg(long, value_name = "PATH")]
    screenshot: Option<PathBuf>,
    /// Exit after the screenshot is written.
    #[arg(long, requires = "screenshot")]
    exit_after_screenshot: bool,
    /// Ignore keyboard and mouse, so automated screenshot runs stay deterministic. Also disables
    /// the debug UI.
    #[arg(long)]
    no_input: bool,
    /// Start with the debug UI hidden (F1 shows it).
    #[arg(long)]
    no_ui: bool,
    /// Directory of the PNGs of the debug UI's Screenshot button (screenshot-<frame>.png).
    /// Default: the directory of the executable.
    #[arg(long, value_name = "DIR")]
    screenshot_dir: Option<PathBuf>,
    /// Debug UI demo: every window and section starts open, and at frame 60 entity 0 is turned to a
    /// yaw of 15 degrees (for captures of the debug UI).
    #[arg(long, conflicts_with_all = ["no_ui", "no_input"])]
    ui_demo: bool,
    /// Test aid: write frame-<frame number>.png into --screenshot-dir every N frames, starting
    /// --warmup frames (at least 3) after the world is complete (without --world at frame 10).
    #[arg(long, value_name = "N")]
    screenshot_every: Option<u64>,
    /// Add a synthetic test entity (a hollow 24^3 box with coloured faces, emissive corners and a
    /// hole in its +Z face) that orbits and spins, to judge motion vectors and NRD (Test window).
    #[arg(long)]
    test_entity: bool,
    /// Orbit centre of the test entity. Default: 120 units in front of the start camera, 10
    /// units below the eye.
    #[arg(long, value_name = "X,Y,Z", value_parser = parse_vec3, allow_hyphen_values = true)]
    test_entity_at: Option<[f32; 3]>,
    /// The test entity starts paused (no transform change, motion vectors 0).
    #[arg(long)]
    test_paused: bool,
    /// Orbit radius of the test entity in world units.
    #[arg(long, value_name = "UNITS", default_value_t = 20.0)]
    test_orbit_radius: f32,
    /// Orbit speed of the test entity in degrees per second.
    #[arg(long, value_name = "DEG_PER_S", default_value_t = 30.0, allow_hyphen_values = true)]
    test_orbit_speed: f32,
    /// Spin (yaw) speed of the test entity in degrees per second; the pitch turns at 0.37 times it.
    #[arg(long, value_name = "DEG_PER_S", default_value_t = 45.0, allow_hyphen_values = true)]
    test_spin_speed: f32,
    /// Timed brick edits on the test entity (adds the test entity when --test-entity is absent).
    #[arg(long)]
    test_edits: bool,
    /// Frames between two test edits (1..120).
    #[arg(long, value_name = "N", default_value_t = 10)]
    test_edit_interval: u32,
    /// Test edit pattern: wave (brick columns under the box are uploaded and removed) or flicker
    /// (the emissive corners are replaced).
    #[arg(long, value_enum, default_value_t = EditPattern::Wave)]
    test_edit_pattern: EditPattern,
}

/// Everything that lives while the window exists. Drop order matters: Drop
/// stops the decoder, waits for the device, destroys the renderer, the scene and
/// the library's device, then the fields drop in declaration order (Vulkan
/// context, then window).
struct State {
    loader: Option<world::WorldLoader>,
    /// The debug UI; `None` with --no-input.
    ui: Option<ui::Ui>,
    renderer: *mut CaelixRenderer,
    scene: *mut CaelixScene,
    caelix_device: *mut CaelixDevice,
    /// Swapchain generation the renderer's cached image views belong to.
    views_generation: u64,
    context: caelix_vk::Context,
    window: Window,
}

impl Drop for State {
    fn drop(&mut self) {
        if let Some(loader) = self.loader.as_mut() {
            loader.stop();
        }
        self.context.wait_idle();
        // SAFETY: the device is idle; every pointer came from its create call or is null
        // (ignored); the renderer goes before the scene and the scene before its device.
        unsafe { caelix_renderer_destroy(self.renderer) };
        unsafe { caelix_scene_destroy(self.scene) };
        log::debug!("caelix renderer and scene destroyed");
        unsafe { caelix_device_destroy(self.caelix_device) };
        log::debug!("caelix device destroyed");
    }
}

/// Keyboard and mouse state of the free camera.
struct Input {
    keys: HashSet<KeyCode>,
    looking: bool,
    speed: f32,
}

/// Who receives the pending readback.
enum ReadbackOwner {
    /// The --screenshot flow.
    Cli,
    /// The debug UI's Screenshot button: the PNG path.
    Ui(PathBuf),
    /// --screenshot-every: the PNG path.
    Every(PathBuf),
}

/// Screenshot progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screenshot {
    /// Waiting for the world (or for frame 10).
    Waiting,
    /// The world is complete; the readback is requested at this presented-frame count.
    RequestAt(u64),
    /// Requested; waiting for the pixels.
    Requested,
    Done,
}

/// Frame-rate and GPU time display.
struct Timing {
    window_start: Instant,
    window_frames: u64,
    fps: f64,
}

/// Reference accumulation as the host sees it.
struct Accumulation {
    /// The camera of the previous frame, to detect changes.
    last_camera: Option<CaelixCamera>,
    /// Frames accumulated since the last reset, this frame included.
    frames: u64,
}

struct App {
    args: Args,
    /// The render options: initial values from the command line, edited by the debug UI.
    settings: RenderSettings,
    /// The options of the previous frame (reference accumulation restarts when they change).
    last_options: Option<CaelixRenderOptions>,
    accumulation: Accumulation,
    /// The scene entities as the debug UI edits them (the world's, then the test entity).
    entities: Vec<ui::EntityRow>,
    /// The test entity and its edits (--test-entity, --test-edits, Test window).
    test: TestScene,
    /// --screenshot-every: the presented-frame count of the first capture.
    every_base: Option<u64>,
    /// The debug UI's Screenshot button was pressed; the copy is recorded when no readback is pending.
    ui_screenshot_requested: bool,
    /// Receiver of the readback that `readback_now` recorded; `None` when nothing is pending.
    readback_owner: Option<ReadbackOwner>,
    /// Shown in the Render window: the newest screenshot of the button.
    screenshot_note: String,
    state: Option<State>,
    camera: Camera,
    input: Input,
    last_frame: Instant,
    frames_presented: u64,
    resize_requested: bool,
    screenshot: Screenshot,
    timing: Timing,
    /// Set when the loaded scene does not hold every decoded brick.
    load_mismatch: bool,
    /// Frames rendered since the renderer reported an NRD instance (the info line is logged at the
    /// second one, whose dispatch count no longer includes the first frame's pool clears).
    nrd_frames: u32,
    error: Option<anyhow::Error>,
}

impl App {
    fn create_state(&mut self, event_loop: &ActiveEventLoop) -> anyhow::Result<State> {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Caelix Renderer")
                    .with_inner_size(PhysicalSize::new(self.args.size.0, self.args.size.1)),
            )
            .context("cannot create the window")?;
        let size = window.inner_size();

        let requirements = device_requirements();
        let mut context = caelix_vk::Context::new(
            &window,
            size.width,
            size.height,
            &requirements,
            cfg!(debug_assertions),
        )
        .context("cannot create the Vulkan context")?;
        // The debug UI: none with --no-input (deterministic runs never touch the overlay).
        let ui = if self.args.no_input {
            None
        } else {
            context.init_overlay().context("cannot create the egui overlay")?;
            let test_open = self.args.test_entity || self.args.test_edits;
            Some(ui::Ui::new(&window, !self.args.no_ui, self.args.ui_demo, test_open))
        };

        let handles = context.adopt_handles();
        let desc = CaelixDeviceAdoptDesc {
            get_instance_proc_addr: handles.get_instance_proc_addr as *mut c_void,
            instance: handles.instance,
            physical_device: handles.physical_device,
            device: handles.device,
            graphics_queue: handles.graphics_queue,
            graphics_queue_family: handles.graphics_queue_family,
        };
        let mut caelix_device: *mut CaelixDevice = ptr::null_mut();
        // SAFETY: desc holds live handles that outlive the adopted device (State drop order).
        let result = unsafe { caelix_device_adopt(&desc, &mut caelix_device) };
        if result != CaelixResult_CAELIX_OK || caelix_device.is_null() {
            bail!("caelix_device_adopt failed: CaelixResult {result}");
        }
        log::debug!("{} frames in flight", context.frames_in_flight());
        // From here on, an error drops `state`, which releases what exists so far.
        let views_generation = context.swapchain_generation();
        let mut state = State {
            loader: None,
            ui,
            renderer: ptr::null_mut(),
            scene: ptr::null_mut(),
            caelix_device,
            views_generation,
            context,
            window,
        };

        let scene_desc = CaelixSceneDesc {
            staging_bytes: self.args.staging_mib << 20,
            records_per_page: 0,
            group_shift: self.args.group_shift,
            block_bytes: 0,
            encoding: self.args.encoding.raw(),
            scratch_bytes: 0, // 128 MiB of acceleration structure build scratch
        };
        // SAFETY: the device is adopted; scene_desc and the out pointer are valid for the call.
        let result = unsafe { caelix_scene_create(state.caelix_device, &scene_desc, &mut state.scene) };
        if result != CaelixResult_CAELIX_OK || state.scene.is_null() {
            bail!("caelix_scene_create failed: CaelixResult {result}");
        }
        log::info!(
            "scene created: staging {} MiB, group shift {}, encoding {:?}",
            self.args.staging_mib,
            self.args.group_shift,
            self.args.encoding
        );
        if let Some(path) = self.args.materials.as_deref() {
            materials::apply(state.scene, path)?;
        }
        // SAFETY: the device and the scene are live; the out pointer is valid for the call.
        let result = unsafe { caelix_renderer_create(state.caelix_device, state.scene, &mut state.renderer) };
        if result != CaelixResult_CAELIX_OK || state.renderer.is_null() {
            bail!("caelix_renderer_create failed: CaelixResult {result}");
        }

        if let Some(path) = self.args.world.as_deref() {
            let loader = world::WorldLoader::open(state.scene, path)?;
            self.entities = loader.entities().iter().map(ui::EntityRow::new).collect();
            state.loader = Some(loader);
        }
        // Camera: --pose, else the world's player pose, else a fixed pose.
        self.camera = match (self.args.pose, state.loader.as_ref().and_then(|l| l.player_pose())) {
            (Some(pose), _) => Camera::from_pose(pose.position, pose.pitch, pose.yaw, self.args.fov),
            (None, Some(player)) => Camera::from_quaternion(player.position, player.rotation, self.args.fov),
            (None, None) => Camera::from_pose([0.0, 64.0, -128.0], 15.0, 0.0, self.args.fov),
        };
        log::info!(
            "camera: position {:?}, pitch {:.4}, yaw {:.4}, vertical fov {}",
            self.camera.position,
            self.camera.pitch,
            self.camera.yaw,
            self.camera.fov_y
        );
        Ok(state)
    }

    fn draw(&mut self, event_loop: &ActiveEventLoop) -> anyhow::Result<()> {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.25);
        self.last_frame = now;
        self.move_camera(dt);

        let Some(state) = self.state.as_mut() else {
            return Ok(());
        };
        if let Some(frame) = state.context.begin_frame()? {
            // A recreated swapchain has new images: drop the views of the old ones. The
            // recreation waited for the device, and nothing was submitted since.
            if state.context.swapchain_generation() != state.views_generation {
                // SAFETY: the renderer is live and no frame that used the views is pending.
                unsafe { caelix_renderer_release_target_views(state.renderer) };
                state.views_generation = state.context.swapchain_generation();
            }

            // Debug UI: one egui pass before the scene work, so that its edits (settings, camera,
            // entity transforms) apply to this frame. Its stats are the previous frame's.
            if let Some(loader) = state.loader.as_ref() {
                for (row, entry) in self.entities.iter_mut().zip(loader.entities()) {
                    row.bricks = entry.bricks_received;
                }
            }
            if self.args.ui_demo && self.frames_presented == UI_DEMO_FRAME {
                if let Some(entity) = self.entities.first_mut() {
                    log::info!("ui demo: entity 0 yaw set to {UI_DEMO_YAW} degrees");
                    entity.set_euler([entity.euler[0], UI_DEMO_YAW, entity.euler[2]]);
                }
            }
            let mut ui_frame = None;
            let mut ui_reset = false;
            if let Some(ui) = state.ui.as_mut().filter(|ui| ui.visible) {
                let stats = ui::Stats {
                    timings: gpu_timings(state.renderer),
                    fps: self.timing.fps,
                    frame_number: frame.frame_number,
                    target: (frame.width, frame.height),
                    scene: scene_stats(state.scene),
                    nrd: nrd_info(state.renderer),
                    load_state: state.loader.as_ref().map_or_else(|| "no world".into(), |l| l.status_line()),
                    accumulated: self.accumulation.frames,
                    screenshot_note: self.screenshot_note.clone(),
                };
                let mut panel = ui::Panel {
                    stats: &stats,
                    settings: &mut self.settings,
                    camera: &mut self.camera,
                    speed: &mut self.input.speed,
                    entities: &mut self.entities,
                    test: &mut self.test,
                    reset_accumulation: false,
                    screenshot: false,
                };
                ui_frame = Some(ui.run(&state.window, &mut panel));
                ui_reset = panel.reset_accumulation;
                if panel.screenshot {
                    self.ui_screenshot_requested = true;
                }
            }
            // Test entity: added or removed, animated (its row becomes dirty when the pose changed)
            // and edited, before the transforms are sent.
            let test_edited = self.test.update(state.scene, dt, &self.camera, &mut self.entities)?;
            let mut entity_moved = false;
            for entity in self.entities.iter_mut().filter(|e| e.dirty) {
                entity.dirty = false;
                entity_moved = true;
                // SAFETY: the scene is live; both arrays are valid for the call.
                let result = unsafe {
                    caelix_scene_set_entity_transform(
                        state.scene,
                        entity.scene_id,
                        entity.position.as_ptr(),
                        entity.rotation.as_ptr(),
                    )
                };
                if result != CaelixResult_CAELIX_OK {
                    bail!("caelix_scene_set_entity_transform({}) failed: CaelixResult {result}", entity.scene_id);
                }
            }

            // SAFETY: the scene is live; the frame's slot fence confirmed safe_frame_number.
            let result = unsafe { caelix_scene_recycle(state.scene, frame.safe_frame_number) };
            if result != CaelixResult_CAELIX_OK {
                bail!("caelix_scene_recycle({}) failed: CaelixResult {result}", frame.safe_frame_number);
            }
            // The test entity's few bricks go first, so a loading world cannot starve them.
            self.test.upload_pending(state.scene)?;
            let uploaded = match state.loader.as_mut() {
                Some(loader) => loader.upload(state.scene, frame.frame_number)?,
                None => false,
            };
            let frame_desc = CaelixFrameDesc {
                command_buffer: frame.command_buffer,
                frame_number: frame.frame_number,
                safe_frame_number: frame.safe_frame_number,
            };
            // SAFETY: the command buffer is recording on the graphics queue.
            let result = unsafe { caelix_scene_flush(state.scene, &frame_desc) };
            if result != CaelixResult_CAELIX_OK {
                bail!(
                    "caelix_scene_flush(frame {}, safe {}) failed: CaelixResult {result}",
                    frame.frame_number,
                    frame.safe_frame_number
                );
            }
            if let Some(loader) = state.loader.as_mut() {
                let test_bricks = self.test.bricks_in_scene();
                if let world::LoadOutcome::Mismatch =
                    loader.after_flush(state.scene, uploaded, frame.frame_number, test_bricks)
                {
                    self.load_mismatch = true;
                }
            }
            // The world is complete on the GPU after this frame's flush.
            let world_ready = state.loader.as_ref().is_none_or(|loader| loader.is_ready());
            self.test.after_flush(&scene_stats(state.scene), &gpu_timings(state.renderer), world_ready);

            let targets = CaelixRenderTargets {
                color: CaelixTargetImage {
                    image: frame.image,
                    format: frame.format,
                    width: frame.width,
                    height: frame.height,
                    layout: frame.layout,
                },
                albedo: NO_TARGET,
                normal: NO_TARGET,
                depth: NO_TARGET,
                motion: NO_TARGET,
                view_z: NO_TARGET,
            };
            let aspect = frame.width as f32 / frame.height.max(1) as f32;
            let camera = self.camera.to_caelix(aspect, DEPTH_NEAR, DEPTH_FAR);
            // A new random stream every frame.
            let mut options = self.settings.options(false, frame.frame_number as u32);
            // Reference mode: restart while the camera moves, the options or an entity change, or
            // the scene still loads. Pt mode: NRD follows the camera and the entities through the
            // motion vectors; its history restarts only while the world is still loading. The
            // Reset accumulation button restarts both.
            let camera_changed = self.accumulation.last_camera.is_none_or(|last| !same_camera(&last, &camera));
            let options_changed = self.last_options.is_none_or(|last| !same_options(&last, &options));
            self.accumulation.last_camera = Some(camera);
            self.last_options = Some(options);
            let reset_accumulation = ui_reset
                || !world_ready
                || (self.settings.mode == Mode::Ref
                    && (camera_changed || options_changed || entity_moved || test_edited));
            self.accumulation.frames = if reset_accumulation { 1 } else { self.accumulation.frames + 1 };
            options.reset_accumulation = u32::from(reset_accumulation);
            // SAFETY: the command buffer is recording after this frame's flush; the image is
            // the acquired one; every struct is valid for the call.
            let result = unsafe { caelix_renderer_render(state.renderer, &frame_desc, &targets, &camera, &options) };
            if result != CaelixResult_CAELIX_OK {
                bail!("caelix_renderer_render(frame {}) failed: CaelixResult {result}", frame.frame_number);
            }

            // Screenshot: wait for the world (or frame 10), then 3 more frames; with
            // --accumulate, wait for N accumulated frames of the complete world. The copy is
            // recorded here, after the render and before the overlay, so it excludes the debug UI.
            let mut cli_due = false;
            if let (Some(n), Screenshot::Waiting) = (self.args.accumulate, self.screenshot) {
                cli_due = self.args.screenshot.is_some() && world_ready && self.accumulation.frames >= n;
            } else if self.args.screenshot.is_some() {
                if self.screenshot == Screenshot::Waiting {
                    let ready = match state.loader.as_ref() {
                        Some(loader) => loader.is_ready(),
                        None => self.frames_presented + 1 >= SCREENSHOT_FRAME_WITHOUT_WORLD,
                    };
                    if ready {
                        let extra = if state.loader.is_some() { SCREENSHOT_EXTRA_FRAMES } else { 0 };
                        let extra = if self.settings.mode == Mode::Pt { extra.max(self.args.warmup) } else { extra };
                        self.screenshot = Screenshot::RequestAt(self.frames_presented + extra);
                    }
                }
                if let Screenshot::RequestAt(at) = self.screenshot {
                    cli_due = self.frames_presented >= at;
                }
            }
            if cli_due && state.context.readback_now(&frame)? {
                self.readback_owner = Some(ReadbackOwner::Cli);
                self.screenshot = Screenshot::Requested;
                if self.args.accumulate.is_some() {
                    log::info!(
                        "screenshot requested at frame {} after {} accumulated frames",
                        frame.frame_number,
                        self.accumulation.frames
                    );
                } else {
                    log::info!("screenshot requested at frame {}", frame.frame_number);
                }
            } else if every_due(&self.args, &mut self.every_base, self.frames_presented, world_ready)
                && state.context.readback_now(&frame)?
            {
                let path = screenshot_dir(&self.args).join(format!("frame-{}.png", frame.frame_number));
                log::info!("screenshot (every) requested at frame {}", frame.frame_number);
                self.readback_owner = Some(ReadbackOwner::Every(path));
            } else if self.ui_screenshot_requested && state.context.readback_now(&frame)? {
                self.ui_screenshot_requested = false;
                let path = screenshot_dir(&self.args).join(format!("screenshot-{}.png", frame.frame_number));
                self.screenshot_note = format!("writing {}", path.display());
                self.readback_owner = Some(ReadbackOwner::Ui(path));
            }

            if let Some(output) = ui_frame {
                state.context.draw_overlay(&frame, output.textures_delta, &output.primitives, output.pixels_per_point)?;
            }
            state.context.end_frame(frame)?;
            self.frames_presented += 1;
        } else {
            log::debug!("frame skipped: swapchain recreated or window size is 0");
        }

        if self.readback_owner.is_some() {
            if let Some((pixels, width, height)) = state.context.take_readback()? {
                match self.readback_owner.take().expect("checked above") {
                    ReadbackOwner::Cli => {
                        let path = self.args.screenshot.clone().expect("screenshot path");
                        write_png(&path, &pixels, width, height)?;
                        log::info!("screenshot written: {} ({width}x{height})", path.display());
                        // Before the first full second the fps of the open window is the best value.
                        let fps = if self.timing.fps > 0.0 {
                            self.timing.fps
                        } else {
                            self.timing.window_frames as f64
                                / self.timing.window_start.elapsed().as_secs_f64().max(1e-3)
                        };
                        log::info!("{}", timing_line(state.renderer, fps));
                        self.screenshot = Screenshot::Done;
                        if self.args.exit_after_screenshot {
                            event_loop.exit();
                            return Ok(());
                        }
                    }
                    ReadbackOwner::Every(path) => {
                        write_png(&path, &pixels, width, height)?;
                        log::info!("screenshot written: {} ({width}x{height})", path.display());
                    }
                    ReadbackOwner::Ui(path) => {
                        // A failed write is reported in the Render window, not fatal.
                        match write_png(&path, &pixels, width, height) {
                            Ok(()) => {
                                log::info!("screenshot written: {} ({width}x{height})", path.display());
                                self.screenshot_note = format!("written: {}", path.display());
                            }
                            Err(e) => {
                                log::error!("screenshot failed: {e:#}");
                                self.screenshot_note = format!("failed: {e:#}");
                            }
                        }
                    }
                }
            }
        }

        // NRD: one info line at the second frame of the renderer's NRD instance (or at an NRD error).
        if self.nrd_frames < 2 {
            // SAFETY: CaelixNrdInfo is a plain C struct of integers; all zero is a valid value.
            let mut info: CaelixNrdInfo = unsafe { std::mem::zeroed() };
            // SAFETY: the renderer is live and info is writable.
            if unsafe { caelix_renderer_get_nrd_info(state.renderer, &mut info) } == CaelixResult_CAELIX_OK
                && (info.width != 0 || info.failed != 0)
            {
                self.nrd_frames += 1;
            }
            if self.nrd_frames == 2 || (info.failed != 0 && self.nrd_frames == 1) {
                self.nrd_frames = 2;
                log::info!(
                    "NRD {}.{}.{}: REBLUR_DIFFUSE_SPECULAR at {}x{}, {} pipelines, {} permanent textures ({:.1} MiB), \
                     {} transient textures ({:.1} MiB), {} dispatches per frame{}",
                    info.version_major,
                    info.version_minor,
                    info.version_build,
                    info.width,
                    info.height,
                    info.pipeline_count,
                    info.permanent_texture_count,
                    info.permanent_bytes as f64 / (1024.0 * 1024.0),
                    info.transient_texture_count,
                    info.transient_bytes as f64 / (1024.0 * 1024.0),
                    info.dispatch_count,
                    if info.failed != 0 { ", FAILED (denoising disabled)" } else { "" }
                );
            }
        }

        // Window title once per second, log line every LOG_EVERY_FRAMES frames.
        self.timing.window_frames += 1;
        let elapsed = self.timing.window_start.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            self.timing.fps = self.timing.window_frames as f64 / elapsed;
            self.timing.window_frames = 0;
            self.timing.window_start = Instant::now();
            state.window.set_title(&timing_line(state.renderer, self.timing.fps));
        }
        if self.frames_presented > 0 && self.frames_presented.is_multiple_of(LOG_EVERY_FRAMES) {
            log::info!("{}", timing_line(state.renderer, self.timing.fps));
            if self.test.entity().is_some() {
                log::info!("{}", self.test.summary(&self.test.rates, &scene_stats(state.scene)));
            }
        }

        if let Some(at) = self.args.resize_at {
            if !self.resize_requested && self.frames_presented >= at {
                self.resize_requested = true;
                log::info!("resize test: requesting 960x540 after {} frames", self.frames_presented);
                let _ = state.window.request_inner_size(PhysicalSize::new(960, 540));
            }
        }
        if let Some(limit) = self.args.frames {
            if self.frames_presented >= limit {
                if let Some(loader) = state.loader.as_ref() {
                    loader.log_incomplete(state.scene);
                }
                if self.args.screenshot.is_some() && self.screenshot != Screenshot::Done {
                    log::warn!("exiting before the screenshot was taken");
                }
                log::info!("{} frames presented, exiting", self.frames_presented);
                event_loop.exit();
                return Ok(());
            }
        }
        state.window.request_redraw();
        Ok(())
    }

    /// WASD along the view and right axes, Q/E down/up along world Y, Shift x8.
    fn move_camera(&mut self, dt: f32) {
        let held = |key| self.input.keys.contains(&key);
        let axis = |plus, minus| f32::from(u8::from(held(plus))) - f32::from(u8::from(held(minus)));
        let forward = axis(KeyCode::KeyW, KeyCode::KeyS);
        let right = axis(KeyCode::KeyD, KeyCode::KeyA);
        let up = axis(KeyCode::KeyE, KeyCode::KeyQ);
        if forward == 0.0 && right == 0.0 && up == 0.0 {
            return;
        }
        let fast = held(KeyCode::ShiftLeft) || held(KeyCode::ShiftRight);
        let step = self.input.speed * if fast { FAST_FACTOR } else { 1.0 } * dt;
        self.camera.translate(forward * step, right * step, up * step);
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: anyhow::Error) {
        log::error!("{error:#}");
        self.error = Some(error);
        event_loop.exit();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match self.create_state(event_loop) {
            Ok(state) => {
                state.window.request_redraw();
                self.state = Some(state);
                self.last_frame = Instant::now();
                self.timing.window_start = Instant::now();
            }
            Err(e) => self.fail(event_loop, e),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // egui sees every window event first (while the UI is visible); the free camera then
        // ignores the pointer and the keys that egui uses.
        if let Some(state) = self.state.as_mut() {
            if let Some(ui) = state.ui.as_mut() {
                ui.on_window_event(&state.window, &event);
            }
        }
        let ui = self.state.as_ref().and_then(|state| state.ui.as_ref());
        let ui_pointer = ui.is_some_and(|ui| ui.wants_pointer());
        let ui_keyboard = ui.is_some_and(|ui| ui.wants_keyboard());
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = self.state.as_mut() {
                    log::debug!("window resized to {}x{}", size.width, size.height);
                    state.context.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } if !self.args.no_input => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    if !pressed {
                        // Always: a key pressed before egui took the focus must not stick.
                        self.input.keys.remove(&code);
                    } else if code == KeyCode::F1 {
                        // F1 shows and hides the debug UI.
                        if !event.repeat {
                            if let Some(ui) = self.state.as_mut().and_then(|state| state.ui.as_mut()) {
                                ui.visible = !ui.visible;
                            }
                        }
                    } else if !ui_keyboard {
                        self.input.keys.insert(code);
                        // P prints the pose in --pose form so a view can be reproduced.
                        if code == KeyCode::KeyP && !event.repeat {
                            log::info!("pose: {}", self.camera.pose_args());
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } if !self.args.no_input => {
                // A press over an egui window does not start the mouse look; a release always ends it.
                self.input.looking = state == ElementState::Pressed && !ui_pointer;
            }
            WindowEvent::MouseWheel { delta, .. } if !self.args.no_input && !ui_pointer => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                };
                self.input.speed = (self.input.speed * SCROLL_FACTOR.powf(notches)).clamp(1.0, 100_000.0);
                log::info!("camera speed {:.1} units/s", self.input.speed);
            }
            WindowEvent::Focused(false) => {
                self.input.keys.clear();
                self.input.looking = false;
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.draw(event_loop) {
                    self.fail(event_loop, e);
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            let ui_pointer = self
                .state
                .as_ref()
                .and_then(|state| state.ui.as_ref())
                .is_some_and(|ui| ui.wants_pointer());
            if self.input.looking && !self.args.no_input && !ui_pointer {
                self.camera
                    .rotate(dx as f32 * MOUSE_DEGREES_PER_PIXEL, dy as f32 * MOUSE_DEGREES_PER_PIXEL);
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // The test entity's numbers of the whole run (since the world was complete).
        if let (Some(state), Some(_)) = (self.state.as_ref(), self.test.entity()) {
            let stats = scene_stats(state.scene);
            match self.test.run_rates(&stats) {
                Some((rates, seconds)) => log::info!(
                    "{} (run of {seconds:.1} s since the world was complete)",
                    self.test.summary(&rates, &stats)
                ),
                None => log::info!(
                    "{} (last 1 s window; the world never completed)",
                    self.test.summary(&self.test.rates, &stats)
                ),
            }
        }
        // wait idle, caelix_renderer_destroy, caelix_scene_destroy, caelix_device_destroy, then
        // the context (State::drop).
        self.state = None;
    }
}

/// Directory of the debug UI's and --screenshot-every's PNGs: --screenshot-dir, else the
/// directory of the executable.
fn screenshot_dir(args: &Args) -> PathBuf {
    match &args.screenshot_dir {
        Some(dir) => dir.clone(),
        None => std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf))
            .unwrap_or_default(),
    }
}

/// --screenshot-every: a capture is due in this frame. The first one is --warmup frames (at
/// least 3) after the world is complete (without a world, after frame 10), then every N.
fn every_due(args: &Args, every_base: &mut Option<u64>, frames_presented: u64, world_ready: bool) -> bool {
    let Some(every) = args.screenshot_every.filter(|&n| n > 0) else {
        return false;
    };
    if every_base.is_none() {
        let ready = if args.world.is_some() {
            world_ready
        } else {
            frames_presented + 1 >= SCREENSHOT_FRAME_WITHOUT_WORLD
        };
        if ready {
            *every_base = Some(frames_presented + args.warmup.max(SCREENSHOT_EXTRA_FRAMES));
        }
    }
    every_base
        .is_some_and(|base| frames_presented >= base && (frames_presented - base).is_multiple_of(every))
}

fn gpu_timings(renderer: *const CaelixRenderer) -> CaelixGpuTimings {
    let mut timings =
        CaelixGpuTimings { trace_ms: 0.0, flush_ms: 0.0, frame_number: 0, composite_ms: 0.0, denoise_ms: 0.0 };
    // SAFETY: the renderer is live and timings is writable.
    unsafe { caelix_renderer_get_gpu_timings(renderer, &mut timings) };
    timings
}

fn scene_stats(scene: *const CaelixScene) -> CaelixSceneStats {
    // SAFETY: CaelixSceneStats holds only integers, so all-zero is a valid value.
    let mut stats: CaelixSceneStats = unsafe { std::mem::zeroed() };
    // SAFETY: the scene is live and stats is writable.
    unsafe { caelix_scene_get_stats(scene, &mut stats) };
    stats
}

/// All zero when the call fails.
fn nrd_info(renderer: *const CaelixRenderer) -> CaelixNrdInfo {
    // SAFETY: CaelixNrdInfo is a plain C struct of integers; all zero is a valid value.
    let mut info: CaelixNrdInfo = unsafe { std::mem::zeroed() };
    // SAFETY: the renderer is live and info is writable.
    if unsafe { caelix_renderer_get_nrd_info(renderer, &mut info) } != CaelixResult_CAELIX_OK {
        // SAFETY: as above.
        info = unsafe { std::mem::zeroed() };
    }
    info
}

/// "Caelix Renderer | trace 1.23 ms | denoise 0.40 ms | composite 0.05 ms | flush 0.45 ms | 144 fps"
fn timing_line(renderer: *const CaelixRenderer, fps: f64) -> String {
    let timings = gpu_timings(renderer);
    format!(
        "Caelix Renderer | trace {:.2} ms | denoise {:.2} ms | composite {:.2} ms | flush {:.2} ms | {:.0} fps",
        timings.trace_ms, timings.denoise_ms, timings.composite_ms, timings.flush_ms, fps
    )
}

/// Whether two cameras give the same image (reference accumulation).
fn same_camera(a: &CaelixCamera, b: &CaelixCamera) -> bool {
    a.position == b.position
        && a.right == b.right
        && a.up == b.up
        && a.forward == b.forward
        && a.tan_half_fov_y == b.tan_half_fov_y
        && a.aspect == b.aspect
}

/// Writes an RGBA PNG with alpha forced to 255: in the lighting modes the swapchain alpha is the
/// coverage (0 on the sky), which image viewers would show as transparent.
fn write_png(path: &Path, rgba: &[u8], width: u32, height: u32) -> anyhow::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    }
    let file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().context("PNG header")?;
    let mut opaque = rgba.to_vec();
    for pixel in opaque.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    writer.write_image_data(&opaque).context("PNG data")?;
    writer.finish().context("PNG finish")?;
    Ok(())
}

/// Forwards library log messages to `log` with target "caelix".
unsafe extern "C" fn caelix_log(level: CaelixLogLevel, message: *const c_char, _user_data: *mut c_void) {
    if message.is_null() {
        return;
    }
    // SAFETY: the library passes a NUL-terminated string valid during the call.
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    // bindgen emits the C enum as plain integer constants, so compare instead of match.
    let level = if level == CaelixLogLevel_CAELIX_LOG_DEBUG {
        log::Level::Debug
    } else if level == CaelixLogLevel_CAELIX_LOG_INFO {
        log::Level::Info
    } else if level == CaelixLogLevel_CAELIX_LOG_WARNING {
        log::Level::Warn
    } else {
        log::Level::Error
    };
    log::log!(target: "caelix", level, "{message}");
}

/// Converts the library's requirements to `caelix_vk` types. This lives in the
/// app so that `caelix-vk` stays independent of `caelix-sys`.
fn device_requirements() -> caelix_vk::DeviceRequirements {
    let mut raw = CaelixDeviceRequirements {
        api_version: 0,
        device_extension_count: 0,
        device_extensions: ptr::null(),
        feature_flags: 0,
    };
    // SAFETY: raw is writable; the extension strings are static in the library.
    unsafe { caelix_get_device_requirements(&mut raw) };
    let device_extensions = (0..raw.device_extension_count as usize)
        .map(|i| unsafe { CStr::from_ptr(*raw.device_extensions.add(i)) }.to_owned())
        .collect();
    let has = |bit: CaelixDeviceFeatureBits| raw.feature_flags & bit as u32 != 0;
    caelix_vk::DeviceRequirements {
        api_version: raw.api_version,
        device_extensions,
        features: caelix_vk::Features {
            buffer_device_address: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS),
            descriptor_indexing: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_DESCRIPTOR_INDEXING),
            synchronization2: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_SYNCHRONIZATION2),
            dynamic_rendering: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_DYNAMIC_RENDERING),
            timeline_semaphore: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_TIMELINE_SEMAPHORE),
            acceleration_structure: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_ACCELERATION_STRUCTURE),
            ray_query: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_RAY_QUERY),
            shader_int64: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_SHADER_INT64),
            scalar_block_layout: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT),
            storage_image_write_without_format: has(
                CaelixDeviceFeatureBits_CAELIX_FEATURE_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT,
            ),
            storage_image_read_without_format: has(
                CaelixDeviceFeatureBits_CAELIX_FEATURE_STORAGE_IMAGE_READ_WITHOUT_FORMAT,
            ),
        },
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let default_filter = if args.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter)).init();

    let mut version = CaelixVersion { major: 0, minor: 0, patch: 0, abi: 0 };
    // SAFETY: version is writable; the version string is static in the library.
    let version_string = unsafe {
        caelix_get_version(&mut version);
        CStr::from_ptr(caelix_version_string())
    };
    log::info!("caelix_renderer {} (abi {})", version_string.to_string_lossy(), version.abi);
    if version.abi != CAELIX_ABI_VERSION {
        bail!(
            "ABI mismatch: library reports abi {}, bindings expect abi {}",
            version.abi,
            CAELIX_ABI_VERSION
        );
    }
    // SAFETY: caelix_log is a valid callback for the whole process lifetime.
    unsafe { caelix_set_log_callback(Some(caelix_log), ptr::null_mut()) };

    if args.bounces > CAELIX_MAX_BOUNCES {
        bail!("--bounces {} is above the maximum {}", args.bounces, CAELIX_MAX_BOUNCES);
    }
    let mode = args.mode.unwrap_or(if args.world.is_some() { Mode::Pt } else { Mode::Primary });
    if args.accumulate.is_some() && mode != Mode::Ref {
        bail!("--accumulate needs --mode ref");
    }
    if args.bounces_transparent > CAELIX_MAX_BOUNCES_TRANSPARENT {
        bail!(
            "--bounces-transparent {} is above the maximum {}",
            args.bounces_transparent,
            CAELIX_MAX_BOUNCES_TRANSPARENT
        );
    }
    if args.view.needs_pt() && mode != Mode::Pt {
        bail!("--view {:?} needs --mode pt", args.view);
    }
    if args.view == View::Material && mode != Mode::Primary {
        bail!("--view material needs --mode primary");
    }
    log::info!(
        "mode {mode:?}, bounces {}, transparent bounces {}, exposure {}, sky on miss {}, sun radiance {:?}, \
         sun NEE {}, denoise {}, \
         window {}x{}",
        args.bounces,
        args.bounces_transparent,
        args.exposure,
        !args.no_sky_on_miss,
        args.sun_radiance,
        !args.no_sun_nee,
        mode == Mode::Pt && !args.no_denoise,
        args.size.0,
        args.size.1
    );

    let (sun_color, sun_intensity) = RenderSettings::split_radiance(args.sun_radiance);
    let settings = RenderSettings {
        mode,
        view: args.view,
        bounces: args.bounces,
        bounces_transparent: args.bounces_transparent,
        denoise: !args.no_denoise,
        sun_nee: !args.no_sun_nee,
        checkerboard: !args.no_checkerboard,
        sky_on_miss: !args.no_sky_on_miss,
        shading: !args.no_shading,
        exposure: args.exposure,
        sun_dir: args.sun,
        sun_radiance: args.sun_radiance,
        sun_color,
        sun_intensity,
        sun_angular_radius_deg: args.sun_angular_radius,
        sky_zenith: args.sky_zenith,
        sky_horizon: args.sky_horizon,
        sky_ground: args.sky_ground,
    };

    if !(1..=120).contains(&args.test_edit_interval) {
        bail!("--test-edit-interval {} is outside 1..120", args.test_edit_interval);
    }
    let mut test = TestScene::new(
        args.test_entity_at,
        args.test_paused,
        args.test_orbit_radius,
        args.test_orbit_speed,
        args.test_spin_speed,
        args.test_edits,
        args.test_edit_interval,
        args.test_edit_pattern,
    );
    // The entity is added by the first frame's update, with the start camera.
    test.enabled = args.test_entity || args.test_edits;

    let event_loop = EventLoop::new().context("cannot create the event loop")?;
    let fov = args.fov;
    let mut app = App {
        args,
        settings,
        last_options: None,
        accumulation: Accumulation { last_camera: None, frames: 0 },
        entities: Vec::new(),
        test,
        every_base: None,
        ui_screenshot_requested: false,
        readback_owner: None,
        screenshot_note: String::new(),
        state: None,
        camera: Camera::from_pose([0.0; 3], 0.0, 0.0, fov),
        input: Input { keys: HashSet::new(), looking: false, speed: DEFAULT_SPEED },
        last_frame: Instant::now(),
        frames_presented: 0,
        resize_requested: false,
        screenshot: Screenshot::Waiting,
        timing: Timing { window_start: Instant::now(), window_frames: 0, fps: 0.0 },
        load_mismatch: false,
        nrd_frames: 0,
        error: None,
    };
    let run_result = event_loop.run_app(&mut app);
    // In case the loop returned without `exiting` (error path).
    app.state = None;
    // SAFETY: disables library logging before the process tears down.
    unsafe { caelix_set_log_callback(None, ptr::null_mut()) };
    run_result.map_err(|e| anyhow!("event loop error: {e}"))?;
    if let Some(error) = app.error {
        return Err(error);
    }
    if app.load_mismatch {
        bail!("the scene does not hold every decoded brick (see the error above)");
    }
    Ok(())
}
