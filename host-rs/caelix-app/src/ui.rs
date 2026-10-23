//! Debug UI: six egui windows (Stats, Render, Camera, Entities, Test, Sequencer) in egui's default style,
//! drawn by `caelix-vk` as an overlay on the swapchain image after the library has written it.
//!
//! egui-winit turns the window events into egui input; this module holds the egui context and
//! builds the windows. They edit the host's state directly (render settings, camera, entity
//! transforms) and report one-shot requests (reset accumulation, screenshot, a file dialog) in
//! [`Panel`]. Every window starts collapsed except Entities without a world (it holds the world's
//! Load button), Test with `--test-*` and Sequencer with `--sequence`.

use std::path::Path;

use caelix_sys::*;
use egui::{CollapsingHeader, ComboBox, DragValue, Grid, ScrollArea, Slider};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::camera::{self, Camera};
use crate::sequence::{Interpolation, Playback, Sequencer, MIN_DURATION};
use crate::settings::{Mode, RenderSettings, UpscaleQuality, Upscaler, View, MAX_UPSCALE_PRESET};
use crate::testscene::{EditPattern, TestScene};
use crate::world::EntityEntry;

/// Size of the window titles (egui's Heading text style), in points. egui's default is 18; the
/// body text is 12.5.
const TITLE_SIZE: f32 = 15.5;

/// Values the Stats window shows, gathered before the egui pass.
pub struct Stats {
    pub timings: CaelixGpuTimings,
    pub fps: f64,
    pub frame_number: u64,
    /// Size of the render target (the swapchain image).
    pub target: (u32, u32),
    pub scene: CaelixSceneStats,
    /// The light tree (`enabled` = 0 without --light-tree).
    pub light: CaelixLightStats,
    /// The renderer's light NEE (`enabled` = 0 with --no-light-nee).
    pub light_nee: CaelixLightNeeStats,
    pub nrd: CaelixNrdInfo,
    pub load_state: String,
    /// Reference mode: frames accumulated since the last reset.
    pub accumulated: u64,
    /// The newest screenshot of the "Screenshot" button, or a waiting note.
    pub screenshot_note: String,
    /// The upscaler queries at the current quality and target size (`None` before the first).
    pub upscaler_sr: Option<CaelixUpscalerQueryResult>,
    pub upscaler_rr: Option<CaelixUpscalerQueryResult>,
    /// The renderer's current upscaler (`caelix_renderer_get_upscaler_info`).
    pub upscaler_info: CaelixUpscalerInfo,
    /// File name of the loaded world (`None` before the first load).
    pub world_name: Option<String>,
    /// The error of the last world load, else empty.
    pub world_note: String,
    /// File name of the sky image of the cubemap (`None` without one).
    pub sky_name: Option<String>,
    /// The error of the last sky load, else empty.
    pub sky_note: String,
}

/// A file dialog that a button asked for. The host runs it after the frame (the dialog blocks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileRequest {
    /// Open a `.cxw` world in place of the loaded one.
    LoadWorld,
    /// Open a lat-long `.exr` / `.hdr` sky image in place of the cubemap.
    LoadSky,
    /// Open a sequence JSON file.
    LoadSequence,
    /// Write the sequence to a JSON file (the dialog starts at the current file).
    SaveSequence,
    /// An empty sequence at the default file of the world (no dialog, nothing written).
    NewSequence,
}

/// Why an upscaler query reported "unsupported", in words (`CaelixUpscalerUnsupportedReason`).
pub fn unsupported_reason(result: &CaelixUpscalerQueryResult) -> String {
    let reason = result.reason;
    if reason == CaelixUpscalerUnsupportedReason_CAELIX_UPSCALER_OK as u32 {
        "supported".into()
    } else if reason == CaelixUpscalerUnsupportedReason_CAELIX_UPSCALER_NOT_BUILT as u32 {
        "the library was built without DLSS (CAELIX_DLSS=OFF)".into()
    } else if reason == CaelixUpscalerUnsupportedReason_CAELIX_UPSCALER_NO_NGX as u32 {
        "NGX did not start: no NVIDIA driver, a DLL or a device extension is missing (--no-dlss?)".into()
    } else if reason == CaelixUpscalerUnsupportedReason_CAELIX_UPSCALER_FEATURE_UNAVAILABLE as u32 {
        "the GPU or the NGX runtime lacks the feature".into()
    } else if reason == CaelixUpscalerUnsupportedReason_CAELIX_UPSCALER_DRIVER_TOO_OLD as u32 {
        format!("the driver is too old (needs {}.{})", result.min_driver_major, result.min_driver_minor)
    } else if reason == CaelixUpscalerUnsupportedReason_CAELIX_UPSCALER_QUALITY_UNAVAILABLE as u32 {
        "the feature has no render size for this quality".into()
    } else if reason == CaelixUpscalerUnsupportedReason_CAELIX_UPSCALER_BAD_ARGUMENT as u32 {
        "the window is below 64 x 32".into()
    } else {
        format!("unknown reason {reason}")
    }
}

/// One scene entity in the Entities window: its file transform and the edited transform.
pub struct EntityRow {
    pub scene_id: CaelixEntityId,
    pub guid: String,
    pub bricks: u64,
    pub original_position: [f32; 3],
    pub original_rotation: [f32; 4],
    pub position: [f32; 3],
    /// Unity Euler angles (pitch, yaw, roll) in degrees, as edited.
    pub euler: [f32; 3],
    /// The quaternion that the scene receives: the original one until the Euler angles change.
    pub rotation: [f32; 4],
    /// The transform changed and has not been sent to the scene yet.
    pub dirty: bool,
}

impl EntityRow {
    pub fn new(entry: &EntityEntry) -> Self {
        Self {
            scene_id: entry.scene_id,
            guid: entry.guid.clone(),
            bricks: entry.bricks_received,
            original_position: entry.position,
            original_rotation: entry.rotation,
            position: entry.position,
            euler: camera::euler_from_quat(entry.rotation),
            rotation: entry.rotation,
            dirty: false,
        }
    }

    /// A row for an entity that the host created itself (the test entity).
    pub fn synthetic(
        scene_id: CaelixEntityId,
        guid: &str,
        position: [f32; 3],
        rotation: [f32; 4],
        bricks: u64,
    ) -> Self {
        Self {
            scene_id,
            guid: guid.into(),
            bricks,
            original_position: position,
            original_rotation: rotation,
            position,
            euler: camera::euler_from_quat(rotation),
            rotation,
            dirty: false,
        }
    }

    pub fn set_euler(&mut self, euler: [f32; 3]) {
        self.euler = euler;
        self.rotation = camera::quat_from_euler(euler[0], euler[1], euler[2]);
        self.dirty = true;
    }

    /// Back to the file's transform (the original quaternion, not a round trip through Euler angles).
    pub fn reset(&mut self) {
        self.position = self.original_position;
        self.rotation = self.original_rotation;
        self.euler = camera::euler_from_quat(self.original_rotation);
        self.dirty = true;
    }
}

/// What the debug windows edit for one frame, and their one-shot requests.
pub struct Panel<'a> {
    pub stats: &'a Stats,
    pub settings: &'a mut RenderSettings,
    pub camera: &'a mut Camera,
    /// Free camera speed in units per second.
    pub speed: &'a mut f32,
    pub entities: &'a mut [EntityRow],
    /// The test entity and its edits (Test window).
    pub test: &'a mut TestScene,
    /// The camera sequence, its playback and its export (Sequencer window).
    pub sequencer: &'a mut Sequencer,
    /// Output: the "Reset accumulation" button was pressed.
    pub reset_accumulation: bool,
    /// Output: the "Screenshot" button was pressed.
    pub screenshot: bool,
    /// Output: a Load / Save / New button was pressed.
    pub file_request: Option<FileRequest>,
}

/// egui's output of one frame, for `caelix_vk::Context::draw_overlay`.
pub struct UiFrame {
    pub textures_delta: egui::TexturesDelta,
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub pixels_per_point: f32,
}

pub struct Ui {
    ctx: egui::Context,
    winit: egui_winit::State,
    /// The windows are drawn (F1 toggles them). A hidden UI receives no input.
    pub visible: bool,
    /// `--ui-demo`: every window and section starts open.
    demo: bool,
    /// The Entities window starts open (no world at start).
    entities_open: bool,
    /// The Test window starts open (`--test-entity`, `--test-edits`).
    test_open: bool,
    /// The Sequencer window starts open (`--sequence`).
    sequencer_open: bool,
}

impl Ui {
    pub fn new(
        window: &Window,
        visible: bool,
        demo: bool,
        entities_open: bool,
        test_open: bool,
        sequencer_open: bool,
    ) -> Self {
        let ctx = egui::Context::default();
        // egui's default style with two changes (owner's choice): no window or popup shadows, and
        // smaller window titles (the title uses the Heading text style, 18 by default).
        ctx.all_styles_mut(|style| {
            style.visuals.window_shadow = egui::epaint::Shadow::NONE;
            style.visuals.popup_shadow = egui::epaint::Shadow::NONE;
            style.text_styles.insert(
                egui::TextStyle::Heading,
                egui::FontId::new(TITLE_SIZE, egui::FontFamily::Proportional),
            );
        });
        let winit = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        Self { ctx, winit, visible, demo, entities_open, test_open, sequencer_open }
    }

    /// Feeds a window event to egui (only while the UI is visible).
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
        if self.visible {
            let _ = self.winit.on_window_event(window, event);
        }
    }

    /// egui uses the pointer (a hovered window or a drag on a widget): the free camera ignores it.
    pub fn wants_pointer(&self) -> bool {
        self.visible && self.ctx.egui_wants_pointer_input()
    }

    /// A widget has keyboard focus: the free camera ignores the keys.
    pub fn wants_keyboard(&self) -> bool {
        self.visible && self.ctx.egui_wants_keyboard_input()
    }

    /// Runs one egui pass with the debug windows and tessellates it.
    pub fn run(&mut self, window: &Window, panel: &mut Panel<'_>) -> UiFrame {
        let raw_input = self.winit.take_egui_input(window);
        let open = StartOpen {
            demo: self.demo,
            entities: self.entities_open,
            test: self.test_open,
            sequencer: self.sequencer_open,
        };
        let output = self.ctx.run_ui(raw_input, |root| debug_windows(root.ctx(), panel, open));
        self.winit.handle_platform_output(window, output.platform_output);
        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        UiFrame { textures_delta: output.textures_delta, primitives, pixels_per_point: output.pixels_per_point }
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

/// Which windows start open (egui reads it on the first frame only).
#[derive(Clone, Copy)]
struct StartOpen {
    /// `--ui-demo`: every window.
    demo: bool,
    entities: bool,
    test: bool,
    sequencer: bool,
}

/// Six separate egui windows: Stats at the top left, Camera at the bottom left (with `--ui-demo`,
/// where Stats is fully open, at the bottom centre, left of the Sequencer), Render at the top right,
/// Entities at the top centre, Test at the bottom right, Sequencer at the bottom centre. Each is
/// draggable and collapsible (title bar) and starts collapsed unless `open` says otherwise; egui
/// keeps the positions after the first frame.
fn debug_windows(ctx: &egui::Context, panel: &mut Panel<'_>, open: StartOpen) {
    let demo = open.demo;
    let screen = ctx.content_rect();
    let margin = 8.0;
    egui::Window::new("Stats")
        .default_pos([screen.min.x + margin, screen.min.y + margin])
        .default_open(demo)
        .show(ctx, |ui| stats_window(ui, panel.stats, demo));
    let (camera_pivot, camera_x) = if demo {
        (egui::Align2::CENTER_BOTTOM, screen.center().x - 260.0)
    } else {
        (egui::Align2::LEFT_BOTTOM, screen.min.x + margin)
    };
    egui::Window::new("Camera")
        .pivot(camera_pivot)
        .default_pos([camera_x, screen.max.y - margin])
        .default_open(demo)
        .show(ctx, |ui| camera_window(ui, panel, demo));
    egui::Window::new("Render")
        .pivot(egui::Align2::RIGHT_TOP)
        .default_pos([screen.max.x - margin, screen.min.y + margin])
        .default_open(demo)
        .show(ctx, |ui| render_window(ui, panel, demo));
    egui::Window::new(format!("Entities ({})", panel.entities.len()))
        .id(egui::Id::new("entities_window"))
        .pivot(egui::Align2::CENTER_TOP)
        .default_pos([screen.center().x, screen.min.y + margin])
        .default_open(open.entities || demo)
        .show(ctx, |ui| entities_window(ui, panel, demo));
    egui::Window::new("Test")
        .pivot(egui::Align2::RIGHT_BOTTOM)
        .default_pos([screen.max.x - margin, screen.max.y - margin])
        .default_open(open.test || demo)
        .show(ctx, |ui| test_window(ui, panel));
    let sequencer_x = if demo { screen.center().x + 220.0 } else { screen.center().x };
    egui::Window::new("Sequencer")
        .pivot(egui::Align2::CENTER_BOTTOM)
        .default_pos([sequencer_x, screen.max.y - margin])
        .default_open(open.sequencer || demo)
        .show(ctx, |ui| sequencer_window(ui, panel));
}

/// Two-column grid of label / value rows.
fn value_grid(ui: &mut egui::Ui, id: &str, rows: &[(&str, String)]) {
    Grid::new(id).num_columns(2).striped(true).show(ui, |ui| {
        for (name, value) in rows {
            ui.label(*name);
            ui.label(value);
            ui.end_row();
        }
    });
}

fn stats_window(ui: &mut egui::Ui, stats: &Stats, demo: bool) {
    ui.label("F1 hides the debug UI.");
    let t = &stats.timings;
    let s = &stats.scene;
    CollapsingHeader::new("Frame").default_open(true).show(ui, |ui| {
        value_grid(
            ui,
            "stats_frame",
            &[
                ("fps", format!("{:.0}", stats.fps)),
                ("frame", stats.frame_number.to_string()),
                ("target", format!("{} x {}", stats.target.0, stats.target.1)),
            ],
        );
    });
    CollapsingHeader::new("GPU times").default_open(true).show(ui, |ui| {
        value_grid(
            ui,
            "stats_gpu",
            &[
                ("of frame", t.frame_number.to_string()),
                ("trace", format!("{:.2} ms", t.trace_ms)),
                ("denoise", format!("{:.2} ms", t.denoise_ms)),
                ("upscale", format!("{:.2} ms", t.upscale_ms)),
                ("composite", format!("{:.2} ms", t.composite_ms)),
                ("flush", format!("{:.2} ms", t.flush_ms)),
            ],
        );
    });
    CollapsingHeader::new("Scene").default_open(demo).show(ui, |ui| {
        value_grid(
            ui,
            "stats_scene",
            &[
                ("entities", s.entities.to_string()),
                ("groups", s.groups.to_string()),
                ("bricks resident", s.bricks_resident.to_string()),
                ("records", format!("{:.1} MiB", mib(s.record_bytes))),
                ("slot tables", format!("{:.1} MiB", mib(s.slot_table_bytes))),
                ("AS builds pending", s.as_builds_pending.to_string()),
            ],
        );
    });
    let l = &stats.light;
    if l.enabled != 0 {
        CollapsingHeader::new("Light tree").default_open(true).show(ui, |ui| {
            value_grid(
                ui,
                "stats_light",
                &[
                    ("light groups", l.light_groups.to_string()),
                    ("light bricks", l.light_bricks.to_string()),
                    ("light faces", l.light_faces.to_string()),
                    ("bricks dropped", l.bricks_dropped.to_string()),
                    ("heap used / capacity", format!("{:.1} / {:.1} MiB", mib(l.heap_bytes_used), mib(l.heap_bytes_capacity))),
                    ("passes (failed)", format!("{} ({})", l.passes, l.failed_passes)),
                    ("records recomputed / copied", format!("{} / {}", l.records_recomputed, l.records_copied)),
                    (
                        "static BVH leaves / nodes / builds",
                        format!("{} / {} / {}", l.tree_leaves[0], l.tree_nodes[0], l.tree_builds[0]),
                    ),
                    (
                        "dynamic BVH leaves / nodes / builds",
                        format!("{} / {} / {}", l.tree_leaves[1], l.tree_nodes[1], l.tree_builds[1]),
                    ),
                    ("light update", format!("{:.3} ms", l.gpu_ms)),
                ],
            );
        });
    }
    let n = &stats.light_nee;
    if n.enabled != 0 {
        CollapsingHeader::new("Light NEE").default_open(true).show(ui, |ui| {
            let mode = match n.reservoir_mode {
                CAELIX_LIGHT_NEE_RESERVOIRS_OFF => "off",
                CAELIX_LIGHT_NEE_RESERVOIRS_ROTATE => "rotate",
                _ => "-",
            };
            value_grid(
                ui,
                "stats_light_nee",
                &[
                    ("running", if n.running != 0 { "yes" } else { "no (no light tree or primary mode)" }.to_string()),
                    ("cells live / active / capacity", format!("{} / {} / {}", n.live_cells, n.active_cells, n.cell_capacity)),
                    ("inserted / pending", format!("{} / {}", n.inserted, n.pending)),
                    ("failed slot / payload", format!("{} / {}", n.failed_slot, n.failed_payload)),
                    ("evicted / rebuild dropped", format!("{} / {}", n.evicted, n.rebuild_dropped)),
                    ("fresh / norm draws", format!("{} / {}", n.fresh_draws, n.normalisation_draws)),
                    ("stale refreshed", n.stale_refreshed.to_string()),
                    ("reservoirs / candidates", format!("{mode} / {}", n.candidates)),
                    ("footprint (scale)", format!("{:.3} ({:.2})", n.footprint, n.controller_scale)),
                    ("maintenance / fill", format!("{:.3} / {:.3} ms", n.maintenance_ms, n.fill_ms)),
                    ("cell memory", format!("{:.1} MiB", mib(n.cell_bytes))),
                ],
            );
        });
    }
    CollapsingHeader::new("NRD and world").default_open(demo).show(ui, |ui| {
        let nrd = &stats.nrd;
        let version = format!("{}.{}.{}", nrd.version_major, nrd.version_minor, nrd.version_build);
        let nrd_line = if nrd.failed != 0 {
            format!("NRD {version}: FAILED (denoising disabled)")
        } else if nrd.width == 0 {
            format!("NRD {version}: no instance yet")
        } else {
            format!(
                "NRD {version}: {}x{}, pools {:.1} MiB, {} dispatches",
                nrd.width,
                nrd.height,
                mib(nrd.permanent_bytes + nrd.transient_bytes),
                nrd.dispatch_count
            )
        };
        ui.label(nrd_line);
        ui.label(format!("world: {}", stats.load_state));
    });
}

/// The upscaler rows of the Mode grid: upscaler, quality, preset and the current render size.
/// Pt mode only; an unsupported upscaler is shown disabled with the reason as hover text.
fn upscaler_rows(ui: &mut egui::Ui, stats: &Stats, settings: &mut RenderSettings) {
    let pt = settings.mode == Mode::Pt;
    let upscaling = pt && settings.upscaler != Upscaler::None;

    ui.label("upscaler");
    ui.add_enabled_ui(pt, |ui| {
        ComboBox::from_id_salt("upscaler").selected_text(settings.upscaler.label()).show_ui(ui, |ui| {
            for upscaler in Upscaler::ALL {
                let result = match upscaler {
                    Upscaler::None => None,
                    Upscaler::DlssSr => stats.upscaler_sr,
                    Upscaler::DlssRr => stats.upscaler_rr,
                };
                let unsupported = result.filter(|r| r.supported == 0);
                let response = ui
                    .add_enabled_ui(unsupported.is_none(), |ui| {
                        ui.selectable_value(&mut settings.upscaler, upscaler, upscaler.label())
                    })
                    .inner;
                if let Some(r) = unsupported {
                    response.on_disabled_hover_text(unsupported_reason(&r));
                }
            }
        });
    });
    ui.end_row();

    ui.label("quality");
    ui.add_enabled_ui(upscaling, |ui| {
        ComboBox::from_id_salt("upscale_quality")
            .selected_text(settings.upscale_quality.label())
            .show_ui(ui, |ui| {
                for quality in UpscaleQuality::ALL {
                    ui.selectable_value(&mut settings.upscale_quality, quality, quality.label());
                }
            });
    });
    ui.end_row();

    ui.label("preset");
    ui.add_enabled(upscaling, DragValue::new(&mut settings.upscale_preset).range(0..=MAX_UPSCALE_PRESET))
        .on_hover_text(PRESET_HINT);
    ui.end_row();

    ui.label("upscale");
    let info = &stats.upscaler_info;
    let text = if !upscaling {
        "off".to_string()
    } else if info.upscaler == 0 {
        "no upscaled frame yet".to_string()
    } else {
        format!(
            "render {}x{} -> output {}x{}{}",
            info.render_width,
            info.render_height,
            info.output_width,
            info.output_height,
            if info.failed != 0 { " FAILED" } else { "" }
        )
    };
    ui.label(text);
    ui.end_row();
}

/// Pitch (elevation above the horizon, positive = up) and yaw (0 = +Z, 90 = +X) of a direction,
/// in degrees.
fn direction_angles(dir: [f32; 3]) -> (f32, f32) {
    let length = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if length == 0.0 {
        return (90.0, 0.0); // the library uses (0, 1, 0) for a zero vector
    }
    let pitch = (dir[1] / length).clamp(-1.0, 1.0).asin().to_degrees();
    let yaw = dir[0].atan2(dir[2]).to_degrees();
    (pitch, yaw)
}

fn direction_from_angles(pitch: f32, yaw: f32) -> [f32; 3] {
    let (sp, cp) = pitch.to_radians().sin_cos();
    let (sy, cy) = yaw.to_radians().sin_cos();
    [cp * sy, sp, cp * cy]
}

/// Hover text of the preset value.
const PRESET_HINT: &str = "0 = the NGX default; else the raw NGX preset value of the chosen quality mode.\n\
    DLSS SR: J = 10, K = 11, L = 12, M = 13.\nDLSS RR: D = 4, E = 5, F = 6.";

fn render_window(ui: &mut egui::Ui, panel: &mut Panel<'_>, demo: bool) {
    let stats = panel.stats;
    let settings = &mut *panel.settings;
    let lighting = settings.lighting();
    CollapsingHeader::new("Mode").default_open(true).show(ui, |ui| {
        Grid::new("render_mode").num_columns(2).show(ui, |ui| {
            ui.label("mode");
            ComboBox::from_id_salt("mode").selected_text(settings.mode.label()).show_ui(ui, |ui| {
                for mode in Mode::ALL {
                    ui.selectable_value(&mut settings.mode, mode, mode.label());
                }
            });
            ui.end_row();

            ui.label("debug view");
            ComboBox::from_id_salt("view").selected_text(settings.view.label()).show_ui(ui, |ui| {
                for view in View::ALL {
                    ui.selectable_value(&mut settings.view, view, view.label());
                }
            });
            ui.end_row();

            ui.label("bounces");
            ui.add_enabled(lighting, Slider::new(&mut settings.bounces, 0..=CAELIX_MAX_BOUNCES));
            ui.end_row();

            ui.label("transparent bounces");
            ui.add_enabled(
                lighting,
                Slider::new(&mut settings.bounces_transparent, 0..=CAELIX_MAX_BOUNCES_TRANSPARENT),
            );
            ui.end_row();

            ui.label("exposure");
            ui.add_enabled(lighting, Slider::new(&mut settings.exposure, 0.05..=8.0).logarithmic(true));
            ui.end_row();

            upscaler_rows(ui, stats, settings);
        });
        ui.add_enabled(settings.mode == Mode::Pt, egui::Checkbox::new(&mut settings.denoise, "denoise (NRD, pt mode)"));
        ui.add_enabled(lighting, egui::Checkbox::new(&mut settings.sun_nee, "sun NEE"));
        ui.add_enabled(
            settings.mode == Mode::Pt,
            egui::Checkbox::new(&mut settings.checkerboard, "glass checkerboard (pt mode)"),
        );
        ui.add_enabled(
            settings.mode == Mode::Pt && settings.upscaler == Upscaler::DlssSr,
            egui::Checkbox::new(&mut settings.upscale_auto_exposure, "auto exposure (SR)"),
        );
        ui.add_enabled(lighting, egui::Checkbox::new(&mut settings.sky_on_miss, "sky on miss"));
        ui.add_enabled(!lighting, egui::Checkbox::new(&mut settings.shading, "Lambert (primary mode)"));
    });

    CollapsingHeader::new("Sun").default_open(demo).show(ui, |ui| {
        let (mut pitch, mut yaw) = direction_angles(settings.effective_sun_dir());
        Grid::new("render_sun").num_columns(2).show(ui, |ui| {
            ui.label("pitch");
            let a = ui.add(Slider::new(&mut pitch, -90.0..=90.0).suffix(" deg"));
            ui.end_row();
            ui.label("yaw");
            let b = ui.add(Slider::new(&mut yaw, -180.0..=180.0).suffix(" deg"));
            ui.end_row();
            if a.changed() || b.changed() {
                settings.sun_dir = Some(direction_from_angles(pitch, yaw));
            }
            let dir = settings.effective_sun_dir();
            ui.label("direction");
            ui.label(format!("({:.3}, {:.3}, {:.3})", dir[0], dir[1], dir[2]));
            ui.end_row();

            ui.label("colour");
            let a = ui.add_enabled(lighting, |ui: &mut egui::Ui| ui.color_edit_button_rgb(&mut settings.sun_color));
            ui.end_row();
            ui.label("intensity");
            let b = ui.add_enabled(
                lighting,
                Slider::new(&mut settings.sun_intensity, 0.0..=1e5).logarithmic(true).smallest_positive(1.0),
            );
            ui.end_row();
            if a.changed() || b.changed() {
                settings.sun_radiance = settings.sun_color.map(|c| c * settings.sun_intensity);
            }
            let r = settings.sun_radiance;
            ui.label("radiance");
            ui.label(format!("({:.4e}, {:.4e}, {:.4e})", r[0], r[1], r[2]));
            ui.end_row();

            ui.label("angular radius");
            ui.add_enabled(lighting, Slider::new(&mut settings.sun_angular_radius_deg, 0.1..=10.0).suffix(" deg"));
            ui.end_row();
        });
    });

    CollapsingHeader::new("Sky").default_open(demo).show(ui, |ui| {
        let cubemap = settings.sky_cubemap;
        ui.horizontal(|ui| {
            ui.label("image");
            file_label(ui, stats.sky_name.as_deref(), "none (procedural sky)");
            let load = ui.button("Load").on_hover_text("open a lat-long .exr or .hdr sky image without the sun disk");
            if load.clicked() {
                panel.file_request = Some(FileRequest::LoadSky);
            }
        });
        if !stats.sky_note.is_empty() {
            ui.label(&stats.sky_note);
        }
        ui.horizontal(|ui| {
            let checkbox = ui.add_enabled(
                lighting && cubemap.is_some(),
                egui::Checkbox::new(&mut settings.use_sky_cubemap, "cubemap"),
            );
            match cubemap {
                Some(c) => {
                    ui.label(format!("face {}, {} mips", c.size, c.mip_levels));
                }
                None => {
                    checkbox.on_disabled_hover_text("no sky image loaded");
                }
            }
        });
        // The procedural colours are greyed out while the cubemap replaces them.
        let procedural = lighting && !(cubemap.is_some() && settings.use_sky_cubemap);
        Grid::new("render_sky").num_columns(2).show(ui, |ui| {
            for (name, color) in [
                ("zenith", &mut settings.sky_zenith),
                ("horizon", &mut settings.sky_horizon),
                ("ground", &mut settings.sky_ground),
            ] {
                ui.label(name);
                ui.horizontal(|ui| {
                    ui.add_enabled(procedural, |ui: &mut egui::Ui| ui.color_edit_button_rgb(color));
                    ui.label(format!("{:.3}, {:.3}, {:.3}", color[0], color[1], color[2]));
                });
                ui.end_row();
            }
        });
    });

    CollapsingHeader::new("Fog").default_open(demo).show(ui, |ui| {
        Grid::new("render_fog").num_columns(2).show(ui, |ui| {
            ui.label("density");
            ui.add_enabled(
                lighting,
                Slider::new(&mut settings.fog_density, 0.0..=1.0)
                    .logarithmic(true)
                    .smallest_positive(1e-5)
                    .suffix(" /unit"),
            );
            ui.end_row();
            ui.label("colour");
            ui.horizontal(|ui| {
                let color = &mut settings.fog_color;
                ui.add_enabled(lighting, |ui: &mut egui::Ui| ui.color_edit_button_rgb(color));
                ui.label(format!("{:.3}, {:.3}, {:.3}", color[0], color[1], color[2]));
            });
            ui.end_row();
        });
    });

    CollapsingHeader::new("Materials").default_open(demo).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("emission scale");
            ui.add_enabled(
                lighting,
                Slider::new(&mut settings.emission_scale, 0.01..=64.0).logarithmic(true),
            )
            .on_hover_text("multiplies every material's emission (0.01 = almost off; the library treats exactly 0 as 1)");
        });
    });

    CollapsingHeader::new("Accumulation and screenshot").default_open(true).show(ui, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Reset accumulation").clicked() {
                panel.reset_accumulation = true;
            }
            if settings.mode == Mode::Ref {
                ui.label(format!("accumulated: {} frames", panel.stats.accumulated));
            }
        });
        if ui.button("Screenshot").on_hover_text("PNG of the image without the debug UI").clicked() {
            panel.screenshot = true;
        }
        if !panel.stats.screenshot_note.is_empty() {
            ui.label(&panel.stats.screenshot_note);
        }
    });
}

fn camera_window(ui: &mut egui::Ui, panel: &mut Panel<'_>, demo: bool) {
    let camera = &mut *panel.camera;
    CollapsingHeader::new("Pose").default_open(true).show(ui, |ui| {
        Grid::new("camera_pose").num_columns(2).show(ui, |ui| {
            ui.label("position");
            ui.horizontal(|ui| {
                for value in &mut camera.position {
                    ui.add(DragValue::new(value).speed(1.0).fixed_decimals(1));
                }
            });
            ui.end_row();
            ui.label("pitch");
            ui.add(DragValue::new(&mut camera.pitch).speed(0.2).range(-89.0..=89.0).suffix(" deg"));
            ui.end_row();
            ui.label("yaw");
            if ui.add(DragValue::new(&mut camera.yaw).speed(0.2).suffix(" deg")).changed() {
                camera.yaw = camera::wrap_degrees(camera.yaw);
            }
            ui.end_row();
            ui.label("vertical fov");
            ui.add(DragValue::new(&mut camera.fov_y).speed(0.2).range(1.0..=170.0).suffix(" deg"));
            ui.end_row();
        });
        ui.horizontal(|ui| {
            if ui.button("Copy --pose").clicked() {
                ui.ctx().copy_text(camera.pose_args());
            }
            ui.label("(P logs it)");
        });
    });
    CollapsingHeader::new("Movement").default_open(demo).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("speed");
            ui.add(Slider::new(&mut *panel.speed, 1.0..=100_000.0).logarithmic(true).suffix(" /s"));
        });
        ui.label("WASD move, Q/E down/up, Shift x8, right mouse button looks, wheel scales the speed.");
    });
}

/// A file name in strong text, or the weak hint `none` when there is no file.
fn file_label(ui: &mut egui::Ui, name: Option<&str>, none: &str) {
    match name {
        Some(name) => ui.strong(name),
        None => ui.weak(none),
    };
}

/// The world row (file name, Load), the load state and the entity list.
fn entities_window(ui: &mut egui::Ui, panel: &mut Panel<'_>, demo: bool) {
    let stats = panel.stats;
    ui.horizontal(|ui| {
        ui.label("world");
        file_label(ui, stats.world_name.as_deref(), "none");
        // The export renders the loaded world; a new one would break it.
        let load = ui.add_enabled(!panel.sequencer.is_exporting(), egui::Button::new("Load"));
        if load.on_hover_text("open a .cxw world in place of the loaded one").clicked() {
            panel.file_request = Some(FileRequest::LoadWorld);
        }
    });
    if !stats.world_note.is_empty() {
        ui.label(&stats.world_note);
    } else if stats.world_name.is_some() {
        ui.label(&stats.load_state);
    } else {
        ui.label("Load opens a .cxw world.");
    }
    if panel.entities.is_empty() {
        return;
    }
    ui.separator();
    ScrollArea::vertical().max_height(300.0).show(ui, |ui| entities_list(ui, panel.entities, demo));
}

fn entities_list(ui: &mut egui::Ui, entities: &mut [EntityRow], demo: bool) {
    for (index, entity) in entities.iter_mut().enumerate() {
        let guid_short = entity.guid.get(..8).unwrap_or(&entity.guid);
        CollapsingHeader::new(format!("#{index}  {guid_short}  {} bricks", entity.bricks))
            .id_salt(("entity", index))
            .default_open(demo && index == 0)
            .show(ui, |ui| {
                ui.label(format!("scene id {}, guid {}", entity.scene_id, entity.guid));
                let mut euler = entity.euler;
                let mut rotated = false;
                Grid::new(("entity_grid", index)).num_columns(2).show(ui, |ui| {
                    ui.label("position");
                    ui.horizontal(|ui| {
                        for value in &mut entity.position {
                            if ui.add(DragValue::new(value).speed(1.0).fixed_decimals(1)).changed() {
                                entity.dirty = true;
                            }
                        }
                    });
                    ui.end_row();
                    for (axis, name) in ["pitch (X)", "yaw (Y)", "roll (Z)"].into_iter().enumerate() {
                        ui.label(name);
                        rotated |= ui
                            .add(DragValue::new(&mut euler[axis]).speed(0.5).fixed_decimals(2).suffix(" deg"))
                            .changed();
                        ui.end_row();
                    }
                });
                if rotated {
                    entity.set_euler(euler);
                }
                ui.horizontal(|ui| {
                    if ui.button("Reset").clicked() {
                        entity.reset();
                    }
                    let q = entity.rotation;
                    ui.label(format!("q ({:.3}, {:.3}, {:.3}, {:.3})", q[0], q[1], q[2], q[3]));
                });
            });
    }
}

/// The test entity (enable, pause, orbit, spin), its timed edits and the scene stats that matter
/// for them.
fn test_window(ui: &mut egui::Ui, panel: &mut Panel<'_>) {
    let test = &mut *panel.test;
    CollapsingHeader::new("Entity").default_open(true).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.checkbox(&mut test.enabled, "enabled")
                .on_hover_text("adds the box in front of the camera (or at --test-entity-at); off removes it");
            ui.checkbox(&mut test.paused, "paused");
        });
        Grid::new("test_entity").num_columns(2).show(ui, |ui| {
            ui.label("orbit radius");
            ui.add(Slider::new(&mut test.orbit_radius, 0.0..=100.0));
            ui.end_row();
            ui.label("orbit speed");
            ui.add(Slider::new(&mut test.orbit_speed, -360.0..=360.0).suffix(" deg/s"));
            ui.end_row();
            ui.label("spin speed");
            ui.add(Slider::new(&mut test.spin_speed, -360.0..=360.0).suffix(" deg/s"));
            ui.end_row();
        });
        ui.horizontal(|ui| {
            if ui.add_enabled(test.entity().is_some(), egui::Button::new("Reset to start")).clicked() {
                test.reset_requested = true;
            }
            match test.entity() {
                Some(id) => {
                    let c = test.centre();
                    ui.label(format!(
                        "scene id {id}, t {:.2} s, centre ({:.1}, {:.1}, {:.1})",
                        test.time(),
                        c[0],
                        c[1],
                        c[2]
                    ))
                }
                None => ui.label("no test entity"),
            };
        });
    });
    CollapsingHeader::new("Edits").default_open(true).show(ui, |ui| {
        ui.checkbox(&mut test.edits_enabled, "enabled").on_hover_text("creates the test entity when it is absent");
        Grid::new("test_edits").num_columns(2).show(ui, |ui| {
            ui.label("interval");
            ui.add(Slider::new(&mut test.interval, 1..=120).suffix(" frames"));
            ui.end_row();
            ui.label("pattern");
            ComboBox::from_id_salt("test_pattern").selected_text(test.pattern.label()).show_ui(ui, |ui| {
                for pattern in EditPattern::ALL {
                    ui.selectable_value(&mut test.pattern, pattern, pattern.label());
                }
            });
            ui.end_row();
        });
        if ui.add_enabled(test.entity().is_some(), egui::Button::new("Step once")).clicked() {
            test.step_once = true;
        }
        value_grid(
            ui,
            "test_counts",
            &[
                ("ticks", test.ticks.to_string()),
                ("bricks uploaded", test.uploads.to_string()),
                ("bricks removed", test.removes.to_string()),
                ("uploads queued", test.pending_uploads().to_string()),
                ("entity bricks", test.brick_count().to_string()),
            ],
        );
    });
    CollapsingHeader::new("Stats").default_open(true).show(ui, |ui| {
        let r = &test.rates;
        value_grid(
            ui,
            "test_stats",
            &[
                ("flush (newest)", format!("{:.3} ms", panel.stats.timings.flush_ms)),
                ("flush (1 s mean / max)", format!("{:.3} / {:.3} ms", r.flush_ms_mean, r.flush_ms_max)),
                ("AS builds", format!("{:.1} /s", r.as_builds_per_s)),
                ("AS builds pending", panel.stats.scene.as_builds_pending.to_string()),
                ("bricks finalized", format!("{:.1} /s", r.finalized_per_s)),
            ],
        );
    });
}

/// A button of a keyframe row; applied after the grid, which borrows the list.
enum RowAction {
    Select(usize),
    Go(usize),
    Set(usize),
    Duplicate(usize),
    Remove(usize),
}

/// The camera sequencer: file, transport, keyframe list and export. File and keyframe edits are
/// disabled during an export.
fn sequencer_window(ui: &mut egui::Ui, panel: &mut Panel<'_>) {
    let seq = &mut *panel.sequencer;
    let camera = &mut *panel.camera;
    let exporting = seq.is_exporting();
    let total = seq.sequence.total();
    let has_keyframes = !seq.sequence.keyframes.is_empty();

    CollapsingHeader::new("File").default_open(true).show(ui, |ui| {
        ui.add_enabled_ui(!exporting, |ui| {
            ui.horizontal(|ui| {
                ui.label("file").on_hover_text(&seq.file);
                let name = Path::new(&seq.file).file_name().map(|n| n.to_string_lossy());
                file_label(ui, name.as_deref(), "none");
                // The host runs the dialogs after the frame; the result is in the note.
                if ui.button("Load").on_hover_text("open a sequence JSON file").clicked() {
                    panel.file_request = Some(FileRequest::LoadSequence);
                }
                if ui.button("Save").on_hover_text("write the sequence to a JSON file").clicked() {
                    panel.file_request = Some(FileRequest::SaveSequence);
                }
                let new = ui.button("New").on_hover_text("an empty sequence at the world's default file");
                if new.clicked() {
                    panel.file_request = Some(FileRequest::NewSequence);
                }
            });
            Grid::new("sequencer_file").num_columns(2).show(ui, |ui| {
                ui.label("name");
                ui.text_edit_singleline(&mut seq.sequence.name);
                ui.end_row();
                ui.label("interpolation");
                ComboBox::from_id_salt("sequencer_interpolation")
                    .selected_text(seq.sequence.interpolation.label())
                    .show_ui(ui, |ui| {
                        for interpolation in Interpolation::ALL {
                            ui.selectable_value(&mut seq.sequence.interpolation, interpolation, interpolation.label());
                        }
                    });
                ui.end_row();
                ui.label("loop");
                ui.checkbox(&mut seq.sequence.looped, "");
                ui.end_row();
                ui.label("fps");
                ui.add(DragValue::new(&mut seq.sequence.fps).range(1..=240)).on_hover_text("the export frame rate");
                ui.end_row();
            });
        });
        // During an export the Export section shows the progress note.
        if !exporting && !seq.note.is_empty() {
            ui.label(&seq.note);
        }
    });

    CollapsingHeader::new("Transport").default_open(true).show(ui, |ui| {
        ui.horizontal(|ui| {
            if ui.add_enabled(!exporting, egui::Button::new("|<")).clicked() {
                seq.seek(0.0, camera);
            }
            let label = if seq.playback == Playback::Playing { "Pause" } else { "Play" };
            if ui.add_enabled(has_keyframes && !exporting, egui::Button::new(label)).clicked() {
                seq.toggle_play();
            }
            if ui.add_enabled(!exporting, egui::Button::new("Stop")).clicked() {
                seq.stop();
            }
        });
        ui.horizontal(|ui| {
            let mut time = seq.time;
            let slider = Slider::new(&mut time, 0.0..=total.max(MIN_DURATION)).suffix(" s");
            if ui.add_enabled(!exporting && total > 0.0, slider).changed() {
                seq.seek(time, camera);
            }
            ui.label(format!("{:.2} / {total:.2} s", seq.time));
        });
        ui.label("K adds a keyframe at the camera, Space plays / pauses, moving the camera stops the playback.");
    });

    CollapsingHeader::new(format!("Keyframes ({})", seq.sequence.keyframes.len()))
        .id_salt("sequencer_keyframes")
        .default_open(true)
        .show(ui, |ui| {
            ui.add_enabled_ui(!exporting, |ui| {
                let mut action = None;
                ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                    Grid::new("sequencer_keyframe_grid").num_columns(6).striped(true).show(ui, |ui| {
                        for header in ["#", "time", "hold", "travel", "pose", ""] {
                            ui.label(header);
                        }
                        ui.end_row();
                        let n = seq.sequence.keyframes.len();
                        let looped = seq.sequence.looped;
                        for i in 0..n {
                            if ui.selectable_label(seq.selected == Some(i), format!("#{i}")).clicked() {
                                action = Some(RowAction::Select(i));
                            }
                            ui.label(format!("{:.2}", seq.sequence.keyframe_time(i)));
                            let k = &mut seq.sequence.keyframes[i];
                            ui.add(DragValue::new(&mut k.hold).speed(0.1).range(0.0..=600.0).suffix(" s"));
                            // The last keyframe of a sequence that does not loop travels nowhere.
                            ui.add_enabled(
                                looped || i + 1 < n,
                                DragValue::new(&mut k.duration).speed(0.1).range(MIN_DURATION..=600.0).suffix(" s"),
                            );
                            let p = k.position;
                            ui.label(format!(
                                "{:.0}, {:.0}, {:.0}  {:.0}/{:.0}  fov {:.0}",
                                p[0], p[1], p[2], k.pitch, k.yaw, k.fov
                            ));
                            ui.horizontal(|ui| {
                                if ui.button("Go").clicked() {
                                    action = Some(RowAction::Go(i));
                                }
                                let set = ui.button("Set").on_hover_text("overwrite the pose with the current camera");
                                if set.clicked() {
                                    action = Some(RowAction::Set(i));
                                }
                                if ui.button("Dup").clicked() {
                                    action = Some(RowAction::Duplicate(i));
                                }
                                if ui.button("x").clicked() {
                                    action = Some(RowAction::Remove(i));
                                }
                            });
                            ui.end_row();
                        }
                    });
                });
                match action {
                    Some(RowAction::Select(i)) => seq.selected = Some(i),
                    Some(RowAction::Go(i)) => seq.go_to(i, camera),
                    Some(RowAction::Set(i)) => seq.set_keyframe(i, camera),
                    Some(RowAction::Duplicate(i)) => seq.duplicate(i),
                    Some(RowAction::Remove(i)) => seq.remove(i),
                    None => {}
                }
                ui.horizontal(|ui| {
                    if ui.button("Add at camera").clicked() {
                        seq.add_keyframe(camera);
                    }
                    let selected = seq.selected;
                    if ui.add_enabled(selected.is_some(), egui::Button::new("Up")).clicked() {
                        if let Some(i) = selected {
                            seq.move_up(i);
                        }
                    }
                    if ui.add_enabled(selected.is_some(), egui::Button::new("Down")).clicked() {
                        if let Some(i) = selected {
                            seq.move_down(i);
                        }
                    }
                });
            });
        });

    CollapsingHeader::new("Export").default_open(true).show(ui, |ui| {
        Grid::new("sequencer_export").num_columns(2).show(ui, |ui| {
            ui.label("dir");
            ui.add_enabled(!exporting, egui::TextEdit::singleline(&mut seq.export_dir).desired_width(320.0));
            ui.end_row();
            ui.label("warmup");
            ui.add_enabled(!exporting, DragValue::new(&mut seq.warmup).range(0..=1000).suffix(" frames"));
            ui.end_row();
            ui.label("samples");
            ui.add_enabled(!exporting, DragValue::new(&mut seq.samples).range(1..=4096).suffix(" per frame"))
                .on_hover_text(
                    "renders per exported frame; the last is captured. Reference mode: accumulated samples per frame.",
                );
            ui.end_row();
            ui.label("size");
            let (w, h) = panel.stats.target;
            ui.label(format!("{w} x {h} (the window; --size sets it)"));
            ui.end_row();
            ui.label("frames");
            ui.label(format!("{} ({total:.2} s)", seq.sequence.frame_count(seq.sequence.fps)));
            ui.end_row();
        });
        ui.horizontal(|ui| {
            if ui.add_enabled(!exporting && has_keyframes, egui::Button::new("Export")).clicked() {
                if let Err(e) = seq.start_export() {
                    seq.note = format!("{e:#}");
                }
            }
            if ui.add_enabled(exporting, egui::Button::new("Cancel")).clicked() {
                seq.cancel_export();
            }
        });
        if exporting && !seq.note.is_empty() {
            ui.label(&seq.note);
        }
        if !seq.ffmpeg.is_empty() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&seq.ffmpeg).monospace());
                if ui.button("Copy").clicked() {
                    ui.ctx().copy_text(seq.ffmpeg.clone());
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_angles_round_trip() {
        let dir = direction_from_angles(35.0, -120.0);
        let (pitch, yaw) = direction_angles(dir);
        assert!((pitch - 35.0).abs() < 1e-3 && (yaw + 120.0).abs() < 1e-3);
        // The default lighting direction: up and toward +X +Z.
        let (pitch, yaw) = direction_angles(crate::settings::LIGHTING_SUN_DIR);
        assert!(pitch > 0.0 && yaw > 0.0 && yaw < 90.0);
    }
}
