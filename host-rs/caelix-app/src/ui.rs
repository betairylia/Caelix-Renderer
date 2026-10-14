//! Debug UI: four egui windows (Stats, Render, Camera, Entities) in egui's default style, drawn by
//! `caelix-vk` as an overlay on the swapchain image after the library has written it.
//!
//! egui-winit turns the window events into egui input; this module holds the egui context and
//! builds the windows. They edit the host's state directly (render settings, camera, entity
//! transforms) and report one-shot requests (reset accumulation, screenshot) in [`Panel`].

use caelix_sys::*;
use egui::{CollapsingHeader, ComboBox, DragValue, Grid, ScrollArea, Slider};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::camera::{self, Camera};
use crate::settings::{Mode, RenderSettings, View};
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
    pub nrd: CaelixNrdInfo,
    pub load_state: String,
    /// Reference mode: frames accumulated since the last reset.
    pub accumulated: u64,
    /// The newest screenshot of the "Screenshot" button, or a waiting note.
    pub screenshot_note: String,
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
    /// Output: the "Reset accumulation" button was pressed.
    pub reset_accumulation: bool,
    /// Output: the "Screenshot" button was pressed.
    pub screenshot: bool,
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
}

impl Ui {
    pub fn new(window: &Window, visible: bool, demo: bool) -> Self {
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
        Self { ctx, winit, visible, demo }
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
        let demo = self.demo;
        let output = self.ctx.run_ui(raw_input, |root| debug_windows(root.ctx(), panel, demo));
        self.winit.handle_platform_output(window, output.platform_output);
        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        UiFrame { textures_delta: output.textures_delta, primitives, pixels_per_point: output.pixels_per_point }
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

/// Four separate egui windows: Stats at the top left, Camera at the bottom left (with `--ui-demo`,
/// where Stats is fully open, at the bottom centre), Render at the top right, Entities at the top
/// centre. Each is draggable and collapsible
/// (title bar); egui keeps the positions after the first frame.
fn debug_windows(ctx: &egui::Context, panel: &mut Panel<'_>, demo: bool) {
    let screen = ctx.content_rect();
    let margin = 8.0;
    egui::Window::new("Stats")
        .default_pos([screen.min.x + margin, screen.min.y + margin])
        .show(ctx, |ui| stats_window(ui, panel.stats, demo));
    let (camera_pivot, camera_x) = if demo {
        (egui::Align2::CENTER_BOTTOM, screen.center().x)
    } else {
        (egui::Align2::LEFT_BOTTOM, screen.min.x + margin)
    };
    egui::Window::new("Camera")
        .pivot(camera_pivot)
        .default_pos([camera_x, screen.max.y - margin])
        .show(ctx, |ui| camera_window(ui, panel, demo));
    egui::Window::new("Render")
        .pivot(egui::Align2::RIGHT_TOP)
        .default_pos([screen.max.x - margin, screen.min.y + margin])
        .show(ctx, |ui| render_window(ui, panel, demo));
    egui::Window::new(format!("Entities ({})", panel.entities.len()))
        .id(egui::Id::new("entities_window"))
        .pivot(egui::Align2::CENTER_TOP)
        .default_pos([screen.center().x, screen.min.y + margin])
        .default_open(demo)
        .show(ctx, |ui| entities_window(ui, panel.entities, demo));
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

fn render_window(ui: &mut egui::Ui, panel: &mut Panel<'_>, demo: bool) {
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

            ui.label("exposure");
            ui.add_enabled(lighting, Slider::new(&mut settings.exposure, 0.05..=8.0).logarithmic(true));
            ui.end_row();
        });
        ui.add_enabled(settings.mode == Mode::Pt, egui::Checkbox::new(&mut settings.denoise, "denoise (NRD, pt mode)"));
        ui.add_enabled(lighting, egui::Checkbox::new(&mut settings.sun_nee, "sun NEE"));
        ui.add_enabled(lighting, egui::Checkbox::new(&mut settings.sky_on_miss, "sky on miss"));
        ui.add_enabled(!lighting, egui::Checkbox::new(&mut settings.shading, "Lambert (primary mode)"));
    });

    CollapsingHeader::new("Sun").default_open(true).show(ui, |ui| {
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
        Grid::new("render_sky").num_columns(2).show(ui, |ui| {
            for (name, color) in [
                ("zenith", &mut settings.sky_zenith),
                ("horizon", &mut settings.sky_horizon),
                ("ground", &mut settings.sky_ground),
            ] {
                ui.label(name);
                ui.horizontal(|ui| {
                    ui.add_enabled(lighting, |ui: &mut egui::Ui| ui.color_edit_button_rgb(color));
                    ui.label(format!("{:.3}, {:.3}, {:.3}", color[0], color[1], color[2]));
                });
                ui.end_row();
            }
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

fn entities_window(ui: &mut egui::Ui, entities: &mut [EntityRow], demo: bool) {
    if entities.is_empty() {
        ui.label("no world loaded");
        return;
    }
    ScrollArea::vertical().max_height(300.0).show(ui, |ui| entities_list(ui, entities, demo));
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
