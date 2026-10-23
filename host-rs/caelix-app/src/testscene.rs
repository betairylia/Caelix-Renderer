//! Test scene: a synthetic moving entity and timed brick edits, to judge the motion vectors and
//! NRD under (a) a moving entity and (b) brick edits over time (`--test-entity`, `--test-edits`,
//! the Test window of the debug UI).
//!
//! The entity is a hollow box of 24^3 blocks with 2-block walls. Its entity-local origin is the
//! centre of the box: the box covers blocks -12..=11 on every axis, which is bricks -2..=1 (4 per
//! axis). The 8 inner bricks (keys -1..=0 on every axis) are air and are not uploaded, so the box
//! has 56 bricks. Face colours (packed opaque rgb5): +X red, -X green, +Y blue, -Y yellow,
//! +Z cyan, -Z magenta. A 4x4 hole (x, y in -2..=1) goes through the +Z wall. The 2x2x2 corner
//! cubes of the four top corners (y 10..=11, x and z in {-12..=-11, 10..=11}) are emissive white
//! (packed emissive rgb4 15,15,15, strength 1: emission 8).
//!
//! Animation (every frame, before the flush): `t += min(dt, 0.1)` while not paused;
//! position = centre + orbit_radius * (cos(t * orbit_speed), 0, sin(t * orbit_speed)),
//! rotation = yaw(t * spin_speed) * pitch(t * spin_speed * 0.37) (Unity `Quaternion.Euler`
//! order, see `camera::quat_from_euler`). The pose is written to the entity's row of the
//! Entities window, which `main.rs` sends with `caelix_scene_set_entity_transform`, and only when
//! it differs from the pose sent last: a paused entity gets no call, so the next flush makes the
//! previous transform equal to the current one and its motion vectors are 0.
//!
//! Edits (every `interval` frames, or "Step once"):
//! - wave: columns of one brick at y = -3 (entity-local blocks -24..-17, under the box), keys
//!   x in -4..=4, z in -1..=1 (9 x 3 columns). A column is up while
//!   `(x + z - phase) mod 6 < 3`; each tick advances the phase by 1, removes the columns that
//!   fall (`caelix_scene_remove_bricks`) and uploads the ones that rise
//!   (`caelix_scene_upload_bricks`, solid orange).
//! - flicker: switches the emissive corner cubes off (dark grey) or on by uploading their 4
//!   bricks again (a replace: no brick is added or removed).
//!
//! Uploads go through a queue: when the staging ring is full (for example while a world loads),
//! the rest is uploaded in a later frame. A remove drops a queued upload of the same key.

use std::collections::HashSet;
use std::ffi::c_void;
use std::time::Instant;

use anyhow::bail;
use caelix_sys::*;
use clap::ValueEnum;

use crate::camera::{self, Camera};
use crate::ui::EntityRow;

/// Blocks per brick.
const BRICK_BLOCKS: usize = 512;
/// The box covers blocks -BOX_HALF..BOX_HALF-1 on every axis.
const BOX_HALF: i32 = 12;
/// Wall thickness in blocks.
const WALL: i32 = 2;
/// The hole in the +Z wall covers x, y in -HOLE_HALF..HOLE_HALF-1.
const HOLE_HALF: i32 = 2;
/// Brick keys of the box on every axis.
const BOX_KEYS: std::ops::RangeInclusive<i32> = -2..=1;
/// Wave columns: brick keys x in -4..=4, z in -1..=1, at y = -3.
const WAVE_X: std::ops::RangeInclusive<i32> = -4..=4;
const WAVE_Z: std::ops::RangeInclusive<i32> = -1..=1;
const WAVE_Y: i32 = -3;
/// Wave period in columns; a column is up for half of it.
const WAVE_PERIOD: i32 = 6;
/// Pitch speed of the spin relative to its yaw speed.
const PITCH_RATIO: f32 = 0.37;
/// Largest animation step per frame, in seconds.
const MAX_STEP: f32 = 0.1;
/// Default position: this far in front of the camera along `forward`, this far below the eye.
const AHEAD: f32 = 120.0;
const BELOW: f32 = 10.0;
/// GUID stored with the entity (the library does not interpret it).
const GUID: [u8; 16] = *b"caelix-test-box\0";

/// Packed opaque rgb5.
const fn opaque(r: u16, g: u16, b: u16) -> u16 {
    0x8000 | (r << 10) | (g << 5) | b
}
/// Packed emissive rgb4 with strength 0..3 (emission = albedo * 2^(3 * strength)).
const fn emissive(r: u16, g: u16, b: u16, strength: u16) -> u16 {
    0x4000 | (r << 10) | (g << 6) | (b << 2) | strength
}

/// Face colours in the order +X, -X, +Y, -Y, +Z, -Z.
const FACE_COLORS: [u16; 6] = [
    opaque(31, 0, 0),
    opaque(0, 31, 0),
    opaque(0, 0, 31),
    opaque(31, 31, 0),
    opaque(0, 31, 31),
    opaque(31, 0, 31),
];
const CORNER_ON: u16 = emissive(15, 15, 15, 1);
const CORNER_OFF: u16 = opaque(4, 4, 4);
const WAVE_COLOR: u16 = opaque(31, 16, 0);

/// Edit pattern of `--test-edits` and the Test window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EditPattern {
    /// Brick columns under the box rise and fall in a travelling wave (uploads and removals).
    Wave,
    /// The emissive corner cubes switch off and on (bricks replaced).
    Flicker,
}

impl EditPattern {
    pub const ALL: [EditPattern; 2] = [EditPattern::Wave, EditPattern::Flicker];

    pub fn label(self) -> &'static str {
        match self {
            EditPattern::Wave => "wave",
            EditPattern::Flicker => "flicker",
        }
    }
}

type Brick = Box<[u16; BRICK_BLOCKS]>;

/// Values per one-second window, shown in the Test window and logged.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rates {
    /// `as_builds_total` increase per second.
    pub as_builds_per_s: f64,
    /// `finalized_bricks_total` increase per second.
    pub finalized_per_s: f64,
    /// Mean and maximum `flush_ms` of the timed frames in the window.
    pub flush_ms_mean: f64,
    pub flush_ms_max: f64,
}

/// Accumulates the scene counters over one window.
struct RateWindow {
    start: Instant,
    builds: u64,
    finalized: u64,
    flush_sum: f64,
    flush_max: f64,
    flush_count: u64,
}

impl RateWindow {
    fn new(stats: &CaelixSceneStats) -> Self {
        Self {
            start: Instant::now(),
            builds: stats.as_builds_total,
            finalized: stats.finalized_bricks_total,
            flush_sum: 0.0,
            flush_max: 0.0,
            flush_count: 0,
        }
    }

    fn add_flush(&mut self, ms: f32) {
        self.flush_sum += f64::from(ms);
        self.flush_max = self.flush_max.max(f64::from(ms));
        self.flush_count += 1;
    }

    fn rates(&self, stats: &CaelixSceneStats) -> Rates {
        let seconds = self.start.elapsed().as_secs_f64().max(1e-6);
        Rates {
            as_builds_per_s: (stats.as_builds_total - self.builds) as f64 / seconds,
            finalized_per_s: (stats.finalized_bricks_total - self.finalized) as f64 / seconds,
            flush_ms_mean: if self.flush_count > 0 { self.flush_sum / self.flush_count as f64 } else { 0.0 },
            flush_ms_max: self.flush_max,
        }
    }
}

/// The test entity, its animation and its edits. Always present in the app; the entity exists
/// in the scene only while `enabled`.
pub struct TestScene {
    /// The entity is in the scene (Test window checkbox). Turning it on adds the entity, turning
    /// it off removes it.
    pub enabled: bool,
    /// The animation stands still: no transform call, motion vectors 0.
    pub paused: bool,
    pub orbit_radius: f32,
    /// Degrees per second.
    pub orbit_speed: f32,
    /// Degrees per second (yaw; the pitch turns at 0.37 times this).
    pub spin_speed: f32,
    /// One-shot: back to t = 0.
    pub reset_requested: bool,
    /// `--test-entity-at`: the orbit centre of a new entity; else in front of the camera.
    at: Option<[f32; 3]>,
    /// The scene entity while enabled.
    entity: Option<CaelixEntityId>,
    /// Orbit centre of the current entity.
    centre: [f32; 3],
    /// Animation time in seconds.
    t: f32,
    /// The pose sent last (position, rotation).
    sent: Option<([f32; 3], [f32; 4])>,

    /// Timed edits are on.
    pub edits_enabled: bool,
    /// Frames between two edit ticks.
    pub interval: u32,
    pub pattern: EditPattern,
    /// One-shot: one edit tick now.
    pub step_once: bool,
    frames_since_tick: u32,
    wave_phase: i32,
    /// Wave columns (brick keys) that are up.
    wave_up: HashSet<[i32; 3]>,
    corners_on: bool,
    /// Edit ticks done.
    pub ticks: u64,
    /// Bricks uploaded (accepted by the library) and removed since the entity was created.
    pub uploads: u64,
    pub removes: u64,
    /// Bricks of the entity (box and wave), as uploaded or queued.
    resident: HashSet<[i32; 3]>,
    /// Bricks of the entity that the library accepted and that were not removed since.
    in_scene: HashSet<[i32; 3]>,
    /// Uploads that wait for staging ring space, in order.
    pending: Vec<([i32; 3], Brick)>,

    window: Option<RateWindow>,
    /// The values of the last complete one-second window.
    pub rates: Rates,
    /// `CaelixGpuTimings::frame_number` of the newest flush time counted.
    last_timed_frame: u64,
    /// Run totals since the world was complete (`main.rs` reports it).
    run: Option<RateWindow>,
}

impl TestScene {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        at: Option<[f32; 3]>,
        paused: bool,
        orbit_radius: f32,
        orbit_speed: f32,
        spin_speed: f32,
        edits: bool,
        interval: u32,
        pattern: EditPattern,
    ) -> Self {
        Self {
            enabled: false,
            paused,
            orbit_radius,
            orbit_speed,
            spin_speed,
            reset_requested: false,
            at,
            entity: None,
            centre: [0.0; 3],
            t: 0.0,
            sent: None,
            edits_enabled: edits,
            interval: interval.max(1),
            pattern,
            step_once: false,
            frames_since_tick: 0,
            wave_phase: 0,
            wave_up: HashSet::new(),
            corners_on: true,
            ticks: 0,
            uploads: 0,
            removes: 0,
            resident: HashSet::new(),
            in_scene: HashSet::new(),
            pending: Vec::new(),
            window: None,
            rates: Rates::default(),
            last_timed_frame: 0,
            run: None,
        }
    }

    /// The scene id of the entity while it exists.
    pub fn entity(&self) -> Option<CaelixEntityId> {
        self.entity
    }

    /// Bricks of the entity (uploaded or queued).
    pub fn brick_count(&self) -> usize {
        self.resident.len()
    }

    /// Bricks of the entity that the library holds (accepted uploads minus removals).
    pub fn bricks_in_scene(&self) -> u64 {
        self.in_scene.len() as u64
    }

    /// Uploads that wait for staging ring space.
    pub fn pending_uploads(&self) -> usize {
        self.pending.len()
    }

    pub fn centre(&self) -> [f32; 3] {
        self.centre
    }

    pub fn time(&self) -> f32 {
        self.t
    }

    /// The pose at animation time `t`: position and unit quaternion (x, y, z, w).
    fn pose(&self, t: f32) -> ([f32; 3], [f32; 4]) {
        let orbit = (t * self.orbit_speed).to_radians();
        let position = [
            self.centre[0] + self.orbit_radius * orbit.cos(),
            self.centre[1],
            self.centre[2] + self.orbit_radius * orbit.sin(),
        ];
        let yaw = t * self.spin_speed;
        let rotation = camera::quat_from_euler(yaw * PITCH_RATIO, yaw, 0.0);
        (position, rotation)
    }

    /// Once per frame, before the transforms are sent and before `upload_pending`: adds or
    /// removes the entity when `enabled` changed, advances the animation (`dt` seconds, at most
    /// 0.1), writes a changed pose into the entity's row (`dirty`, sent by the caller) and runs
    /// the edit tick. Returns true when bricks changed this frame (reference accumulation
    /// restarts).
    pub fn update(
        &mut self,
        scene: *mut CaelixScene,
        dt: f32,
        camera: &Camera,
        rows: &mut Vec<EntityRow>,
    ) -> anyhow::Result<bool> {
        // Edits need the entity: turning edits on adds it; turning the entity off also turns the
        // edits off (below).
        if self.edits_enabled && !self.enabled && self.entity.is_none() {
            self.enabled = true;
        }
        match (self.enabled, self.entity) {
            (true, None) => self.create(scene, camera, rows)?,
            (false, Some(_)) => {
                self.edits_enabled = false;
                self.destroy(scene, rows)?;
                return Ok(true);
            }
            _ => {}
        }
        let Some(entity) = self.entity else {
            return Ok(false);
        };

        // Animation.
        if self.reset_requested {
            self.reset_requested = false;
            self.t = 0.0;
        } else if !self.paused {
            self.t += dt.clamp(0.0, MAX_STEP);
        }
        let (position, rotation) = self.pose(self.t);
        if self.sent != Some((position, rotation)) {
            self.sent = Some((position, rotation));
            if let Some(row) = rows.iter_mut().find(|row| row.scene_id == entity) {
                row.position = position;
                row.rotation = rotation;
                row.euler = camera::euler_from_quat(rotation);
                row.dirty = true;
            }
        }

        // Edits.
        let mut tick = std::mem::take(&mut self.step_once);
        if self.edits_enabled {
            self.frames_since_tick += 1;
            if self.frames_since_tick >= self.interval.max(1) {
                tick = true;
            }
        }
        if tick {
            self.frames_since_tick = 0;
            self.edit_tick(scene, entity)?;
        }
        if let Some(row) = rows.iter_mut().find(|row| row.scene_id == entity) {
            row.bricks = self.resident.len() as u64;
        }
        Ok(tick)
    }

    /// Adds the entity at `--test-entity-at` or in front of the camera and queues its bricks.
    fn create(&mut self, scene: *mut CaelixScene, camera: &Camera, rows: &mut Vec<EntityRow>) -> anyhow::Result<()> {
        self.centre = self.at.unwrap_or_else(|| {
            let (_, _, forward) = camera.basis();
            let p = camera.position;
            [p[0] + forward[0] * AHEAD, p[1] + forward[1] * AHEAD - BELOW, p[2] + forward[2] * AHEAD]
        });
        self.t = 0.0;
        let (position, rotation) = self.pose(0.0);
        let desc = CaelixEntityDesc { guid: GUID, position, rotation };
        let mut id: CaelixEntityId = CAELIX_INVALID_ENTITY;
        // SAFETY: the scene is live; desc and id are valid for the call.
        let result = unsafe { caelix_scene_add_entity(scene, &desc, &mut id) };
        if result != CaelixResult_CAELIX_OK {
            bail!("caelix_scene_add_entity (test entity) failed: CaelixResult {result}");
        }
        // It moves every frame: its lights belong in the dynamic light BVH.
        // SAFETY: the scene is live and the entity was just added.
        let result = unsafe { caelix_scene_set_entity_motion(scene, id, CAELIX_ENTITY_DYNAMIC) };
        if result != CaelixResult_CAELIX_OK {
            bail!("caelix_scene_set_entity_motion (test entity) failed: CaelixResult {result}");
        }
        self.entity = Some(id);
        self.sent = Some((position, rotation));
        self.uploads = 0;
        self.removes = 0;
        self.ticks = 0;
        self.frames_since_tick = 0;
        self.wave_phase = 0;
        self.wave_up.clear();
        self.corners_on = true;
        self.resident.clear();
        self.in_scene.clear();
        self.pending.clear();
        for z in BOX_KEYS {
            for y in BOX_KEYS {
                for x in BOX_KEYS {
                    let key = [x, y, z];
                    let brick = box_brick(key, self.corners_on);
                    if brick.iter().any(|&v| v != 0) {
                        self.queue_upload(key, brick);
                    }
                }
            }
        }
        log::info!(
            "test entity: scene id {id}, orbit centre {:?}, {} bricks, orbit radius {} at {} deg/s, spin {} deg/s{}",
            self.centre,
            self.pending.len(),
            self.orbit_radius,
            self.orbit_speed,
            self.spin_speed,
            if self.paused { ", paused" } else { "" }
        );
        rows.push(EntityRow::synthetic(id, "test-box (synthetic)", position, rotation, self.pending.len() as u64));
        Ok(())
    }

    fn destroy(&mut self, scene: *mut CaelixScene, rows: &mut Vec<EntityRow>) -> anyhow::Result<()> {
        let Some(id) = self.entity.take() else {
            return Ok(());
        };
        // SAFETY: the scene is live and the entity exists.
        let result = unsafe { caelix_scene_remove_entity(scene, id) };
        if result != CaelixResult_CAELIX_OK {
            bail!("caelix_scene_remove_entity({id}) (test entity) failed: CaelixResult {result}");
        }
        rows.retain(|row| row.scene_id != id);
        self.pending.clear();
        self.resident.clear();
        self.in_scene.clear();
        self.wave_up.clear();
        self.sent = None;
        log::info!("test entity removed (scene id {id})");
        Ok(())
    }

    /// One edit of the current pattern.
    fn edit_tick(&mut self, scene: *mut CaelixScene, entity: CaelixEntityId) -> anyhow::Result<()> {
        self.ticks += 1;
        match self.pattern {
            EditPattern::Wave => {
                self.wave_phase += 1;
                let up: HashSet<[i32; 3]> = wave_columns(self.wave_phase).collect();
                let falling: Vec<[i32; 3]> = self.wave_up.difference(&up).copied().collect();
                let rising: Vec<[i32; 3]> = up.difference(&self.wave_up).copied().collect();
                self.remove(scene, entity, &falling)?;
                for key in rising {
                    self.queue_upload(key, Box::new([WAVE_COLOR; BRICK_BLOCKS]));
                }
                self.wave_up = up;
            }
            EditPattern::Flicker => {
                self.corners_on = !self.corners_on;
                for x in [-2, 1] {
                    for z in [-2, 1] {
                        let key = [x, 1, z];
                        self.queue_upload(key, box_brick(key, self.corners_on));
                    }
                }
            }
        }
        Ok(())
    }

    fn queue_upload(&mut self, key: [i32; 3], brick: Brick) {
        self.pending.retain(|(k, _)| *k != key);
        self.pending.push((key, brick));
        self.resident.insert(key);
    }

    fn remove(&mut self, scene: *mut CaelixScene, entity: CaelixEntityId, keys: &[[i32; 3]]) -> anyhow::Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        self.pending.retain(|(k, _)| !keys.contains(k));
        let keys: Vec<CaelixBrickKey> = keys.iter().map(|k| CaelixBrickKey { x: k[0], y: k[1], z: k[2] }).collect();
        let count = u32::try_from(keys.len())?;
        // SAFETY: the scene is live; keys holds `count` keys for the call.
        let result = unsafe { caelix_scene_remove_bricks(scene, entity, count, keys.as_ptr()) };
        if result != CaelixResult_CAELIX_OK {
            bail!("caelix_scene_remove_bricks (test entity, {count} bricks) failed: CaelixResult {result}");
        }
        for key in &keys {
            self.resident.remove(&[key.x, key.y, key.z]);
            self.in_scene.remove(&[key.x, key.y, key.z]);
        }
        self.removes += u64::from(count);
        Ok(())
    }

    /// Uploads queued bricks until the staging ring is full. Call after `caelix_scene_recycle`
    /// and before the frame's flush.
    pub fn upload_pending(&mut self, scene: *mut CaelixScene) -> anyhow::Result<()> {
        let Some(entity) = self.entity else {
            return Ok(());
        };
        if self.pending.is_empty() {
            return Ok(());
        }
        let keys: Vec<CaelixBrickKey> =
            self.pending.iter().map(|(k, _)| CaelixBrickKey { x: k[0], y: k[1], z: k[2] }).collect();
        let mut blocks = Vec::with_capacity(self.pending.len() * BRICK_BLOCKS);
        for (_, brick) in &self.pending {
            blocks.extend_from_slice(&brick[..]);
        }
        let count = u32::try_from(keys.len())?;
        let mut consumed = 0u32;
        // SAFETY: the scene is live; keys holds `count` keys and blocks count * 512 u16 values,
        // both alive for the call, which only borrows them.
        let result = unsafe {
            let blocks = blocks.as_ptr().cast::<c_void>();
            caelix_scene_upload_bricks(scene, entity, count, keys.as_ptr(), blocks, &mut consumed)
        };
        if result != CaelixResult_CAELIX_OK {
            bail!("caelix_scene_upload_bricks (test entity, {count} bricks) failed: CaelixResult {result}");
        }
        for (key, _) in self.pending.drain(..consumed as usize) {
            self.in_scene.insert(key);
        }
        self.uploads += u64::from(consumed);
        Ok(())
    }

    /// Once per frame after the flush: updates the one-second rates and, once the world is
    /// complete, the run totals.
    pub fn after_flush(&mut self, stats: &CaelixSceneStats, timings: &CaelixGpuTimings, world_ready: bool) {
        let window = self.window.get_or_insert_with(|| RateWindow::new(stats));
        if world_ready && self.run.is_none() {
            self.run = Some(RateWindow::new(stats));
        }
        if timings.frame_number != 0 && timings.frame_number != self.last_timed_frame {
            self.last_timed_frame = timings.frame_number;
            window.add_flush(timings.flush_ms);
            if let Some(run) = self.run.as_mut() {
                run.add_flush(timings.flush_ms);
            }
        }
        if window.start.elapsed().as_secs_f64() >= 1.0 {
            self.rates = window.rates(stats);
            self.window = Some(RateWindow::new(stats));
        }
    }

    /// The rates since the world was complete (`None` before), and that time in seconds.
    pub fn run_rates(&self, stats: &CaelixSceneStats) -> Option<(Rates, f64)> {
        self.run.as_ref().map(|run| (run.rates(stats), run.start.elapsed().as_secs_f64()))
    }

    /// One line for the log.
    pub fn summary(&self, rates: &Rates, stats: &CaelixSceneStats) -> String {
        format!(
            "test scene: entity {}, t {:.2} s{}, edits {} ({}, every {} frames, {} ticks, {} uploads, {} removes, \
             {} queued), AS builds {:.1}/s, {} pending, finalized {:.1} bricks/s, flush mean {:.3} ms max {:.3} ms",
            self.entity.map_or_else(|| "none".into(), |id| id.to_string()),
            self.t,
            if self.paused { " (paused)" } else { "" },
            if self.edits_enabled { "on" } else { "off" },
            self.pattern.label(),
            self.interval,
            self.ticks,
            self.uploads,
            self.removes,
            self.pending.len(),
            rates.as_builds_per_s,
            stats.as_builds_pending,
            rates.finalized_per_s,
            rates.flush_ms_mean,
            rates.flush_ms_max
        )
    }
}

/// The wave columns that are up at `phase`.
fn wave_columns(phase: i32) -> impl Iterator<Item = [i32; 3]> {
    WAVE_Z.flat_map(move |z| {
        WAVE_X.filter_map(move |x| {
            ((x + z - phase).rem_euclid(WAVE_PERIOD) < WAVE_PERIOD / 2).then_some([x, WAVE_Y, z])
        })
    })
}

/// The block of the box at entity-local block `p` (0 = air).
fn box_block(p: [i32; 3], corners_on: bool) -> u16 {
    let inside = |c: i32| (-BOX_HALF..BOX_HALF).contains(&c);
    let wall = |c: i32| !(-BOX_HALF + WALL..BOX_HALF - WALL).contains(&c);
    if !p.iter().all(|&c| inside(c)) || !p.iter().any(|&c| wall(c)) {
        return 0;
    }
    // The hole in the +Z wall.
    if p[2] >= BOX_HALF - WALL && (-HOLE_HALF..HOLE_HALF).contains(&p[0]) && (-HOLE_HALF..HOLE_HALF).contains(&p[1]) {
        return 0;
    }
    // The 2x2x2 cubes of the four top corners.
    if p[1] >= BOX_HALF - WALL && wall(p[0]) && wall(p[2]) {
        return if corners_on { CORNER_ON } else { CORNER_OFF };
    }
    // The face: the axis on which the block centre is farthest from the box centre (ties go to
    // the first axis).
    let mut axis = 0;
    for a in 1..3 {
        if (p[a] as f32 + 0.5).abs() > (p[axis] as f32 + 0.5).abs() {
            axis = a;
        }
    }
    let negative = p[axis] < 0;
    FACE_COLORS[axis * 2 + usize::from(negative)]
}

/// The 512 blocks of the box brick `key` (voxel index x + 8 y + 64 z).
fn box_brick(key: [i32; 3], corners_on: bool) -> Brick {
    let mut brick = Box::new([0u16; BRICK_BLOCKS]);
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                let p = [key[0] * 8 + x, key[1] * 8 + y, key[2] * 8 + z];
                brick[(x + 8 * y + 64 * z) as usize] = box_block(p, corners_on);
            }
        }
    }
    brick
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_layout() {
        let mut bricks = 0;
        let mut blocks = 0usize;
        for z in BOX_KEYS {
            for y in BOX_KEYS {
                for x in BOX_KEYS {
                    let brick = box_brick([x, y, z], true);
                    let n = brick.iter().filter(|&&v| v != 0).count();
                    if n > 0 {
                        bricks += 1;
                    }
                    blocks += n;
                }
            }
        }
        // 4^3 bricks minus the 8 inner air bricks.
        assert_eq!(bricks, 56);
        // 24^3 - 20^3 shell minus the 4x4x2 hole.
        assert_eq!(blocks, 24 * 24 * 24 - 20 * 20 * 20 - 32);
        // Faces, hole and corners.
        assert_eq!(box_block([11, 0, 0], true), FACE_COLORS[0]);
        assert_eq!(box_block([-12, 0, 0], true), FACE_COLORS[1]);
        assert_eq!(box_block([0, 11, 0], true), FACE_COLORS[2]);
        assert_eq!(box_block([0, -12, 0], true), FACE_COLORS[3]);
        assert_eq!(box_block([5, 5, 10], true), FACE_COLORS[4]);
        assert_eq!(box_block([0, 0, -11], true), FACE_COLORS[5]);
        assert_eq!(box_block([0, 0, 11], true), 0);
        assert_eq!(box_block([0, 0, 0], true), 0);
        assert_eq!(box_block([-12, 11, 11], true), CORNER_ON);
        assert_eq!(box_block([10, 10, -11], false), CORNER_OFF);
        assert_eq!(box_block([10, -12, -11], true), FACE_COLORS[3]);
    }

    #[test]
    fn wave_moves() {
        let a: HashSet<_> = wave_columns(0).collect();
        let b: HashSet<_> = wave_columns(1).collect();
        assert!(!a.is_empty() && a.len() < 27);
        assert!(a != b);
        // A full period brings the wave back.
        let c: HashSet<_> = wave_columns(WAVE_PERIOD).collect();
        assert_eq!(a, c);
    }
}
