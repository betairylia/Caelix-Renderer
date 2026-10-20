//! Camera sequencer: keyframed camera tours, previewed in real time and exported as PNG frames.
//!
//! A [`Sequence`] is a list of [`Keyframe`]s (pose, field of view, hold time, travel time) that a
//! JSON file stores. The timeline is the holds and the travels in list order: keyframe i rests for
//! `hold` seconds, then travels `duration` seconds to keyframe i + 1 (to keyframe 0 when the
//! sequence loops). [`Sequence::sample`] interpolates the 6-vector (position, pitch, yaw, fov):
//! smooth = a Catmull-Rom spline with non-uniform knots (velocity is continuous at a keyframe; the
//! camera is at rest at a hold and at the ends of a sequence that does not loop), or linear. Yaw is
//! always taken on the shortest arc.
//!
//! [`Sequencer`] holds the sequence, the UI fields, the playback state and the export: the draw
//! loop calls [`Sequencer::advance`] once per frame, which writes the camera and names the frame
//! to capture during an export; [`Sequencer::frame_captured`] hands the pixels to a writer thread.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{bail, Context as _};
use serde::{Deserialize, Serialize};

use crate::camera::{wrap_degrees, Camera};

/// Version of the JSON format that this host writes; a newer file is rejected.
pub const FORMAT_VERSION: u32 = 1;
/// Travel time of a new keyframe when there is none to copy from.
pub const DEFAULT_DURATION: f32 = 3.0;
/// A travel segment is never shorter than this (keeps the spline parametrization finite).
pub const MIN_DURATION: f32 = 0.01;
/// Export frame rate of a new sequence.
pub const DEFAULT_FPS: u32 = 60;

/// Position, pitch, yaw and fov: the values that the spline interpolates.
type Vec6 = [f32; 6];

/// How the camera travels between two keyframes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Interpolation {
    /// Catmull-Rom spline with non-uniform knots.
    Smooth,
    /// Straight lines, constant speed per segment.
    Linear,
}

impl Interpolation {
    pub const ALL: [Interpolation; 2] = [Interpolation::Smooth, Interpolation::Linear];

    pub fn label(self) -> &'static str {
        match self {
            Interpolation::Smooth => "smooth",
            Interpolation::Linear => "linear",
        }
    }
}

/// One camera pose of a sequence and its times.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    pub position: [f32; 3],
    /// Unity eulerAngles.x, degrees, positive looks down.
    pub pitch: f32,
    /// Unity eulerAngles.y, degrees.
    pub yaw: f32,
    /// Vertical field of view, degrees.
    pub fov: f32,
    /// Seconds the camera stays at this pose before it travels on.
    #[serde(default)]
    pub hold: f32,
    /// Seconds of travel to the next keyframe (to the first one when the sequence loops;
    /// unused for the last keyframe otherwise).
    pub duration: f32,
}

impl Keyframe {
    /// The camera's pose with the given times.
    pub fn from_camera(camera: &Camera, hold: f32, duration: f32) -> Self {
        Self {
            position: camera.position,
            pitch: camera.pitch,
            yaw: camera.yaw,
            fov: camera.fov_y,
            hold,
            duration,
        }
    }

    /// The camera at this keyframe.
    pub fn camera(&self) -> Camera {
        Camera::from_pose(self.position, self.pitch, self.yaw, self.fov)
    }
}

/// A camera tour: the JSON file of the sequencer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sequence {
    pub version: u32,
    pub name: String,
    pub interpolation: Interpolation,
    #[serde(rename = "loop")]
    pub looped: bool,
    /// Frame rate of the export.
    pub fps: u32,
    pub keyframes: Vec<Keyframe>,
}

impl Sequence {
    /// An empty sequence: smooth, not looped, 60 fps.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            version: FORMAT_VERSION,
            name: name.into(),
            interpolation: Interpolation::Smooth,
            looped: false,
            fps: DEFAULT_FPS,
            keyframes: Vec::new(),
        }
    }

    /// Reads a JSON sequence. A newer format version is an error; durations below
    /// [`MIN_DURATION`], negative holds and an fps of 0 are clamped.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
        let mut sequence: Sequence =
            serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))?;
        if sequence.version > FORMAT_VERSION {
            bail!(
                "{}: sequence format version {} is newer than this host's {FORMAT_VERSION}",
                path.display(),
                sequence.version
            );
        }
        for k in &mut sequence.keyframes {
            k.duration = k.duration.max(MIN_DURATION);
            k.hold = k.hold.max(0.0);
        }
        sequence.fps = sequence.fps.max(1);
        Ok(sequence)
    }

    /// Writes the sequence as pretty-printed JSON, creating the parent directory.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(self).context("cannot serialize the sequence")?;
        std::fs::write(path, text).with_context(|| format!("cannot write {}", path.display()))
    }

    /// Start time of keyframe `index` (the start of its hold).
    pub fn keyframe_time(&self, index: usize) -> f32 {
        // The same additions in the same order as `sample`, so that `sample(keyframe_time(i))` lands
        // exactly on keyframe i.
        let mut t = 0.0;
        for k in &self.keyframes[..index.min(self.keyframes.len())] {
            t += k.hold;
            t += k.duration.max(MIN_DURATION);
        }
        t
    }

    /// Length of the sequence: every hold and every travel; the last keyframe's travel (back to
    /// the first) only when looping. 0 without keyframes.
    pub fn total(&self) -> f32 {
        let n = self.keyframes.len();
        let mut t = 0.0;
        for (i, k) in self.keyframes.iter().enumerate() {
            t += k.hold;
            if i + 1 < n || self.looped {
                t += k.duration.max(MIN_DURATION);
            }
        }
        t
    }

    /// Number of frames of an export at `fps`: every frame time in [0, total], or [0, total) when
    /// looping (so the video loops without a repeated frame).
    pub fn frame_count(&self, fps: u32) -> u64 {
        let frames = f64::from(self.total()) * f64::from(fps);
        if self.looped {
            frames.round().max(1.0) as u64
        } else {
            frames.floor() as u64 + 1
        }
    }

    /// The camera at time `t` (seconds). `None` without keyframes. `t` wraps when the sequence
    /// loops, otherwise it is clamped to [0, total].
    pub fn sample(&self, t: f32) -> Option<Camera> {
        let n = self.keyframes.len();
        if n == 0 {
            return None;
        }
        let total = self.total();
        let t = if self.looped && total > 0.0 { t.rem_euclid(total) } else { t.clamp(0.0, total) };
        let mut start = 0.0;
        for i in 0..n {
            let k = &self.keyframes[i];
            if t < start + k.hold || (i + 1 == n && !self.looped) {
                return Some(k.camera());
            }
            start += k.hold;
            let d = k.duration.max(MIN_DURATION);
            if t < start + d || i + 1 == n {
                let u = ((t - start) / d).clamp(0.0, 1.0);
                let j = (i + 1) % n;
                let p0 = vec6(k);
                let p1 = vec6_near(&self.keyframes[j], k.yaw);
                let v = match self.interpolation {
                    Interpolation::Linear => lerp(p0, p1, u),
                    Interpolation::Smooth => {
                        // Tangents are per second; the Hermite form wants them per unit of u.
                        let m0 = self.tangent(i).map(|m| m * d);
                        let m1 = self.tangent(j).map(|m| m * d);
                        hermite(p0, m0, p1, m1, u)
                    }
                };
                return Some(Camera::from_pose([v[0], v[1], v[2]], v[3], v[4], v[5]));
            }
            start += d;
        }
        // Float safety net.
        Some(self.keyframes[n - 1].camera())
    }

    /// Velocity of the 6-vector at keyframe `i`, per second. Zero (the camera is at rest) at a
    /// keyframe with a hold, and at the first and the last keyframe of a sequence that does not
    /// loop. Elsewhere the finite difference of the two neighbours over their travel times
    /// (a Catmull-Rom spline with non-uniform knots: velocity is continuous at the keyframe).
    fn tangent(&self, i: usize) -> Vec6 {
        let n = self.keyframes.len();
        let k = &self.keyframes[i];
        if k.hold > 0.0 || n < 2 || (!self.looped && (i == 0 || i + 1 == n)) {
            return [0.0; 6];
        }
        let prev = &self.keyframes[(i + n - 1) % n];
        let next = &self.keyframes[(i + 1) % n];
        let a = vec6_near(prev, k.yaw);
        let b = vec6_near(next, k.yaw);
        let span = prev.duration.max(MIN_DURATION) + k.duration.max(MIN_DURATION);
        std::array::from_fn(|c| (b[c] - a[c]) / span)
    }
}

/// `[x, y, z, pitch, yaw, fov]` of a keyframe.
fn vec6(k: &Keyframe) -> Vec6 {
    [k.position[0], k.position[1], k.position[2], k.pitch, k.yaw, k.fov]
}

/// [`vec6`] with the yaw on the shortest arc from `base_yaw`.
fn vec6_near(k: &Keyframe, base_yaw: f32) -> Vec6 {
    let mut v = vec6(k);
    v[4] = base_yaw + wrap_degrees(k.yaw - base_yaw);
    v
}

fn lerp(a: Vec6, b: Vec6, u: f32) -> Vec6 {
    std::array::from_fn(|c| a[c] + (b[c] - a[c]) * u)
}

/// Cubic Hermite curve from `p0` (tangent `m0`) to `p1` (tangent `m1`), tangents per unit of `u`.
fn hermite(p0: Vec6, m0: Vec6, p1: Vec6, m1: Vec6, u: f32) -> Vec6 {
    let (u2, u3) = (u * u, u * u * u);
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    std::array::from_fn(|c| h00 * p0[c] + h10 * m0[c] + h01 * p1[c] + h11 * m1[c])
}

/// What the sequencer does with the camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Playback {
    /// The free camera moves; the timeline stays.
    Stopped,
    /// The timeline advances in real time and drives the camera.
    Playing,
    /// The timeline advances one frame per capture and drives the camera.
    Exporting,
}

/// One PNG for the writer thread: path, RGBA8 pixels, width, height.
type PngJob = (PathBuf, Vec<u8>, u32, u32);

/// A running export: the frame counters and the PNG writer thread.
struct Export {
    /// Next frame index to capture.
    frame: u64,
    count: u64,
    warmup_left: u32,
    samples_left: u32,
    /// Start of the export time: the last frame before the world was complete.
    started: Instant,
    /// `None` once the export finishes (dropping it ends the writer's loop).
    sender: Option<SyncSender<PngJob>>,
    writer: Option<JoinHandle<()>>,
    /// The writer's first error.
    error: Arc<Mutex<Option<String>>>,
    dir: PathBuf,
    fps: u32,
}

impl Export {
    /// The writer's first error, if any.
    fn writer_error(&self) -> Option<String> {
        self.error.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Drops the sender and waits for the writer to write the queued frames.
    fn join(&mut self) {
        self.sender = None;
        if let Some(writer) = self.writer.take() {
            if writer.join().is_err() {
                let mut error = self.error.lock().unwrap_or_else(|e| e.into_inner());
                error.get_or_insert_with(|| "the PNG writer thread panicked".into());
            }
        }
    }
}

impl Drop for Export {
    fn drop(&mut self) {
        self.join();
    }
}

/// The sequence, the Sequencer window's fields, the playback and the export.
pub struct Sequencer {
    pub sequence: Sequence,
    /// Path of the JSON file (the UI text field; Load reads it, Save writes it).
    pub file: String,
    /// Directory of the exported frames (UI text field).
    pub export_dir: String,
    /// Export: frames rendered at the first frame's pose before the first capture (NRD / upscaler
    /// history).
    pub warmup: u32,
    /// Export: renders per exported frame; the last one is captured. In reference mode they
    /// accumulate (clean offline frames); in pt mode NRD just sees repeated frames.
    pub samples: u32,
    /// Current time on the timeline, seconds.
    pub time: f32,
    pub playback: Playback,
    pub selected: Option<usize>,
    /// Status line of the Sequencer window (load/save result, export progress or result).
    pub note: String,
    /// After an export: the ffmpeg command that turns the frames into an mp4.
    pub ffmpeg: String,
    /// Exit the host when the export finishes (--sequence-export).
    pub exit_after_export: bool,
    export: Option<Export>,
    /// Why the last export failed; taken once by [`Sequencer::export_failed`].
    failure: Option<String>,
}

impl Sequencer {
    pub fn new(sequence: Sequence, file: String, export_dir: String, warmup: u32, samples: u32) -> Self {
        Self {
            sequence,
            file,
            export_dir,
            warmup,
            samples,
            time: 0.0,
            playback: Playback::Stopped,
            selected: None,
            note: String::new(),
            ffmpeg: String::new(),
            exit_after_export: false,
            export: None,
            failure: None,
        }
    }

    // ---- Editing ----

    /// Adds a keyframe at the camera after the selected one (else at the end) and selects it.
    /// Its travel time is the selected keyframe's, else the last one's, else
    /// [`DEFAULT_DURATION`]. Returns its index.
    pub fn add_keyframe(&mut self, camera: &Camera) -> usize {
        let keys = &self.sequence.keyframes;
        let selected = self.selected.filter(|&i| i < keys.len());
        let duration = selected.or(keys.len().checked_sub(1)).map_or(DEFAULT_DURATION, |i| keys[i].duration);
        let index = selected.map_or(keys.len(), |i| i + 1);
        self.sequence.keyframes.insert(index, Keyframe::from_camera(camera, 0.0, duration));
        self.selected = Some(index);
        log::info!("sequencer: keyframe #{index} added at {}", camera.pose_args());
        index
    }

    /// Overwrites the pose of keyframe `i` with the camera; its hold and travel stay.
    pub fn set_keyframe(&mut self, i: usize, camera: &Camera) {
        if let Some(k) = self.sequence.keyframes.get_mut(i) {
            *k = Keyframe::from_camera(camera, k.hold, k.duration);
        }
    }

    /// Inserts a copy of keyframe `i` after it and selects the copy.
    pub fn duplicate(&mut self, i: usize) {
        if let Some(k) = self.sequence.keyframes.get(i).cloned() {
            self.sequence.keyframes.insert(i + 1, k);
            self.selected = Some(i + 1);
        }
    }

    /// Removes keyframe `i`; the selection follows the keyframe it pointed at.
    pub fn remove(&mut self, i: usize) {
        if i >= self.sequence.keyframes.len() {
            return;
        }
        self.sequence.keyframes.remove(i);
        self.selected = match self.selected {
            Some(s) if s == i => None,
            Some(s) if s > i => Some(s - 1),
            other => other,
        };
        if self.sequence.keyframes.is_empty() && self.playback == Playback::Playing {
            self.playback = Playback::Stopped;
        }
    }

    /// Swaps keyframe `i` with the one before it; the selection follows.
    pub fn move_up(&mut self, i: usize) {
        if i > 0 && i < self.sequence.keyframes.len() {
            self.swap(i, i - 1);
        }
    }

    /// Swaps keyframe `i` with the one after it; the selection follows.
    pub fn move_down(&mut self, i: usize) {
        if i + 1 < self.sequence.keyframes.len() {
            self.swap(i, i + 1);
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.sequence.keyframes.swap(a, b);
        self.selected = match self.selected {
            Some(s) if s == a => Some(b),
            Some(s) if s == b => Some(a),
            other => other,
        };
    }

    /// Puts the camera at keyframe `i`, moves the time to its start and selects it. Stops a
    /// playback; does nothing while exporting.
    pub fn go_to(&mut self, i: usize, camera: &mut Camera) {
        if self.is_exporting() {
            return;
        }
        let Some(k) = self.sequence.keyframes.get(i) else {
            return;
        };
        *camera = k.camera();
        self.time = self.sequence.keyframe_time(i);
        self.selected = Some(i);
        self.playback = Playback::Stopped;
    }

    // ---- Transport ----

    /// Starts the playback (from 0 at the end of a sequence that does not loop). Does nothing
    /// without keyframes or while exporting.
    pub fn play(&mut self) {
        if self.sequence.keyframes.is_empty() || self.is_exporting() {
            return;
        }
        if !self.sequence.looped && self.time >= self.sequence.total() {
            self.time = 0.0;
        }
        self.playback = Playback::Playing;
    }

    /// Stops the playback; the time stays.
    pub fn pause(&mut self) {
        if self.playback == Playback::Playing {
            self.playback = Playback::Stopped;
        }
    }

    pub fn toggle_play(&mut self) {
        match self.playback {
            Playback::Playing => self.pause(),
            Playback::Stopped => self.play(),
            Playback::Exporting => {}
        }
    }

    /// Stops the playback and moves the time to 0. Does nothing while exporting.
    pub fn stop(&mut self) {
        if self.is_exporting() {
            return;
        }
        self.playback = Playback::Stopped;
        self.time = 0.0;
    }

    /// Moves the time to `t` and the camera to the sequence at `t`. A playback continues from
    /// there. Does nothing while exporting.
    pub fn seek(&mut self, t: f32, camera: &mut Camera) {
        if self.is_exporting() {
            return;
        }
        self.time = t.clamp(0.0, self.sequence.total());
        if let Some(c) = self.sequence.sample(self.time) {
            *camera = c;
        }
    }

    /// Manual camera input took over: a playback stops, the time stays.
    pub fn stop_if_playing(&mut self) {
        self.pause();
    }

    // ---- File ----

    /// Reads `file` into the sequence (stops the playback, clears the selection, time 0). The
    /// note tells the result. Does nothing while exporting.
    pub fn load(&mut self) -> anyhow::Result<()> {
        if self.is_exporting() {
            bail!("cannot load while exporting");
        }
        match Sequence::load(Path::new(&self.file)) {
            Ok(sequence) => {
                self.sequence = sequence;
                self.playback = Playback::Stopped;
                self.selected = None;
                self.time = 0.0;
                self.note = format!("loaded {} keyframes from {}", self.sequence.keyframes.len(), self.file);
                log::info!("sequencer: {}", self.note);
                Ok(())
            }
            Err(e) => {
                self.note = format!("{e:#}");
                log::warn!("sequencer: {}", self.note);
                Err(e)
            }
        }
    }

    /// Writes the sequence to `file`. The note tells the result.
    pub fn save(&mut self) -> anyhow::Result<()> {
        match self.sequence.save(Path::new(&self.file)) {
            Ok(()) => {
                self.note = format!("saved {} keyframes to {}", self.sequence.keyframes.len(), self.file);
                log::info!("sequencer: {}", self.note);
                Ok(())
            }
            Err(e) => {
                self.note = format!("{e:#}");
                log::warn!("sequencer: {}", self.note);
                Err(e)
            }
        }
    }

    /// Replaces the sequence with an empty one named after the file stem. Does nothing while
    /// exporting.
    pub fn new_sequence(&mut self) {
        if self.is_exporting() {
            return;
        }
        let name = Path::new(&self.file).file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        self.sequence = Sequence::new(name);
        self.playback = Playback::Stopped;
        self.selected = None;
        self.time = 0.0;
        self.note = "new sequence".into();
    }

    // ---- Export ----

    /// Starts exporting the sequence into `export_dir` at the sequence's fps. The frames are
    /// captured by the draw loop through [`Sequencer::advance`] and [`Sequencer::frame_captured`].
    pub fn start_export(&mut self) -> anyhow::Result<()> {
        if self.is_exporting() {
            bail!("an export is running");
        }
        if self.sequence.keyframes.is_empty() {
            bail!("export: the sequence has no keyframes");
        }
        let fps = self.sequence.fps;
        if fps == 0 {
            bail!("export: fps is 0");
        }
        let dir = PathBuf::from(&self.export_dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("export: cannot create {}", dir.display()))?;
        let count = self.sequence.frame_count(fps);

        let (sender, receiver) = mpsc::sync_channel::<PngJob>(8);
        let error = Arc::new(Mutex::new(None));
        let writer_error = Arc::clone(&error);
        let writer = std::thread::Builder::new()
            .name("sequence-png".into())
            .spawn(move || {
                for (path, pixels, width, height) in receiver {
                    if let Err(e) = crate::write_png(&path, &pixels, width, height) {
                        let mut slot = writer_error.lock().unwrap_or_else(|e| e.into_inner());
                        slot.get_or_insert_with(|| format!("{e:#}"));
                    }
                }
            })
            .context("export: cannot start the PNG writer thread")?;

        self.export = Some(Export {
            frame: 0,
            count,
            warmup_left: self.warmup,
            samples_left: self.samples.max(1),
            started: Instant::now(),
            sender: Some(sender),
            writer: Some(writer),
            error,
            dir,
            fps,
        });
        self.time = 0.0;
        self.playback = Playback::Exporting;
        self.ffmpeg.clear();
        self.failure = None;
        self.note = format!("exporting {count} frames at {fps} fps to {}", self.export_dir);
        log::info!(
            "sequencer: {} (warmup {} frames, {} samples per frame)",
            self.note,
            self.warmup,
            self.samples.max(1)
        );
        Ok(())
    }

    /// Stops the export; the frames written so far stay.
    pub fn cancel_export(&mut self) {
        if let Some(export) = self.export.take() {
            let frames = export.frame;
            // Dropping the export joins the writer after the sender is dropped.
            drop(export);
            self.playback = Playback::Stopped;
            self.note = format!("export cancelled after {frames} frames");
            log::info!("sequencer: {}", self.note);
        }
    }

    pub fn is_exporting(&self) -> bool {
        self.export.is_some()
    }

    /// Why the last export failed, once (the draw loop turns it into an error with
    /// --sequence-export).
    pub fn export_failed(&mut self) -> Option<String> {
        self.failure.take()
    }

    /// Ends the export with an error in the note.
    fn abort_export(&mut self, error: String) {
        self.export = None;
        self.playback = Playback::Stopped;
        self.note = format!("export failed: {error}");
        log::error!("sequencer: {}", self.note);
        self.failure = Some(error);
    }

    // ---- Per frame ----

    /// Advances the timeline by `dt` seconds while playing, or by one export step, and writes
    /// the camera. Returns the PNG path of the frame to capture, when this frame is an export
    /// capture.
    pub fn advance(&mut self, dt: f32, world_ready: bool, camera: &mut Camera) -> Option<PathBuf> {
        match self.playback {
            Playback::Stopped => None,
            Playback::Playing => {
                let total = self.sequence.total();
                self.time += dt;
                if self.time >= total {
                    if self.sequence.looped && total > 0.0 {
                        self.time = self.time.rem_euclid(total);
                    } else {
                        self.time = total;
                        self.playback = Playback::Stopped;
                    }
                }
                match self.sequence.sample(self.time) {
                    Some(c) => *camera = c,
                    None => self.playback = Playback::Stopped,
                }
                None
            }
            Playback::Exporting => {
                let Some(export) = self.export.as_mut() else {
                    self.playback = Playback::Stopped;
                    return None;
                };
                if let Some(error) = export.writer_error() {
                    self.abort_export(error);
                    return None;
                }
                self.time = (export.frame as f64 / f64::from(export.fps)) as f32;
                if let Some(c) = self.sequence.sample(self.time) {
                    *camera = c;
                }
                if !world_ready {
                    // The export time counts from the complete world (warmup included).
                    export.started = Instant::now();
                    self.note = "export: waiting for the world".into();
                    return None;
                }
                if export.warmup_left > 0 {
                    export.warmup_left -= 1;
                    return None;
                }
                export.samples_left = export.samples_left.saturating_sub(1);
                if export.samples_left == 0 {
                    Some(export.dir.join(format!("frame-{:06}.png", export.frame)))
                } else {
                    None
                }
            }
        }
    }

    /// Hands the captured frame to the PNG writer and moves the export to the next frame.
    /// Returns `true` when the export just finished.
    pub fn frame_captured(&mut self, path: PathBuf, pixels: Vec<u8>, width: u32, height: u32) -> bool {
        let Some(export) = self.export.as_mut() else {
            return false;
        };
        let sent = export.sender.as_ref().is_some_and(|s| s.send((path, pixels, width, height)).is_ok());
        if !sent {
            let error = export.writer_error().unwrap_or_else(|| "the PNG writer thread stopped".into());
            self.abort_export(error);
            return false;
        }
        export.frame += 1;
        export.samples_left = self.samples.max(1);
        let elapsed = export.started.elapsed().as_secs_f64();
        self.note = format!("exporting frame {} / {} ({elapsed:.1} s)", export.frame, export.count);
        if export.frame < export.count {
            return false;
        }

        // The last frame: wait for the writer, then report.
        let mut export = self.export.take().expect("checked above");
        export.join();
        if let Some(error) = export.writer_error() {
            self.abort_export(error);
            return false;
        }
        let seconds = export.started.elapsed().as_secs_f64();
        let dir = export.dir.display().to_string();
        let dir = dir.trim_end_matches(['/', '\\']);
        self.note = format!("exported {} frames to {dir} in {seconds:.1} s ({width} x {height})", export.count);
        self.ffmpeg = format!(
            "ffmpeg -framerate {} -i \"{dir}/frame-%06d.png\" -c:v libx264 -pix_fmt yuv420p -crf 18 \"{dir}.mp4\"",
            export.fps
        );
        log::info!("sequencer: {}", self.note);
        log::info!("sequencer: {}", self.ffmpeg);
        self.playback = Playback::Stopped;
        self.time = self.sequence.total();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(position: [f32; 3], pitch: f32, yaw: f32, hold: f32, duration: f32) -> Keyframe {
        Keyframe { position, pitch, yaw, fov: 60.0, hold, duration }
    }

    fn sequence(keyframes: Vec<Keyframe>, interpolation: Interpolation, looped: bool) -> Sequence {
        Sequence { interpolation, looped, keyframes, ..Sequence::new("test") }
    }

    fn assert_pose(camera: &Camera, k: &Keyframe) {
        let expected = k.camera();
        for c in 0..3 {
            assert!((camera.position[c] - expected.position[c]).abs() < 1e-4, "{camera:?} vs {expected:?}");
        }
        assert!((camera.pitch - expected.pitch).abs() < 1e-4, "{camera:?} vs {expected:?}");
        assert!(wrap_degrees(camera.yaw - expected.yaw).abs() < 1e-4, "{camera:?} vs {expected:?}");
        assert!((camera.fov_y - expected.fov_y).abs() < 1e-4, "{camera:?} vs {expected:?}");
    }

    #[test]
    fn samples_keyframes_and_holds() {
        for interpolation in Interpolation::ALL {
            let s = sequence(
                vec![
                    key([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 2.0),
                    key([10.0, 5.0, -4.0], 10.0, 30.0, 1.0, 3.0),
                    key([30.0, 0.0, 10.0], -5.0, 60.0, 0.0, 1.0),
                ],
                interpolation,
                false,
            );
            for i in 0..3 {
                assert_pose(&s.sample(s.keyframe_time(i)).unwrap(), &s.keyframes[i]);
            }
            // Inside the hold of keyframe 1 the pose stays.
            let t1 = s.keyframe_time(1);
            for dt in [0.25, 0.5, 0.99] {
                assert_pose(&s.sample(t1 + dt).unwrap(), &s.keyframes[1]);
            }
            // Past the end: the last keyframe.
            assert_pose(&s.sample(s.total() + 5.0).unwrap(), &s.keyframes[2]);
        }
    }

    #[test]
    fn yaw_takes_the_shortest_arc() {
        let s = sequence(
            vec![key([0.0; 3], 0.0, 170.0, 0.0, 2.0), key([0.0; 3], 0.0, -170.0, 0.0, 2.0)],
            Interpolation::Linear,
            false,
        );
        let middle = s.sample(1.0).unwrap();
        assert!((middle.yaw.abs() - 180.0).abs() < 1e-3, "yaw {}", middle.yaw);
    }

    #[test]
    fn smooth_velocity_is_continuous_and_ends_rest() {
        let s = sequence(
            vec![
                key([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 2.0),
                key([4.0, 2.0, -2.0], 0.0, 0.0, 0.0, 5.0),
                key([10.0, 0.0, 4.0], 0.0, 0.0, 0.0, 1.0),
            ],
            Interpolation::Smooth,
            false,
        );
        let h = 1e-3;
        let t1 = s.keyframe_time(1);
        let at = |t: f32| s.sample(t).unwrap().position;
        let (before, mid, after) = (at(t1 - h), at(t1), at(t1 + h));
        for c in 0..3 {
            let v_before = (mid[c] - before[c]) / h;
            let v_after = (after[c] - mid[c]) / h;
            assert!((v_before - v_after).abs() < 1e-2, "axis {c}: {v_before} vs {v_after}");
        }
        // At rest at both ends.
        let distance = |a: [f32; 3], b: [f32; 3]| (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f32>().sqrt();
        assert!(distance(at(0.0), at(1e-3)) < 1e-4);
        let total = s.total();
        assert!(distance(at(total - 1e-3), at(total)) < 1e-4);
    }

    #[test]
    fn loop_includes_the_last_travel() {
        let s = sequence(
            vec![
                key([0.0, 0.0, 0.0], 0.0, 0.0, 0.5, 2.0),
                key([10.0, 0.0, 0.0], 0.0, 90.0, 0.0, 3.0),
                key([10.0, 0.0, 10.0], 0.0, 180.0, 0.0, 4.0),
            ],
            Interpolation::Smooth,
            true,
        );
        assert!((s.total() - 9.5).abs() < 1e-5);
        assert_pose(&s.sample(s.total()).unwrap(), &s.keyframes[0]);
        assert_pose(&s.sample(0.0).unwrap(), &s.keyframes[0]);
        // Wraps: one period later is the same pose.
        let a = s.sample(3.7).unwrap();
        let b = s.sample(3.7 + s.total()).unwrap();
        assert!((0..3).all(|c| (a.position[c] - b.position[c]).abs() < 1e-3));
    }

    #[test]
    fn json_round_trip() {
        let s = sequence(
            vec![key([1.0, 2.0, 3.0], 10.0, -20.0, 0.5, 2.0), key([4.0, 5.0, 6.0], 0.0, 45.0, 0.0, 1.0)],
            Interpolation::Linear,
            true,
        );
        let text = serde_json::to_string_pretty(&s).unwrap();
        assert!(text.contains("\"loop\": true") && text.contains("\"linear\""), "{text}");
        let back: Sequence = serde_json::from_str(&text).unwrap();
        assert_eq!(back.keyframes, s.keyframes);
        assert!(back.looped && back.interpolation == Interpolation::Linear && back.fps == s.fps);

        // `hold` is optional.
        let text = r#"{"version": 1, "name": "x", "interpolation": "smooth", "loop": false, "fps": 30,
            "keyframes": [{"position": [0, 1, 2], "pitch": 5, "yaw": 6, "fov": 60, "duration": 2}]}"#;
        let parsed: Sequence = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.keyframes[0].hold, 0.0);
        assert!(!parsed.looped && parsed.interpolation == Interpolation::Smooth && parsed.fps == 30);
    }

    #[test]
    fn frame_count_rule() {
        let keyframes = vec![key([0.0; 3], 0.0, 0.0, 0.0, 1.5), key([1.0; 3], 0.0, 0.0, 0.0, 1.5)];
        let s = sequence(keyframes.clone(), Interpolation::Smooth, false);
        assert_eq!(s.frame_count(60), 91);
        let s = sequence(keyframes, Interpolation::Smooth, true);
        assert_eq!(s.frame_count(60), 180);
        assert_eq!(sequence(Vec::new(), Interpolation::Smooth, false).frame_count(60), 1);
    }
}
