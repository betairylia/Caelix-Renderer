//! Streams a `.cxw` world into a `CaelixScene`.
//!
//! One background thread runs [`CxwFile::decode_all`] and sends every decoded
//! sector through a bounded channel. The decoder is parallel internally; when
//! the channel is full its sink blocks, so the decoder cannot run far ahead of
//! the uploads. The render thread drains the channel once per frame into
//! `caelix_scene_upload_bricks` until the staging ring is full (`consumed <
//! count`); the rest of that sector waits for the next frame.

use std::ffi::c_void;
use std::mem::{align_of, size_of};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context as _};
use caelix_cxw::{CxwFile, DecodedSector, BLOCKS_IN_BRICK};
use caelix_sys::*;

// The decoder's keys are passed to the library without a copy.
const _: () = assert!(size_of::<CaelixBrickKey>() == 12);
const _: () = assert!(size_of::<CaelixBrickKey>() == size_of::<[i32; 3]>());
const _: () = assert!(align_of::<CaelixBrickKey>() == align_of::<[i32; 3]>());

/// Decoded sectors that may wait in the channel.
const CHANNEL_SECTORS: usize = 32;
/// Decoded sectors whose first key is queried after the load.
const PROBE_COUNT: usize = 3;

type DecodeOutput = (Result<(), caelix_cxw::Error>, Duration);

/// What the load produced, for the caller's exit code.
pub enum LoadOutcome {
    /// Still loading.
    Loading,
    /// The load completed in this frame and the resident brick count matches.
    Complete,
    /// The load completed in this frame and the resident brick count does not match.
    Mismatch,
}

pub struct WorldLoader {
    /// `None` after `stop`.
    rx: Option<Receiver<DecodedSector>>,
    /// `None` after the thread was joined.
    decoder: Option<JoinHandle<DecodeOutput>>,
    /// Scene entity id per file entity.
    entity_ids: Vec<CaelixEntityId>,
    /// A sector that the staging ring could not take completely, and its first
    /// brick that is not uploaded yet.
    pending: Option<(DecodedSector, usize)>,
    decode_time: Option<Duration>,
    complete: bool,
    sectors_received: u64,
    bricks_received: u64,
    bricks_uploaded: u64,
    upload_frames: u64,
    /// Frame numbers of the first and the last frame that uploaded bricks.
    upload_frame_range: Option<(u64, u64)>,
    upload_start: Option<Instant>,
    upload_end: Option<Instant>,
    /// (file entity, first key) of the first decoded sectors with bricks.
    probes: Vec<(usize, [i32; 3])>,
}

impl WorldLoader {
    /// Opens the file, adds every entity to the scene and starts the decoder thread.
    pub fn open(scene: *mut CaelixScene, path: &Path) -> anyhow::Result<Self> {
        let file = CxwFile::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let world = file.world();
        log::info!("world {}: format version {}", path.display(), world.version);
        match world.player_pose {
            Some(p) => log::info!("player pose: position {:?} rotation {:?}", p.position, p.rotation),
            None => log::info!("player pose: none"),
        }
        log::info!("{} entities, {} sectors", world.entities.len(), file.sectors().len());

        let mut entity_ids = Vec::with_capacity(world.entities.len());
        for (index, entity) in world.entities.iter().enumerate() {
            let desc = CaelixEntityDesc {
                guid: entity.guid,
                position: entity.position,
                rotation: entity.rotation,
            };
            let mut id: CaelixEntityId = CAELIX_INVALID_ENTITY;
            // SAFETY: the scene is live; desc and id are valid for the call.
            let result = unsafe { caelix_scene_add_entity(scene, &desc, &mut id) };
            if result != CaelixResult_CAELIX_OK {
                bail!("caelix_scene_add_entity failed for file entity {index}: CaelixResult {result}");
            }
            log::debug!(
                "entity [{index}] {} -> scene id {id}, {} sectors, position {:?}",
                entity.guid_string(),
                entity.sector_count,
                entity.position
            );
            entity_ids.push(id);
        }

        let (tx, rx) = mpsc::sync_channel::<DecodedSector>(CHANNEL_SECTORS);
        let decoder = thread::Builder::new()
            .name("cxw-decode".into())
            .spawn(move || {
                let start = Instant::now();
                let result = file.decode_all(|sector| {
                    // A send error means the receiver is gone (the app is exiting):
                    // drop the sector. decode_all still decodes the rest.
                    let _ = tx.send(sector);
                });
                (result, start.elapsed())
            })
            .context("cannot start the decoder thread")?;

        Ok(Self {
            rx: Some(rx),
            decoder: Some(decoder),
            entity_ids,
            pending: None,
            decode_time: None,
            complete: false,
            sectors_received: 0,
            bricks_received: 0,
            bricks_uploaded: 0,
            upload_frames: 0,
            upload_frame_range: None,
            upload_start: None,
            upload_end: None,
            probes: Vec::new(),
        })
    }

    /// Uploads decoded bricks until the staging ring is full or the channel is empty.
    /// Call once per frame before `caelix_scene_flush`. Returns true when bricks were
    /// uploaded.
    pub fn upload(&mut self, scene: *mut CaelixScene, frame_number: u64) -> anyhow::Result<bool> {
        let mut uploaded = 0u64;
        loop {
            let (sector, offset) = match self.pending.take() {
                Some(pending) => pending,
                None => match self.receive()? {
                    Some(sector) => (sector, 0),
                    None => break,
                },
            };
            let remaining = sector.keys.len() - offset;
            if remaining == 0 {
                continue;
            }
            let count = u32::try_from(remaining).context("sector brick count exceeds u32")?;
            let entity = self.entity_ids[sector.entity];
            let keys = sector.keys[offset..].as_ptr().cast::<CaelixBrickKey>();
            let blocks = sector.blocks[offset * BLOCKS_IN_BRICK..].as_ptr().cast::<c_void>();
            if self.upload_start.is_none() {
                self.upload_start = Some(Instant::now());
            }
            let mut consumed = 0u32;
            // SAFETY: keys holds `count` keys ([i32; 3] has the layout of CaelixBrickKey,
            // checked above) and blocks holds count * 512 u16 values; both stay alive for
            // the call, which only borrows them.
            let result =
                unsafe { caelix_scene_upload_bricks(scene, entity, count, keys, blocks, &mut consumed) };
            if result != CaelixResult_CAELIX_OK {
                bail!(
                    "caelix_scene_upload_bricks failed: CaelixResult {result} \
                     (entity {entity}, sector {:?}, {count} bricks, {consumed} consumed)",
                    sector.coord
                );
            }
            uploaded += u64::from(consumed);
            let offset = offset + consumed as usize;
            if offset < sector.keys.len() {
                // The staging ring is full: the rest waits for a later frame.
                self.pending = Some((sector, offset));
                break;
            }
        }
        if uploaded > 0 {
            self.bricks_uploaded += uploaded;
            self.upload_frames += 1;
            let first = self.upload_frame_range.map_or(frame_number, |(first, _)| first);
            self.upload_frame_range = Some((first, frame_number));
        }
        Ok(uploaded > 0)
    }

    /// Call after the frame's `caelix_scene_flush`. `uploaded` is the result of this
    /// frame's `upload`. Logs the summary once when the load has completed.
    pub fn after_flush(&mut self, scene: *const CaelixScene, uploaded: bool) -> LoadOutcome {
        if uploaded {
            self.upload_end = Some(Instant::now());
        }
        if self.complete || self.decoder.is_some() || self.pending.is_some() {
            return LoadOutcome::Loading;
        }
        self.complete = true;
        self.log_summary(scene)
    }

    /// Logs the progress of an incomplete load.
    pub fn log_incomplete(&self) {
        if !self.complete {
            log::warn!(
                "world load incomplete: {} of {} received bricks uploaded, {} sectors received, decoder {}",
                self.bricks_uploaded,
                self.bricks_received,
                self.sectors_received,
                if self.decoder.is_some() { "running" } else { "finished" }
            );
        }
    }

    /// Stops the decoder: drops the receiver so the sink's sends fail, then joins the
    /// thread. decode_all cannot be cancelled, so the join waits until it has decoded
    /// the remaining sectors (each is dropped right away).
    pub fn stop(&mut self) {
        self.rx = None;
        self.pending = None;
        if let Some(decoder) = self.decoder.take() {
            log::info!("waiting for the decoder thread to stop");
            match decoder.join() {
                Ok((Ok(()), time)) => log::debug!("decoder thread stopped after {:.3} s", time.as_secs_f64()),
                Ok((Err(e), _)) => log::debug!("decoder thread stopped with: {e}"),
                Err(_) => log::error!("the decoder thread panicked"),
            }
        }
    }

    /// Next decoded sector, or None when the channel is empty or the decoder finished.
    /// Joins the decoder when the channel is disconnected and returns its error.
    fn receive(&mut self) -> anyhow::Result<Option<DecodedSector>> {
        let Some(rx) = self.rx.as_ref() else {
            return Ok(None);
        };
        match rx.try_recv() {
            Ok(sector) => {
                self.sectors_received += 1;
                self.bricks_received += sector.keys.len() as u64;
                if self.probes.len() < PROBE_COUNT {
                    if let Some(key) = sector.keys.first() {
                        self.probes.push((sector.entity, *key));
                    }
                }
                Ok(Some(sector))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.rx = None;
                if let Some(decoder) = self.decoder.take() {
                    let (result, time) = decoder.join().map_err(|_| anyhow!("the decoder thread panicked"))?;
                    result.context("decoding the world failed")?;
                    self.decode_time = Some(time);
                }
                Ok(None)
            }
        }
    }

    fn log_summary(&self, scene: *const CaelixScene) -> LoadOutcome {
        // SAFETY: CaelixSceneStats holds only integers, so all-zero is a valid value.
        let mut stats: CaelixSceneStats = unsafe { std::mem::zeroed() };
        // SAFETY: the scene is live and stats is writable.
        unsafe { caelix_scene_get_stats(scene, &mut stats) };

        let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
        let decode_secs = self.decode_time.map_or(0.0, |t| t.as_secs_f64());
        let upload_secs = match (self.upload_start, self.upload_end) {
            (Some(start), Some(end)) => end.duration_since(start).as_secs_f64(),
            _ => 0.0,
        };
        let frames_spanned = self.upload_frame_range.map_or(0, |(first, last)| last - first + 1);
        let bricks_per_second =
            if upload_secs > 0.0 { self.bricks_uploaded as f64 / upload_secs } else { 0.0 };

        log::info!(
            "world loaded: {} entities, {} groups, {} bricks resident, {} record pages",
            stats.entities,
            stats.groups,
            stats.bricks_resident,
            stats.record_pages
        );
        log::info!(
            "GPU memory: records {:.1} MiB, slot tables {:.1} MiB, staging {:.1} MiB",
            mib(stats.record_bytes),
            mib(stats.slot_table_bytes),
            mib(stats.staging_bytes)
        );
        log::info!(
            "decode {:.3} s ({} sectors, {} bricks; wall time includes channel back-pressure), \
             upload {:.3} s, {} frames uploaded (of {} frames spanned), {:.0} bricks/s",
            decode_secs,
            self.sectors_received,
            self.bricks_received,
            upload_secs,
            self.upload_frames,
            frames_spanned,
            bricks_per_second
        );
        log::info!("scene stats: {stats:?}");

        for &(file_entity, key) in &self.probes {
            let entity = self.entity_ids[file_entity];
            // SAFETY: CaelixBrickInfo holds only integers, so all-zero is a valid value.
            let mut info: CaelixBrickInfo = unsafe { std::mem::zeroed() };
            let brick_key = CaelixBrickKey { x: key[0], y: key[1], z: key[2] };
            // SAFETY: the scene is live and info is writable.
            let result = unsafe { caelix_scene_query_brick(scene, entity, brick_key, &mut info) };
            if result != CaelixResult_CAELIX_OK {
                log::error!("caelix_scene_query_brick(entity {entity}, key {key:?}) failed: CaelixResult {result}");
            } else {
                log::info!(
                    "query entity {entity} key {key:?}: resident {} record_index {}",
                    info.resident,
                    info.record_index
                );
            }
        }

        if stats.bricks_resident != self.bricks_received {
            log::error!(
                "bricks_resident {} differs from the {} bricks the decoder produced",
                stats.bricks_resident,
                self.bricks_received
            );
            LoadOutcome::Mismatch
        } else {
            LoadOutcome::Complete
        }
    }
}
