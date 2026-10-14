//! Reader for Caelix `.cxw` world files (format version 8 and older).
//!
//! Byte-level reference: `ScenesPlayground/docs/cxw-format.md`. The C# code in
//! `Caelix-Core/Runtime/IO/` is the source of truth; this crate follows its read rules.
//!
//! The file is memory-mapped. [`CxwFile::open`] parses the header, the entity table and every
//! sector index, and checks every region size. [`CxwFile::decode`] inflates one sector and
//! returns its non-air bricks; [`CxwFile::decode_all`] does that for every sector on the rayon
//! thread pool and streams the results into a sink.
//!
//! Brick values are returned exactly as stored. This crate does not interpret packed colors
//! or block IDs.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use flate2::{Decompress, FlushDecompress, Status};
use memmap2::Mmap;
use rayon::prelude::*;

pub mod summary;

/// `"CXLS"`, the current file magic.
pub const FILE_MAGIC: u32 = 0x534C_5843;
/// `"VXLS"`, the legacy file magic. Accepted on read.
pub const LEGACY_FILE_MAGIC: u32 = 0x534C_5856;
/// Highest format version this reader accepts.
pub const CURRENT_VERSION: u16 = 8;
/// The only valid header flags value (`Deflate`).
pub const SAVE_FLAG_DEFLATE: u16 = 1;
/// `"VXS2"`, the magic of an inflated sector payload.
pub const SECTOR_PAYLOAD_MAGIC: u32 = 0x3253_5856;

/// Entity flag bit 0: the entity has a physics body.
pub const ENTITY_FLAG_HAS_BODY: u8 = 1;
/// Entity flag bit 1: the entity never moves.
pub const ENTITY_FLAG_STATIC: u8 = 2;
const ENTITY_FLAGS_ALL: u8 = ENTITY_FLAG_HAS_BODY | ENTITY_FLAG_STATIC;
const WORLD_FLAG_HAS_PLAYER_POSE: u8 = 1;

/// Bricks per sector axis.
pub const BRICKS_PER_AXIS: i32 = 16;
/// Bricks in one sector (16^3).
pub const BRICKS_IN_SECTOR: usize = 4096;
/// Voxels in one brick (8^3).
pub const BLOCKS_IN_BRICK: usize = 512;

const HEADER_BYTES: u64 = 64;
const PREVIEW_BYTES: u64 = BRICKS_IN_SECTOR as u64 * 4;
/// Preview + u32 payload size. `regionSize == REGION_OVERHEAD + payloadSize`.
const REGION_OVERHEAD: u64 = PREVIEW_BYTES + 4;
const SECTOR_INDEX_ENTRY_BYTES: u64 = 24;
/// Smallest entity record (version < 3): guid + transform + require flags + index.
const MIN_ENTITY_RECORD_BYTES: usize = 16 + 28 + 2 + 8 + 4;
const MAX_SLOTS: u32 = 16;
const SLOT_BLOCK: u8 = 0;
const BLOCK_STRIDE: u16 = 2;
/// Upper bound for one inflated payload. A full sector with Block and PhysicsInfo slots is
/// about 6 MB; anything far above that is a broken or hostile file.
const MAX_PAYLOAD_BYTES: usize = 1 << 30;

// Version gates of SingleFileSaveStorage.LoadIndex.
const VERSION_ENTITY_FLAGS: u16 = 3;
const VERSION_VELOCITY: u16 = 4;
const VERSION_PROTECTED: u16 = 5;
const VERSION_FLAGS_V6: u16 = 6;
const VERSION_WORLD_RECORD: u16 = 7;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad file magic 0x{0:08X} (expected CXLS or VXLS)")]
    BadMagic(u32),
    #[error("format version {0} is not supported (newest supported: {CURRENT_VERSION})")]
    UnsupportedVersion(u16),
    #[error("unsupported save flags 0x{0:04X} (must be 1 = Deflate)")]
    BadFlags(u16),
    /// `offset` is a file offset, except for `what` values that start with "payload", where it
    /// is an offset into the inflated sector payload.
    #[error("truncated data reading {what} at offset {offset}")]
    Truncated { what: &'static str, offset: u64 },
    #[error(
        "sector {coord:?}: region size is {actual}, expected {expected} (16388 + payload size)"
    )]
    BadRegionSize {
        coord: [i32; 3],
        expected: u64,
        actual: u32,
    },
    #[error("bad sector payload magic 0x{0:08X} (expected VXS2)")]
    BadSectorMagic(u32),
    #[error("invalid brick map counters: capacity {capacity}, count {count}, free {free_count}")]
    BadCounts {
        capacity: u32,
        count: u32,
        free_count: u32,
    },
    #[error("inflate failed: {0}")]
    Inflate(String),
    #[error("sector {0:?} appears twice in one entity's sector index")]
    DuplicateSector([i32; 3]),
    /// A payload structure error that the variants above do not cover (slot table, brick
    /// index table).
    #[error("sector {coord:?}: invalid payload: {reason}")]
    BadPayload {
        coord: [i32; 3],
        reason: &'static str,
    },
}

/// One entity record, after the version migrations of the C# reader.
#[derive(Clone, Debug)]
pub struct EntityInfo {
    /// The 16 GUID bytes as stored (C# `Guid128` = four little-endian u32 lanes x, y, z, w).
    pub guid: [u8; 16],
    pub position: [f32; 3],
    /// Quaternion `x, y, z, w`.
    pub rotation: [f32; 4],
    /// Bit 0 = HasBody, bit 1 = Static (migrated for versions below 6).
    pub flags: u8,
    pub protected: bool,
    /// Linear xyz, then angular xyz. Zero before version 4.
    pub velocity: [f32; 6],
    /// Raw entity require-update flags as stored. The version < 8 automata bit conversion of
    /// the engine loader is NOT applied.
    pub require_update_flags: u16,
    pub sector_count: u32,
}

impl EntityInfo {
    pub fn has_body(&self) -> bool {
        self.flags & ENTITY_FLAG_HAS_BODY != 0
    }

    pub fn is_static(&self) -> bool {
        self.flags & ENTITY_FLAG_STATIC != 0
    }

    /// The four `Guid128` lanes `x, y, z, w` (little-endian u32 each).
    pub fn guid_lanes(&self) -> [u32; 4] {
        std::array::from_fn(|i| u32::from_le_bytes(self.guid[i * 4..i * 4 + 4].try_into().unwrap()))
    }

    /// Canonical dashed UUID string. The 16 bytes are read like .NET `Guid(byte[])` and Python
    /// `uuid.UUID(bytes_le=...)` (u32 LE, u16 LE, u16 LE, 8 bytes in order). This matches the
    /// GUID strings in the scene writer's JSON reports.
    pub fn guid_string(&self) -> String {
        let g = &self.guid;
        let a = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
        let b = u16::from_le_bytes([g[4], g[5]]);
        let c = u16::from_le_bytes([g[6], g[7]]);
        format!(
            "{a:08x}-{b:04x}-{c:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            g[8], g[9], g[10], g[11], g[12], g[13], g[14], g[15]
        )
    }
}

/// The player (free camera) pose of a version 7+ world record.
#[derive(Clone, Copy, Debug)]
pub struct PlayerPose {
    pub position: [f32; 3],
    /// Quaternion `x, y, z, w`.
    pub rotation: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct World {
    pub version: u16,
    /// `None` before version 7 and when the world record has no pose.
    pub player_pose: Option<PlayerPose>,
    pub entities: Vec<EntityInfo>,
}

/// One sector index entry. `entity` indexes [`World::entities`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SectorRef {
    pub entity: usize,
    pub coord: [i32; 3],
    pub region_offset: u64,
    pub region_size: u32,
}

/// The non-air bricks of one sector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedSector {
    pub entity: usize,
    pub coord: [i32; 3],
    /// Entity-local brick keys (`sector_coord * 16 + brick_in_sector`), one per non-air brick,
    /// in ascending absolute brick index order. A brick whose 512 values are all 0 is dropped.
    pub keys: Vec<[i32; 3]>,
    /// `keys.len() * 512` raw file values; voxel index `x + 8y + 64z` inside each brick.
    pub blocks: Vec<u16>,
    /// The preview word of each kept brick, in the same order as `keys`. See [`summary`].
    pub summaries: Vec<u32>,
    /// Number of non-zero block values in the kept bricks.
    pub voxels: u64,
}

/// An open `.cxw` file: the memory map and the parsed index.
pub struct CxwFile {
    mmap: Mmap,
    world: World,
    sectors: Vec<SectorRef>,
}

impl CxwFile {
    /// Maps the file and parses the header, the entity table and all sector indices. Checks
    /// every region size against its payload size and rejects a sector coord that appears
    /// twice in one entity.
    ///
    /// The file must not be modified while the `CxwFile` is open (it is memory-mapped).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len < HEADER_BYTES {
            return Err(Error::Truncated {
                what: "header",
                offset: 0,
            });
        }
        // SAFETY: the map is read-only. Like every memory-mapped reader, this assumes that no
        // other process truncates or rewrites the file while it is open (documented above).
        let mmap = unsafe { Mmap::map(&file)? };
        let (world, sectors) = parse_index(&mmap)?;
        Ok(Self {
            mmap,
            world,
            sectors,
        })
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    /// Every sector of every entity, in entity order and then sector index order.
    pub fn sectors(&self) -> &[SectorRef] {
        &self.sectors
    }

    /// Inflates and decodes one sector.
    pub fn decode(&self, sector: SectorRef) -> Result<DecodedSector, Error> {
        self.decode_with(sector, &mut Scratch::new())
    }

    /// Decodes every sector on the rayon thread pool and passes each to `sink` as it completes
    /// (any order). Stops at the first error and returns it.
    pub fn decode_all(&self, sink: impl Fn(DecodedSector) + Sync) -> Result<(), Error> {
        self.sectors
            .par_iter()
            .try_for_each_init(Scratch::new, |scratch, sector| {
                sink(self.decode_with(*sector, scratch)?);
                Ok(())
            })
    }

    fn decode_with(
        &self,
        sector: SectorRef,
        scratch: &mut Scratch,
    ) -> Result<DecodedSector, Error> {
        let (preview, compressed) = region(
            &self.mmap,
            sector.coord,
            sector.region_offset,
            sector.region_size,
        )?;
        scratch.inflate(compressed, sector.coord)?;
        decode_payload(&scratch.buffer, preview, sector.entity, sector.coord)
    }
}

// ------------------------------------------------------------------------------------ index

/// Bounds-checked little-endian reader over a byte slice.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len())
            .ok_or(Error::Truncated {
                what,
                offset: self.pos as u64,
            })?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], Error> {
        Ok(self.take(N, what)?.try_into().unwrap())
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, Error> {
        Ok(self.array::<1>(what)?[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.array(what)?))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.array(what)?))
    }

    fn i32x3(&mut self, what: &'static str) -> Result<[i32; 3], Error> {
        let b: [u8; 12] = self.array(what)?;
        Ok(std::array::from_fn(|i| {
            i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
        }))
    }

    fn f32s<const N: usize>(&mut self, what: &'static str) -> Result<[f32; N], Error> {
        let b = self.take(N * 4, what)?;
        Ok(std::array::from_fn(|i| {
            f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
        }))
    }
}

/// Pre-v6 `bodyState` byte -> entity flags (`WorldSaveFormat.EntityFlagsFromLegacyBodyState`).
fn flags_from_legacy_body_state(body_state: u8) -> u8 {
    match body_state {
        1 => ENTITY_FLAG_HAS_BODY | ENTITY_FLAG_STATIC,
        2 => ENTITY_FLAG_HAS_BODY,
        _ => ENTITY_FLAG_STATIC,
    }
}

fn parse_index(data: &[u8]) -> Result<(World, Vec<SectorRef>), Error> {
    let len = data.len() as u64;
    let mut header = Cursor::new(data, 0);
    let magic = header.u32("header")?;
    if magic != FILE_MAGIC && magic != LEGACY_FILE_MAGIC {
        return Err(Error::BadMagic(magic));
    }
    let version = header.u16("header")?;
    if version > CURRENT_VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    let flags = header.u16("header")?;
    if flags != SAVE_FLAG_DEFLATE {
        return Err(Error::BadFlags(flags));
    }
    let table = header.u64("header")?;
    if table < HEADER_BYTES || table > len {
        return Err(Error::Truncated {
            what: "entity table",
            offset: table,
        });
    }

    let mut c = Cursor::new(data, table as usize);
    let mut player_pose = None;
    if version >= VERSION_WORLD_RECORD {
        let world_flags = c.u8("world record")?;
        let position = c.f32s::<3>("world record")?;
        let rotation = c.f32s::<4>("world record")?;
        if world_flags & WORLD_FLAG_HAS_PLAYER_POSE != 0 {
            player_pose = Some(PlayerPose { position, rotation });
        }
    }
    let count = c.u32("entity count")? as usize;
    let remaining = data.len().saturating_sub(c.pos);
    let mut entities = Vec::with_capacity(count.min(remaining / MIN_ENTITY_RECORD_BYTES));
    let mut index_offsets = Vec::with_capacity(entities.capacity());
    for _ in 0..count {
        let what = "entity record";
        let guid = c.array::<16>(what)?;
        let position = c.f32s::<3>(what)?;
        let rotation = c.f32s::<4>(what)?;
        let require_update_flags = c.u16(what)?;
        let flags = if version >= VERSION_FLAGS_V6 {
            c.u8(what)? & ENTITY_FLAGS_ALL
        } else if version >= VERSION_ENTITY_FLAGS {
            flags_from_legacy_body_state(c.u8(what)?)
        } else {
            ENTITY_FLAG_STATIC
        };
        let velocity = if version >= VERSION_VELOCITY {
            c.f32s::<6>(what)?
        } else {
            [0.0; 6]
        };
        let protected = version >= VERSION_PROTECTED && c.u8(what)? != 0;
        let sector_index_offset = c.u64(what)?;
        let sector_count = c.u32(what)?;
        entities.push(EntityInfo {
            guid,
            position,
            rotation,
            flags,
            protected,
            velocity,
            require_update_flags,
            sector_count,
        });
        index_offsets.push(sector_index_offset);
    }

    let mut sectors = Vec::new();
    let mut seen = HashSet::new();
    for (entity, (info, &start)) in entities.iter().zip(&index_offsets).enumerate() {
        let bytes = info.sector_count as u64 * SECTOR_INDEX_ENTRY_BYTES;
        if start < HEADER_BYTES || start.checked_add(bytes).is_none_or(|end| end > len) {
            return Err(Error::Truncated {
                what: "sector index",
                offset: start,
            });
        }
        let mut c = Cursor::new(data, start as usize);
        seen.clear();
        sectors.reserve(info.sector_count as usize);
        for _ in 0..info.sector_count {
            let coord = c.i32x3("sector index")?;
            let region_offset = c.u64("sector index")?;
            let region_size = c.u32("sector index")?;
            region(data, coord, region_offset, region_size)?;
            if !seen.insert(coord) {
                return Err(Error::DuplicateSector(coord));
            }
            sectors.push(SectorRef {
                entity,
                coord,
                region_offset,
                region_size,
            });
        }
    }

    Ok((
        World {
            version,
            player_pose,
            entities,
        },
        sectors,
    ))
}

/// Checks one region and returns (preview bytes, compressed payload).
fn region(data: &[u8], coord: [i32; 3], offset: u64, size: u32) -> Result<(&[u8], &[u8]), Error> {
    let len = data.len() as u64;
    if offset
        .checked_add(REGION_OVERHEAD)
        .is_none_or(|end| end > len)
    {
        return Err(Error::Truncated {
            what: "sector region",
            offset,
        });
    }
    let start = offset as usize;
    let preview = &data[start..start + PREVIEW_BYTES as usize];
    let size_at = start + PREVIEW_BYTES as usize;
    let payload_size = u32::from_le_bytes(data[size_at..size_at + 4].try_into().unwrap());
    let expected = REGION_OVERHEAD + payload_size as u64;
    if expected != size as u64 {
        return Err(Error::BadRegionSize {
            coord,
            expected,
            actual: size,
        });
    }
    let payload_start = offset + REGION_OVERHEAD;
    if payload_start + payload_size as u64 > len {
        return Err(Error::Truncated {
            what: "sector payload",
            offset: payload_start,
        });
    }
    let payload_start = payload_start as usize;
    Ok((
        preview,
        &data[payload_start..payload_start + payload_size as usize],
    ))
}

// ---------------------------------------------------------------------------------- payload

/// Per-thread decode state: the raw-deflate inflater and the output buffer, both reused.
struct Scratch {
    inflater: Decompress,
    buffer: Vec<u8>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            inflater: Decompress::new(false),
            buffer: Vec::new(),
        }
    }

    /// Inflates a raw deflate stream (no zlib header) into `self.buffer`.
    fn inflate(&mut self, input: &[u8], coord: [i32; 3]) -> Result<(), Error> {
        let fail = |message: &str| Error::Inflate(format!("sector {coord:?}: {message}"));
        self.inflater.reset(false);
        self.buffer.clear();
        let guess = input
            .len()
            .saturating_mul(4)
            .clamp(64 * 1024, MAX_PAYLOAD_BYTES);
        self.buffer.reserve(guess);
        loop {
            if self.buffer.len() == self.buffer.capacity() {
                if self.buffer.len() >= MAX_PAYLOAD_BYTES {
                    return Err(fail("payload inflates to more than 1 GiB"));
                }
                self.buffer.reserve(self.buffer.len());
            }
            let in_before = self.inflater.total_in();
            let out_before = self.inflater.total_out();
            let status = self
                .inflater
                .decompress_vec(
                    &input[in_before as usize..],
                    &mut self.buffer,
                    FlushDecompress::None,
                )
                .map_err(|e| fail(&e.to_string()))?;
            if status == Status::StreamEnd {
                return Ok(());
            }
            let stalled =
                self.inflater.total_in() == in_before && self.inflater.total_out() == out_before;
            if stalled && self.buffer.len() < self.buffer.capacity() {
                return Err(fail("deflate stream ends before its final block"));
            }
        }
    }
}

fn decode_payload(
    raw: &[u8],
    preview: &[u8],
    entity: usize,
    coord: [i32; 3],
) -> Result<DecodedSector, Error> {
    let bad = |reason| Error::BadPayload { coord, reason };
    let mut c = Cursor::new(raw, 0);
    let magic = c.u32("payload header")?;
    if magic != SECTOR_PAYLOAD_MAGIC {
        return Err(Error::BadSectorMagic(magic));
    }
    let capacity = c.u32("payload header")?;
    let count = c.u32("payload header")?;
    let free_count = c.u32("payload header")?;
    if capacity as usize > BRICKS_IN_SECTOR
        || count > capacity
        || free_count > capacity
        || count + free_count != capacity
    {
        return Err(Error::BadCounts {
            capacity,
            count,
            free_count,
        });
    }
    let indices = c.take(BRICKS_IN_SECTOR * 2, "payload brick indices")?;
    c.take(free_count as usize * 2, "payload free list")?;
    c.take(2 + BRICKS_IN_SECTOR * 2, "payload require-update flags")?;
    let slot_count = c.u32("payload slot record count")?;
    if slot_count > MAX_SLOTS {
        return Err(bad("more than 16 slot records"));
    }
    let mut seen_slots = 0u32;
    let mut block_data = None;
    for _ in 0..slot_count {
        let slot_id = c.u8("payload slot record header")?;
        let stride = c.u16("payload slot record header")?;
        if slot_id as u32 >= MAX_SLOTS || seen_slots & (1 << slot_id) != 0 {
            return Err(bad("invalid or duplicate slot id"));
        }
        seen_slots |= 1 << slot_id;
        if stride == 0 {
            return Err(bad("slot stride is 0"));
        }
        let data = c.take(
            capacity as usize * BLOCKS_IN_BRICK * stride as usize,
            "payload slot data",
        )?;
        if slot_id == SLOT_BLOCK {
            if stride != BLOCK_STRIDE {
                return Err(bad("Block slot stride is not 2"));
            }
            block_data = Some(data);
        }
    }
    // Trailing bytes are tolerated, like SectorSerializer.Unpack.

    let base = coord.map(|v| v.wrapping_mul(BRICKS_PER_AXIS));
    let mut keys = Vec::with_capacity(count as usize);
    let mut summaries = Vec::with_capacity(count as usize);
    let mut blocks: Vec<u16> = Vec::with_capacity(count as usize * BLOCKS_IN_BRICK);
    let mut voxels = 0u64;
    for (bi, id) in indices.as_chunks::<2>().0.iter().enumerate() {
        let id = i16::from_le_bytes(*id);
        if id == -1 {
            continue;
        }
        if id < -1 || id as u32 >= capacity {
            return Err(bad("brick index table entry outside -1..capacity-1"));
        }
        let data = block_data.ok_or_else(|| bad("allocated bricks but no Block slot"))?;
        let brick_bytes = BLOCKS_IN_BRICK * BLOCK_STRIDE as usize;
        let src = &data[id as usize * brick_bytes..][..brick_bytes];
        let start = blocks.len();
        blocks.extend(
            src.as_chunks::<2>()
                .0
                .iter()
                .map(|&b| u16::from_le_bytes(b)),
        );
        let non_zero = blocks[start..].iter().filter(|&&v| v != 0).count();
        if non_zero == 0 {
            blocks.truncate(start);
            continue;
        }
        voxels += non_zero as u64;
        let bi = bi as i32;
        keys.push([
            base[0].wrapping_add(bi & 15),
            base[1].wrapping_add((bi >> 4) & 15),
            base[2].wrapping_add(bi >> 8),
        ]);
        let p = bi as usize * 4;
        summaries.push(u32::from_le_bytes(preview[p..p + 4].try_into().unwrap()));
    }

    Ok(DecodedSector {
        entity,
        coord,
        keys,
        blocks,
        summaries,
        voxels,
    })
}
