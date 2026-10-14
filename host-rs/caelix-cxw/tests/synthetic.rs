//! Hand-built files: version gates and migrations (v2..v8), payload details the fixture does
//! not cover (free list, brick id remapping, all-air bricks, extra slots, negative coords),
//! and error cases.
//!
//! The writer below puts the entity table FIRST and the regions LAST (the C# writer does the
//! opposite), so these tests also check that the reader only follows offsets.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use caelix_cxw::{CxwFile, Error, FILE_MAGIC, LEGACY_FILE_MAGIC};
use flate2::write::DeflateEncoder;
use flate2::Compression;

// ------------------------------------------------------------------------------- writer

#[derive(Clone)]
struct TSector {
    coord: [i32; 3],
    /// (absolute brick index, 512 values).
    bricks: Vec<(usize, Vec<u16>)>,
    /// Free brick ids, placed below the used ids so used ids are remapped.
    free: u32,
    /// Writes a PhysicsInfo slot (stride 1) BEFORE the Block slot.
    physics_slot_first: bool,
    payload_magic: u32,
    /// Replaces the compressed payload (for corrupt-stream tests).
    compressed_override: Option<Vec<u8>>,
}

impl TSector {
    fn new(coord: [i32; 3], bricks: Vec<(usize, Vec<u16>)>) -> Self {
        Self {
            coord,
            bricks,
            free: 0,
            physics_slot_first: false,
            payload_magic: 0x3253_5856,
            compressed_override: None,
        }
    }
}

#[derive(Clone)]
struct TEntity {
    guid: [u8; 16],
    position: [f32; 3],
    rotation: [f32; 4],
    require: u16,
    /// The raw flags byte (legacy body state for v3..v5).
    flags_byte: u8,
    velocity: [f32; 6],
    protected: u8,
    sectors: Vec<TSector>,
}

impl TEntity {
    fn new(sectors: Vec<TSector>) -> Self {
        Self {
            guid: std::array::from_fn(|i| i as u8 + 1),
            position: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            require: 0,
            flags_byte: 2,
            velocity: [0.0; 6],
            protected: 0,
            sectors,
        }
    }
}

struct TFile {
    magic: u32,
    version: u16,
    flags: u16,
    /// (world flags, position, rotation), written for v7+.
    world: (u8, [f32; 3], [f32; 4]),
    entities: Vec<TEntity>,
}

impl TFile {
    fn new(version: u16, entities: Vec<TEntity>) -> Self {
        Self {
            magic: FILE_MAGIC,
            version,
            flags: 1,
            world: (0, [0.0; 3], [0.0, 0.0, 0.0, 1.0]),
            entities,
        }
    }
}

fn preview_word(bi: usize) -> u32 {
    0xA5A0_0000 | (bi as u32) << 8 | 0x0F
}

fn payload(s: &TSector) -> Vec<u8> {
    let n = s.bricks.len() as u32;
    let capacity = n + s.free;
    let mut indices = vec![-1i16; 4096];
    // Used ids from the top down: brick i gets id capacity-1-i; free ids are 0..free.
    for (i, (bi, _)) in s.bricks.iter().enumerate() {
        indices[*bi] = (capacity - 1 - i as u32) as i16;
    }
    let mut out = Vec::new();
    for v in [s.payload_magic, capacity, n, s.free] {
        out.extend(v.to_le_bytes());
    }
    for v in &indices {
        out.extend(v.to_le_bytes());
    }
    for id in 0..s.free {
        out.extend((id as i16).to_le_bytes());
    }
    out.extend(0x0001u16.to_le_bytes()); // sector require-update flags (ignored by the reader)
    out.extend(std::iter::repeat_n(0u8, 4096 * 2));
    let slot_count: u32 = if s.physics_slot_first { 2 } else { 1 };
    out.extend(slot_count.to_le_bytes());
    if s.physics_slot_first {
        out.push(1);
        out.extend(1u16.to_le_bytes());
        out.extend(std::iter::repeat_n(0x7Eu8, capacity as usize * 512));
    }
    out.push(0);
    out.extend(2u16.to_le_bytes());
    // Free ids hold garbage that must never be read.
    let mut block = vec![0xFFFFu16; capacity as usize * 512];
    for (i, (_, values)) in s.bricks.iter().enumerate() {
        let id = (capacity - 1 - i as u32) as usize;
        block[id * 512..(id + 1) * 512].copy_from_slice(values);
    }
    for v in block {
        out.extend(v.to_le_bytes());
    }
    out
}

fn deflate(raw: &[u8]) -> Vec<u8> {
    let mut e = DeflateEncoder::new(Vec::new(), Compression::fast());
    e.write_all(raw).unwrap();
    e.finish().unwrap()
}

fn record_bytes(version: u16) -> usize {
    16 + 28
        + 2
        + usize::from(version >= 3)
        + if version >= 4 { 24 } else { 0 }
        + usize::from(version >= 5)
        + 12
}

/// Returns the file bytes. Layout: header | world record + entity table | per entity: sector
/// index, then its regions.
fn build(f: &TFile) -> Vec<u8> {
    let table_offset = 64usize;
    let table_len =
        if f.version >= 7 { 29 } else { 0 } + 4 + f.entities.len() * record_bytes(f.version);
    let mut body = Vec::new(); // everything after the table
    let mut index_offsets = Vec::new();
    for e in &f.entities {
        let index_offset = table_offset + table_len + body.len();
        index_offsets.push(index_offset as u64);
        let regions_start = index_offset + e.sectors.len() * 24;
        let mut index = Vec::new();
        let mut regions = Vec::new();
        for s in &e.sectors {
            let compressed = s
                .compressed_override
                .clone()
                .unwrap_or_else(|| deflate(&payload(s)));
            let region_offset = (regions_start + regions.len()) as u64;
            let mut preview = vec![0u32; 4096];
            for (bi, _) in &s.bricks {
                preview[*bi] = preview_word(*bi);
            }
            for w in preview {
                regions.extend(w.to_le_bytes());
            }
            regions.extend((compressed.len() as u32).to_le_bytes());
            regions.extend(&compressed);
            for c in s.coord {
                index.extend(c.to_le_bytes());
            }
            index.extend(region_offset.to_le_bytes());
            index.extend((16388 + compressed.len() as u32).to_le_bytes());
        }
        body.extend(index);
        body.extend(regions);
    }

    let mut out = Vec::new();
    out.extend(f.magic.to_le_bytes());
    out.extend(f.version.to_le_bytes());
    out.extend(f.flags.to_le_bytes());
    out.extend((table_offset as u64).to_le_bytes());
    out.resize(64, 0);
    if f.version >= 7 {
        out.push(f.world.0);
        for v in f.world.1.iter().chain(&f.world.2) {
            out.extend(v.to_le_bytes());
        }
    }
    out.extend((f.entities.len() as u32).to_le_bytes());
    for (e, index_offset) in f.entities.iter().zip(index_offsets) {
        out.extend(e.guid);
        for v in e.position.iter().chain(&e.rotation) {
            out.extend(v.to_le_bytes());
        }
        out.extend(e.require.to_le_bytes());
        if f.version >= 3 {
            out.push(e.flags_byte);
        }
        if f.version >= 4 {
            for v in e.velocity {
                out.extend(v.to_le_bytes());
            }
        }
        if f.version >= 5 {
            out.push(e.protected);
        }
        out.extend(index_offset.to_le_bytes());
        out.extend((e.sectors.len() as u32).to_le_bytes());
    }
    assert_eq!(out.len(), table_offset + table_len);
    out.extend(body);
    out
}

fn temp_path() -> PathBuf {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("caelix-cxw-{}-{n}.cxw", std::process::id()))
}

/// Writes `bytes` to a temp file, opens it, runs `f`, deletes the file.
fn with_file<R>(bytes: &[u8], f: impl FnOnce(Result<CxwFile, Error>) -> R) -> R {
    let path = temp_path();
    std::fs::write(&path, bytes).unwrap();
    let result = f(CxwFile::open(&path));
    std::fs::remove_file(&path).unwrap();
    result
}

fn brick(seed: u16, non_zero: usize) -> Vec<u16> {
    (0..512)
        .map(|i| {
            if i < non_zero {
                seed.wrapping_add(i as u16) | 0x8000
            } else {
                0
            }
        })
        .collect()
}

fn simple_sector(coord: [i32; 3]) -> TSector {
    TSector::new(coord, vec![(0, brick(1, 10))])
}

// -------------------------------------------------------------------------------- tests

#[test]
fn v8_world_record_and_payload_details() {
    let b_far = brick(100, 512); // full brick
    let b_some = brick(7, 3);
    let mut sector = TSector::new(
        [-2, 0, 3],
        vec![
            (15 + 256, b_far.clone()),                // b = (15, 0, 1)
            (4 + 16 * 2 + 256 * 15, vec![0u16; 512]), // allocated all-air brick: dropped
            (16 * 15, b_some.clone()),                // b = (0, 15, 0)
        ],
    );
    sector.free = 2;
    sector.physics_slot_first = true;
    let mut e = TEntity::new(vec![sector, simple_sector([0, 0, 0])]);
    e.flags_byte = 0b1111_1101; // HasBody, unknown bits masked off
    e.velocity = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    e.protected = 1;
    e.require = 0x0003;
    let mut f = TFile::new(8, vec![e, TEntity::new(vec![])]);
    f.world = (0xFF, [10.0, 20.0, 30.0], [0.0, 0.6, 0.0, 0.8]);

    with_file(&build(&f), |file| {
        let file = file.unwrap();
        let w = file.world();
        assert_eq!(w.version, 8);
        let pose = w.player_pose.expect("pose");
        assert_eq!(pose.position, [10.0, 20.0, 30.0]);
        assert_eq!(pose.rotation, [0.0, 0.6, 0.0, 0.8]);
        assert_eq!(w.entities.len(), 2);
        let e = &w.entities[0];
        assert_eq!(e.flags, 1);
        assert!(e.has_body() && !e.is_static());
        assert_eq!(e.velocity, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(e.protected);
        assert_eq!(e.require_update_flags, 3, "raw value, no v<8 conversion");
        assert_eq!(e.position, [1.0, 2.0, 3.0]);
        assert_eq!(e.sector_count, 2);
        assert_eq!(w.entities[1].sector_count, 0);
        assert_eq!(e.guid_string(), "04030201-0605-0807-090a-0b0c0d0e0f10");

        assert_eq!(file.sectors().len(), 2);
        let s = file.decode(file.sectors()[0]).unwrap();
        assert_eq!(s.coord, [-2, 0, 3]);
        assert_eq!(s.entity, 0);
        // Ascending absolute brick index: (0,15,0) = 240 before (15,0,1) = 271.
        assert_eq!(s.keys, vec![[-32, 15, 48], [-17, 0, 49]]);
        assert_eq!(s.summaries, vec![preview_word(240), preview_word(271)]);
        assert_eq!(&s.blocks[..512], &b_some[..]);
        assert_eq!(&s.blocks[512..], &b_far[..]);
        assert_eq!(s.voxels, 3 + 512);

        let all = Mutex::new(Vec::new());
        file.decode_all(|d| all.lock().unwrap().push(d)).unwrap();
        let mut all = all.into_inner().unwrap();
        all.sort_by_key(|d| d.coord);
        assert_eq!(all[0], s);
        assert_eq!(all[1].keys, vec![[0, 0, 0]]);
        assert_eq!(all[1].voxels, 10);
    });
}

#[test]
fn v7_without_pose() {
    let mut f = TFile::new(7, vec![TEntity::new(vec![simple_sector([0, 0, 0])])]);
    f.world = (0, [5.0; 3], [0.0, 0.0, 0.0, 1.0]);
    with_file(&build(&f), |file| {
        let file = file.unwrap();
        assert!(file.world().player_pose.is_none());
        assert_eq!(file.world().entities[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(file.decode(file.sectors()[0]).unwrap().voxels, 10);
    });
}

#[test]
fn legacy_flag_migration_v3_to_v5() {
    for version in [3u16, 4, 5] {
        let entities = [0u8, 1, 2, 9]
            .iter()
            .map(|&state| {
                let mut e = TEntity::new(vec![simple_sector([state as i32, 0, 0])]);
                e.flags_byte = state;
                e.velocity = [0.5; 6];
                e.protected = 1;
                e
            })
            .collect();
        let f = TFile::new(version, entities);
        with_file(&build(&f), |file| {
            let file = file.unwrap();
            let w = file.world();
            assert!(w.player_pose.is_none());
            let flags: Vec<u8> = w.entities.iter().map(|e| e.flags).collect();
            // off -> Static, static -> HasBody|Static, dynamic -> HasBody, unknown -> Static.
            assert_eq!(flags, vec![2, 3, 1, 2], "version {version}");
            for e in &w.entities {
                let velocity = if version >= 4 { [0.5; 6] } else { [0.0; 6] };
                assert_eq!(e.velocity, velocity, "version {version}");
                assert_eq!(e.protected, version >= 5, "version {version}");
            }
            assert_eq!(file.sectors().len(), 4);
            for s in file.sectors() {
                assert_eq!(file.decode(*s).unwrap().keys, vec![[s.coord[0] * 16, 0, 0]]);
            }
        });
    }
}

#[test]
fn v2_has_no_flags_byte() {
    let f = TFile::new(2, vec![TEntity::new(vec![simple_sector([0, -1, 0])])]);
    with_file(&build(&f), |file| {
        let file = file.unwrap();
        let e = &file.world().entities[0];
        assert_eq!(e.flags, 2);
        assert!(!e.protected);
        assert_eq!(e.velocity, [0.0; 6]);
        assert_eq!(
            file.decode(file.sectors()[0]).unwrap().keys,
            vec![[0, -16, 0]]
        );
    });
}

#[test]
fn legacy_magic_accepted() {
    let mut f = TFile::new(6, vec![TEntity::new(vec![simple_sector([0, 0, 0])])]);
    f.magic = LEGACY_FILE_MAGIC;
    with_file(&build(&f), |file| {
        assert_eq!(file.unwrap().world().version, 6)
    });
}

fn base() -> TFile {
    TFile::new(
        8,
        vec![TEntity::new(vec![
            simple_sector([0, 0, 0]),
            simple_sector([1, 0, 0]),
        ])],
    )
}

#[test]
fn bad_magic() {
    let mut f = base();
    f.magic = 0x1234_5678;
    with_file(&build(&f), |r| {
        assert!(
            matches!(r, Err(Error::BadMagic(0x1234_5678))),
            "{r:?}",
            r = r.err()
        )
    });
}

#[test]
fn bad_flags() {
    for flags in [0u16, 3] {
        let mut f = base();
        f.flags = flags;
        with_file(&build(&f), |r| {
            assert!(
                matches!(r, Err(Error::BadFlags(v)) if v == flags),
                "{:?}",
                r.err()
            )
        });
    }
}

#[test]
fn unsupported_version() {
    let mut bytes = build(&base());
    bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
    with_file(&bytes, |r| {
        assert!(
            matches!(r, Err(Error::UnsupportedVersion(99))),
            "{:?}",
            r.err()
        )
    });
}

#[test]
fn file_shorter_than_header() {
    with_file(&[0x43, 0x58, 0x4C, 0x53], |r| {
        assert!(
            matches!(r, Err(Error::Truncated { what: "header", .. })),
            "{:?}",
            r.err()
        )
    });
}

#[test]
fn truncated_region_payload() {
    let bytes = build(&base());
    // The last region ends at the end of the file: cut into its payload.
    with_file(&bytes[..bytes.len() - 5], |r| {
        assert!(
            matches!(
                r,
                Err(Error::Truncated {
                    what: "sector payload",
                    ..
                })
            ),
            "{:?}",
            r.err()
        )
    });
}

#[test]
fn truncated_region_preview() {
    let bytes = build(&base());
    // Cut inside the last region's preview (16 KB before its payload).
    with_file(&bytes[..bytes.len() - 9000], |r| {
        assert!(
            matches!(
                r,
                Err(Error::Truncated {
                    what: "sector region",
                    ..
                })
            ),
            "{:?}",
            r.err()
        )
    });
}

#[test]
fn truncated_entity_table() {
    let mut bytes = build(&base());
    // Point the entity table near the end so the world record runs past the file.
    let off = (bytes.len() - 10) as u64;
    bytes[8..16].copy_from_slice(&off.to_le_bytes());
    with_file(&bytes, |r| {
        assert!(
            matches!(
                r,
                Err(Error::Truncated {
                    what: "world record",
                    ..
                })
            ),
            "{:?}",
            r.err()
        )
    });
}

#[test]
fn bad_region_size() {
    let f = base();
    let mut bytes = build(&f);
    // First sector index entry starts right after the table: 64 + 29 + 4 + 84.
    let entry = 64 + 29 + 4 + 84;
    let size_at = entry + 20;
    let size = u32::from_le_bytes(bytes[size_at..size_at + 4].try_into().unwrap());
    bytes[size_at..size_at + 4].copy_from_slice(&(size + 1).to_le_bytes());
    with_file(&bytes, |r| match r {
        Err(Error::BadRegionSize {
            coord,
            expected,
            actual,
        }) => {
            assert_eq!(coord, [0, 0, 0]);
            assert_eq!(actual, size + 1);
            assert_eq!(expected, size as u64);
        }
        other => panic!("{:?}", other.err()),
    });
}

#[test]
fn duplicate_sector() {
    let f = TFile::new(
        8,
        vec![TEntity::new(vec![
            simple_sector([2, 3, 4]),
            simple_sector([2, 3, 4]),
        ])],
    );
    with_file(&build(&f), |r| {
        assert!(
            matches!(r, Err(Error::DuplicateSector([2, 3, 4]))),
            "{:?}",
            r.err()
        )
    });
    // The same coord in two entities is fine.
    let f = TFile::new(
        8,
        vec![
            TEntity::new(vec![simple_sector([2, 3, 4])]),
            TEntity::new(vec![simple_sector([2, 3, 4])]),
        ],
    );
    with_file(&build(&f), |r| assert_eq!(r.unwrap().sectors().len(), 2));
}

#[test]
fn bad_sector_magic_and_decode_all_error() {
    let mut bad = simple_sector([1, 1, 1]);
    bad.payload_magic = 0xDEAD_BEEF;
    let mut sectors: Vec<_> = (0..20).map(|i| simple_sector([i, 0, 0])).collect();
    sectors.push(bad);
    let f = TFile::new(8, vec![TEntity::new(sectors)]);
    with_file(&build(&f), |r| {
        let file = r.unwrap();
        let last = *file.sectors().last().unwrap();
        assert!(matches!(
            file.decode(last),
            Err(Error::BadSectorMagic(0xDEAD_BEEF))
        ));
        assert!(matches!(
            file.decode_all(|_| {}),
            Err(Error::BadSectorMagic(0xDEAD_BEEF))
        ));
    });
}

#[test]
fn bad_counts() {
    let mut raw = payload(&simple_sector([0, 0, 0]));
    raw[8..12].copy_from_slice(&5u32.to_le_bytes()); // count 5 > capacity 1
    let mut s = simple_sector([0, 0, 0]);
    s.compressed_override = Some(deflate(&raw));
    let f = TFile::new(8, vec![TEntity::new(vec![s])]);
    with_file(&build(&f), |r| {
        let file = r.unwrap();
        let e = file.decode(file.sectors()[0]);
        assert!(
            matches!(
                e,
                Err(Error::BadCounts {
                    capacity: 1,
                    count: 5,
                    free_count: 0
                })
            ),
            "{:?}",
            e.err()
        );
    });
}

#[test]
fn corrupt_deflate() {
    let mut s = simple_sector([0, 0, 0]);
    let mut compressed = deflate(&payload(&s));
    compressed.truncate(compressed.len() / 2); // stream ends early
    s.compressed_override = Some(compressed);
    let mut s2 = simple_sector([1, 0, 0]);
    s2.compressed_override = Some(vec![0xFF; 64]); // invalid block type
    let f = TFile::new(8, vec![TEntity::new(vec![s, s2])]);
    with_file(&build(&f), |r| {
        let file = r.unwrap();
        for s in file.sectors() {
            let e = file.decode(*s);
            assert!(matches!(e, Err(Error::Inflate(_))), "{:?}", e.err());
        }
    });
}

#[test]
fn truncated_payload_slot_data() {
    let mut raw = payload(&simple_sector([0, 0, 0]));
    raw.truncate(raw.len() - 100);
    let mut s = simple_sector([0, 0, 0]);
    s.compressed_override = Some(deflate(&raw));
    let f = TFile::new(8, vec![TEntity::new(vec![s])]);
    with_file(&build(&f), |r| {
        let file = r.unwrap();
        let e = file.decode(file.sectors()[0]);
        assert!(
            matches!(
                e,
                Err(Error::Truncated {
                    what: "payload slot data",
                    ..
                })
            ),
            "{:?}",
            e.err()
        );
    });
}

#[test]
fn large_payload_grows_inflate_buffer() {
    // 4096 * 1 KB of Block data inflates far past the initial 64 KB buffer guess, so the
    // inflater must resume across several output buffer growths.
    let bricks: Vec<_> = (0..4096)
        .map(|bi| (bi, brick(bi as u16, 1 + bi % 512)))
        .collect();
    let expected_voxels: u64 = (0..4096u64).map(|bi| 1 + bi % 512).sum();
    let f = TFile::new(
        8,
        vec![TEntity::new(vec![TSector::new([-1, -1, -1], bricks)])],
    );
    with_file(&build(&f), |r| {
        let file = r.unwrap();
        let s = file.decode(file.sectors()[0]).unwrap();
        assert_eq!(s.keys.len(), 4096);
        assert_eq!(s.voxels, expected_voxels);
        assert_eq!(s.keys[0], [-16, -16, -16]);
        assert_eq!(s.keys[4095], [-1, -1, -1]);
        assert_eq!(&s.blocks[4095 * 512..], &brick(4095, 1 + 4095 % 512)[..]);
    });
}
