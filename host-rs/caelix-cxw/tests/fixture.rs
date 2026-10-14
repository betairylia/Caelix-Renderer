//! Checks against `Caelix/Tests/Editor/Fixtures/packed-materials.cxw` (version 6), with the
//! numbers of the Python reader in `ScenesPlayground/src/cxw` as the oracle.
//!
//! Set `CAELIX_FIXTURE_DIR` to the fixture directory if the repos are not side by side.

use std::path::PathBuf;
use std::sync::Mutex;

use caelix_cxw::{summary, CxwFile, DecodedSector, ENTITY_FLAG_STATIC};

const FIXTURE: &str = "packed-materials.cxw";

fn fixture() -> Option<CxwFile> {
    let dir = std::env::var_os("CAELIX_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../Caelix/Tests/Editor/Fixtures"
            ))
        });
    let path = dir.join(FIXTURE);
    if !path.is_file() {
        eprintln!(
            "SKIPPED: fixture {} not found (set CAELIX_FIXTURE_DIR)",
            path.display()
        );
        return None;
    }
    Some(CxwFile::open(&path).expect("fixture opens"))
}

fn decode_all_sorted(file: &CxwFile) -> Vec<DecodedSector> {
    let out = Mutex::new(Vec::new());
    file.decode_all(|s| out.lock().unwrap().push(s))
        .expect("decode_all");
    let mut out = out.into_inner().unwrap();
    out.sort_by_key(|s| (s.entity, s.coord));
    out
}

#[test]
fn header_and_world() {
    let Some(file) = fixture() else { return };
    let world = file.world();
    assert_eq!(world.version, 6);
    assert!(world.player_pose.is_none(), "version 6 has no world record");
    assert_eq!(world.entities.len(), 1);
    let e = &world.entities[0];
    assert_eq!(e.guid_string(), "324814f8-b552-4c3b-92cd-efd4f360f3b0");
    assert_eq!(
        e.guid,
        [
            0xf8, 0x14, 0x48, 0x32, 0x52, 0xb5, 0x3b, 0x4c, 0x92, 0xcd, 0xef, 0xd4, 0xf3, 0x60,
            0xf3, 0xb0
        ]
    );
    assert_eq!(
        e.guid_lanes(),
        [0x3248_14f8, 0x4c3b_b552, 0xd4ef_cd92, 0xb0f3_60f3]
    );
    assert_eq!(e.flags, ENTITY_FLAG_STATIC);
    assert!(e.is_static() && !e.has_body());
    assert!(e.protected);
    assert_eq!(e.position, [0.0; 3]);
    assert_eq!(e.rotation, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(e.velocity, [0.0; 6]);
    assert_eq!(e.sector_count, 8);
}

#[test]
fn sector_count() {
    let Some(file) = fixture() else { return };
    assert_eq!(file.sectors().len(), 8);
    assert!(file.sectors().iter().all(|s| s.entity == 0));
    let mut coords: Vec<_> = file.sectors().iter().map(|s| s.coord).collect();
    coords.sort();
    let mut expected = Vec::new();
    for x in -1..=0 {
        for y in -1..=0 {
            for z in -1..=0 {
                expected.push([x, y, z]);
            }
        }
    }
    assert_eq!(coords, expected);
}

#[test]
fn totals() {
    let Some(file) = fixture() else { return };
    let sectors = decode_all_sorted(&file);
    assert_eq!(sectors.len(), 8);
    let bricks: usize = sectors.iter().map(|s| s.keys.len()).sum();
    let voxels: u64 = sectors.iter().map(|s| s.voxels).sum();
    assert_eq!(bricks, 243);
    assert_eq!(voxels, 49_284);
    for s in &sectors {
        assert_eq!(s.blocks.len(), s.keys.len() * 512);
        assert_eq!(
            s.voxels,
            s.blocks.iter().filter(|&&v| v != 0).count() as u64
        );
    }
}

#[test]
fn brick_key_and_voxel_bounds() {
    let Some(file) = fixture() else { return };
    let mut key_min = [i32::MAX; 3];
    let mut key_max = [i32::MIN; 3];
    let mut vox_min = [i32::MAX; 3];
    let mut vox_max = [i32::MIN; 3];
    for s in decode_all_sorted(&file) {
        for (i, key) in s.keys.iter().enumerate() {
            // A key belongs to its sector: floor(key / 16) == coord.
            for a in 0..3 {
                assert_eq!(key[a].div_euclid(16), s.coord[a]);
                key_min[a] = key_min[a].min(key[a]);
                key_max[a] = key_max[a].max(key[a]);
            }
            for (vi, &v) in s.blocks[i * 512..(i + 1) * 512].iter().enumerate() {
                if v == 0 {
                    continue;
                }
                let local = [vi as i32 & 7, (vi as i32 >> 3) & 7, vi as i32 >> 6];
                for a in 0..3 {
                    let p = key[a] * 8 + local[a];
                    vox_min[a] = vox_min[a].min(p);
                    vox_max[a] = vox_max[a].max(p);
                }
            }
        }
    }
    assert_eq!((key_min, key_max), ([-5, -5, -1], [3, 3, 1]));
    // The voxel extent of those bricks.
    assert_eq!(key_min.map(|k| k * 8), [-40, -40, -8]);
    assert_eq!(key_max.map(|k| k * 8 + 7), [31, 31, 15]);
    assert_eq!((vox_min, vox_max), ([-33, -33, -2], [30, 30, 10]));
}

#[test]
fn summaries_match_bricks() {
    let Some(file) = fixture() else { return };
    for s in decode_all_sorted(&file) {
        assert_eq!(s.summaries.len(), s.keys.len());
        for (i, &word) in s.summaries.iter().enumerate() {
            let occupancy = summary::coarse_occupancy(word);
            assert_ne!(occupancy, 0, "sector {:?} brick {:?}", s.coord, s.keys[i]);
            let mut expected = 0u8;
            for (vi, &v) in s.blocks[i * 512..(i + 1) * 512].iter().enumerate() {
                if v != 0 {
                    let (x, y, z) = (vi & 7, (vi >> 3) & 7, vi >> 6);
                    expected |= 1 << ((x >> 2) | ((y >> 2) << 1) | ((z >> 2) << 2));
                }
            }
            assert_eq!(
                occupancy, expected,
                "sector {:?} brick {:?}",
                s.coord, s.keys[i]
            );
            assert_eq!(summary::emission_mask(word) & !occupancy, 0);
            assert_eq!(summary::rgb565(word) & 0x20, 0, "green LSB is always 0");
        }
    }
}

#[test]
fn decode_matches_decode_all() {
    let Some(file) = fixture() else { return };
    let all = decode_all_sorted(&file);
    let mut single: Vec<_> = file
        .sectors()
        .iter()
        .map(|&s| file.decode(s).unwrap())
        .collect();
    single.sort_by_key(|s| (s.entity, s.coord));
    assert_eq!(single, all);
    // And a second parallel run gives the same result.
    assert_eq!(decode_all_sorted(&file), all);
}
