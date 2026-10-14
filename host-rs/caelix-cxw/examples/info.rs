//! Prints the index of a `.cxw` file, then decodes every sector and prints totals and timing.
//!
//! `cargo run -p caelix-cxw --release --example info -- <file.cxw>`

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use caelix_cxw::CxwFile;

fn kind(flags: u8) -> &'static str {
    match flags & 3 {
        2 => "static",
        3 => "static-collider",
        1 => "dynamic",
        _ => "dynamic-no-body",
    }
}

fn main() {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: info <file.cxw>");
        std::process::exit(2);
    };

    let open_start = Instant::now();
    let file = match CxwFile::open(&path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let open_time = open_start.elapsed();

    let world = file.world();
    println!("file:     {}", path.to_string_lossy());
    println!("version:  {}", world.version);
    match world.player_pose {
        Some(p) => println!(
            "player:   position {:?} rotation {:?}",
            p.position, p.rotation
        ),
        None => println!("player:   (no pose)"),
    }
    println!("entities: {}", world.entities.len());
    for (i, e) in world.entities.iter().enumerate() {
        println!(
            "  [{i}] {} flags {} ({}){} sectors {} position {:?}",
            e.guid_string(),
            e.flags,
            kind(e.flags),
            if e.protected { " protected" } else { "" },
            e.sector_count,
            e.position,
        );
    }
    println!("index parsed in {:.1} ms", open_time.as_secs_f64() * 1e3);

    let sectors = AtomicU64::new(0);
    let bricks = AtomicU64::new(0);
    let voxels = AtomicU64::new(0);
    let block_bytes = AtomicU64::new(0);
    let start = Instant::now();
    let result = file.decode_all(|s| {
        sectors.fetch_add(1, Ordering::Relaxed);
        bricks.fetch_add(s.keys.len() as u64, Ordering::Relaxed);
        voxels.fetch_add(s.voxels, Ordering::Relaxed);
        block_bytes.fetch_add(s.blocks.len() as u64 * 2, Ordering::Relaxed);
    });
    let elapsed = start.elapsed().as_secs_f64();
    if let Err(e) = result {
        eprintln!("decode error: {e}");
        std::process::exit(1);
    }

    let bricks = bricks.into_inner();
    println!("decode_all ({} threads):", rayon::current_num_threads());
    println!("  sectors      {}", sectors.into_inner());
    println!("  bricks       {bricks}");
    println!("  voxels       {}", voxels.into_inner());
    println!(
        "  block bytes  {} ({:.1} MiB)",
        block_bytes.load(Ordering::Relaxed),
        block_bytes.load(Ordering::Relaxed) as f64 / (1 << 20) as f64
    );
    println!("  wall time    {:.3} s", elapsed);
    println!("  throughput   {:.0} bricks/s", bricks as f64 / elapsed);
}
