//! `--materials <path.json>`: a host material table for the scene (`caelix_scene_set_materials`).
//!
//! The file is a JSON array of entries:
//!
//! ```json
//! [{ "id": 1, "albedo": [0.9, 0.9, 0.9], "smoothness": 1.0, "metallic": 1.0,
//!    "emission": [0, 0, 0], "ior": 1.5, "extinction": 0.0, "transparent": false,
//!    "sss_model": 0, "sss_probability": 0.0, "sss_albedo": [0, 0, 0], "sss_anisotropy": 0.0,
//!    "sss_steps": 0, "sss_params": [], "texture_set": 0, "connect_group": 0 }]
//! ```
//!
//! Every field except `id` is optional (defaults: albedo 1, smoothness 0, metallic 0, emission 0,
//! ior 1.5, extinction 0, transparent false; the subsurface and texture keys 0, `sss_params` up to
//! 8 numbers with the missing ones 0). The library checks the ranges (`caelix.h`, "Materials"; no
//! kernel reads the subsurface and texture fields yet). Mapping to block codes (the block-ID encoding): an
//! opaque entry goes to code `0x8000 | id` (id 0..32767), a transparent entry to code `id` (id
//! 0..511). The rest of the table keeps the library's built-in content: the host reads every
//! entry with `caelix_scene_get_material`, replaces the listed ones and writes the whole table.
//! In a packed scene the same codes are written, which is rarely useful (packed codes carry their
//! colour); the glass palette of a packed scene is `caelix_scene_set_glass_palette`.

use std::path::Path;

use anyhow::{bail, Context as _};
use caelix_sys::*;
use serde_json::Value;

const ENTRIES: usize = 65536;

fn number(entry: &Value, key: &str, default: f32) -> anyhow::Result<f32> {
    match entry.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .map(|v| v as f32)
            .with_context(|| format!("\"{key}\" is not a number")),
    }
}

fn unsigned(entry: &Value, key: &str, default: u32) -> anyhow::Result<u32> {
    match entry.get(key) {
        None => Ok(default),
        Some(value) => {
            let v = value.as_u64().with_context(|| format!("\"{key}\" is not an unsigned integer"))?;
            u32::try_from(v).with_context(|| format!("\"{key}\" is above {}", u32::MAX))
        }
    }
}

/// `sss_params`: 0..8 numbers, the missing ones 0.
fn params8(entry: &Value, key: &str) -> anyhow::Result<[f32; 8]> {
    let mut out = [0.0f32; 8];
    if let Some(value) = entry.get(key) {
        let items = value.as_array().with_context(|| format!("\"{key}\" is not an array"))?;
        if items.len() > 8 {
            bail!("\"{key}\" holds {} numbers, at most 8", items.len());
        }
        for (slot, item) in out.iter_mut().zip(items) {
            *slot = item.as_f64().with_context(|| format!("\"{key}\" holds a non-number"))? as f32;
        }
    }
    Ok(out)
}

fn vec3(entry: &Value, key: &str, default: [f32; 3]) -> anyhow::Result<[f32; 3]> {
    match entry.get(key) {
        None => Ok(default),
        Some(value) => {
            let items = value.as_array().with_context(|| format!("\"{key}\" is not an array"))?;
            if items.len() != 3 {
                bail!("\"{key}\" needs 3 numbers, has {}", items.len());
            }
            let mut out = [0.0f32; 3];
            for (slot, item) in out.iter_mut().zip(items) {
                *slot = item.as_f64().with_context(|| format!("\"{key}\" holds a non-number"))? as f32;
            }
            Ok(out)
        }
    }
}

/// The block code and the material of one JSON entry.
fn parse_entry(entry: &Value) -> anyhow::Result<(usize, CaelixMaterial)> {
    let id = entry.get("id").and_then(Value::as_u64).context("\"id\" is missing or not an unsigned integer")?;
    let transparent = match entry.get("transparent") {
        None => false,
        Some(value) => value.as_bool().context("\"transparent\" is not a boolean")?,
    };
    let code = if transparent {
        if id > 0x1FF {
            bail!("transparent id {id} is above 511");
        }
        id as usize
    } else {
        if id > 0x7FFF {
            bail!("opaque id {id} is above 32767");
        }
        0x8000 | id as usize
    };
    let material = CaelixMaterial {
        albedo: vec3(entry, "albedo", [1.0; 3])?,
        smoothness: number(entry, "smoothness", 0.0)?,
        emission: vec3(entry, "emission", [0.0; 3])?,
        metallic: number(entry, "metallic", 0.0)?,
        ior: number(entry, "ior", 1.5)?,
        extinction: number(entry, "extinction", 0.0)?,
        sss_model: unsigned(entry, "sss_model", 0)?,
        sss_probability: number(entry, "sss_probability", 0.0)?,
        sss_albedo: vec3(entry, "sss_albedo", [0.0; 3])?,
        sss_anisotropy: number(entry, "sss_anisotropy", 0.0)?,
        sss_steps: unsigned(entry, "sss_steps", 0)?,
        sss_params: params8(entry, "sss_params")?,
        texture_set: unsigned(entry, "texture_set", 0)?,
        connect_group: unsigned(entry, "connect_group", 0)?,
        reserved: 0,
    };
    Ok((code, material))
}

/// Reads `path` and writes the resulting table into `scene`.
pub fn apply(scene: *mut CaelixScene, path: &Path) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let json: Value = serde_json::from_str(&text).with_context(|| format!("{} is not valid JSON", path.display()))?;
    let entries = json.as_array().with_context(|| format!("{}: the top level is not an array", path.display()))?;

    let empty = CaelixMaterial {
        albedo: [0.0; 3],
        smoothness: 0.0,
        emission: [0.0; 3],
        metallic: 0.0,
        ior: 0.0,
        extinction: 0.0,
        sss_model: 0,
        sss_probability: 0.0,
        sss_albedo: [0.0; 3],
        sss_anisotropy: 0.0,
        sss_steps: 0,
        sss_params: [0.0; 8],
        texture_set: 0,
        connect_group: 0,
        reserved: 0,
    };
    let mut table = vec![empty; ENTRIES];
    for (code, slot) in table.iter_mut().enumerate() {
        // SAFETY: the scene is live; `slot` is a valid CaelixMaterial for the call.
        let result = unsafe { caelix_scene_get_material(scene, code as u32, slot) };
        if result != CaelixResult_CAELIX_OK {
            bail!("caelix_scene_get_material({code:#06x}) failed: CaelixResult {result}");
        }
    }
    for (index, entry) in entries.iter().enumerate() {
        let (code, material) = parse_entry(entry).with_context(|| format!("{}: entry {index}", path.display()))?;
        table[code] = material;
    }
    // SAFETY: the scene is live; `table` holds ENTRIES materials for the call.
    let result = unsafe { caelix_scene_set_materials(scene, table.as_ptr(), ENTRIES as u32) };
    if result != CaelixResult_CAELIX_OK {
        bail!("caelix_scene_set_materials failed: CaelixResult {result} (see the library log)");
    }
    log::info!("materials: {} entries from {}", entries.len(), path.display());
    Ok(())
}
