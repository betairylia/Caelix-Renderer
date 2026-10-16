//! Sky image: a lat-long (equirectangular) HDR image, `.exr` or `.hdr`, converted on the CPU to a
//! mipmapped cubemap that `caelix-vk` uploads once and the library samples as `CaelixRenderOptions::sky`.
//!
//! Orientation of the lat-long image (world Y up, +Z forward): the centre of the image faces +Z,
//! u (left to right) grows toward +X, the top row is straight up and the bottom row straight down.
//! A direction d maps to `u = 0.5 + atan2(d.x, d.z) / (2 pi) + rotation / 360` (wrapped to [0, 1)) and
//! `v = acos(d.y) / pi`; `--sky-rotation` turns the image around Y.
//!
//! Cube faces are in Vulkan order (+X, -X, +Y, -Y, +Z, -Z). The texel (x, y) of a face of N x N
//! texels (y = row from the top) has s = (x + 0.5) / N * 2 - 1, t = (y + 0.5) / N * 2 - 1 and the
//! direction (before normalization): +X (1, -t, -s), -X (-1, -t, s), +Y (s, 1, t), -Y (s, -1, -t),
//! +Z (s, -t, 1), -Z (-s, -t, -1). This is the inverse of the Vulkan spec's cube face selection, so
//! the library can sample the cube with the world direction.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context as _};

/// A lat-long image in linear radiance, row 0 = top of the image.
pub struct LatLong {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<[f32; 3]>,
}

/// A cubemap with every mip level down to 1 x 1.
pub struct Cubemap {
    /// Face width = height of mip 0 in texels.
    pub size: u32,
    /// `mips[m]`: the 6 faces of `(size >> m)^2` texels each, face-major (face 0 first), row-major
    /// inside a face, alpha 1.
    pub mips: Vec<Vec<[f32; 4]>>,
}

impl Cubemap {
    /// Bytes of every mip level as RGBA32F.
    pub fn byte_size(&self) -> u64 {
        self.mips.iter().map(|mip| mip.len() as u64 * 16).sum()
    }
}

/// Reads a lat-long sky image. The decoder follows the extension (case-insensitive): `.exr`
/// (first RGBA layer) or `.hdr` (Radiance RGBE). Negative or non-finite texels become 0.
pub fn load_latlong(path: &Path) -> anyhow::Result<LatLong> {
    let extension = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    let mut image = match extension.as_deref() {
        Some("exr") => load_exr(path)?,
        Some("hdr") => load_hdr(path)?,
        _ => bail!("{}: unsupported sky image type (expected a .exr or .hdr file)", path.display()),
    };
    if image.width == 0 || image.height == 0 {
        bail!("{}: the image is empty ({}x{})", path.display(), image.width, image.height);
    }
    for texel in &mut image.rgb {
        for c in texel {
            if !c.is_finite() || *c < 0.0 {
                *c = 0.0;
            }
        }
    }
    Ok(image)
}

fn load_exr(path: &Path) -> anyhow::Result<LatLong> {
    use exr::prelude::*;
    let image = read_first_rgba_layer_from_file(
        path,
        |resolution: Vec2<usize>, _: &RgbaChannels| LatLong {
            width: resolution.width() as u32,
            height: resolution.height() as u32,
            rgb: vec![[0.0; 3]; resolution.width() * resolution.height()],
        },
        |image: &mut LatLong, position: Vec2<usize>, (r, g, b, _a): (f32, f32, f32, f32)| {
            let index = position.y() * image.width as usize + position.x();
            image.rgb[index] = [r, g, b];
        },
    )
    .with_context(|| format!("cannot read the OpenEXR file {}", path.display()))?;
    Ok(image.layer_data.channel_data.pixels)
}

fn load_hdr(path: &Path) -> anyhow::Result<LatLong> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let decoder = image::codecs::hdr::HdrDecoder::new(BufReader::new(file))
        .with_context(|| format!("cannot read the Radiance HDR header of {}", path.display()))?;
    let image = image::DynamicImage::from_decoder(decoder)
        .with_context(|| format!("cannot decode the Radiance HDR file {}", path.display()))?
        .into_rgb32f();
    let (width, height) = image.dimensions();
    let rgb = image.pixels().map(|p| p.0).collect();
    Ok(LatLong { width, height, rgb })
}

/// The unit direction of the texel centre (s, t in [-1, 1]) of a cube face (Vulkan face order).
fn face_direction(face: usize, s: f32, t: f32) -> [f32; 3] {
    let d = match face {
        0 => [1.0, -t, -s],
        1 => [-1.0, -t, s],
        2 => [s, 1.0, t],
        3 => [s, -1.0, -t],
        4 => [s, -t, 1.0],
        5 => [-s, -t, -1.0],
        _ => unreachable!("a cube has 6 faces"),
    };
    let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    d.map(|c| c / length)
}

impl LatLong {
    /// Bilinear sample at (u, v) in [0, 1): u wraps, v is clamped. Texel centres are at
    /// ((x + 0.5) / width, (y + 0.5) / height).
    fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let (w, h) = (i64::from(self.width), i64::from(self.height));
        let x = u * w as f32 - 0.5;
        let y = v * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let x0 = x0 as i64;
        let y0 = y0 as i64;
        let xs = [x0.rem_euclid(w), (x0 + 1).rem_euclid(w)];
        let ys = [y0.clamp(0, h - 1), (y0 + 1).clamp(0, h - 1)];
        let texel = |x: i64, y: i64| self.rgb[(y * w + x) as usize];
        let mut out = [0.0f32; 3];
        for (row, wy) in [(ys[0], 1.0 - fy), (ys[1], fy)] {
            for (column, wx) in [(xs[0], 1.0 - fx), (xs[1], fx)] {
                let t = texel(column, row);
                for c in 0..3 {
                    out[c] += t[c] * wx * wy;
                }
            }
        }
        out
    }

    /// The radiance in the world direction `d` (unit length), with the image turned by
    /// `rotation_deg` around Y.
    fn radiance(&self, d: [f32; 3], rotation_deg: f32) -> [f32; 3] {
        let u = 0.5 + d[0].atan2(d[2]) / std::f32::consts::TAU + rotation_deg / 360.0;
        let u = u - u.floor();
        let v = d[1].clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
        self.sample(u, v)
    }
}

/// Converts a lat-long image into a cubemap of `face_size` (a power of two, at least 4) with every
/// mip level down to 1 x 1 (2 x 2 box filter per face). See the module comment for the orientation.
pub fn to_cubemap(latlong: &LatLong, face_size: u32, rotation_deg: f32) -> Cubemap {
    assert!(face_size >= 4 && face_size.is_power_of_two(), "face size {face_size} must be a power of two >= 4");
    let n = face_size as usize;
    let mut base = Vec::with_capacity(6 * n * n);
    for face in 0..6 {
        for y in 0..n {
            let t = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            for x in 0..n {
                let s = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let [r, g, b] = latlong.radiance(face_direction(face, s, t), rotation_deg);
                base.push([r, g, b, 1.0]);
            }
        }
    }
    let mip_levels = face_size.trailing_zeros() + 1;
    let mut mips = Vec::with_capacity(mip_levels as usize);
    mips.push(base);
    for level in 1..mip_levels as usize {
        let prev = &mips[level - 1];
        let pn = n >> (level - 1);
        let cn = n >> level;
        let mut mip = Vec::with_capacity(6 * cn * cn);
        for face in 0..6 {
            let face_base = face * pn * pn;
            for y in 0..cn {
                for x in 0..cn {
                    let mut sum = [0.0f32; 4];
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let t = prev[face_base + (2 * y + dy) * pn + 2 * x + dx];
                        for c in 0..4 {
                            sum[c] += t[c];
                        }
                    }
                    mip.push(sum.map(|c| c * 0.25));
                }
            }
        }
        mips.push(mip);
    }
    Cubemap { size: face_size, mips }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Vulkan spec's cube face selection: the face of the largest-magnitude axis and
    /// (sc / |ma|, tc / |ma|), the face coordinates in [-1, 1].
    fn vulkan_face(d: [f32; 3]) -> (usize, f32, f32) {
        let [rx, ry, rz] = d;
        let (face, sc, tc, ma) = if rx.abs() >= ry.abs() && rx.abs() >= rz.abs() {
            if rx >= 0.0 {
                (0, -rz, -ry, rx)
            } else {
                (1, rz, -ry, rx)
            }
        } else if ry.abs() >= rz.abs() {
            if ry >= 0.0 {
                (2, rx, rz, ry)
            } else {
                (3, rx, -rz, ry)
            }
        } else if rz >= 0.0 {
            (4, rx, -ry, rz)
        } else {
            (5, -rx, -ry, rz)
        };
        (face, sc / ma.abs(), tc / ma.abs())
    }

    #[test]
    fn face_table_round_trip() {
        let n = 8usize;
        for face in 0..6 {
            for (x, y) in [(0, 0), (7, 0), (0, 7), (7, 7), (3, 4), (5, 2)] {
                let s = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let t = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let (got_face, got_s, got_t) = vulkan_face(face_direction(face, s, t));
                assert_eq!(got_face, face, "texel ({x}, {y})");
                assert!((got_s - s).abs() < 1e-6 && (got_t - t).abs() < 1e-6, "face {face} texel ({x}, {y})");
            }
        }
    }

    #[test]
    fn constant_image_gives_constant_faces_and_mips() {
        let value = [0.25, 1.5, 3.0];
        let latlong = LatLong { width: 16, height: 8, rgb: vec![value; 16 * 8] };
        let cubemap = to_cubemap(&latlong, 16, 37.0);
        assert_eq!(cubemap.mips.len(), 5);
        for (level, mip) in cubemap.mips.iter().enumerate() {
            let side = 16usize >> level;
            assert_eq!(mip.len(), 6 * side * side);
            for texel in mip {
                for c in 0..3 {
                    assert!((texel[c] - value[c]).abs() < 1e-5, "mip {level}: {texel:?}");
                }
                assert_eq!(texel[3], 1.0);
            }
        }
    }

    /// A file in the temp directory, removed on drop.
    struct TempFile(std::path::PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn temp_file(name: &str) -> TempFile {
        TempFile(std::env::temp_dir().join(format!("caelix-sky-test-{}-{name}", std::process::id())))
    }

    /// Texel (x, y) of the 4x2 test images; one negative texel checks the clamp to 0.
    fn test_texel(x: usize, y: usize) -> [f32; 3] {
        if (x, y) == (3, 1) {
            [-1.0, 2.0, 0.5]
        } else {
            [x as f32 + 1.0, y as f32 * 4.0 + 0.5, 0.25]
        }
    }

    fn check_loaded(image: &LatLong) {
        assert_eq!((image.width, image.height), (4, 2));
        for y in 0..2 {
            for x in 0..4 {
                let expected = test_texel(x, y).map(|c| c.max(0.0));
                let got = image.rgb[y * 4 + x];
                for c in 0..3 {
                    // RGBE keeps 8 mantissa bits relative to the largest component.
                    assert!((got[c] - expected[c]).abs() <= 0.02 * expected[c].max(1.0), "({x}, {y}): {got:?}");
                }
            }
        }
    }

    #[test]
    fn reads_exr() {
        let file = temp_file("a.EXR");
        exr::prelude::write_rgb_file(&file.0, 4, 2, |x, y| {
            let [r, g, b] = test_texel(x, y);
            (r, g, b)
        })
        .unwrap();
        check_loaded(&load_latlong(&file.0).unwrap());
    }

    #[test]
    fn reads_hdr() {
        let file = temp_file("a.hdr");
        let pixels: Vec<image::Rgb<f32>> =
            (0..2).flat_map(|y| (0..4).map(move |x| image::Rgb(test_texel(x, y).map(|c| c.max(0.0))))).collect();
        let writer = std::io::BufWriter::new(File::create(&file.0).unwrap());
        image::codecs::hdr::HdrEncoder::new(writer).encode(&pixels, 4, 2).unwrap();
        check_loaded(&load_latlong(&file.0).unwrap());
    }

    #[test]
    fn rejects_other_extensions() {
        let error = load_latlong(Path::new("sky.png")).err().unwrap();
        assert!(error.to_string().contains("expected a .exr or .hdr file"));
    }

    #[test]
    fn image_centre_faces_plus_z() {
        // Every texel of column x holds x: the sample's value is the column coordinate.
        let width = 8u32;
        let rgb = (0..4).flat_map(|_| (0..width).map(|x| [x as f32; 3])).collect();
        let latlong = LatLong { width, height: 4, rgb };
        let column = |d: [f32; 3], rotation: f32| latlong.radiance(d, rotation)[0];
        // +Z sees the image centre (u = 0.5, between columns 3 and 4), +X a quarter turn right of it.
        assert!((column([0.0, 0.0, 1.0], 0.0) - 3.5).abs() < 1e-4);
        assert!((column([1.0, 0.0, 0.0], 0.0) - 5.5).abs() < 1e-4);
        // A rotation of 90 degrees moves what +X saw to +Z.
        assert!((column([0.0, 0.0, 1.0], 90.0) - 5.5).abs() < 1e-4);
    }
}
