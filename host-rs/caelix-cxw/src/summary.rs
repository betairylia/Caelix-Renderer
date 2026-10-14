//! Fields of a sector preview word (one `u32` per brick, section 4 of the format doc).
//!
//! Sub-brick index inside a brick: `sx | sy << 1 | sz << 2`, with `s = voxel >> 2` per axis
//! (the brick split into 2x2x2 sub-bricks of 4^3 voxels).

/// Bits 0..7: which of the 8 sub-bricks hold a non-zero block.
pub fn coarse_occupancy(word: u32) -> u8 {
    word as u8
}

/// Bits 8..15: which of the 8 sub-bricks hold an emissive block.
pub fn emission_mask(word: u32) -> u8 {
    (word >> 8) as u8
}

/// Bits 16..31: RGB565 average color of the brick's non-empty voxels (green LSB always 0).
pub fn rgb565(word: u32) -> u16 {
    (word >> 16) as u16
}
