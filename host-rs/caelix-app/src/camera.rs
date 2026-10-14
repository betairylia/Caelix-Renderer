//! Free camera with Unity's conventions.
//!
//! The voxel world is Y up and left-handed like Unity: +X right, +Y up, +Z forward.
//! Orientation is yaw and pitch in degrees, as Unity's `eulerAngles.y` and
//! `eulerAngles.x` (roll is 0):
//!
//! ```text
//! forward = Ry(yaw) * Rx(pitch) * (0, 0, 1) = (sin(yaw) cos(pitch), -sin(pitch), cos(yaw) cos(pitch))
//! right   = Ry(yaw) * (1, 0, 0)             = (cos(yaw), 0, -sin(yaw))
//! up      = Ry(yaw) * Rx(pitch) * (0, 1, 0) = (sin(yaw) sin(pitch), cos(pitch), cos(yaw) sin(pitch))
//! ```
//!
//! So yaw 90 looks along +X, and a positive pitch looks down.

use caelix_sys::CaelixCamera;

/// Pitch is clamped to this range, so the basis never degenerates.
const PITCH_LIMIT: f32 = 89.0;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: [f32; 3],
    /// Degrees, Unity `eulerAngles.y`.
    pub yaw: f32,
    /// Degrees, Unity `eulerAngles.x`. Positive looks down.
    pub pitch: f32,
    /// Vertical field of view in degrees.
    pub fov_y: f32,
}

impl Camera {
    /// A camera from a Unity pose. Angles above 180 (Unity reports -10 as 350) are
    /// wrapped to negative values; the pitch is clamped to +-89.
    pub fn from_pose(position: [f32; 3], pitch: f32, yaw: f32, fov_y: f32) -> Self {
        let mut camera = Self { position, yaw: wrap_degrees(yaw), pitch: wrap_degrees(pitch), fov_y };
        camera.pitch = camera.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
        camera
    }

    /// A camera from a position and a rotation quaternion (x, y, z, w) such as the
    /// player pose of a world file. Roll is dropped.
    pub fn from_quaternion(position: [f32; 3], q: [f32; 4], fov_y: f32) -> Self {
        let [x, y, z, w] = q;
        // forward = q * (0, 0, 1)
        let forward = [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)];
        let yaw = forward[0].atan2(forward[2]).to_degrees();
        let pitch = (-forward[1]).clamp(-1.0, 1.0).asin().to_degrees();
        Self::from_pose(position, pitch, yaw, fov_y)
    }

    /// right, up, forward.
    pub fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let (sy, cy) = self.yaw.to_radians().sin_cos();
        let (sp, cp) = self.pitch.to_radians().sin_cos();
        let forward = [sy * cp, -sp, cy * cp];
        let right = [cy, 0.0, -sy];
        let up = [sy * sp, cp, cy * sp];
        (right, up, forward)
    }

    /// Mouse look: `dx`, `dy` in degrees (right and down are positive).
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw = wrap_degrees(self.yaw + dx);
        self.pitch = (self.pitch + dy).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Moves by `forward` along the view direction, `right` along the right axis
    /// and `up` along world +Y.
    pub fn translate(&mut self, forward: f32, right: f32, up: f32) {
        let (r, _, f) = self.basis();
        for (i, p) in self.position.iter_mut().enumerate() {
            *p += f[i] * forward + r[i] * right;
        }
        self.position[1] += up;
    }

    pub fn to_caelix(self, aspect: f32, near_z: f32, far_z: f32) -> CaelixCamera {
        let (right, up, forward) = self.basis();
        CaelixCamera {
            position: self.position,
            pad0: 0.0,
            right,
            pad1: 0.0,
            up,
            pad2: 0.0,
            forward,
            pad3: 0.0,
            tan_half_fov_y: (self.fov_y.to_radians() * 0.5).tan(),
            aspect,
            near_z,
            far_z,
            projection: perspective_reverse_z(self.fov_y, aspect, near_z),
        }
    }
}

/// Column-major reverse-Z infinite perspective for the camera's view space
/// (+z forward): clip = (x / (tan * aspect), y / tan, near, z), so the device
/// depth is near / z (1 at `near`, 0 at infinity). `fov_y` is the vertical field
/// of view in degrees.
pub fn perspective_reverse_z(fov_y: f32, aspect: f32, near: f32) -> [f32; 16] {
    let tan_half = (fov_y.to_radians() * 0.5).tan();
    let mut m = [0.0; 16];
    m[0] = 1.0 / (tan_half * aspect);
    m[5] = 1.0 / tan_half;
    m[11] = 1.0; // column 2, row 3: clip.w = z
    m[14] = near; // column 3, row 2: clip.z = near
    m
}

/// Wraps degrees to (-180, 180].
fn wrap_degrees(degrees: f32) -> f32 {
    let wrapped = degrees.rem_euclid(360.0);
    if wrapped > 180.0 {
        wrapped - 360.0
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn basis_matches_unity() {
        // Identity: Unity's forward/right/up.
        let (r, u, f) = Camera::from_pose([0.0; 3], 0.0, 0.0, 60.0).basis();
        assert!(close(r, [1.0, 0.0, 0.0]) && close(u, [0.0, 1.0, 0.0]) && close(f, [0.0, 0.0, 1.0]));
        // Yaw 90 looks along +X; right is then -Z.
        let (r, _, f) = Camera::from_pose([0.0; 3], 0.0, 90.0, 60.0).basis();
        assert!(close(f, [1.0, 0.0, 0.0]) && close(r, [0.0, 0.0, -1.0]));
        // Positive pitch looks down.
        let (_, u, f) = Camera::from_pose([0.0; 3], 90.0 - 1e-3, 0.0, 60.0).basis();
        assert!(f[1] < -0.99 && u[2] > 0.99);
    }

    #[test]
    fn quaternion_round_trip() {
        // Unity Quaternion.Euler(20, 45, 0) = Ry(45) * Rx(20).
        let (hy, hx) = (45f32.to_radians() * 0.5, 20f32.to_radians() * 0.5);
        let q = [hx.sin() * hy.cos(), hy.sin() * hx.cos(), -hx.sin() * hy.sin(), hx.cos() * hy.cos()];
        let camera = Camera::from_quaternion([0.0; 3], q, 60.0);
        assert!((camera.yaw - 45.0).abs() < 1e-3 && (camera.pitch - 20.0).abs() < 1e-3);
    }

    #[test]
    fn reverse_z_depth() {
        let m = perspective_reverse_z(60.0, 1.0, 0.5);
        // clip.z / clip.w at view depth z (row r, column c = m[c * 4 + r]).
        let depth = |z: f32| (m[10] * z + m[14]) / (m[11] * z + m[15]);
        assert!((depth(0.5) - 1.0).abs() < 1e-6 && (depth(50.0) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn wraps_unity_angles() {
        let camera = Camera::from_pose([0.0; 3], 348.94, 55.6, 60.0);
        assert!((camera.pitch + 11.06).abs() < 1e-3);
    }
}
