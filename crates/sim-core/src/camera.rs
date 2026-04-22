use nalgebra::Vector3;
use rayon::prelude::*;

use crate::environment::Arena;

/// Camera intrinsic parameters (pinhole model).
#[derive(Debug, Clone)]
pub struct CameraParams {
    pub width: usize,
    pub height: usize,
    /// Horizontal field of view in radians.
    pub fov: f64,
    /// Maximum ray depth in meters.
    pub max_depth: f64,
    /// Render rate in Hz.
    pub render_hz: f64,
    // Derived
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl CameraParams {
    pub fn new(width: usize, height: usize, fov_deg: f64, max_depth: f64, render_hz: f64) -> Self {
        let fov = fov_deg.to_radians();
        let fx = width as f64 / (2.0 * (fov / 2.0).tan());
        let fy = fx; // square pixels
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        Self {
            width,
            height,
            fov,
            max_depth,
            render_hz,
            fx,
            fy,
            cx,
            cy,
        }
    }

    /// Default camera: 320x240, 90° FOV, 15m max depth, 30 Hz.
    pub fn default_fpv() -> Self {
        Self::new(320, 240, 90.0, 15.0, 30.0)
    }
}

/// A rendered depth frame from the camera.
#[derive(Debug, Clone)]
pub struct CameraFrame {
    /// Row-major depth values (width * height).
    pub depth: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// Simulation time when this frame was rendered.
    pub timestamp: f64,
}

/// Render a depth image via sphere tracing against the arena SDF.
///
/// The camera is at `origin` with attitude rotation matrix columns derived
/// from the drone's unit quaternion. Body frame: +X forward, +Y left, +Z up.
///
/// `opponent_pos` and `opponent_radius` add the opponent drone as a sphere
/// primitive to the SDF scene so it appears (and occludes) naturally.
pub fn render_depth(
    params: &CameraParams,
    arena: &Arena,
    origin: &Vector3<f64>,
    rotation: &nalgebra::Rotation3<f64>,
    opponent_pos: &Vector3<f64>,
    opponent_radius: f64,
    timestamp: f64,
) -> CameraFrame {
    let width = params.width;
    let height = params.height;
    let fx = params.fx;
    let fy = params.fy;
    let cx = params.cx;
    let cy = params.cy;
    let max_depth = params.max_depth;

    // Parallelize over rows
    let depth: Vec<f32> = (0..height)
        .into_par_iter()
        .flat_map(|v| {
            let mut row = Vec::with_capacity(width);
            for u in 0..width {
                // Ray direction in body frame: +X forward, +Y left, +Z up
                let dir_body =
                    Vector3::new(1.0, -(u as f64 - cx) / fx, -(v as f64 - cy) / fy).normalize();

                // Transform to world frame
                let dir_world = rotation * dir_body;

                let d = sphere_trace(
                    arena,
                    origin,
                    &dir_world,
                    max_depth,
                    opponent_pos,
                    opponent_radius,
                );
                row.push(d as f32);
            }
            row
        })
        .collect();

    CameraFrame {
        depth,
        width,
        height,
        timestamp,
    }
}

/// Sphere-trace a single ray against the arena SDF + opponent sphere.
fn sphere_trace(
    arena: &Arena,
    origin: &Vector3<f64>,
    direction: &Vector3<f64>,
    max_depth: f64,
    opponent_pos: &Vector3<f64>,
    opponent_radius: f64,
) -> f64 {
    let epsilon = 0.001; // 1mm hit threshold
    let mut t = 0.0;

    while t < max_depth {
        let p = origin + direction * t;

        // Combined SDF: arena geometry + opponent as sphere
        let d_arena = arena.sdf(&p);
        let d_opponent = (p - opponent_pos).norm() - opponent_radius;
        let d = d_arena.min(d_opponent);

        if d < epsilon {
            return t;
        }

        t += d.max(epsilon);
    }

    max_depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Obstacle;
    use nalgebra::{Rotation3, UnitQuaternion, Vector3};

    fn identity_rotation() -> Rotation3<f64> {
        UnitQuaternion::identity().to_rotation_matrix()
    }

    fn test_arena() -> Arena {
        let mut arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        for (x, y) in [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)] {
            arena.obstacles.push(Obstacle::Box {
                center: Vector3::new(x, y, 1.5),
                half_extents: Vector3::new(0.5, 0.5, 1.5),
            });
        }
        arena
    }

    #[test]
    fn test_center_pixel_is_forward() {
        let params = CameraParams::default_fpv();
        let cx = params.cx;
        let cy = params.cy;
        let fx = params.fx;
        let fy = params.fy;

        // Center pixel direction in body frame
        let dir = Vector3::new(1.0, -(cx - cx) / fx, -(cy - cy) / fy).normalize();

        // Should be purely +X (forward)
        assert!((dir.x - 1.0).abs() < 1e-10);
        assert!(dir.y.abs() < 1e-10);
        assert!(dir.z.abs() < 1e-10);
    }

    #[test]
    fn test_corner_pixels_at_fov_edges() {
        let params = CameraParams::default_fpv();
        let fx = params.fx;

        // Left edge pixel (u=0): direction should angle left at ~45°
        let dir = Vector3::new(1.0, -(0.0 - params.cx) / fx, 0.0).normalize();
        let angle = dir.y.atan2(dir.x);
        // Should be close to 45° (half of 90° FOV)
        assert!((angle - std::f64::consts::FRAC_PI_4).abs() < 0.01);
    }

    #[test]
    fn test_known_depth_to_obstacle() {
        let params = CameraParams::default_fpv();
        let arena = test_arena();

        // Place camera at (3.5, 5.0, 1.5) looking forward (+X).
        // The center pillar at (5.0, 5.0) has half-extent 0.5,
        // so its near face is at x=4.5. Distance = 4.5 - 3.5 = 1.0.
        let origin = Vector3::new(3.5, 5.0, 1.5);
        let rot = identity_rotation();
        let far_opponent = Vector3::new(100.0, 100.0, 100.0);

        let frame = render_depth(&params, &arena, &origin, &rot, &far_opponent, 0.05, 0.0);

        // Center pixel depth should be ~1.0 (pillar face at 1m away)
        let center_idx = (params.height / 2) * params.width + (params.width / 2);
        let depth = frame.depth[center_idx];
        assert!(
            (depth - 1.0).abs() < 0.05,
            "Expected ~1.0m depth to pillar, got {depth}"
        );
    }

    #[test]
    fn test_opponent_appears_in_depth() {
        let params = CameraParams::default_fpv();
        let arena = Arena::new(Vector3::new(20.0, 20.0, 10.0)); // empty arena

        let origin = Vector3::new(5.0, 10.0, 5.0);
        let rot = identity_rotation();
        let opponent = Vector3::new(8.0, 10.0, 5.0); // 3m ahead

        let frame = render_depth(&params, &arena, &origin, &rot, &opponent, 0.05, 0.0);

        let center_idx = (params.height / 2) * params.width + (params.width / 2);
        let depth = frame.depth[center_idx];
        // Should hit opponent at ~3.0m (minus radius)
        assert!(
            (depth - 3.0).abs() < 0.1,
            "Expected ~3.0m depth to opponent, got {depth}"
        );
    }

    #[test]
    fn test_occluded_opponent_not_visible() {
        let params = CameraParams::default_fpv();
        let arena = test_arena();

        // Camera at (3.0, 5.0, 1.5) looking forward (+X).
        // Center pillar at x=4.5..5.5. Opponent behind pillar at x=7.0.
        let origin = Vector3::new(3.0, 5.0, 1.5);
        let rot = identity_rotation();
        let opponent = Vector3::new(7.0, 5.0, 1.5);

        let frame = render_depth(&params, &arena, &origin, &rot, &opponent, 0.05, 0.0);

        let center_idx = (params.height / 2) * params.width + (params.width / 2);
        let depth = frame.depth[center_idx];
        // Should hit pillar (~1.5m), not opponent (4m)
        assert!(depth < 2.0, "Should hit pillar, not opponent. Got {depth}");
    }
}
