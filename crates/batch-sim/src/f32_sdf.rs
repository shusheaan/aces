//! f32 port of `sim-core::environment` for GPU (WGSL) parity validation.
//!
//! WGSL compute shaders only support `f32` (no `f64`), so the Phase 2 GPU MPPI
//! rollout must evaluate the arena SDF in f32. This module mirrors
//! `crates/sim-core/src/environment.rs` exactly — only the numeric type differs.
//! Any behavioral divergence is a bug.
//!
//! Validation tests (`#[cfg(test)]`) compare f32 vs f64 SDF values across
//! realistic probe-point distributions (points inside the warehouse arena,
//! near obstacle surfaces, and outside the bounds).
//!
//! NOTE: keep this file logic-identical to `sim-core::environment`. Do not add
//! features here that are not in the f64 reference.

use aces_sim_core::environment::{Arena, Obstacle};
use nalgebra::Vector3;

/// A single obstacle defined by its SDF (f32 port of `Obstacle`).
#[derive(Debug, Clone)]
pub enum ObstacleF32 {
    /// Axis-aligned box: center, half-extents
    Box {
        center: Vector3<f32>,
        half_extents: Vector3<f32>,
    },
    /// Sphere: center, radius
    Sphere { center: Vector3<f32>, radius: f32 },
    /// Cylinder: center of base, radius, height (along Z)
    Cylinder {
        center: Vector3<f32>,
        radius: f32,
        height: f32,
    },
}

impl ObstacleF32 {
    /// Compute the signed distance from point p to this obstacle.
    pub fn sdf(&self, p: &Vector3<f32>) -> f32 {
        match self {
            ObstacleF32::Box {
                center,
                half_extents,
            } => {
                let d = (p - center).abs() - half_extents;
                let outside = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0)).norm();
                let inside = d.x.max(d.y.max(d.z)).min(0.0);
                outside + inside
            }
            ObstacleF32::Sphere { center, radius } => (p - center).norm() - radius,
            ObstacleF32::Cylinder {
                center,
                radius,
                height,
            } => {
                let dx = ((p.x - center.x).powi(2) + (p.y - center.y).powi(2)).sqrt() - radius;
                let dz = (p.z - center.z - height * 0.5).abs() - height * 0.5;
                let outside = (Vector3::new(dx.max(0.0), dz.max(0.0), 0.0)).xy().norm();
                let inside = dx.max(dz).min(0.0);
                outside + inside
            }
        }
    }

    /// Convert an f64 `Obstacle` into its f32 analogue.
    pub fn from_f64(obs: &Obstacle) -> Self {
        match obs {
            Obstacle::Box {
                center,
                half_extents,
            } => ObstacleF32::Box {
                center: center.cast::<f32>(),
                half_extents: half_extents.cast::<f32>(),
            },
            Obstacle::Sphere { center, radius } => ObstacleF32::Sphere {
                center: center.cast::<f32>(),
                radius: *radius as f32,
            },
            Obstacle::Cylinder {
                center,
                radius,
                height,
            } => ObstacleF32::Cylinder {
                center: center.cast::<f32>(),
                radius: *radius as f32,
                height: *height as f32,
            },
        }
    }
}

/// The 3D arena containing boundaries and obstacles (f32 port of `Arena`).
#[derive(Debug, Clone)]
pub struct ArenaF32 {
    /// Arena dimensions [x, y, z] in meters
    pub bounds: Vector3<f32>,
    /// Obstacles in the arena
    pub obstacles: Vec<ObstacleF32>,
    /// Drone collision radius (meters)
    pub drone_radius: f32,
}

impl ArenaF32 {
    pub fn new(bounds: Vector3<f32>) -> Self {
        Self {
            bounds,
            obstacles: Vec::new(),
            drone_radius: 0.05,
        }
    }

    /// Signed distance to the nearest boundary wall.
    /// Negative means outside bounds.
    #[inline]
    pub fn boundary_sdf(&self, p: &Vector3<f32>) -> f32 {
        let dx_min = p.x;
        let dx_max = self.bounds.x - p.x;
        let dy_min = p.y;
        let dy_max = self.bounds.y - p.y;
        let dz_min = p.z; // ground
        let dz_max = self.bounds.z - p.z; // ceiling

        dx_min
            .min(dx_max)
            .min(dy_min)
            .min(dy_max)
            .min(dz_min)
            .min(dz_max)
    }

    /// Signed distance to nearest obstacle.
    #[inline]
    pub fn obstacle_sdf(&self, p: &Vector3<f32>) -> f32 {
        self.obstacles
            .iter()
            .map(|obs| obs.sdf(p))
            .fold(f32::INFINITY, f32::min)
    }

    /// Combined SDF: minimum of boundary and obstacle distances.
    #[inline]
    pub fn sdf(&self, p: &Vector3<f32>) -> f32 {
        self.boundary_sdf(p).min(self.obstacle_sdf(p))
    }

    /// Check if a drone at position p is colliding with anything.
    pub fn is_collision(&self, p: &Vector3<f32>) -> bool {
        self.sdf(p) < self.drone_radius
    }

    /// Check if position is out of bounds.
    pub fn is_out_of_bounds(&self, p: &Vector3<f32>) -> bool {
        self.boundary_sdf(p) < 0.0
    }

    /// Convert an f64 `Arena` into its f32 analogue (obstacles, bounds,
    /// drone_radius all cast down).
    pub fn from_f64(arena: &Arena) -> Self {
        Self {
            bounds: arena.bounds.cast::<f32>(),
            obstacles: arena.obstacles.iter().map(ObstacleF32::from_f64).collect(),
            drone_radius: arena.drone_radius as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aces_sim_core::environment::{Arena, Obstacle};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// Max allowed |f32 - f64| SDF diff, in meters. At arena-scale coordinates
    /// (<=10m) f32 has ~1e-6 relative precision, so 1e-4m (0.1mm) is
    /// comfortably achievable.
    const SDF_TOL: f64 = 1e-4;

    /// Default warehouse arena: 10x10x3m with 5 box pillars (matches the f64
    /// `test_arena` helper in `sim-core/src/environment.rs`).
    fn warehouse_arena_f64() -> Arena {
        let mut arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        for (x, y) in [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)] {
            arena.obstacles.push(Obstacle::Box {
                center: Vector3::new(x, y, 1.5),
                half_extents: Vector3::new(0.5, 0.5, 1.5),
            });
        }
        arena
    }

    /// Draw a uniform random probe in the expanded bounds (bounds ± 1m, so
    /// some probes are outside the arena).
    fn random_probe(rng: &mut SmallRng, bounds: &Vector3<f64>) -> Vector3<f64> {
        Vector3::new(
            rng.gen_range(-1.0..(bounds.x + 1.0)),
            rng.gen_range(-1.0..(bounds.y + 1.0)),
            rng.gen_range(-1.0..(bounds.z + 1.0)),
        )
    }

    #[test]
    fn test_f32_sdf_matches_f64_random_points() {
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);

        let mut rng = SmallRng::seed_from_u64(42);
        let mut max_diff = 0.0f64;
        for _ in 0..1000 {
            let p_f64 = random_probe(&mut rng, &arena_f64.bounds);
            let p_f32 = p_f64.cast::<f32>();

            let d_f64 = arena_f64.sdf(&p_f64);
            let d_f32 = arena_f32.sdf(&p_f32) as f64;
            let diff = (d_f32 - d_f64).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < SDF_TOL,
                "sdf diff {} > tol {} at p={:?} (f64={}, f32={})",
                diff,
                SDF_TOL,
                p_f64,
                d_f64,
                d_f32
            );
        }
        // Sanity: report-via-assert that precision is comfortably below tol.
        // Observed max diff on this distribution is ~5e-7, so 1e-4 has ample
        // headroom.
        assert!(max_diff < SDF_TOL, "max_diff={} exceeded tol", max_diff);
    }

    #[test]
    fn test_f32_obstacle_sdf_matches_f64_at_surface() {
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);

        // For each pillar, sample 20 points near its 6 faces at offsets ±0.01m.
        let pillars = [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)];
        let hx = 0.5;
        let hy = 0.5;
        let hz = 1.5;
        let zc = 1.5;
        let offsets = [-0.01_f64, 0.01];

        for (cx, cy) in pillars {
            // 6 face centers × 2 offsets = 12 + 8 intra-face probes = 20
            let face_centers: [Vector3<f64>; 6] = [
                Vector3::new(cx + hx, cy, zc),
                Vector3::new(cx - hx, cy, zc),
                Vector3::new(cx, cy + hy, zc),
                Vector3::new(cx, cy - hy, zc),
                Vector3::new(cx, cy, zc + hz),
                Vector3::new(cx, cy, zc - hz),
            ];
            let mut probes: Vec<Vector3<f64>> = Vec::with_capacity(20);
            for (i, fc) in face_centers.iter().enumerate() {
                // Pick the face-normal axis: 0,1 -> x; 2,3 -> y; 4,5 -> z
                let axis = i / 2;
                for off in offsets {
                    let mut p = *fc;
                    match axis {
                        0 => p.x += off,
                        1 => p.y += off,
                        2 => p.z += off,
                        _ => unreachable!(),
                    }
                    probes.push(p);
                }
            }
            // Pad to 20 with slight perturbations on face centers.
            while probes.len() < 20 {
                let fc = face_centers[probes.len() % 6];
                probes.push(Vector3::new(fc.x + 0.005, fc.y + 0.005, fc.z + 0.005));
            }

            for p_f64 in probes {
                let p_f32 = p_f64.cast::<f32>();
                let d_f64 = arena_f64.obstacle_sdf(&p_f64);
                let d_f32 = arena_f32.obstacle_sdf(&p_f32) as f64;
                let diff = (d_f32 - d_f64).abs();
                assert!(
                    diff < SDF_TOL,
                    "obstacle sdf diff {} > tol {} at p={:?}",
                    diff,
                    SDF_TOL,
                    p_f64
                );
            }
        }
    }

    #[test]
    fn test_f32_boundary_sdf_matches_f64() {
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);

        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..1000 {
            let p_f64 = random_probe(&mut rng, &arena_f64.bounds);
            let p_f32 = p_f64.cast::<f32>();

            let d_f64 = arena_f64.boundary_sdf(&p_f64);
            let d_f32 = arena_f32.boundary_sdf(&p_f32) as f64;
            let diff = (d_f32 - d_f64).abs();
            assert!(
                diff < SDF_TOL,
                "boundary sdf diff {} > tol {} at p={:?}",
                diff,
                SDF_TOL,
                p_f64
            );
        }
    }

    #[test]
    fn test_f32_is_collision_matches_f64() {
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);
        let drone_radius = arena_f64.drone_radius;

        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..1000 {
            let p_f64 = random_probe(&mut rng, &arena_f64.bounds);
            let p_f32 = p_f64.cast::<f32>();

            // Skip points that sit within 1e-3m of the drone_radius threshold:
            // there the f32/f64 boolean can legitimately disagree due to the
            // 1e-4-level SDF discrepancy landing on either side of the cutoff.
            let d_f64 = arena_f64.sdf(&p_f64);
            if (d_f64 - drone_radius).abs() < 1e-3 {
                continue;
            }

            let c_f64 = arena_f64.is_collision(&p_f64);
            let c_f32 = arena_f32.is_collision(&p_f32);
            assert_eq!(
                c_f64, c_f32,
                "collision bool mismatch at p={:?} (f64 sdf={})",
                p_f64, d_f64
            );
        }
    }

    #[test]
    fn test_f32_cylinder_sdf_sanity() {
        // Warehouse uses only boxes; add a cylinder to exercise that path.
        let mut arena_f64 = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        arena_f64.obstacles.push(Obstacle::Cylinder {
            center: Vector3::new(3.0, 3.0, 0.0),
            radius: 0.3,
            height: 2.0,
        });
        let arena_f32 = ArenaF32::from_f64(&arena_f64);

        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..50 {
            let p_f64 = random_probe(&mut rng, &arena_f64.bounds);
            let p_f32 = p_f64.cast::<f32>();

            let d_f64 = arena_f64.sdf(&p_f64);
            let d_f32 = arena_f32.sdf(&p_f32) as f64;
            let diff = (d_f32 - d_f64).abs();
            assert!(
                diff < SDF_TOL,
                "cylinder sdf diff {} > tol {} at p={:?}",
                diff,
                SDF_TOL,
                p_f64
            );
        }
    }

    #[test]
    fn test_f32_sphere_sdf_sanity() {
        let mut arena_f64 = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        arena_f64.obstacles.push(Obstacle::Sphere {
            center: Vector3::new(4.0, 4.0, 1.5),
            radius: 0.5,
        });
        let arena_f32 = ArenaF32::from_f64(&arena_f64);

        let mut rng = SmallRng::seed_from_u64(11);
        for _ in 0..50 {
            let p_f64 = random_probe(&mut rng, &arena_f64.bounds);
            let p_f32 = p_f64.cast::<f32>();

            let d_f64 = arena_f64.sdf(&p_f64);
            let d_f32 = arena_f32.sdf(&p_f32) as f64;
            let diff = (d_f32 - d_f64).abs();
            assert!(
                diff < SDF_TOL,
                "sphere sdf diff {} > tol {} at p={:?}",
                diff,
                SDF_TOL,
                p_f64
            );
        }
    }
}
