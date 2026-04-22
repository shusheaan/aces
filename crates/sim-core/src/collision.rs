use nalgebra::Vector3;

use crate::environment::Arena;

/// Result of a line-of-sight check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Visibility {
    /// Target is visible (no obstruction)
    Visible,
    /// Target is occluded by an obstacle
    Occluded,
}

/// Check line-of-sight between two points using sphere tracing.
///
/// Returns whether `target` is visible from `origin` through the arena.
pub fn check_line_of_sight(
    arena: &Arena,
    origin: &Vector3<f64>,
    target: &Vector3<f64>,
) -> Visibility {
    let direction = target - origin;
    let total_dist = direction.norm();

    if total_dist < 1e-12 {
        return Visibility::Visible;
    }

    let dir_normalized = direction / total_dist;
    let mut t = 0.0;
    let epsilon = 0.001; // 1mm tolerance

    while t < total_dist {
        let p = origin + dir_normalized * t;
        let d = arena.obstacle_sdf(&p);

        if d < epsilon {
            return Visibility::Occluded;
        }

        // Sphere tracing: safe to advance by SDF distance
        t += d.max(epsilon);
    }

    Visibility::Visible
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Obstacle;

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
    fn test_clear_line_of_sight() {
        let arena = test_arena();
        let a = Vector3::new(1.0, 1.0, 1.5);
        let b = Vector3::new(1.0, 3.0, 1.5);
        assert_eq!(check_line_of_sight(&arena, &a, &b), Visibility::Visible);
    }

    #[test]
    fn test_occluded_by_pillar() {
        let arena = test_arena();
        // Line passes through the center pillar at (5,5)
        let a = Vector3::new(3.0, 5.0, 1.5);
        let b = Vector3::new(7.0, 5.0, 1.5);
        assert_eq!(check_line_of_sight(&arena, &a, &b), Visibility::Occluded);
    }
}
