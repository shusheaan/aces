use nalgebra::Vector3;

/// A single obstacle defined by its SDF.
#[derive(Debug, Clone)]
pub enum Obstacle {
    /// Axis-aligned box: center, half-extents
    Box {
        center: Vector3<f64>,
        half_extents: Vector3<f64>,
    },
    /// Sphere: center, radius
    Sphere { center: Vector3<f64>, radius: f64 },
    /// Cylinder: center of base, radius, height (along Z)
    Cylinder {
        center: Vector3<f64>,
        radius: f64,
        height: f64,
    },
}

impl Obstacle {
    /// Compute the signed distance from point p to this obstacle.
    pub fn sdf(&self, p: &Vector3<f64>) -> f64 {
        match self {
            Obstacle::Box {
                center,
                half_extents,
            } => {
                let d = (p - center).abs() - half_extents;
                let outside = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0)).norm();
                let inside = d.x.max(d.y.max(d.z)).min(0.0);
                outside + inside
            }
            Obstacle::Sphere { center, radius } => (p - center).norm() - radius,
            Obstacle::Cylinder {
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
}

/// The 3D arena containing boundaries and obstacles.
#[derive(Debug, Clone)]
pub struct Arena {
    /// Arena dimensions [x, y, z] in meters
    pub bounds: Vector3<f64>,
    /// Obstacles in the arena
    pub obstacles: Vec<Obstacle>,
    /// Drone collision radius (meters)
    pub drone_radius: f64,
}

impl Arena {
    pub fn new(bounds: Vector3<f64>) -> Self {
        Self {
            bounds,
            obstacles: Vec::new(),
            drone_radius: 0.05,
        }
    }

    /// Signed distance to the nearest boundary wall.
    /// Negative means outside bounds.
    pub fn boundary_sdf(&self, p: &Vector3<f64>) -> f64 {
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
    pub fn obstacle_sdf(&self, p: &Vector3<f64>) -> f64 {
        self.obstacles
            .iter()
            .map(|obs| obs.sdf(p))
            .fold(f64::INFINITY, f64::min)
    }

    /// Combined SDF: minimum of boundary and obstacle distances.
    pub fn sdf(&self, p: &Vector3<f64>) -> f64 {
        self.boundary_sdf(p).min(self.obstacle_sdf(p))
    }

    /// Check if a drone at position p is colliding with anything.
    pub fn is_collision(&self, p: &Vector3<f64>) -> bool {
        self.sdf(p) < self.drone_radius
    }

    /// Check if position is out of bounds.
    pub fn is_out_of_bounds(&self, p: &Vector3<f64>) -> bool {
        self.boundary_sdf(p) < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_sdf_inside() {
        let obs = Obstacle::Box {
            center: Vector3::new(5.0, 5.0, 1.5),
            half_extents: Vector3::new(0.5, 0.5, 1.5),
        };
        let p = Vector3::new(5.0, 5.0, 1.5); // center
        assert!(obs.sdf(&p) < 0.0);
    }

    #[test]
    fn test_box_sdf_outside() {
        let obs = Obstacle::Box {
            center: Vector3::new(5.0, 5.0, 1.5),
            half_extents: Vector3::new(0.5, 0.5, 1.5),
        };
        let p = Vector3::new(7.0, 5.0, 1.5); // far away
        assert!(obs.sdf(&p) > 0.0);
    }

    #[test]
    fn test_arena_boundary() {
        let arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        let inside = Vector3::new(5.0, 5.0, 1.5);
        let outside = Vector3::new(-1.0, 5.0, 1.5);

        assert!(arena.boundary_sdf(&inside) > 0.0);
        assert!(arena.boundary_sdf(&outside) < 0.0);
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
    fn test_warehouse_collision() {
        let arena = test_arena();
        // Center of a pillar
        let in_pillar = Vector3::new(5.0, 5.0, 1.5);
        assert!(arena.is_collision(&in_pillar));

        // Open space
        let free = Vector3::new(3.5, 3.5, 1.5);
        assert!(!arena.is_collision(&free));
    }
}
