use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Full state of a quadrotor drone (13-dimensional).
///
/// State vector: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
#[derive(Debug, Clone)]
pub struct DroneState {
    /// Position in world frame (meters)
    pub position: Vector3<f64>,
    /// Velocity in world frame (m/s)
    pub velocity: Vector3<f64>,
    /// Attitude quaternion (body-to-world rotation)
    pub attitude: UnitQuaternion<f64>,
    /// Angular velocity in body frame (rad/s)
    pub angular_velocity: Vector3<f64>,
}

impl DroneState {
    pub fn new(
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        attitude: UnitQuaternion<f64>,
        angular_velocity: Vector3<f64>,
    ) -> Self {
        Self {
            position,
            velocity,
            attitude,
            angular_velocity,
        }
    }

    /// Create a hovering state at the given position.
    pub fn hover_at(position: Vector3<f64>) -> Self {
        Self {
            position,
            velocity: Vector3::zeros(),
            attitude: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        }
    }

    /// Forward direction unit vector in world frame.
    #[inline]
    pub fn forward(&self) -> Vector3<f64> {
        self.attitude * Vector3::x()
    }

    /// Distance to another drone.
    #[inline]
    pub fn distance_to(&self, other: &DroneState) -> f64 {
        (other.position - self.position).norm()
    }

    /// Angle between forward direction and vector to other drone (radians).
    #[inline]
    pub fn angle_to(&self, other: &DroneState) -> f64 {
        let to_other = other.position - self.position;
        let dist = to_other.norm();
        if dist < 1e-12 {
            return 0.0;
        }
        let dir = to_other / dist;
        let dot = self.forward().dot(&dir).clamp(-1.0, 1.0);
        dot.acos()
    }

    /// Convert to flat array [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz].
    pub fn to_array(&self) -> [f64; 13] {
        let q = self.attitude.quaternion();
        [
            self.position.x,
            self.position.y,
            self.position.z,
            self.velocity.x,
            self.velocity.y,
            self.velocity.z,
            q.w,
            q.i,
            q.j,
            q.k,
            self.angular_velocity.x,
            self.angular_velocity.y,
            self.angular_velocity.z,
        ]
    }

    /// Reconstruct from flat array [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz].
    pub fn from_array(arr: &[f64; 13]) -> Self {
        Self {
            position: Vector3::new(arr[0], arr[1], arr[2]),
            velocity: Vector3::new(arr[3], arr[4], arr[5]),
            attitude: UnitQuaternion::from_quaternion(Quaternion::new(
                arr[6], arr[7], arr[8], arr[9],
            )),
            angular_velocity: Vector3::new(arr[10], arr[11], arr[12]),
        }
    }
}

impl Default for DroneState {
    fn default() -> Self {
        Self::hover_at(Vector3::zeros())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_from_array_round_trip() {
        let state = DroneState::hover_at(Vector3::new(1.0, 2.0, 3.0));
        let arr = state.to_array();
        let reconstructed = DroneState::from_array(&arr);
        let arr2 = reconstructed.to_array();
        for i in 0..13 {
            assert!((arr[i] - arr2[i]).abs() < 1e-12, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_forward_rotated_90_yaw() {
        // Yaw 90° around Z: forward (+X body) should become +Y world
        let yaw_90 = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::z()),
            std::f64::consts::FRAC_PI_2,
        );
        let state = DroneState::new(Vector3::zeros(), Vector3::zeros(), yaw_90, Vector3::zeros());
        let fwd = state.forward();
        assert!((fwd.x).abs() < 1e-10, "fwd.x should be ~0, got {}", fwd.x);
        assert!(
            (fwd.y - 1.0).abs() < 1e-10,
            "fwd.y should be ~1, got {}",
            fwd.y
        );
        assert!((fwd.z).abs() < 1e-10, "fwd.z should be ~0, got {}", fwd.z);
    }

    #[test]
    fn test_angle_to_symmetry() {
        // Targets at symmetric positions (left vs right) should return equal angles
        let state = DroneState::hover_at(Vector3::new(0.0, 0.0, 1.0));

        let left = DroneState::hover_at(Vector3::new(1.0, 1.0, 1.0));
        let right = DroneState::hover_at(Vector3::new(1.0, -1.0, 1.0));

        let angle_left = state.angle_to(&left);
        let angle_right = state.angle_to(&right);

        assert!(
            (angle_left - angle_right).abs() < 1e-10,
            "left angle ({}) != right angle ({})",
            angle_left,
            angle_right
        );
    }
}
