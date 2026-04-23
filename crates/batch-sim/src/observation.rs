use aces_sim_core::environment::Arena;
use aces_sim_core::state::DroneState;

/// Build a 21-dimensional observation vector for one drone.
///
/// Layout:
///   [0:3]   own velocity
///   [3:6]   own angular velocity
///   [6:9]   relative position to opponent
///   [9:12]  relative velocity
///   [12:15] own attitude (roll, pitch, yaw)
///   [15]    nearest obstacle distance (SDF)
///   [16]    lock-on progress (self → opponent)
///   [17]    being-locked progress (opponent → self)
///   [18]    opponent visible (0 or 1)
///   [19]    belief uncertainty (0 for MPPI-vs-MPPI, no PF)
///   [20]    time since last seen (0 for always-visible)
pub fn build_observation(
    self_state: &DroneState,
    opponent_state: &DroneState,
    arena: &Arena,
    lock_progress: f64,
    being_locked_progress: f64,
    opponent_visible: bool,
) -> [f64; 21] {
    let rel_pos = opponent_state.position - self_state.position;
    let rel_vel = opponent_state.velocity - self_state.velocity;

    // Euler angles from quaternion
    let (roll, pitch, yaw) = quaternion_to_euler(&self_state.attitude);

    let sdf = arena.sdf(&self_state.position);

    [
        self_state.velocity.x,
        self_state.velocity.y,
        self_state.velocity.z,
        self_state.angular_velocity.x,
        self_state.angular_velocity.y,
        self_state.angular_velocity.z,
        rel_pos.x,
        rel_pos.y,
        rel_pos.z,
        rel_vel.x,
        rel_vel.y,
        rel_vel.z,
        roll,
        pitch,
        yaw,
        sdf,
        lock_progress,
        being_locked_progress,
        if opponent_visible { 1.0 } else { 0.0 },
        0.0, // belief_uncertainty: 0 for MPPI-vs-MPPI (no particle filter)
        0.0, // time_since_last_seen: 0 for always-visible
    ]
}

/// Extract Euler angles (roll, pitch, yaw) from a unit quaternion.
fn quaternion_to_euler(q: &nalgebra::UnitQuaternion<f64>) -> (f64, f64, f64) {
    let q = q.quaternion();
    let (w, x, y, z) = (q.w, q.i, q.j, q.k);

    // Roll (x-axis rotation)
    let sinr_cosp = 2.0 * (w * x + y * z);
    let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    let roll = sinr_cosp.atan2(cosr_cosp);

    // Pitch (y-axis rotation)
    let sinp = 2.0 * (w * y - z * x);
    let pitch = if sinp.abs() >= 1.0 {
        std::f64::consts::FRAC_PI_2.copysign(sinp)
    } else {
        sinp.asin()
    };

    // Yaw (z-axis rotation)
    let siny_cosp = 2.0 * (w * z + x * y);
    let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    let yaw = siny_cosp.atan2(cosy_cosp);

    (roll, pitch, yaw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_observation_dimensions() {
        let arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        let a = DroneState::hover_at(Vector3::new(1.0, 1.0, 1.5));
        let b = DroneState::hover_at(Vector3::new(9.0, 9.0, 1.5));

        let obs = build_observation(&a, &b, &arena, 0.0, 0.0, true);
        assert_eq!(obs.len(), 21);
    }

    #[test]
    fn test_observation_relative_position() {
        let arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        let a = DroneState::hover_at(Vector3::new(1.0, 2.0, 1.5));
        let b = DroneState::hover_at(Vector3::new(4.0, 6.0, 1.5));

        let obs = build_observation(&a, &b, &arena, 0.0, 0.0, true);

        // rel_pos = b - a = (3, 4, 0)
        assert!((obs[6] - 3.0).abs() < 1e-10);
        assert!((obs[7] - 4.0).abs() < 1e-10);
        assert!((obs[8] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_euler_identity() {
        let q = nalgebra::UnitQuaternion::identity();
        let (roll, pitch, yaw) = quaternion_to_euler(&q);
        assert!((roll).abs() < 1e-10);
        assert!((pitch).abs() < 1e-10);
        assert!((yaw).abs() < 1e-10);
    }
}
