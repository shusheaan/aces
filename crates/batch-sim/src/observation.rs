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

    /// Compare `quaternion_to_euler` (hand-rolled in batch-sim/observation.rs)
    /// with nalgebra's `UnitQuaternion::euler_angles()` (used in py-bridge).
    ///
    /// Both produce intrinsic ZYX / extrinsic XYZ Tait-Bryan angles.  They
    /// must agree to within 1e-6 for any quaternion that is **not** at
    /// gimbal-lock (|sin(pitch)| < 1).  At exact gimbal-lock (pitch = ±π/2)
    /// the two implementations choose different conventions for the
    /// degenerate roll/yaw split, so we skip those cases below.
    #[test]
    fn test_euler_convention_consistency() {
        use nalgebra::UnitQuaternion;
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

        // Helper: build a unit quaternion from roll, pitch, yaw (ZYX, intrinsic).
        let rpy_to_q = |r: f64, p: f64, y: f64| -> UnitQuaternion<f64> {
            UnitQuaternion::from_euler_angles(r, p, y)
        };

        // (roll_in, pitch_in, yaw_in, near_gimbal_lock)
        let cases: &[(f64, f64, f64, bool)] = &[
            (0.0, 0.0, 0.0, false),                   // identity
            (PI / 6.0, 0.0, 0.0, false),              // 30° roll
            (0.0, PI / 3.0, 0.0, false),              // 60° pitch
            (0.0, 0.0, PI / 4.0, false),              // 45° yaw
            (FRAC_PI_4, FRAC_PI_4, FRAC_PI_4, false), // combined non-singular
            // Near-vertical (pitch = 89°): technically not gimbal-lock, but
            // close enough that small quaternion errors may trigger different
            // branches — include to verify tolerance holds.
            (0.0, 89.0_f64.to_radians(), 0.0, false),
            // Exact gimbal lock: pitch = +π/2. The two implementations choose
            // arbitrary roll/yaw splits here; we flag them and skip the assert.
            (0.0, FRAC_PI_2, 0.0, true),
            // Exact gimbal lock: pitch = -π/2.
            (0.0, -FRAC_PI_2, 0.0, true),
        ];

        for &(roll_in, pitch_in, yaw_in, near_lock) in cases {
            let q = rpy_to_q(roll_in, pitch_in, yaw_in);

            // Path A: hand-rolled (batch-sim)
            let (roll_a, pitch_a, yaw_a) = quaternion_to_euler(&q);

            // Path B: nalgebra (py-bridge)
            let (roll_b, pitch_b, yaw_b) = q.euler_angles();

            if near_lock {
                // At gimbal-lock the roll/yaw split is arbitrary; only pitch
                // must agree (it is well-defined as ±π/2).
                assert!(
                    (pitch_a - pitch_b).abs() < 1e-6,
                    "gimbal-lock pitch mismatch: batch-sim={pitch_a} nalgebra={pitch_b} \
                     (input pitch={pitch_in})"
                );
                // Document that roll/yaw disagree — this is expected behaviour.
                // No assertion on roll_a/yaw_a vs roll_b/yaw_b here.
            } else {
                // Wrap angle difference into [-π, π] before comparing.
                let wrap = |d: f64| -> f64 {
                    let d = d % (2.0 * PI);
                    if d > PI {
                        d - 2.0 * PI
                    } else if d < -PI {
                        d + 2.0 * PI
                    } else {
                        d
                    }
                };
                let tol = 1e-6;
                assert!(
                    wrap(roll_a - roll_b).abs() < tol,
                    "roll mismatch (input r={roll_in:.3} p={pitch_in:.3} y={yaw_in:.3}): \
                     batch-sim={roll_a} nalgebra={roll_b}"
                );
                assert!(
                    wrap(pitch_a - pitch_b).abs() < tol,
                    "pitch mismatch (input r={roll_in:.3} p={pitch_in:.3} y={yaw_in:.3}): \
                     batch-sim={pitch_a} nalgebra={pitch_b}"
                );
                assert!(
                    wrap(yaw_a - yaw_b).abs() < tol,
                    "yaw mismatch (input r={roll_in:.3} p={pitch_in:.3} y={yaw_in:.3}): \
                     batch-sim={yaw_a} nalgebra={yaw_b}"
                );
            }
        }
    }
}
