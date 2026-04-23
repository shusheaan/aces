//! f32 port of `sim-core::dynamics` for GPU (WGSL) parity validation.
//!
//! WGSL compute shaders only support `f32` (no `f64`), so the Phase 2 GPU MPPI
//! rollout must match the f64 CPU integrator within acceptable tolerances.
//! This module mirrors `crates/sim-core/src/dynamics.rs` exactly — only the
//! numeric type differs. Any behavioral divergence is a bug.
//!
//! Validation tests (`#[cfg(test)]`) compare f32 vs f64 over 500-step rollouts
//! (a typical MPPI horizon: H=50 × S=10 substeps at dt_sim=0.001s).
//!
//! NOTE: keep this file logic-identical to `sim-core::dynamics`. Do not add
//! features here that are not in the f64 reference.

use aces_sim_core::state::DroneState;
use nalgebra::{Quaternion, UnitQuaternion, Vector3, Vector4};

/// Physical parameters for a quadrotor drone (f32 port of `DroneParams`).
#[derive(Debug, Clone)]
pub struct DroneParamsF32 {
    /// Mass (kg)
    pub mass: f32,
    /// Arm length — motor to center of gravity (m)
    pub arm_length: f32,
    /// Inertia tensor diagonal [Ixx, Iyy, Izz] (kg*m^2)
    pub inertia: Vector3<f32>,
    /// Maximum thrust per motor (N)
    pub max_thrust: f32,
    /// Torque coefficient (counter-torque / thrust ratio)
    pub torque_coeff: f32,
    /// Linear drag coefficient
    pub drag_coeff: f32,
    /// Gravitational acceleration (m/s^2)
    pub gravity: f32,
}

impl DroneParamsF32 {
    /// Crazyflie 2.1 reference parameters (f32).
    pub fn crazyflie() -> Self {
        Self {
            mass: 0.027,
            arm_length: 0.04,
            inertia: Vector3::new(1.4e-5, 1.4e-5, 2.17e-5),
            max_thrust: 0.15,
            torque_coeff: 0.005964,
            drag_coeff: 0.01,
            gravity: 9.81,
        }
    }

    /// Hover thrust per motor.
    pub fn hover_thrust(&self) -> f32 {
        self.mass * self.gravity / 4.0
    }

    /// Compute total thrust and torques from motor forces [f1, f2, f3, f4].
    /// Returns (total_thrust, torque_vector). X-configuration mixing matrix.
    pub fn motor_mixing(&self, motors: &Vector4<f32>) -> (f32, Vector3<f32>) {
        let d = self.arm_length;
        let c = self.torque_coeff;
        let s = std::f32::consts::FRAC_1_SQRT_2;

        let total_thrust = motors.sum();
        let tau_x = d * s * (motors[0] - motors[1] - motors[2] + motors[3]);
        let tau_y = d * s * (motors[0] + motors[1] - motors[2] - motors[3]);
        let tau_z = c * (motors[0] - motors[1] + motors[2] - motors[3]);

        (total_thrust, Vector3::new(tau_x, tau_y, tau_z))
    }
}

impl Default for DroneParamsF32 {
    fn default() -> Self {
        Self::crazyflie()
    }
}

/// Full state of a quadrotor drone (f32 port of `DroneState`).
#[derive(Debug, Clone)]
pub struct DroneStateF32 {
    /// Position in world frame (meters)
    pub position: Vector3<f32>,
    /// Velocity in world frame (m/s)
    pub velocity: Vector3<f32>,
    /// Attitude quaternion (body-to-world rotation)
    pub attitude: UnitQuaternion<f32>,
    /// Angular velocity in body frame (rad/s)
    pub angular_velocity: Vector3<f32>,
}

impl DroneStateF32 {
    /// Create a hovering state at the given position.
    pub fn hover_at(position: Vector3<f32>) -> Self {
        Self {
            position,
            velocity: Vector3::zeros(),
            attitude: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
        }
    }

    /// Convert from the f64 reference state (downcast).
    pub fn from_f64(state: &DroneState) -> Self {
        Self {
            position: state.position.cast::<f32>(),
            velocity: state.velocity.cast::<f32>(),
            attitude: state.attitude.cast::<f32>(),
            angular_velocity: state.angular_velocity.cast::<f32>(),
        }
    }
}

/// Compute state derivative for RK4 integration (f32).
///
/// Returns (velocity, acceleration, quaternion_derivative, angular_acceleration).
/// `external_force` is in the world frame (e.g. wind), in Newtons.
#[inline]
pub fn state_derivative_f32(
    state: &DroneStateF32,
    motors: &Vector4<f32>,
    params: &DroneParamsF32,
    external_force: &Vector3<f32>,
) -> (Vector3<f32>, Vector3<f32>, Quaternion<f32>, Vector3<f32>) {
    let (total_thrust, torque) = params.motor_mixing(motors);

    // Position derivative = velocity
    let p_dot = state.velocity;

    // Velocity derivative (world frame)
    // v_dot = [0,0,-g] + (1/m) * R(q) * [0,0,F_total] + (1/m) * F_drag + F_ext/m
    let thrust_body = Vector3::new(0.0, 0.0, total_thrust);
    let thrust_world = state.attitude * thrust_body;
    let gravity = Vector3::new(0.0, 0.0, -params.gravity * params.mass);
    let drag = -params.drag_coeff * state.velocity;
    let v_dot = (thrust_world + gravity + drag + external_force) / params.mass;

    // Quaternion derivative: q_dot = 0.5 * q * [0, wx, wy, wz]
    let w = state.angular_velocity;
    let omega_quat = Quaternion::new(0.0, w.x, w.y, w.z);
    let q = state.attitude.quaternion();
    let q_dot = q * omega_quat * 0.5;

    // Angular velocity derivative (body frame)
    // I * w_dot = tau - w x (I * w)
    let iw = Vector3::new(
        params.inertia.x * w.x,
        params.inertia.y * w.y,
        params.inertia.z * w.z,
    );
    let gyro = w.cross(&iw);
    let w_dot = Vector3::new(
        (torque.x - gyro.x) / params.inertia.x,
        (torque.y - gyro.y) / params.inertia.y,
        (torque.z - gyro.z) / params.inertia.z,
    );

    (p_dot, v_dot, q_dot, w_dot)
}

/// Advance drone state by one simulation step using RK4 integration (f32).
///
/// Motors are clamped to [0, max_thrust]. `external_force` is constant across
/// sub-steps (world-frame wind in Newtons). The quaternion is integrated via
/// the average of the four RK4 stage derivatives and renormalized by
/// `UnitQuaternion::from_quaternion` — explicit renormalization contains the
/// drift that f32 would otherwise accumulate.
pub fn step_rk4_f32(
    state: &DroneStateF32,
    motors: &Vector4<f32>,
    params: &DroneParamsF32,
    dt: f32,
    external_force: &Vector3<f32>,
) -> DroneStateF32 {
    // Clamp motor thrusts
    let motors_clamped = Vector4::new(
        motors[0].clamp(0.0, params.max_thrust),
        motors[1].clamp(0.0, params.max_thrust),
        motors[2].clamp(0.0, params.max_thrust),
        motors[3].clamp(0.0, params.max_thrust),
    );

    // RK4 integration (external force is constant across sub-steps)
    let (k1_p, k1_v, k1_q, k1_w) =
        state_derivative_f32(state, &motors_clamped, params, external_force);

    let s2 = DroneStateF32 {
        position: state.position + k1_p * dt * 0.5,
        velocity: state.velocity + k1_v * dt * 0.5,
        attitude: integrate_quaternion_f32(&state.attitude, &k1_q, dt * 0.5),
        angular_velocity: state.angular_velocity + k1_w * dt * 0.5,
    };
    let (k2_p, k2_v, k2_q, k2_w) =
        state_derivative_f32(&s2, &motors_clamped, params, external_force);

    let s3 = DroneStateF32 {
        position: state.position + k2_p * dt * 0.5,
        velocity: state.velocity + k2_v * dt * 0.5,
        attitude: integrate_quaternion_f32(&state.attitude, &k2_q, dt * 0.5),
        angular_velocity: state.angular_velocity + k2_w * dt * 0.5,
    };
    let (k3_p, k3_v, k3_q, k3_w) =
        state_derivative_f32(&s3, &motors_clamped, params, external_force);

    let s4 = DroneStateF32 {
        position: state.position + k3_p * dt,
        velocity: state.velocity + k3_v * dt,
        attitude: integrate_quaternion_f32(&state.attitude, &k3_q, dt),
        angular_velocity: state.angular_velocity + k3_w * dt,
    };
    let (k4_p, k4_v, k4_q, k4_w) =
        state_derivative_f32(&s4, &motors_clamped, params, external_force);

    // Combine
    let new_pos = state.position + (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * dt / 6.0;
    let new_vel = state.velocity + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * dt / 6.0;
    let new_w = state.angular_velocity + (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) * dt / 6.0;

    // Quaternion integration via average derivative
    let avg_q_dot = (k1_q + k2_q * 2.0 + k3_q * 2.0 + k4_q) * (1.0 / 6.0);
    let new_attitude = integrate_quaternion_f32(&state.attitude, &avg_q_dot, dt);

    let result = DroneStateF32 {
        position: new_pos,
        velocity: new_vel,
        attitude: new_attitude,
        angular_velocity: new_w,
    };

    // Invariant checks (debug builds only — zero cost in release).
    // Quaternion norm tolerance is 1e-3 (vs 1e-4 in the f64 reference) because
    // f32 drift is higher.
    debug_assert!(
        result.position.iter().all(|v| v.is_finite()),
        "NaN/Inf in position after RK4: {:?}",
        result.position
    );
    debug_assert!(
        result.velocity.iter().all(|v| v.is_finite()),
        "NaN/Inf in velocity after RK4: {:?}",
        result.velocity
    );
    debug_assert!(
        result.angular_velocity.iter().all(|v| v.is_finite()),
        "NaN/Inf in angular_velocity after RK4: {:?}",
        result.angular_velocity
    );
    debug_assert!(
        (result.attitude.quaternion().norm() - 1.0).abs() < 1e-3,
        "Quaternion norm drift after RK4: {}",
        result.attitude.quaternion().norm()
    );

    result
}

/// Integrate quaternion: q_new = normalize(q + q_dot * dt).
#[inline]
fn integrate_quaternion_f32(
    q: &UnitQuaternion<f32>,
    q_dot: &Quaternion<f32>,
    dt: f32,
) -> UnitQuaternion<f32> {
    let new_q = q.quaternion() + q_dot * dt;
    UnitQuaternion::from_quaternion(new_q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aces_sim_core::dynamics::{step_rk4, DroneParams};
    use aces_sim_core::state::DroneState;
    use nalgebra::{Vector3, Vector4};

    /// Thresholds from the Phase 2 GPU parity plan: after 500 f64-vs-f32 RK4
    /// steps at dt=0.001s, position must agree to <1mm and attitude to
    /// <0.01 rad.
    const POS_TOL_M: f64 = 1e-3;
    const ATT_TOL_RAD: f64 = 1e-2;

    /// Angle between two unit quaternions (radians, in [0, pi]).
    fn quat_angle_between(a: &UnitQuaternion<f64>, b: &UnitQuaternion<f64>) -> f64 {
        // Compute manually and handle the double-cover by taking |dot|: the
        // rotation between a and b is a * b^-1, whose angle is 2*acos(|dot|).
        let dot = a.quaternion().dot(b.quaternion()).abs().min(1.0);
        2.0 * dot.acos()
    }

    /// Run a 500-step rollout in both f64 and f32 and compare final states.
    fn run_and_compare(
        initial_f64: DroneState,
        motors_f64: Vector4<f64>,
        wind_f64: Vector3<f64>,
        steps: usize,
        dt: f64,
    ) -> (f64, f64) {
        let params_f64 = DroneParams::crazyflie();
        let params_f32 = DroneParamsF32::crazyflie();

        let motors_f32 = motors_f64.cast::<f32>();
        let wind_f32 = wind_f64.cast::<f32>();
        let dt_f32 = dt as f32;

        let mut state_f64 = initial_f64.clone();
        let mut state_f32 = DroneStateF32::from_f64(&initial_f64);

        for _ in 0..steps {
            state_f64 = step_rk4(&state_f64, &motors_f64, &params_f64, dt, &wind_f64);
            state_f32 = step_rk4_f32(&state_f32, &motors_f32, &params_f32, dt_f32, &wind_f32);
        }

        let pos_diff = (state_f32.position.cast::<f64>() - state_f64.position).norm();
        let att_diff = quat_angle_between(&state_f64.attitude, &state_f32.attitude.cast::<f64>());
        (pos_diff, att_diff)
    }

    #[test]
    fn test_f32_rk4_matches_f64_hover() {
        let params = DroneParams::crazyflie();
        let initial = DroneState::hover_at(Vector3::new(0.0, 0.0, 1.0));
        let hover = params.hover_thrust();
        let motors = Vector4::new(hover, hover, hover, hover);
        let wind = Vector3::zeros();

        let (pos_diff, att_diff) = run_and_compare(initial, motors, wind, 500, 0.001);

        assert!(
            pos_diff < POS_TOL_M,
            "hover: f32 vs f64 position diverged by {pos_diff} m after 500 steps (tol {POS_TOL_M})"
        );
        assert!(
            att_diff < ATT_TOL_RAD,
            "hover: f32 vs f64 attitude diverged by {att_diff} rad after 500 steps (tol {ATT_TOL_RAD})"
        );
    }

    #[test]
    fn test_f32_rk4_matches_f64_aggressive() {
        let params = DroneParams::crazyflie();
        let initial = DroneState::hover_at(Vector3::new(0.0, 0.0, 1.5));
        // Asymmetric motors — same pattern as sim-core's
        // test_quaternion_norm_preserved_rk4 so we exercise all three axes.
        let motors = Vector4::new(
            params.max_thrust,
            params.max_thrust * 0.3,
            params.max_thrust * 0.7,
            params.max_thrust * 0.5,
        );
        let wind = Vector3::zeros();

        let (pos_diff, att_diff) = run_and_compare(initial, motors, wind, 500, 0.001);

        assert!(
            pos_diff < POS_TOL_M,
            "aggressive: f32 vs f64 position diverged by {pos_diff} m after 500 steps (tol {POS_TOL_M})"
        );
        assert!(
            att_diff < ATT_TOL_RAD,
            "aggressive: f32 vs f64 attitude diverged by {att_diff} rad after 500 steps (tol {ATT_TOL_RAD})"
        );
    }

    #[test]
    fn test_f32_rk4_matches_f64_with_wind() {
        let params = DroneParams::crazyflie();
        let initial = DroneState::hover_at(Vector3::new(0.0, 0.0, 1.0));
        let hover = params.hover_thrust();
        let motors = Vector4::new(hover, hover, hover, hover);
        let wind = Vector3::new(0.1, 0.05, 0.0);

        let (pos_diff, att_diff) = run_and_compare(initial, motors, wind, 500, 0.001);

        assert!(
            pos_diff < POS_TOL_M,
            "wind: f32 vs f64 position diverged by {pos_diff} m after 500 steps (tol {POS_TOL_M})"
        );
        assert!(
            att_diff < ATT_TOL_RAD,
            "wind: f32 vs f64 attitude diverged by {att_diff} rad after 500 steps (tol {ATT_TOL_RAD})"
        );
    }

    #[test]
    fn test_f32_quaternion_norm_drift_500_steps() {
        // Same aggressive setup as Test B — run pure f32 for 500 steps and
        // confirm explicit renormalization in UnitQuaternion::from_quaternion
        // contains f32 drift to <1e-3.
        //
        // NOTE: On its own this test cannot detect a missing renormalization,
        // because UnitQuaternion::from_quaternion enforces unit norm by
        // construction. Full coverage of "renormalization actually happens"
        // relies on the three f64-vs-f32 parity tests above — if renormalization
        // were skipped, attitude drift would exceed ATT_TOL_RAD there.
        let params = DroneParamsF32::crazyflie();
        let mut state = DroneStateF32::hover_at(Vector3::new(0.0, 0.0, 1.5));
        let motors = Vector4::new(
            params.max_thrust,
            params.max_thrust * 0.3,
            params.max_thrust * 0.7,
            params.max_thrust * 0.5,
        );
        let wind = Vector3::zeros();

        for _ in 0..500 {
            state = step_rk4_f32(&state, &motors, &params, 0.001, &wind);
        }

        let q_norm = state.attitude.quaternion().norm();
        assert!(
            (q_norm - 1.0).abs() < 1e-3,
            "f32 quaternion norm drifted to {q_norm} after 500 steps (tol 1e-3)"
        );
    }
}
