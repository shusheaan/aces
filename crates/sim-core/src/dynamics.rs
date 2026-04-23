use nalgebra::{Quaternion, UnitQuaternion, Vector3, Vector4};

use crate::state::DroneState;

/// Physical parameters for a quadrotor drone.
#[derive(Debug, Clone)]
pub struct DroneParams {
    /// Mass (kg)
    pub mass: f64,
    /// Arm length — motor to center of gravity (m)
    pub arm_length: f64,
    /// Inertia tensor diagonal [Ixx, Iyy, Izz] (kg*m^2)
    pub inertia: Vector3<f64>,
    /// Maximum thrust per motor (N)
    pub max_thrust: f64,
    /// Torque coefficient (counter-torque / thrust ratio)
    pub torque_coeff: f64,
    /// Linear drag coefficient
    pub drag_coeff: f64,
    /// Gravitational acceleration (m/s^2)
    pub gravity: f64,
}

impl DroneParams {
    /// Crazyflie 2.1 reference parameters.
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
    pub fn hover_thrust(&self) -> f64 {
        self.mass * self.gravity / 4.0
    }

    /// Compute total thrust and torques from motor forces [f1, f2, f3, f4].
    /// Returns (total_thrust, torque_vector).
    /// Uses X-configuration mixing matrix.
    pub fn motor_mixing(&self, motors: &Vector4<f64>) -> (f64, Vector3<f64>) {
        let d = self.arm_length;
        let c = self.torque_coeff;
        let s = std::f64::consts::FRAC_1_SQRT_2;

        let total_thrust = motors.sum();
        let tau_x = d * s * (motors[0] - motors[1] - motors[2] + motors[3]);
        let tau_y = d * s * (motors[0] + motors[1] - motors[2] - motors[3]);
        let tau_z = c * (motors[0] - motors[1] + motors[2] - motors[3]);

        (total_thrust, Vector3::new(tau_x, tau_y, tau_z))
    }
}

impl Default for DroneParams {
    fn default() -> Self {
        Self::crazyflie()
    }
}

/// Compute state derivative for RK4 integration.
///
/// # Arguments
/// * `state` - Current drone state
/// * `motors` - Motor thrusts [f1, f2, f3, f4] (N), clamped to [0, max_thrust]
/// * `params` - Drone physical parameters
/// * `external_force` - External force in world frame (e.g. wind), in Newtons
///
/// # Returns
/// Tuple of (velocity, acceleration, quaternion_derivative, angular_acceleration)
#[inline]
pub fn state_derivative(
    state: &DroneState,
    motors: &Vector4<f64>,
    params: &DroneParams,
    external_force: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>, Quaternion<f64>, Vector3<f64>) {
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

/// Advance drone state by one simulation step using RK4 integration.
///
/// # Arguments
/// * `state` - Current state
/// * `motors` - Motor thrusts [f1,f2,f3,f4], will be clamped to [0, max_thrust]
/// * `params` - Drone parameters
/// * `dt` - Timestep (seconds)
/// * `external_force` - External force in world frame (e.g. wind), in Newtons
pub fn step_rk4(
    state: &DroneState,
    motors: &Vector4<f64>,
    params: &DroneParams,
    dt: f64,
    external_force: &Vector3<f64>,
) -> DroneState {
    // Clamp motor thrusts
    let motors_clamped = Vector4::new(
        motors[0].clamp(0.0, params.max_thrust),
        motors[1].clamp(0.0, params.max_thrust),
        motors[2].clamp(0.0, params.max_thrust),
        motors[3].clamp(0.0, params.max_thrust),
    );

    // RK4 integration (external force is constant across sub-steps)
    let (k1_p, k1_v, k1_q, k1_w) = state_derivative(state, &motors_clamped, params, external_force);

    let s2 = DroneState {
        position: state.position + k1_p * dt * 0.5,
        velocity: state.velocity + k1_v * dt * 0.5,
        attitude: integrate_quaternion(&state.attitude, &k1_q, dt * 0.5),
        angular_velocity: state.angular_velocity + k1_w * dt * 0.5,
    };
    let (k2_p, k2_v, k2_q, k2_w) = state_derivative(&s2, &motors_clamped, params, external_force);

    let s3 = DroneState {
        position: state.position + k2_p * dt * 0.5,
        velocity: state.velocity + k2_v * dt * 0.5,
        attitude: integrate_quaternion(&state.attitude, &k2_q, dt * 0.5),
        angular_velocity: state.angular_velocity + k2_w * dt * 0.5,
    };
    let (k3_p, k3_v, k3_q, k3_w) = state_derivative(&s3, &motors_clamped, params, external_force);

    let s4 = DroneState {
        position: state.position + k3_p * dt,
        velocity: state.velocity + k3_v * dt,
        attitude: integrate_quaternion(&state.attitude, &k3_q, dt),
        angular_velocity: state.angular_velocity + k3_w * dt,
    };
    let (k4_p, k4_v, k4_q, k4_w) = state_derivative(&s4, &motors_clamped, params, external_force);

    // Combine
    let new_pos = state.position + (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * dt / 6.0;
    let new_vel = state.velocity + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * dt / 6.0;
    let new_w = state.angular_velocity + (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) * dt / 6.0;

    // Quaternion integration via average derivative
    let avg_q_dot = (k1_q + k2_q * 2.0 + k3_q * 2.0 + k4_q) * (1.0 / 6.0);
    let new_attitude = integrate_quaternion(&state.attitude, &avg_q_dot, dt);

    let result = DroneState {
        position: new_pos,
        velocity: new_vel,
        attitude: new_attitude,
        angular_velocity: new_w,
    };

    // Invariant checks (debug builds only — zero cost in release)
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
        (result.attitude.quaternion().norm() - 1.0).abs() < 1e-4,
        "Quaternion norm drift after RK4: {}",
        result.attitude.quaternion().norm()
    );

    result
}

/// Integrate quaternion: q_new = normalize(q + q_dot * dt)
#[inline]
fn integrate_quaternion(
    q: &UnitQuaternion<f64>,
    q_dot: &Quaternion<f64>,
    dt: f64,
) -> UnitQuaternion<f64> {
    let new_q = q.quaternion() + q_dot * dt;
    UnitQuaternion::from_quaternion(new_q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hover_stationary() {
        let params = DroneParams::crazyflie();
        let state = DroneState::hover_at(Vector3::new(0.0, 0.0, 1.0));
        let hover = params.hover_thrust();
        let motors = Vector4::new(hover, hover, hover, hover);

        let no_wind = Vector3::zeros();
        let new_state = step_rk4(&state, &motors, &params, 0.001, &no_wind);

        assert_relative_eq!(new_state.position.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(new_state.position.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(new_state.position.z, 1.0, epsilon = 1e-10);
        assert_relative_eq!(new_state.velocity.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_freefall() {
        let params = DroneParams::crazyflie();
        let state = DroneState::hover_at(Vector3::new(0.0, 0.0, 10.0));
        let motors = Vector4::zeros();
        let no_wind = Vector3::zeros();

        // One step at 1ms
        let new_state = step_rk4(&state, &motors, &params, 0.001, &no_wind);

        // Should accelerate downward at ~g (drag causes tiny deviation)
        assert!(new_state.velocity.z < 0.0);
        assert_relative_eq!(
            new_state.velocity.z,
            -params.gravity * 0.001,
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_thrust_direction_identity() {
        // At identity attitude (no rotation), full thrust should produce pure +Z world acceleration
        let params = DroneParams::crazyflie();
        let state = DroneState::hover_at(Vector3::new(0.0, 0.0, 1.0));
        let full_thrust = Vector4::new(
            params.max_thrust,
            params.max_thrust,
            params.max_thrust,
            params.max_thrust,
        );
        let no_wind = Vector3::zeros();

        let (_, v_dot, _, _) = state_derivative(&state, &full_thrust, &params, &no_wind);

        // v_dot.x and v_dot.y should be ~0 (only drag, which is 0 at rest)
        assert!(v_dot.x.abs() < 1e-10, "thrust leaked into x: {}", v_dot.x);
        assert!(v_dot.y.abs() < 1e-10, "thrust leaked into y: {}", v_dot.y);
        // v_dot.z should be positive (thrust exceeds gravity)
        assert!(
            v_dot.z > 0.0,
            "full thrust should overcome gravity, got vz_dot={}",
            v_dot.z
        );
    }

    #[test]
    fn test_thrust_direction_pitched_forward() {
        // Pitch 90° forward (rotate around body Y so +Z_body aligns with +X_world)
        // Thrust should now push in +X world direction
        let params = DroneParams::crazyflie();
        let pitch_90 = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::y()),
            std::f64::consts::FRAC_PI_2,
        );
        let state = DroneState::new(
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::zeros(),
            pitch_90,
            Vector3::zeros(),
        );
        let full_thrust = Vector4::new(
            params.max_thrust,
            params.max_thrust,
            params.max_thrust,
            params.max_thrust,
        );
        let no_wind = Vector3::zeros();

        let (_, v_dot, _, _) = state_derivative(&state, &full_thrust, &params, &no_wind);

        // Thrust rotated 90° around Y: body +Z → world +X
        let total_thrust = 4.0 * params.max_thrust;
        let expected_ax = total_thrust / params.mass;
        assert_relative_eq!(v_dot.x, expected_ax, epsilon = 0.1);
        assert!(v_dot.y.abs() < 1e-10, "thrust leaked into y: {}", v_dot.y);
    }

    #[test]
    fn test_quaternion_norm_preserved_rk4() {
        // After 1000 aggressive RK4 steps, quaternion should still be unit
        let params = DroneParams::crazyflie();
        let mut state = DroneState::hover_at(Vector3::new(0.0, 0.0, 1.0));
        // Asymmetric motors to induce rotation
        let motors = Vector4::new(
            params.max_thrust,
            params.max_thrust * 0.3,
            params.max_thrust * 0.7,
            params.max_thrust * 0.5,
        );
        let no_wind = Vector3::zeros();

        for _ in 0..1000 {
            state = step_rk4(&state, &motors, &params, 0.001, &no_wind);
        }

        let q_norm = state.attitude.quaternion().norm();
        assert!(
            (q_norm - 1.0).abs() < 1e-6,
            "quaternion norm drifted to {} after 1000 steps",
            q_norm
        );
    }

    #[test]
    fn test_motor_mixing_roundtrip() {
        // Verify motor_mixing is consistent: known motors → known thrust/torque structure
        let params = DroneParams::crazyflie();

        // Equal motors → zero torque
        let equal = Vector4::new(0.1, 0.1, 0.1, 0.1);
        let (thrust, torque) = params.motor_mixing(&equal);
        assert_relative_eq!(thrust, 0.4, epsilon = 1e-12);
        assert_relative_eq!(torque.norm(), 0.0, epsilon = 1e-12);

        // Motor 0 only → should produce positive tau_x, positive tau_y, positive tau_z
        let m0_only = Vector4::new(0.1, 0.0, 0.0, 0.0);
        let (_, torque) = params.motor_mixing(&m0_only);
        assert!(torque.x > 0.0, "motor 0 should give positive tau_x");
        assert!(torque.y > 0.0, "motor 0 should give positive tau_y");
        assert!(torque.z > 0.0, "motor 0 should give positive tau_z");
    }

    #[test]
    fn test_hover_drifts_with_wind() {
        let params = DroneParams::crazyflie();
        let mut state = DroneState::hover_at(Vector3::new(0.0, 0.0, 1.0));
        let hover = params.hover_thrust();
        let motors = Vector4::new(hover, hover, hover, hover);
        let wind = Vector3::new(0.1, 0.0, 0.0); // constant wind in +x

        // Run 1000 steps (1 second)
        for _ in 0..1000 {
            state = step_rk4(&state, &motors, &params, 0.001, &wind);
        }

        // Drone should have drifted in +x direction
        assert!(
            state.position.x > 0.1,
            "drone should drift with wind, x = {}",
            state.position.x
        );
        // Z should stay approximately at 1.0 (hover compensates gravity)
        assert_relative_eq!(state.position.z, 1.0, epsilon = 0.1);
    }
}
