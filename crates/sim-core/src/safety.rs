use nalgebra::{Vector3, Vector4};

use crate::dynamics::DroneParams;
use crate::environment::Arena;
use crate::state::DroneState;

/// Safety violation severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Ok,
    Warning,
    Critical,
}

/// Individual safety violation.
#[derive(Debug, Clone)]
pub struct Violation {
    pub kind: ViolationKind,
    pub severity: Severity,
    pub value: f64,
    pub limit: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationKind {
    SpeedExceeded,
    TiltExceeded,
    AngularRateExceeded,
    GeofenceBreach,
    NaN,
}

/// Result of a safety check.
#[derive(Debug, Clone)]
pub struct SafetyStatus {
    pub severity: Severity,
    pub violations: Vec<Violation>,
}

impl SafetyStatus {
    pub fn ok() -> Self {
        Self {
            severity: Severity::Ok,
            violations: Vec::new(),
        }
    }

    pub fn is_ok(&self) -> bool {
        self.severity == Severity::Ok
    }

    pub fn is_critical(&self) -> bool {
        self.severity == Severity::Critical
    }
}

/// Safety envelope: detects violations, NaN → hover as only active intervention.
#[derive(Debug, Clone)]
pub struct SafetyEnvelope {
    /// Maximum speed before warning (m/s)
    pub max_velocity: f64,
    /// Maximum tilt angle before warning (radians)
    pub max_tilt: f64,
    /// Maximum angular rate before warning (rad/s)
    pub max_angular_rate: f64,
    /// Minimum distance to arena boundary before warning (meters)
    pub geofence_margin: f64,
}

impl SafetyEnvelope {
    /// Default limits based on Crazyflie capabilities.
    pub fn crazyflie() -> Self {
        Self {
            max_velocity: 5.0,
            max_tilt: std::f64::consts::FRAC_PI_3, // 60 degrees
            max_angular_rate: 20.0,                // ~1145 deg/s
            geofence_margin: 0.2,                  // 20cm from walls
        }
    }

    /// Check drone state against safety limits.
    pub fn check(&self, state: &DroneState, arena: &Arena) -> SafetyStatus {
        let mut violations = Vec::new();
        let mut worst = Severity::Ok;

        // NaN check — critical
        if has_nan(state) {
            violations.push(Violation {
                kind: ViolationKind::NaN,
                severity: Severity::Critical,
                value: f64::NAN,
                limit: 0.0,
            });
            return SafetyStatus {
                severity: Severity::Critical,
                violations,
            };
        }

        // Speed check
        let speed = state.velocity.norm();
        if speed > self.max_velocity {
            let sev = if speed > self.max_velocity * 1.5 {
                Severity::Critical
            } else {
                Severity::Warning
            };
            violations.push(Violation {
                kind: ViolationKind::SpeedExceeded,
                severity: sev,
                value: speed,
                limit: self.max_velocity,
            });
            if sev as u8 > worst as u8 {
                worst = sev;
            }
        }

        // Tilt check: angle between body +Z and world +Z
        let body_up = state.attitude * Vector3::z();
        let tilt = body_up.z.clamp(-1.0, 1.0).acos();
        if tilt > self.max_tilt {
            let sev = if tilt > self.max_tilt * 1.3 {
                Severity::Critical
            } else {
                Severity::Warning
            };
            violations.push(Violation {
                kind: ViolationKind::TiltExceeded,
                severity: sev,
                value: tilt,
                limit: self.max_tilt,
            });
            if sev as u8 > worst as u8 {
                worst = sev;
            }
        }

        // Angular rate check
        let ang_rate = state.angular_velocity.norm();
        if ang_rate > self.max_angular_rate {
            let sev = if ang_rate > self.max_angular_rate * 1.5 {
                Severity::Critical
            } else {
                Severity::Warning
            };
            violations.push(Violation {
                kind: ViolationKind::AngularRateExceeded,
                severity: sev,
                value: ang_rate,
                limit: self.max_angular_rate,
            });
            if sev as u8 > worst as u8 {
                worst = sev;
            }
        }

        // Geofence check
        let boundary_dist = arena.boundary_sdf(&state.position);
        if boundary_dist < self.geofence_margin {
            let sev = if boundary_dist < 0.0 {
                Severity::Critical
            } else {
                Severity::Warning
            };
            violations.push(Violation {
                kind: ViolationKind::GeofenceBreach,
                severity: sev,
                value: boundary_dist,
                limit: self.geofence_margin,
            });
            if sev as u8 > worst as u8 {
                worst = sev;
            }
        }

        SafetyStatus {
            severity: worst,
            violations,
        }
    }

    /// Only active intervention: if state has NaN, return hover motors.
    /// Otherwise return the original motors unchanged.
    pub fn sanitize_motors(
        &self,
        motors: Vector4<f64>,
        state: &DroneState,
        params: &DroneParams,
    ) -> Vector4<f64> {
        if has_nan(state) || motors.iter().any(|v| v.is_nan() || v.is_infinite()) {
            let hover = params.hover_thrust();
            return Vector4::new(hover, hover, hover, hover);
        }
        motors
    }
}

/// Check if any component of the drone state is NaN or infinite.
fn has_nan(state: &DroneState) -> bool {
    let q = state.attitude.quaternion();
    state.position.iter().any(|v| !v.is_finite())
        || state.velocity.iter().any(|v| !v.is_finite())
        || !q.w.is_finite()
        || !q.i.is_finite()
        || !q.j.is_finite()
        || !q.k.is_finite()
        || state.angular_velocity.iter().any(|v| !v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    fn test_arena() -> Arena {
        Arena::new(Vector3::new(10.0, 10.0, 3.0))
    }

    #[test]
    fn test_hover_is_safe() {
        let envelope = SafetyEnvelope::crazyflie();
        let state = DroneState::hover_at(Vector3::new(5.0, 5.0, 1.5));
        let status = envelope.check(&state, &test_arena());
        assert!(status.is_ok());
    }

    #[test]
    fn test_speed_violation() {
        let envelope = SafetyEnvelope::crazyflie();
        let state = DroneState::new(
            Vector3::new(5.0, 5.0, 1.5),
            Vector3::new(6.0, 0.0, 0.0), // 6 m/s > 5 m/s limit
            UnitQuaternion::identity(),
            Vector3::zeros(),
        );
        let status = envelope.check(&state, &test_arena());
        assert!(!status.is_ok());
        assert_eq!(status.violations[0].kind, ViolationKind::SpeedExceeded);
    }

    #[test]
    fn test_tilt_violation() {
        let envelope = SafetyEnvelope::crazyflie();
        // Tilt 80° around Y axis
        let tilted = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::y()),
            80.0_f64.to_radians(),
        );
        let state = DroneState::new(
            Vector3::new(5.0, 5.0, 1.5),
            Vector3::zeros(),
            tilted,
            Vector3::zeros(),
        );
        let status = envelope.check(&state, &test_arena());
        assert!(!status.is_ok());
        assert_eq!(status.violations[0].kind, ViolationKind::TiltExceeded);
    }

    #[test]
    fn test_geofence_violation() {
        let envelope = SafetyEnvelope::crazyflie();
        // Position very close to boundary
        let state = DroneState::hover_at(Vector3::new(0.05, 5.0, 1.5));
        let status = envelope.check(&state, &test_arena());
        assert!(!status.is_ok());
        assert_eq!(status.violations[0].kind, ViolationKind::GeofenceBreach);
    }

    #[test]
    fn test_nan_is_critical() {
        let envelope = SafetyEnvelope::crazyflie();
        let state = DroneState::new(
            Vector3::new(f64::NAN, 5.0, 1.5),
            Vector3::zeros(),
            UnitQuaternion::identity(),
            Vector3::zeros(),
        );
        let status = envelope.check(&state, &test_arena());
        assert!(status.is_critical());
        assert_eq!(status.violations[0].kind, ViolationKind::NaN);
    }

    #[test]
    fn test_sanitize_motors_nan_state() {
        let envelope = SafetyEnvelope::crazyflie();
        let params = DroneParams::crazyflie();
        let state = DroneState::new(
            Vector3::new(f64::NAN, 0.0, 0.0),
            Vector3::zeros(),
            UnitQuaternion::identity(),
            Vector3::zeros(),
        );
        let motors = Vector4::new(0.1, 0.1, 0.1, 0.1);
        let safe = envelope.sanitize_motors(motors, &state, &params);
        let hover = params.hover_thrust();
        for i in 0..4 {
            assert!((safe[i] - hover).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sanitize_motors_nan_input() {
        let envelope = SafetyEnvelope::crazyflie();
        let params = DroneParams::crazyflie();
        let state = DroneState::hover_at(Vector3::new(5.0, 5.0, 1.5));
        let motors = Vector4::new(f64::NAN, 0.1, 0.1, 0.1);
        let safe = envelope.sanitize_motors(motors, &state, &params);
        let hover = params.hover_thrust();
        for i in 0..4 {
            assert!((safe[i] - hover).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sanitize_normal_passthrough() {
        let envelope = SafetyEnvelope::crazyflie();
        let params = DroneParams::crazyflie();
        let state = DroneState::hover_at(Vector3::new(5.0, 5.0, 1.5));
        let motors = Vector4::new(0.05, 0.06, 0.07, 0.08);
        let result = envelope.sanitize_motors(motors, &state, &params);
        assert_eq!(result, motors);
    }
}
