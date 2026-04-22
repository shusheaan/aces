use nalgebra::{Matrix3, Matrix3x6, Matrix6, Matrix6x3, Vector3, Vector6};

/// Extended Kalman Filter for tracking opponent drone.
///
/// State: [px, py, pz, vx, vy, vz] (position + velocity)
/// Assumes constant-acceleration motion model for prediction.
#[derive(Debug, Clone)]
pub struct EKF {
    /// Estimated state [px, py, pz, vx, vy, vz]
    pub state: Vector6<f64>,
    /// State covariance matrix (6x6)
    pub covariance: Matrix6<f64>,
    /// Spectral density of acceleration noise (m²/s³). Replaces old scalar process_noise.
    pub q_accel: f64,
    /// Measurement noise standard deviation (m)
    pub measurement_noise_std: f64,
    /// Most recent innovation (measurement residual), None before first update
    pub last_innovation: Option<Vector3<f64>>,
    /// Most recent Kalman gain, None before first update
    pub last_kalman_gain: Option<Matrix6x3<f64>>,
}

impl EKF {
    pub fn new(initial_position: Vector3<f64>, measurement_noise_std: f64) -> Self {
        let state = Vector6::new(
            initial_position.x,
            initial_position.y,
            initial_position.z,
            0.0,
            0.0,
            0.0,
        );

        // Initial covariance: uncertain about velocity
        let mut covariance = Matrix6::identity() * 0.1;
        covariance[(3, 3)] = 1.0;
        covariance[(4, 4)] = 1.0;
        covariance[(5, 5)] = 1.0;

        Self {
            state,
            covariance,
            q_accel: 4.0,
            measurement_noise_std,
            last_innovation: None,
            last_kalman_gain: None,
        }
    }

    /// Build the discretized process noise matrix Q(dt) for a constant-acceleration model.
    ///
    /// Q(dt) = q_a * [ dt³/3·I₃  dt²/2·I₃ ]
    ///               [ dt²/2·I₃  dt·I₃    ]
    pub fn process_noise_matrix(&self, dt: f64) -> Matrix6<f64> {
        let qa = self.q_accel;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;

        let pp = Matrix3::identity() * (qa * dt3 / 3.0);
        let pv = Matrix3::identity() * (qa * dt2 / 2.0);
        let vv = Matrix3::identity() * (qa * dt);

        let mut q = Matrix6::zeros();
        // Top-left 3×3: position-position
        for i in 0..3 {
            for j in 0..3 {
                q[(i, j)] = pp[(i, j)];
            }
        }
        // Top-right 3×3: position-velocity cross terms
        for i in 0..3 {
            for j in 0..3 {
                q[(i, j + 3)] = pv[(i, j)];
            }
        }
        // Bottom-left 3×3: velocity-position cross terms (symmetric)
        for i in 0..3 {
            for j in 0..3 {
                q[(i + 3, j)] = pv[(i, j)];
            }
        }
        // Bottom-right 3×3: velocity-velocity
        for i in 0..3 {
            for j in 0..3 {
                q[(i + 3, j + 3)] = vv[(i, j)];
            }
        }
        q
    }

    /// Prediction step: advance state by dt assuming constant velocity.
    pub fn predict(&mut self, dt: f64) {
        // State transition: p += v * dt
        let mut f = Matrix6::identity();
        for i in 0..3 {
            f[(i, i + 3)] = dt;
        }

        self.state = f * self.state;

        let q = self.process_noise_matrix(dt);
        self.covariance = f * self.covariance * f.transpose() + q;
    }

    /// Update step: incorporate a position measurement.
    pub fn update(&mut self, measured_position: &Vector3<f64>) {
        // Measurement matrix H: observes position only
        let mut h = Matrix3x6::zeros();
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;

        let r = Matrix3::identity() * self.measurement_noise_std.powi(2);

        // Innovation
        let predicted_pos = Vector3::new(self.state[0], self.state[1], self.state[2]);
        let y = measured_position - predicted_pos;

        // Innovation covariance
        let s = h * self.covariance * h.transpose() + r;
        let s_inv = s.try_inverse().unwrap_or(Matrix3::identity());

        // Kalman gain (6×3)
        let k: Matrix6x3<f64> = self.covariance * h.transpose() * s_inv;

        // State update
        self.state += k * y;

        // Covariance update — numerically stable Joseph form:
        // P = (I - KH) P (I - KH)^T + K R K^T
        let i_kh = Matrix6::identity() - k * h;
        self.covariance = i_kh * self.covariance * i_kh.transpose() + k * r * k.transpose();

        // Store diagnostics
        self.last_innovation = Some(y);
        self.last_kalman_gain = Some(k);
    }

    /// Get estimated position.
    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.state[0], self.state[1], self.state[2])
    }

    /// Get estimated velocity.
    pub fn velocity(&self) -> Vector3<f64> {
        Vector3::new(self.state[3], self.state[4], self.state[5])
    }

    /// Return the diagonal of the state covariance as [pp_x, pp_y, pp_z, vv_x, vv_y, vv_z].
    pub fn covariance_diagonal(&self) -> [f64; 6] {
        [
            self.covariance[(0, 0)],
            self.covariance[(1, 1)],
            self.covariance[(2, 2)],
            self.covariance[(3, 3)],
            self.covariance[(4, 4)],
            self.covariance[(5, 5)],
        ]
    }

    /// Return the most recent innovation (pre-fit residual), or None before first update.
    pub fn last_innovation(&self) -> Option<Vector3<f64>> {
        self.last_innovation
    }

    /// Return selected diagonal elements of the last Kalman gain:
    /// [K[0,0], K[1,1], K[2,2], K[3,0], K[4,1], K[5,2]]
    pub fn last_kalman_gain_diag(&self) -> Option<[f64; 6]> {
        self.last_kalman_gain.map(|k| {
            [k[(0, 0)], k[(1, 1)], k[(2, 2)], k[(3, 0)], k[(4, 1)], k[(5, 2)]]
        })
    }

    /// Reset the filter with a new initial position.
    pub fn reset(&mut self, position: Vector3<f64>) {
        self.state = Vector6::new(position.x, position.y, position.z, 0.0, 0.0, 0.0);
        self.covariance = {
            let mut c = Matrix6::identity() * 0.1;
            c[(3, 3)] = 1.0;
            c[(4, 4)] = 1.0;
            c[(5, 5)] = 1.0;
            c
        };
        self.last_innovation = None;
        self.last_kalman_gain = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn test_ekf_reduces_noise() {
        let noise_std = 0.1;
        let mut ekf = EKF::new(Vector3::new(1.0, 2.0, 1.5), noise_std);
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, noise_std).unwrap();

        // Simulate a target moving at constant velocity
        let velocity = Vector3::new(0.5, -0.3, 0.1);
        let dt = 0.01;

        let mut sum_raw_error = 0.0;
        let mut sum_ekf_error = 0.0;
        let n = 500;

        for i in 0..n {
            let t = i as f64 * dt;
            let true_pos = Vector3::new(1.0, 2.0, 1.5) + velocity * t;

            // Noisy measurement
            let noisy_pos = true_pos
                + Vector3::new(
                    normal.sample(&mut rng),
                    normal.sample(&mut rng),
                    normal.sample(&mut rng),
                );

            ekf.predict(dt);
            ekf.update(&noisy_pos);

            let raw_err = (noisy_pos - true_pos).norm();
            let ekf_err = (ekf.position() - true_pos).norm();

            // Skip first 50 steps (burn-in)
            if i >= 50 {
                sum_raw_error += raw_err;
                sum_ekf_error += ekf_err;
            }
        }

        let avg_raw = sum_raw_error / (n - 50) as f64;
        let avg_ekf = sum_ekf_error / (n - 50) as f64;

        assert!(
            avg_ekf < avg_raw,
            "EKF error ({:.4}) should be less than raw noise ({:.4})",
            avg_ekf,
            avg_raw
        );
    }

    #[test]
    fn test_ekf_tracks_velocity() {
        let mut ekf = EKF::new(Vector3::new(0.0, 0.0, 1.0), 0.05);
        let velocity = Vector3::new(1.0, 0.0, 0.0);
        let dt = 0.01;

        // Feed perfect measurements of a constant-velocity target
        for i in 0..200 {
            let t = i as f64 * dt;
            let pos = Vector3::new(0.0, 0.0, 1.0) + velocity * t;
            ekf.predict(dt);
            ekf.update(&pos);
        }

        let est_vel = ekf.velocity();
        assert!(
            (est_vel - velocity).norm() < 0.1,
            "EKF should estimate velocity, got {:?}",
            est_vel
        );
    }

    #[test]
    fn test_q_matrix_has_cross_terms() {
        let ekf = EKF::new(Vector3::zeros(), 0.1);
        let q = ekf.process_noise_matrix(0.01);

        // Off-diagonal block (top-right, position-velocity cross terms) must be non-zero
        let cross_term = q[(0, 3)]; // Q[px, vx]
        assert!(
            cross_term.abs() > 1e-12,
            "Q matrix position-velocity cross term Q[0,3] should be non-zero, got {}",
            cross_term
        );

        // Diagonal position block
        let pos_diag = q[(0, 0)];
        assert!(
            pos_diag > 0.0,
            "Q matrix position diagonal Q[0,0] should be positive, got {}",
            pos_diag
        );

        // Diagonal velocity block
        let vel_diag = q[(3, 3)];
        assert!(
            vel_diag > 0.0,
            "Q matrix velocity diagonal Q[3,3] should be positive, got {}",
            vel_diag
        );

        // Verify the ratio dt³/3 : dt²/2 : dt for dt=0.01
        let dt = 0.01_f64;
        let qa = ekf.q_accel;
        let expected_pp = qa * dt * dt * dt / 3.0;
        let expected_pv = qa * dt * dt / 2.0;
        let expected_vv = qa * dt;
        assert!(
            (q[(0, 0)] - expected_pp).abs() < 1e-15,
            "Q[0,0] = {}, expected {}",
            q[(0, 0)],
            expected_pp
        );
        assert!(
            (q[(0, 3)] - expected_pv).abs() < 1e-15,
            "Q[0,3] = {}, expected {}",
            q[(0, 3)],
            expected_pv
        );
        assert!(
            (q[(3, 3)] - expected_vv).abs() < 1e-15,
            "Q[3,3] = {}, expected {}",
            q[(3, 3)],
            expected_vv
        );
    }

    #[test]
    fn test_covariance_stays_symmetric_positive_definite() {
        let mut ekf = EKF::new(Vector3::new(5.0, 0.0, 2.0), 0.1);
        let mut rng = StdRng::seed_from_u64(99);
        let normal = Normal::new(0.0, 0.1_f64).unwrap();
        let dt = 0.01;
        let velocity = Vector3::new(0.3, 0.2, -0.1);

        for step in 0..2000 {
            let t = step as f64 * dt;
            let true_pos = Vector3::new(5.0, 0.0, 2.0) + velocity * t;
            let meas = true_pos
                + Vector3::new(
                    normal.sample(&mut rng),
                    normal.sample(&mut rng),
                    normal.sample(&mut rng),
                );

            ekf.predict(dt);
            ekf.update(&meas);

            let p = &ekf.covariance;

            // Check symmetry: max |P - P^T| < 1e-10
            let sym_err = (p - p.transpose()).abs().max();
            assert!(
                sym_err < 1e-10,
                "Covariance not symmetric at step {}: max |P-P^T| = {}",
                step,
                sym_err
            );

            // Check positive definiteness via Cholesky
            assert!(
                p.clone().cholesky().is_some(),
                "Covariance not positive-definite at step {}",
                step
            );
        }
    }

    #[test]
    fn test_last_innovation_available() {
        let mut ekf = EKF::new(Vector3::new(0.0, 0.0, 1.0), 0.1);

        // Before any update, innovation should be None
        assert!(
            ekf.last_innovation().is_none(),
            "Innovation should be None before first update"
        );
        assert!(
            ekf.last_kalman_gain_diag().is_none(),
            "Kalman gain should be None before first update"
        );

        // After predict + update, both should be Some
        ekf.predict(0.01);
        ekf.update(&Vector3::new(0.1, 0.0, 1.0));

        assert!(
            ekf.last_innovation().is_some(),
            "Innovation should be Some after first update"
        );
        assert!(
            ekf.last_kalman_gain_diag().is_some(),
            "Kalman gain should be Some after first update"
        );

        // After reset, both should be None again
        ekf.reset(Vector3::new(1.0, 2.0, 3.0));
        assert!(
            ekf.last_innovation().is_none(),
            "Innovation should be None after reset"
        );
        assert!(
            ekf.last_kalman_gain_diag().is_none(),
            "Kalman gain should be None after reset"
        );
    }
}
