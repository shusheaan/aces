use nalgebra::{Matrix6, Vector3, Vector6};

/// Extended Kalman Filter for tracking opponent drone.
///
/// State: [px, py, pz, vx, vy, vz] (position + velocity)
/// Assumes constant-velocity motion model for prediction.
#[derive(Debug, Clone)]
pub struct EKF {
    /// Estimated state [px, py, pz, vx, vy, vz]
    pub state: Vector6<f64>,
    /// State covariance matrix (6x6)
    pub covariance: Matrix6<f64>,
    /// Process noise covariance
    pub process_noise: Matrix6<f64>,
    /// Measurement noise covariance (position only, 3x3 embedded in 6x6)
    pub measurement_noise_std: f64,
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

        // Process noise: accounts for unknown accelerations
        let mut process_noise = Matrix6::zeros();
        let q_pos = 0.01;
        let q_vel = 1.0;
        for i in 0..3 {
            process_noise[(i, i)] = q_pos;
            process_noise[(i + 3, i + 3)] = q_vel;
        }

        Self {
            state,
            covariance,
            process_noise,
            measurement_noise_std,
        }
    }

    /// Prediction step: advance state by dt assuming constant velocity.
    pub fn predict(&mut self, dt: f64) {
        // State transition: p += v * dt
        let mut f = Matrix6::identity();
        for i in 0..3 {
            f[(i, i + 3)] = dt;
        }

        self.state = f * self.state;
        self.covariance = f * self.covariance * f.transpose() + self.process_noise * dt;
    }

    /// Update step: incorporate a position measurement.
    pub fn update(&mut self, measured_position: &Vector3<f64>) {
        // Measurement matrix H: observes position only
        let mut h = nalgebra::Matrix3x6::zeros();
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;

        let r = nalgebra::Matrix3::identity() * self.measurement_noise_std.powi(2);

        // Innovation
        let predicted_pos = Vector3::new(self.state[0], self.state[1], self.state[2]);
        let y = measured_position - predicted_pos;

        // Innovation covariance
        let s = h * self.covariance * h.transpose() + r;
        let s_inv = s.try_inverse().unwrap_or(nalgebra::Matrix3::identity());

        // Kalman gain
        let k = self.covariance * h.transpose() * s_inv;

        // State update
        self.state += k * y;

        // Covariance update
        let i = Matrix6::identity();
        self.covariance = (i - k * h) * self.covariance;
    }

    /// Get estimated position.
    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.state[0], self.state[1], self.state[2])
    }

    /// Get estimated velocity.
    pub fn velocity(&self) -> Vector3<f64> {
        Vector3::new(self.state[3], self.state[4], self.state[5])
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
}
