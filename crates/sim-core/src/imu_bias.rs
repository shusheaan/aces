use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// IMU bias model: random walk on accelerometer and gyroscope biases.
///
/// At each step: bias += N(0, σ² · dt) per component
/// Applied: accel_meas = accel_true + accel_bias
///          gyro_meas  = gyro_true  + gyro_bias
#[derive(Debug, Clone)]
pub struct ImuBias {
    pub accel_bias: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,
    /// Accel bias random walk std dev (m/s² per √s)
    pub accel_walk_std: f64,
    /// Gyro bias random walk std dev (rad/s per √s)
    pub gyro_walk_std: f64,
    pub enabled: bool,
}

impl ImuBias {
    pub fn new(accel_walk_std: f64, gyro_walk_std: f64) -> Self {
        Self {
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            accel_walk_std,
            gyro_walk_std,
            enabled: accel_walk_std > 0.0 || gyro_walk_std > 0.0,
        }
    }

    pub fn disabled() -> Self {
        Self {
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            accel_walk_std: 0.0,
            gyro_walk_std: 0.0,
            enabled: false,
        }
    }

    /// Advance bias random walk by dt and return biased measurements.
    pub fn apply<R: Rng>(
        &mut self,
        true_accel: &Vector3<f64>,
        true_gyro: &Vector3<f64>,
        dt: f64,
        rng: &mut R,
    ) -> (Vector3<f64>, Vector3<f64>) {
        if !self.enabled {
            return (*true_accel, *true_gyro);
        }

        let sqrt_dt = dt.sqrt();

        if self.accel_walk_std > 0.0 {
            let n = Normal::new(0.0, self.accel_walk_std * sqrt_dt).unwrap();
            self.accel_bias += Vector3::new(n.sample(rng), n.sample(rng), n.sample(rng));
        }
        if self.gyro_walk_std > 0.0 {
            let n = Normal::new(0.0, self.gyro_walk_std * sqrt_dt).unwrap();
            self.gyro_bias += Vector3::new(n.sample(rng), n.sample(rng), n.sample(rng));
        }

        (true_accel + self.accel_bias, true_gyro + self.gyro_bias)
    }

    pub fn reset(&mut self) {
        self.accel_bias = Vector3::zeros();
        self.gyro_bias = Vector3::zeros();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_disabled_is_passthrough() {
        let mut bias = ImuBias::disabled();
        let mut rng = StdRng::seed_from_u64(42);

        let accel = Vector3::new(1.0, 2.0, 3.0);
        let gyro = Vector3::new(0.1, 0.2, 0.3);

        let (meas_accel, meas_gyro) = bias.apply(&accel, &gyro, 0.001, &mut rng);

        assert_eq!(meas_accel, accel);
        assert_eq!(meas_gyro, gyro);
    }

    #[test]
    fn test_bias_accumulates_over_time() {
        // After 10k steps at 1kHz (10 seconds), accel bias norm should grow > 0.05
        let accel_walk_std = 0.1; // m/s² per √s
        let mut bias = ImuBias::new(accel_walk_std, 0.0);
        let mut rng = StdRng::seed_from_u64(123);

        let accel = Vector3::zeros();
        let gyro = Vector3::zeros();
        let dt = 0.001; // 1 kHz

        for _ in 0..10_000 {
            bias.apply(&accel, &gyro, dt, &mut rng);
        }

        let bias_norm = bias.accel_bias.norm();
        assert!(
            bias_norm > 0.05,
            "Expected accel bias norm > 0.05, got {bias_norm}"
        );
    }

    #[test]
    fn test_bias_random_walk_variance() {
        // 500 trials: empirical variance of x-component should be within 20% of σ²·T
        let accel_walk_std = 0.1; // m/s² per √s
        let dt = 0.001;
        let steps = 1000; // T = 1 second
        let trials = 500;
        let expected_variance = accel_walk_std * accel_walk_std * (steps as f64 * dt); // σ²·T

        let mut final_biases = Vec::with_capacity(trials);

        for seed in 0..trials as u64 {
            let mut bias = ImuBias::new(accel_walk_std, 0.0);
            let mut rng = StdRng::seed_from_u64(seed);
            let accel = Vector3::zeros();
            let gyro = Vector3::zeros();

            for _ in 0..steps {
                bias.apply(&accel, &gyro, dt, &mut rng);
            }
            final_biases.push(bias.accel_bias.x);
        }

        let mean: f64 = final_biases.iter().sum::<f64>() / trials as f64;
        let empirical_variance: f64 = final_biases
            .iter()
            .map(|&b| (b - mean).powi(2))
            .sum::<f64>()
            / (trials as f64 - 1.0);

        let ratio = empirical_variance / expected_variance;
        assert!(
            (0.8..=1.2).contains(&ratio),
            "Empirical variance {empirical_variance:.4} not within 20% of expected {expected_variance:.4} (ratio={ratio:.3})"
        );
    }

    #[test]
    fn test_reset_clears_bias() {
        let mut bias = ImuBias::new(0.1, 0.05);
        let mut rng = StdRng::seed_from_u64(999);

        let accel = Vector3::zeros();
        let gyro = Vector3::zeros();

        // Accumulate some bias
        for _ in 0..1000 {
            bias.apply(&accel, &gyro, 0.001, &mut rng);
        }

        // Verify bias is non-zero before reset
        assert!(
            bias.accel_bias.norm() > 0.0,
            "Expected non-zero accel bias before reset"
        );
        assert!(
            bias.gyro_bias.norm() > 0.0,
            "Expected non-zero gyro bias before reset"
        );

        bias.reset();

        assert_eq!(bias.accel_bias, Vector3::zeros(), "accel_bias not zeroed");
        assert_eq!(bias.gyro_bias, Vector3::zeros(), "gyro_bias not zeroed");
    }
}
