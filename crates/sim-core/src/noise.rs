use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Gaussian observation noise model.
///
/// Adds i.i.d. Gaussian noise to position measurements:
///   p_obs = p_true + ε,  ε ~ N(0, σ²·I)
#[derive(Debug, Clone)]
pub struct ObservationNoise {
    /// Standard deviation of position noise (meters)
    pub std_dev: f64,
    /// Whether noise is enabled
    pub enabled: bool,
}

impl ObservationNoise {
    pub fn new(std_dev: f64) -> Self {
        Self {
            std_dev,
            enabled: std_dev > 0.0,
        }
    }

    /// Apply noise to a position measurement.
    pub fn apply<R: Rng>(&self, true_position: &Vector3<f64>, rng: &mut R) -> Vector3<f64> {
        if !self.enabled {
            return *true_position;
        }

        let normal = Normal::new(0.0, self.std_dev).unwrap();
        Vector3::new(
            true_position.x + normal.sample(rng),
            true_position.y + normal.sample(rng),
            true_position.z + normal.sample(rng),
        )
    }

    /// Apply noise to a velocity measurement (same std_dev).
    pub fn apply_velocity<R: Rng>(
        &self,
        true_velocity: &Vector3<f64>,
        rng: &mut R,
    ) -> Vector3<f64> {
        if !self.enabled {
            return *true_velocity;
        }

        let normal = Normal::new(0.0, self.std_dev).unwrap();
        Vector3::new(
            true_velocity.x + normal.sample(rng),
            true_velocity.y + normal.sample(rng),
            true_velocity.z + normal.sample(rng),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_disabled_noise_is_exact() {
        let noise = ObservationNoise::new(0.0);
        let mut rng = StdRng::seed_from_u64(42);
        let pos = Vector3::new(1.0, 2.0, 3.0);

        for _ in 0..100 {
            let obs = noise.apply(&pos, &mut rng);
            assert_eq!(obs, pos);
        }
    }

    #[test]
    fn test_noise_has_expected_variance() {
        let std_dev = 0.1;
        let noise = ObservationNoise::new(std_dev);
        let mut rng = StdRng::seed_from_u64(42);
        let pos = Vector3::new(5.0, 5.0, 1.5);

        let n = 50_000;
        let mut sum_sq_err = 0.0;
        for _ in 0..n {
            let obs = noise.apply(&pos, &mut rng);
            let err = obs - pos;
            sum_sq_err += err.x * err.x;
        }
        let empirical_var = sum_sq_err / n as f64;
        let expected_var = std_dev * std_dev;

        assert!(
            (empirical_var - expected_var).abs() / expected_var < 0.1,
            "empirical var {} vs expected {}",
            empirical_var,
            expected_var
        );
    }

    #[test]
    fn test_noise_is_zero_mean() {
        let noise = ObservationNoise::new(0.1);
        let mut rng = StdRng::seed_from_u64(42);
        let pos = Vector3::new(5.0, 5.0, 1.5);

        let n = 50_000;
        let mut sum = Vector3::zeros();
        for _ in 0..n {
            let obs = noise.apply(&pos, &mut rng);
            sum += obs - pos;
        }
        let mean = sum / n as f64;

        assert!(
            mean.norm() < 0.01,
            "mean error should be ~0, got {}",
            mean.norm()
        );
    }
}
