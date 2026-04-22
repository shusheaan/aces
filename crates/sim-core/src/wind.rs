use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Ornstein-Uhlenbeck wind disturbance process.
///
/// Models correlated wind forces with mean reversion:
///   dw = θ · (μ - w) · dt + σ · dW
///
/// where θ controls reversion speed, μ is the mean wind,
/// σ is the volatility, and dW is a Wiener process increment.
#[derive(Debug, Clone)]
pub struct WindModel {
    /// Current wind force vector (N)
    pub force: Vector3<f64>,
    /// Mean reversion rate (higher = faster return to mean)
    pub theta: f64,
    /// Mean wind force (N)
    pub mu: Vector3<f64>,
    /// Volatility (N)
    pub sigma: f64,
    /// Whether wind is enabled
    pub enabled: bool,
}

impl WindModel {
    pub fn new(theta: f64, mu: Vector3<f64>, sigma: f64) -> Self {
        Self {
            force: mu,
            theta,
            mu,
            sigma,
            enabled: true,
        }
    }

    /// Default wind parameters from plan: θ=2.0, μ=[0,0,0], σ=0.3N
    pub fn default_params() -> Self {
        Self::new(2.0, Vector3::zeros(), 0.3)
    }

    /// Disabled wind (zero force, no randomness).
    pub fn disabled() -> Self {
        Self {
            force: Vector3::zeros(),
            theta: 0.0,
            mu: Vector3::zeros(),
            sigma: 0.0,
            enabled: false,
        }
    }

    /// Advance the OU process by dt and return the current wind force.
    pub fn step<R: Rng>(&mut self, dt: f64, rng: &mut R) -> Vector3<f64> {
        if !self.enabled {
            return Vector3::zeros();
        }

        let normal = Normal::new(0.0, 1.0).unwrap();
        let sqrt_dt = dt.sqrt();

        // Euler-Maruyama discretization of OU process
        for i in 0..3 {
            let drift = self.theta * (self.mu[i] - self.force[i]) * dt;
            let diffusion = self.sigma * sqrt_dt * normal.sample(rng);
            self.force[i] += drift + diffusion;
        }

        self.force
    }

    /// Reset wind to mean value.
    pub fn reset(&mut self) {
        self.force = self.mu;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_disabled_wind_is_zero() {
        let mut wind = WindModel::disabled();
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let f = wind.step(0.001, &mut rng);
            assert_eq!(f, Vector3::zeros());
        }
    }

    #[test]
    fn test_ou_mean_reversion() {
        // Start far from mean, verify it reverts
        let mut wind = WindModel::new(5.0, Vector3::zeros(), 0.01);
        wind.force = Vector3::new(1.0, 1.0, 1.0);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..10_000 {
            wind.step(0.001, &mut rng);
        }

        // After 10s with θ=5, should be close to μ=0
        assert!(
            wind.force.norm() < 0.2,
            "wind should revert to mean, got {}",
            wind.force.norm()
        );
    }

    #[test]
    fn test_ou_stationary_variance() {
        // Stationary variance of OU process = σ² / (2θ)
        let theta = 2.0;
        let sigma = 0.3;
        let expected_var = sigma * sigma / (2.0 * theta);

        let mut wind = WindModel::new(theta, Vector3::zeros(), sigma);
        let mut rng = StdRng::seed_from_u64(123);

        // Burn-in
        for _ in 0..10_000 {
            wind.step(0.001, &mut rng);
        }

        // Collect samples
        let n = 50_000;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            wind.step(0.001, &mut rng);
            sum_sq += wind.force.x * wind.force.x;
        }
        let empirical_var = sum_sq / n as f64;

        // Should be within 50% of expected (statistical test, generous tolerance)
        assert!(
            (empirical_var - expected_var).abs() / expected_var < 0.5,
            "empirical var {} vs expected {}",
            empirical_var,
            expected_var
        );
    }
}
