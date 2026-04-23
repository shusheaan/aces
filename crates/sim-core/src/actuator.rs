use nalgebra::Vector4;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Actuator model: first-order motor delay + multiplicative noise + per-motor bias.
///
/// Models real motor behavior:
/// - First-order lag: thrust tracks command with time constant tau
/// - Multiplicative noise: thrust_out = thrust_actual * (1 + ε), ε ~ N(0, σ²)
/// - Per-motor bias: thrust_out *= (1 + bias_i), set at episode start
#[derive(Debug, Clone)]
pub struct ActuatorModel {
    /// Current actual thrust per motor (N) — lags behind command
    pub thrust_state: Vector4<f64>,
    /// Motor time constant (seconds). 0 = instant response.
    pub time_constant: f64,
    /// Multiplicative noise std dev. 0 = no noise.
    pub noise_std: f64,
    /// Per-motor bias (fraction, e.g. 0.03 = +3%).
    pub bias: Vector4<f64>,
    /// Whether the model is enabled
    pub enabled: bool,
    /// Cached normal distribution for multiplicative noise (avoids recreating on every call)
    noise_normal: Option<Normal<f64>>,
}

impl ActuatorModel {
    pub fn new(time_constant: f64, noise_std: f64) -> Self {
        let noise_normal = if noise_std > 0.0 {
            Some(Normal::new(0.0, noise_std).unwrap())
        } else {
            None
        };
        Self {
            thrust_state: Vector4::zeros(),
            time_constant,
            noise_std,
            bias: Vector4::zeros(),
            enabled: time_constant > 0.0 || noise_std > 0.0,
            noise_normal,
        }
    }

    pub fn disabled() -> Self {
        Self {
            thrust_state: Vector4::zeros(),
            time_constant: 0.0,
            noise_std: 0.0,
            bias: Vector4::zeros(),
            enabled: false,
            noise_normal: None,
        }
    }

    /// Randomize per-motor bias. Call once at episode start.
    /// Automatically enables the model if non-zero bias is applied.
    pub fn randomize_bias<R: Rng>(&mut self, bias_range: f64, rng: &mut R) {
        if bias_range > 0.0 {
            let dist = rand_distr::Uniform::new(-bias_range, bias_range);
            for i in 0..4 {
                self.bias[i] = dist.sample(rng);
            }
            self.enabled = true;
        } else {
            self.bias = Vector4::zeros();
        }
    }

    /// Process commanded thrusts through delay, noise, and bias.
    pub fn apply<R: Rng>(
        &mut self,
        commanded: &Vector4<f64>,
        dt: f64,
        rng: &mut R,
    ) -> Vector4<f64> {
        debug_assert!(
            dt >= 0.0,
            "ActuatorModel::apply called with negative dt: {dt}"
        );
        if !self.enabled {
            return *commanded;
        }

        // First-order lag
        if self.time_constant > 0.0 {
            let alpha = 1.0 - (-dt / self.time_constant).exp();
            self.thrust_state += (commanded - self.thrust_state) * alpha;
        } else {
            self.thrust_state = *commanded;
        }

        let mut output = self.thrust_state;

        // Multiplicative noise
        if let Some(ref normal) = self.noise_normal {
            for i in 0..4 {
                output[i] *= 1.0 + normal.sample(rng);
            }
        }

        // Per-motor bias
        for i in 0..4 {
            output[i] *= 1.0 + self.bias[i];
        }

        // Clamp non-negative
        for i in 0..4 {
            output[i] = output[i].max(0.0);
        }

        output
    }

    pub fn reset(&mut self) {
        self.thrust_state = Vector4::zeros();
        self.bias = Vector4::zeros();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_disabled_is_passthrough() {
        let mut model = ActuatorModel::disabled();
        let mut rng = StdRng::seed_from_u64(42);
        let cmd = Vector4::new(1.0, 2.0, 3.0, 4.0);
        let out = model.apply(&cmd, 0.01, &mut rng);
        assert_relative_eq!(out, cmd, epsilon = 1e-12);
    }

    #[test]
    fn test_first_order_delay_step_response() {
        let tau = 0.05;
        let dt = 0.001;
        let mut model = ActuatorModel::new(tau, 0.0);
        let mut rng = StdRng::seed_from_u64(0);

        let cmd = Vector4::new(1.0, 1.0, 1.0, 1.0);

        // Verify alpha matches formula for first step
        let alpha_expected = 1.0 - (-dt / tau).exp();

        // After first step from zero, thrust_state = alpha * cmd
        let out = model.apply(&cmd, dt, &mut rng);
        assert_relative_eq!(out[0], alpha_expected, epsilon = 1e-10);

        // After 1000 more steps should be very close to command
        for _ in 0..1000 {
            model.apply(&cmd, dt, &mut rng);
        }
        assert_relative_eq!(model.thrust_state[0], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_multiplicative_noise_variance() {
        let noise_std = 0.05;
        let mut model = ActuatorModel::new(0.0, noise_std);
        // Force instant response by setting enabled but zero time_constant
        let mut rng = StdRng::seed_from_u64(12345);

        let cmd = Vector4::new(1.0, 1.0, 1.0, 1.0);
        let n_samples = 50_000;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;

        for _ in 0..n_samples {
            let out = model.apply(&cmd, 0.001, &mut rng);
            sum += out[0];
            sum_sq += out[0] * out[0];
        }

        let mean = sum / n_samples as f64;
        let variance = sum_sq / n_samples as f64 - mean * mean;
        let expected_variance = noise_std * noise_std; // E[(1+ε)²] - 1 ≈ σ² for small σ

        // Allow 15% tolerance
        assert!(
            (variance - expected_variance).abs() / expected_variance < 0.15,
            "Empirical variance {variance:.6} too far from expected {expected_variance:.6}"
        );
    }

    #[test]
    fn test_per_motor_bias() {
        let mut model = ActuatorModel::new(0.0, 0.0);
        // Enable so the bias code path runs; time_constant=0 and noise_std=0 so
        // the only effect is the per-motor bias multiplication.
        model.enabled = true;
        // Manually set bias: [+5%, -5%, 0%, +10%]
        model.bias = Vector4::new(0.05, -0.05, 0.0, 0.10);
        let mut rng = StdRng::seed_from_u64(0);

        let cmd = Vector4::new(1.0, 1.0, 1.0, 1.0);
        let out = model.apply(&cmd, 0.001, &mut rng);

        assert_relative_eq!(out[0], 1.05, epsilon = 1e-10);
        assert_relative_eq!(out[1], 0.95, epsilon = 1e-10);
        assert_relative_eq!(out[2], 1.00, epsilon = 1e-10);
        assert_relative_eq!(out[3], 1.10, epsilon = 1e-10);
    }

    #[test]
    fn test_output_never_negative() {
        // Use high noise to stress-test clamping
        let noise_std = 2.0;
        let mut model = ActuatorModel::new(0.0, noise_std);
        let mut rng = StdRng::seed_from_u64(99);

        let cmd = Vector4::new(0.1, 0.1, 0.1, 0.1);

        for _ in 0..10_000 {
            let out = model.apply(&cmd, 0.001, &mut rng);
            for i in 0..4 {
                assert!(out[i] >= 0.0, "Output[{i}] was negative: {}", out[i]);
            }
        }
    }
}
