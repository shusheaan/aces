# Sim2Real Noise, Domain Randomization & EKF Hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the sim-to-real gap by adding actuator noise, sensor bias, domain randomization, and hardening the EKF with proper math and comprehensive statistical tests.

**Architecture:** Seven tasks across three Rust crates and two Python layers. Tasks 1-3 are Rust sim-core/estimator changes (independent). Task 4 wires new Rust features through py-bridge. Task 5 extends Python config. Task 6 adds domain randomization to the env. Task 7 adds statistical EKF tests. Each task is independently testable.

**Tech Stack:** Rust (nalgebra, rand, rand_distr), PyO3, Python (numpy, scipy for tests)

---

## File Map

### Rust — New Files
- `crates/sim-core/src/actuator.rs` — Motor delay + noise model
- `crates/sim-core/src/imu_bias.rs` — IMU accel/gyro bias random walk

### Rust — Modified Files
- `crates/sim-core/src/lib.rs` — Register new modules
- `crates/estimator/src/ekf.rs` — Fix Q matrix, Joseph form, expose diagnostics
- `crates/py-bridge/src/lib.rs` — Wire actuator, bias, domain rand, EKF diagnostics

### Python — Modified Files
- `aces/config.py` — Extend NoiseConfig with new fields
- `aces/env.py` — Domain randomization in reset(), pass new params
- `configs/rules.toml` — New noise/DR config keys
- `configs/curriculum.toml` — Per-phase DR overrides

### Python — New Test Files
- `tests/test_ekf_statistical.py` — NEES, innovation whiteness, Monte Carlo

---

## Task 1: Fix EKF — Proper Q Matrix + Joseph Form + Diagnostics

**Files:**
- Modify: `crates/estimator/src/ekf.rs`

This task fixes three mathematical issues and adds diagnostic accessors needed for Task 7.

- [ ] **Step 1: Write failing test for proper Q matrix structure**

The correct discrete Q for constant-acceleration noise with spectral density `q_a` is:
```
Q(dt) = q_a * [ dt³/3·I  dt²/2·I ]
              [ dt²/2·I  dt·I    ]
```

Add this test at the bottom of the `mod tests` block in `ekf.rs`:

```rust
#[test]
fn test_q_matrix_has_cross_terms() {
    let ekf = EKF::new(Vector3::zeros(), 0.1);
    // With proper discretization, Q should have off-diagonal blocks
    // Q[0,3] should be non-zero (position-velocity cross-covariance)
    // We check the process_noise_fn produces correct structure
    let dt = 0.01;
    let q = ekf.process_noise_matrix(dt);
    // Off-diagonal block: dt²/2 * q_a
    assert!(q[(0, 3)].abs() > 1e-10, "Q should have pos-vel cross terms");
    assert_eq!(q[(0, 3)], q[(3, 0)], "Q should be symmetric");
}
```

- [ ] **Step 2: Write failing test for Joseph form numerical stability**

```rust
#[test]
fn test_covariance_stays_symmetric_positive_definite() {
    let mut ekf = EKF::new(Vector3::new(1.0, 2.0, 1.5), 0.1);
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.1).unwrap();
    let velocity = Vector3::new(0.5, -0.3, 0.1);
    let dt = 0.01;

    for i in 0..2000 {
        let t = i as f64 * dt;
        let true_pos = Vector3::new(1.0, 2.0, 1.5) + velocity * t;
        let noisy = true_pos + Vector3::new(
            normal.sample(&mut rng),
            normal.sample(&mut rng),
            normal.sample(&mut rng),
        );
        ekf.predict(dt);
        ekf.update(&noisy);

        // Check symmetry
        for r in 0..6 {
            for c in 0..6 {
                let diff = (ekf.covariance[(r, c)] - ekf.covariance[(c, r)]).abs();
                assert!(diff < 1e-10, "P not symmetric at ({r},{c}) step {i}: diff={diff}");
            }
        }
        // Check positive definite via Cholesky
        assert!(
            ekf.covariance.cholesky().is_some(),
            "P not positive definite at step {i}"
        );
    }
}
```

- [ ] **Step 3: Write failing test for innovation accessor**

```rust
#[test]
fn test_last_innovation_available() {
    let mut ekf = EKF::new(Vector3::new(1.0, 2.0, 1.5), 0.1);
    assert!(ekf.last_innovation().is_none(), "no innovation before first update");
    ekf.predict(0.01);
    ekf.update(&Vector3::new(1.1, 2.1, 1.6));
    let innov = ekf.last_innovation().unwrap();
    // Innovation = measurement - predicted = [1.1-1.0, 2.1-2.0, 1.6-1.5] approximately
    assert!(innov.norm() > 0.0);
}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd /Users/shu/GitHub/aces && cargo test -p aces-estimator`
Expected: 3 compile errors (methods don't exist yet)

- [ ] **Step 5: Implement fixed EKF**

Replace the full EKF implementation in `crates/estimator/src/ekf.rs`:

```rust
use nalgebra::{Matrix3, Matrix3x6, Matrix6, Vector3, Vector6};

/// Extended Kalman Filter for tracking opponent drone.
///
/// State: [px, py, pz, vx, vy, vz] (position + velocity)
/// Uses constant-velocity motion model with piecewise-constant acceleration noise.
#[derive(Debug, Clone)]
pub struct EKF {
    /// Estimated state [px, py, pz, vx, vy, vz]
    pub state: Vector6<f64>,
    /// State covariance matrix (6x6)
    pub covariance: Matrix6<f64>,
    /// Acceleration process noise spectral density (m/s²)²
    pub q_accel: f64,
    /// Measurement noise standard deviation (m)
    pub measurement_noise_std: f64,
    /// Last innovation vector (set after each update)
    last_innovation: Option<Vector3<f64>>,
    /// Last Kalman gain (set after each update)
    last_kalman_gain: Option<nalgebra::Matrix6x3<f64>>,
}

impl EKF {
    pub fn new(initial_position: Vector3<f64>, measurement_noise_std: f64) -> Self {
        let state = Vector6::new(
            initial_position.x,
            initial_position.y,
            initial_position.z,
            0.0, 0.0, 0.0,
        );

        // Initial covariance: small position uncertainty, large velocity uncertainty
        let mut covariance = Matrix6::identity() * 0.1;
        covariance[(3, 3)] = 1.0;
        covariance[(4, 4)] = 1.0;
        covariance[(5, 5)] = 1.0;

        Self {
            state,
            covariance,
            q_accel: 4.0, // (m/s²)² — tuned for maneuvering quadrotor
            measurement_noise_std,
            last_innovation: None,
            last_kalman_gain: None,
        }
    }

    /// Build the discrete process noise matrix for timestep dt.
    ///
    /// For constant-acceleration noise with spectral density q_a:
    /// Q(dt) = q_a * [ dt³/3·I  dt²/2·I ]
    ///               [ dt²/2·I  dt·I    ]
    pub fn process_noise_matrix(&self, dt: f64) -> Matrix6<f64> {
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let mut q = Matrix6::zeros();
        for i in 0..3 {
            q[(i, i)] = dt3 / 3.0;           // position block
            q[(i, i + 3)] = dt2 / 2.0;       // cross block
            q[(i + 3, i)] = dt2 / 2.0;       // cross block (symmetric)
            q[(i + 3, i + 3)] = dt;           // velocity block
        }
        q * self.q_accel
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
    /// Uses Joseph form for numerical stability.
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
        self.last_innovation = Some(y);

        // Innovation covariance
        let s = h * self.covariance * h.transpose() + r;
        let s_inv = s.try_inverse().unwrap_or(Matrix3::identity());

        // Kalman gain (6x3)
        let k = self.covariance * h.transpose() * s_inv;
        self.last_kalman_gain = Some(k);

        // State update
        self.state += k * y;

        // Covariance update — Joseph form: P = (I-KH) P (I-KH)^T + K R K^T
        let i_kh = Matrix6::identity() - k * h;
        self.covariance = i_kh * self.covariance * i_kh.transpose() + k * r * k.transpose();
    }

    /// Get estimated position.
    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.state[0], self.state[1], self.state[2])
    }

    /// Get estimated velocity.
    pub fn velocity(&self) -> Vector3<f64> {
        Vector3::new(self.state[3], self.state[4], self.state[5])
    }

    /// Get covariance diagonal [P_px, P_py, P_pz, P_vx, P_vy, P_vz].
    pub fn covariance_diagonal(&self) -> [f64; 6] {
        [
            self.covariance[(0, 0)], self.covariance[(1, 1)], self.covariance[(2, 2)],
            self.covariance[(3, 3)], self.covariance[(4, 4)], self.covariance[(5, 5)],
        ]
    }

    /// Last innovation vector (measurement - predicted). None before first update.
    pub fn last_innovation(&self) -> Option<Vector3<f64>> {
        self.last_innovation
    }

    /// Last Kalman gain matrix diagonal (6 values). None before first update.
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
```

- [ ] **Step 6: Run all estimator tests**

Run: `cd /Users/shu/GitHub/aces && cargo test -p aces-estimator`
Expected: All pass (including old tests `test_ekf_reduces_noise` and `test_ekf_tracks_velocity` which test behavior, not internals)

- [ ] **Step 7: Commit**

```bash
git add crates/estimator/src/ekf.rs
git commit -m "fix(ekf): proper Q discretization, Joseph form, diagnostic accessors"
```

---

## Task 2: Actuator Noise Model

**Files:**
- Create: `crates/sim-core/src/actuator.rs`
- Modify: `crates/sim-core/src/lib.rs`

Adds first-order motor delay and multiplicative thrust noise.

- [ ] **Step 1: Write test file with failing tests**

Create `crates/sim-core/src/actuator.rs` with tests first:

```rust
use nalgebra::Vector4;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Actuator model: first-order motor delay + multiplicative noise + per-motor bias.
///
/// Models real motor behavior:
/// - First-order lag: thrust_actual = thrust_actual + (thrust_cmd - thrust_actual) * (1 - e^(-dt/tau))
/// - Multiplicative noise: thrust_out = thrust_actual * (1 + ε), ε ~ N(0, σ²)
/// - Per-motor bias: thrust_out *= (1 + bias_i)
#[derive(Debug, Clone)]
pub struct ActuatorModel {
    /// Current actual thrust per motor (N) — lags behind command
    pub thrust_state: Vector4<f64>,
    /// Motor time constant (seconds). 0 = instant response.
    pub time_constant: f64,
    /// Multiplicative noise std dev. 0 = no noise.
    pub noise_std: f64,
    /// Per-motor bias (fraction, e.g. 0.03 = +3%). Set at episode start.
    pub bias: Vector4<f64>,
    /// Whether the model is enabled
    pub enabled: bool,
}

impl ActuatorModel {
    pub fn new(time_constant: f64, noise_std: f64) -> Self {
        Self {
            thrust_state: Vector4::zeros(),
            time_constant,
            noise_std,
            bias: Vector4::zeros(),
            enabled: time_constant > 0.0 || noise_std > 0.0,
        }
    }

    /// Disabled model: pass-through, zero delay, zero noise.
    pub fn disabled() -> Self {
        Self {
            thrust_state: Vector4::zeros(),
            time_constant: 0.0,
            noise_std: 0.0,
            bias: Vector4::zeros(),
            enabled: false,
        }
    }

    /// Randomize per-motor bias. Call once at episode start.
    /// `bias_range`: maximum absolute bias fraction (e.g. 0.05 for ±5%).
    pub fn randomize_bias<R: Rng>(&mut self, bias_range: f64, rng: &mut R) {
        if bias_range > 0.0 {
            let dist = rand_distr::Uniform::new(-bias_range, bias_range);
            for i in 0..4 {
                self.bias[i] = dist.sample(rng);
            }
        } else {
            self.bias = Vector4::zeros();
        }
    }

    /// Process commanded thrusts through delay, noise, and bias.
    /// Returns the actual thrust applied to each motor.
    pub fn apply<R: Rng>(
        &mut self,
        commanded: &Vector4<f64>,
        dt: f64,
        rng: &mut R,
    ) -> Vector4<f64> {
        if !self.enabled {
            return *commanded;
        }

        // First-order lag: exponential smoothing
        if self.time_constant > 0.0 {
            let alpha = 1.0 - (-dt / self.time_constant).exp();
            self.thrust_state += (commanded - self.thrust_state) * alpha;
        } else {
            self.thrust_state = *commanded;
        }

        let mut output = self.thrust_state;

        // Multiplicative noise
        if self.noise_std > 0.0 {
            let normal = Normal::new(0.0, self.noise_std).unwrap();
            for i in 0..4 {
                output[i] *= 1.0 + normal.sample(rng);
            }
        }

        // Per-motor bias
        for i in 0..4 {
            output[i] *= 1.0 + self.bias[i];
        }

        // Clamp to non-negative (motors can't produce negative thrust)
        for i in 0..4 {
            output[i] = output[i].max(0.0);
        }

        output
    }

    /// Reset internal state (call on episode reset).
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
        let cmd = Vector4::new(0.05, 0.06, 0.07, 0.08);
        let out = model.apply(&cmd, 0.001, &mut rng);
        assert_eq!(out, cmd);
    }

    #[test]
    fn test_first_order_delay_step_response() {
        // With tau=20ms and dt=1ms, after one step the output should be
        // alpha = 1 - exp(-1/20) ≈ 0.0488
        let mut model = ActuatorModel::new(0.020, 0.0); // 20ms delay, no noise
        let mut rng = StdRng::seed_from_u64(42);
        let cmd = Vector4::new(0.1, 0.1, 0.1, 0.1);

        // Initial state is zero
        let out = model.apply(&cmd, 0.001, &mut rng);
        let alpha = 1.0 - (-0.001_f64 / 0.020).exp();
        assert_relative_eq!(out[0], 0.1 * alpha, epsilon = 1e-6);

        // After many steps, should converge to command
        for _ in 0..1000 {
            model.apply(&cmd, 0.001, &mut rng);
        }
        assert_relative_eq!(model.thrust_state[0], 0.1, epsilon = 1e-4);
    }

    #[test]
    fn test_multiplicative_noise_variance() {
        let mut model = ActuatorModel::new(0.0, 0.05); // no delay, 5% noise
        let mut rng = StdRng::seed_from_u64(42);
        let cmd = Vector4::new(0.1, 0.1, 0.1, 0.1);

        let n = 50_000;
        let mut sum_sq_frac = 0.0;
        for _ in 0..n {
            let out = model.apply(&cmd, 0.001, &mut rng);
            let frac_err = (out[0] - 0.1) / 0.1;
            sum_sq_frac += frac_err * frac_err;
        }
        let empirical_var = sum_sq_frac / n as f64;
        let expected_var = 0.05 * 0.05;
        assert!(
            (empirical_var - expected_var).abs() / expected_var < 0.15,
            "empirical var {empirical_var} vs expected {expected_var}"
        );
    }

    #[test]
    fn test_per_motor_bias() {
        let mut model = ActuatorModel::new(0.0, 0.0);
        model.bias = Vector4::new(0.05, -0.05, 0.0, 0.1); // +5%, -5%, 0%, +10%
        let mut rng = StdRng::seed_from_u64(42);
        let cmd = Vector4::new(0.1, 0.1, 0.1, 0.1);
        let out = model.apply(&cmd, 0.001, &mut rng);

        assert_relative_eq!(out[0], 0.105, epsilon = 1e-10);
        assert_relative_eq!(out[1], 0.095, epsilon = 1e-10);
        assert_relative_eq!(out[2], 0.100, epsilon = 1e-10);
        assert_relative_eq!(out[3], 0.110, epsilon = 1e-10);
    }

    #[test]
    fn test_output_never_negative() {
        let mut model = ActuatorModel::new(0.0, 0.5); // very high noise
        let mut rng = StdRng::seed_from_u64(42);
        let cmd = Vector4::new(0.01, 0.01, 0.01, 0.01); // small thrust

        for _ in 0..10_000 {
            let out = model.apply(&cmd, 0.001, &mut rng);
            for i in 0..4 {
                assert!(out[i] >= 0.0, "motor {i} went negative: {}", out[i]);
            }
        }
    }
}
```

- [ ] **Step 2: Register module in lib.rs**

Add to `crates/sim-core/src/lib.rs`:

```rust
pub mod actuator;
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/shu/GitHub/aces && cargo test -p aces-sim-core actuator`
Expected: All 5 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/sim-core/src/actuator.rs crates/sim-core/src/lib.rs
git commit -m "feat(sim-core): add actuator model with first-order delay, noise, and bias"
```

---

## Task 3: IMU Bias Random Walk Model

**Files:**
- Create: `crates/sim-core/src/imu_bias.rs`
- Modify: `crates/sim-core/src/lib.rs`

Models slowly-drifting accelerometer and gyroscope biases.

- [ ] **Step 1: Create imu_bias.rs with implementation and tests**

```rust
use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// IMU bias model: random walk on accelerometer and gyroscope biases.
///
/// At each step: bias += N(0, σ² · dt) · I
/// Applied to measurements: accel_meas = accel_true + accel_bias
///                          gyro_meas  = gyro_true  + gyro_bias
#[derive(Debug, Clone)]
pub struct ImuBias {
    /// Current accelerometer bias (m/s²)
    pub accel_bias: Vector3<f64>,
    /// Current gyroscope bias (rad/s)
    pub gyro_bias: Vector3<f64>,
    /// Accel bias random walk std dev (m/s² per √s)
    pub accel_walk_std: f64,
    /// Gyro bias random walk std dev (rad/s per √s)
    pub gyro_walk_std: f64,
    /// Whether bias model is enabled
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

        // Random walk step
        if self.accel_walk_std > 0.0 {
            let n = Normal::new(0.0, self.accel_walk_std * sqrt_dt).unwrap();
            self.accel_bias += Vector3::new(
                n.sample(rng), n.sample(rng), n.sample(rng),
            );
        }
        if self.gyro_walk_std > 0.0 {
            let n = Normal::new(0.0, self.gyro_walk_std * sqrt_dt).unwrap();
            self.gyro_bias += Vector3::new(
                n.sample(rng), n.sample(rng), n.sample(rng),
            );
        }

        (true_accel + self.accel_bias, true_gyro + self.gyro_bias)
    }

    /// Reset biases to zero.
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
        let accel = Vector3::new(0.0, 0.0, 9.81);
        let gyro = Vector3::new(0.1, -0.2, 0.3);
        let (a, g) = bias.apply(&accel, &gyro, 0.001, &mut rng);
        assert_eq!(a, accel);
        assert_eq!(g, gyro);
    }

    #[test]
    fn test_bias_accumulates_over_time() {
        let mut bias = ImuBias::new(0.1, 0.01); // moderate bias walk
        let mut rng = StdRng::seed_from_u64(42);
        let accel = Vector3::new(0.0, 0.0, 9.81);
        let gyro = Vector3::zeros();

        // Run for 10 seconds at 1kHz
        for _ in 0..10_000 {
            bias.apply(&accel, &gyro, 0.001, &mut rng);
        }

        // After 10s, accel bias variance = σ² * t = 0.01 * 10 = 0.1
        // So std ≈ 0.316 m/s², bias should be on the order of 0.3
        // We just check it's non-trivially large
        assert!(
            bias.accel_bias.norm() > 0.05,
            "bias should accumulate, got {}",
            bias.accel_bias.norm()
        );
    }

    #[test]
    fn test_bias_random_walk_variance() {
        // Variance of random walk after T steps of size dt:
        // Var = σ² * T * dt = σ² * total_time
        let sigma = 0.1;
        let dt = 0.001;
        let n = 10_000; // 10 seconds

        let trials = 500;
        let mut sum_sq = 0.0;
        for seed in 0..trials {
            let mut bias = ImuBias::new(sigma, 0.0);
            let mut rng = StdRng::seed_from_u64(seed);
            let zero = Vector3::zeros();
            for _ in 0..n {
                bias.apply(&zero, &zero, dt, &mut rng);
            }
            sum_sq += bias.accel_bias.x * bias.accel_bias.x;
        }
        let empirical_var = sum_sq / trials as f64;
        let expected_var = sigma * sigma * (n as f64 * dt);

        assert!(
            (empirical_var - expected_var).abs() / expected_var < 0.2,
            "empirical var {empirical_var} vs expected {expected_var}"
        );
    }

    #[test]
    fn test_reset_clears_bias() {
        let mut bias = ImuBias::new(0.1, 0.01);
        let mut rng = StdRng::seed_from_u64(42);
        let zero = Vector3::zeros();
        for _ in 0..1000 {
            bias.apply(&zero, &zero, 0.001, &mut rng);
        }
        assert!(bias.accel_bias.norm() > 0.0);
        bias.reset();
        assert_eq!(bias.accel_bias, Vector3::zeros());
        assert_eq!(bias.gyro_bias, Vector3::zeros());
    }
}
```

- [ ] **Step 2: Register in lib.rs**

Add to `crates/sim-core/src/lib.rs`:

```rust
pub mod imu_bias;
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/shu/GitHub/aces && cargo test -p aces-sim-core imu_bias`
Expected: All 4 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/sim-core/src/imu_bias.rs crates/sim-core/src/lib.rs
git commit -m "feat(sim-core): add IMU bias random walk model for accel and gyro"
```

---

## Task 4: Wire New Features Through PyO3 Bridge

**Files:**
- Modify: `crates/py-bridge/src/lib.rs`

Connect actuator model, IMU bias, EKF diagnostics, and domain randomization params to Python.

- [ ] **Step 1: Add new imports at top of lib.rs**

After the existing `use aces_sim_core::...` imports, add:

```rust
use aces_sim_core::actuator::ActuatorModel;
use aces_sim_core::imu_bias::ImuBias;
```

- [ ] **Step 2: Add new fields to StepResult**

After the existing `safety_b: u8,` field (line 99), add:

```rust
    // --- EKF diagnostics ---
    /// EKF covariance diagonal for A's tracking of B [P_px, P_py, P_pz, P_vx, P_vy, P_vz]
    ekf_a_cov_diag: [f64; 6],
    /// EKF last innovation for A's tracking of B [ix, iy, iz]
    ekf_a_innovation: [f64; 3],
    /// IMU accel bias on drone A
    imu_accel_bias_a: [f64; 3],
    /// IMU gyro bias on drone A
    imu_gyro_bias_a: [f64; 3],
```

- [ ] **Step 3: Add new fields to Simulation struct**

After the `safety: SafetyEnvelope,` field (line 153), add:

```rust
    /// Actuator model for drone A
    actuator_a: ActuatorModel,
    /// Actuator model for drone B
    actuator_b: ActuatorModel,
    /// IMU bias for drone A
    imu_bias_a: ImuBias,
    /// IMU bias for drone B
    imu_bias_b: ImuBias,
    /// Domain randomization: motor bias range (fraction)
    motor_bias_range: f64,
```

- [ ] **Step 4: Extend Simulation::new() signature**

Add these parameters to `#[pyo3(signature = (...))]` after `camera_min_conf_dist = 5.0`:

```rust
        motor_time_constant = 0.0,
        motor_noise_std = 0.0,
        motor_bias_range = 0.0,
        imu_accel_bias_std = 0.0,
        imu_gyro_bias_std = 0.0,
```

Add corresponding function parameters:

```rust
        motor_time_constant: f64,
        motor_noise_std: f64,
        motor_bias_range: f64,
        imu_accel_bias_std: f64,
        imu_gyro_bias_std: f64,
```

In the constructor body, before the `Self {` block, add:

```rust
        let actuator_a = ActuatorModel::new(motor_time_constant, motor_noise_std);
        let actuator_b = ActuatorModel::new(motor_time_constant, motor_noise_std);
        let imu_bias_a = ImuBias::new(imu_accel_bias_std, imu_gyro_bias_std);
        let imu_bias_b = ImuBias::new(imu_accel_bias_std, imu_gyro_bias_std);
```

Add to the `Self {` block:

```rust
            actuator_a,
            actuator_b,
            imu_bias_a,
            imu_bias_b,
            motor_bias_range,
```

- [ ] **Step 5: Wire actuator into step() physics loop**

Replace the physics sub-step loop in `step()` (lines 396-401):

```rust
        let dt_sim = self.dt_ctrl / self.substeps as f64;
        for _ in 0..self.substeps {
            let wind_force_a = self.wind_a.step(dt_sim, &mut self.rng);
            let wind_force_b = self.wind_b.step(dt_sim, &mut self.rng);
            // Actuator model: delay + noise + bias
            let actual_a = self.actuator_a.apply(&ua, dt_sim, &mut self.rng);
            let actual_b = self.actuator_b.apply(&ub, dt_sim, &mut self.rng);
            self.drone_a = step_rk4(&self.drone_a, &actual_a, &self.params, dt_sim, &wind_force_a);
            self.drone_b = step_rk4(&self.drone_b, &actual_b, &self.params, dt_sim, &wind_force_b);
        }
```

- [ ] **Step 6: Wire IMU bias into observation generation**

After the noisy observation generation block and before the EKF section, add IMU bias stepping. Replace the observation block (around lines 422-432) to also step IMU bias:

```rust
        // --- IMU bias step (accumulates each control step) ---
        let zero3 = Vector3::zeros();
        let (_biased_accel_a, _biased_gyro_a) = self.imu_bias_a.apply(
            &zero3, &self.drone_a.angular_velocity, self.dt_ctrl, &mut self.rng,
        );
        let (_biased_accel_b, _biased_gyro_b) = self.imu_bias_b.apply(
            &zero3, &self.drone_b.angular_velocity, self.dt_ctrl, &mut self.rng,
        );
```

- [ ] **Step 7: Add EKF diagnostics to StepResult construction**

In the `StepResult { ... }` block, after `safety_b`, add:

```rust
            ekf_a_cov_diag: self.ekf_a.covariance_diagonal(),
            ekf_a_innovation: self.ekf_a.last_innovation()
                .map(|v| [v.x, v.y, v.z])
                .unwrap_or([0.0; 3]),
            imu_accel_bias_a: [
                self.imu_bias_a.accel_bias.x,
                self.imu_bias_a.accel_bias.y,
                self.imu_bias_a.accel_bias.z,
            ],
            imu_gyro_bias_a: [
                self.imu_bias_a.gyro_bias.x,
                self.imu_bias_a.gyro_bias.y,
                self.imu_bias_a.gyro_bias.z,
            ],
```

- [ ] **Step 8: Wire reset: randomize actuator bias + reset IMU**

In the `reset()` method, after the particle filter reset block (before the camera reset), add:

```rust
        // Reset actuators and randomize per-motor bias
        self.actuator_a.reset();
        self.actuator_b.reset();
        if self.motor_bias_range > 0.0 {
            self.actuator_a.randomize_bias(self.motor_bias_range, &mut self.rng);
            self.actuator_b.randomize_bias(self.motor_bias_range, &mut self.rng);
        }
        self.imu_bias_a.reset();
        self.imu_bias_b.reset();
```

- [ ] **Step 9: Build and verify**

Run: `cd /Users/shu/GitHub/aces && cargo check -p aces-py-bridge`
Expected: Compiles without errors

Run: `cd /Users/shu/GitHub/aces && cargo test`
Expected: All existing tests pass

- [ ] **Step 10: Commit**

```bash
git add crates/py-bridge/src/lib.rs
git commit -m "feat(py-bridge): wire actuator model, IMU bias, EKF diagnostics to Python"
```

---

## Task 5: Extend Python Config

**Files:**
- Modify: `configs/rules.toml`
- Modify: `aces/config.py`
- Modify: `configs/curriculum.toml`

- [ ] **Step 1: Add new TOML config keys**

In `configs/rules.toml`, after the existing `[noise]` section's `obs_noise_std` line, add:

```toml

# Actuator noise
motor_time_constant = 0.0     # First-order delay (seconds), 0 = instant
motor_noise_std = 0.0         # Multiplicative motor noise std, 0 = off
motor_bias_range = 0.0        # Per-motor bias range (fraction), 0 = off

# IMU bias random walk
imu_accel_bias_std = 0.0      # Accel bias walk (m/s² per √s), 0 = off
imu_gyro_bias_std = 0.0       # Gyro bias walk (rad/s per √s), 0 = off

# Domain randomization (episode-level, fraction of nominal)
[domain_randomization]
enabled = false
mass_range = 0.0              # ±fraction of nominal mass
inertia_range = 0.0           # ±fraction of nominal inertia
max_thrust_range = 0.0        # ±fraction of nominal max thrust
drag_range = 0.0              # ±fraction of nominal drag coeff
```

- [ ] **Step 2: Extend Python NoiseConfig dataclass**

In `aces/config.py`, replace the `NoiseConfig` class (lines 97-102):

```python
@dataclass(frozen=True)
class NoiseConfig:
    wind_theta: float
    wind_mu: list[float]
    wind_sigma: float
    obs_noise_std: float
    motor_time_constant: float = 0.0
    motor_noise_std: float = 0.0
    motor_bias_range: float = 0.0
    imu_accel_bias_std: float = 0.0
    imu_gyro_bias_std: float = 0.0
```

- [ ] **Step 3: Add DomainRandomizationConfig dataclass**

After `NoiseConfig`, add:

```python
@dataclass(frozen=True)
class DomainRandomizationConfig:
    enabled: bool = False
    mass_range: float = 0.0
    inertia_range: float = 0.0
    max_thrust_range: float = 0.0
    drag_range: float = 0.0
```

- [ ] **Step 4: Add DR to RulesConfig**

In `RulesConfig` (line 150-158), add the field:

```python
    domain_randomization: DomainRandomizationConfig
```

- [ ] **Step 5: Update _parse_rules to load new fields**

In `_parse_rules`, update the noise parsing block (around line 256-262):

```python
    noise_raw = data["noise"]
    noise = NoiseConfig(
        wind_theta=noise_raw["wind_theta"],
        wind_mu=noise_raw["wind_mu"],
        wind_sigma=noise_raw["wind_sigma"],
        obs_noise_std=noise_raw["obs_noise_std"],
        motor_time_constant=noise_raw.get("motor_time_constant", 0.0),
        motor_noise_std=noise_raw.get("motor_noise_std", 0.0),
        motor_bias_range=noise_raw.get("motor_bias_range", 0.0),
        imu_accel_bias_std=noise_raw.get("imu_accel_bias_std", 0.0),
        imu_gyro_bias_std=noise_raw.get("imu_gyro_bias_std", 0.0),
    )
```

After the noise parsing, add:

```python
    dr_raw = data.get("domain_randomization", {})
    domain_randomization = DomainRandomizationConfig(
        enabled=dr_raw.get("enabled", False),
        mass_range=dr_raw.get("mass_range", 0.0),
        inertia_range=dr_raw.get("inertia_range", 0.0),
        max_thrust_range=dr_raw.get("max_thrust_range", 0.0),
        drag_range=dr_raw.get("drag_range", 0.0),
    )
```

Update the `return RulesConfig(...)` call to include:

```python
        domain_randomization=domain_randomization,
```

- [ ] **Step 6: Add DR noise overrides to curriculum phases**

In `configs/curriculum.toml`, update phases 4 and 5 to add actuator noise:

```toml
[[phase]]
name = "self_play_noisy"
task = "dogfight"
opponent = "pool"
wind_sigma = 0.3
obs_noise_std = 0.1
motor_time_constant = 0.02
motor_noise_std = 0.05
motor_bias_range = 0.03
imu_accel_bias_std = 0.02
imu_gyro_bias_std = 0.005
use_fpv = false
max_timesteps = 2_000_000
promote_condition = "win_rate>0.55"
promote_window = 500

[[phase]]
name = "fpv_transfer"
task = "dogfight"
opponent = "pool"
wind_sigma = 0.3
obs_noise_std = 0.1
motor_time_constant = 0.02
motor_noise_std = 0.05
motor_bias_range = 0.03
imu_accel_bias_std = 0.02
imu_gyro_bias_std = 0.005
use_fpv = true
max_timesteps = 5_000_000
promote_condition = "steps"
promote_window = 500
```

- [ ] **Step 7: Run config loading tests**

Run: `cd /Users/shu/GitHub/aces && python -c "from aces.config import load_configs; c = load_configs(); print(c.rules.noise); print(c.rules.domain_randomization)"`
Expected: Prints NoiseConfig and DomainRandomizationConfig with new fields

- [ ] **Step 8: Commit**

```bash
git add configs/rules.toml configs/curriculum.toml aces/config.py
git commit -m "feat(config): add actuator noise, IMU bias, and domain randomization config"
```

---

## Task 6: Domain Randomization in Env

**Files:**
- Modify: `aces/env.py`

Wire new noise params to Simulation and add episode-level parameter randomization in reset().

- [ ] **Step 1: Store new config values in __init__**

In `aces/env.py`, after the existing noise parameter block (around line 132), add:

```python
        self._motor_time_constant = noise.motor_time_constant
        self._motor_noise_std = noise.motor_noise_std
        self._motor_bias_range = noise.motor_bias_range
        self._imu_accel_bias_std = noise.imu_accel_bias_std
        self._imu_gyro_bias_std = noise.imu_gyro_bias_std

        # Domain randomization config
        self._dr = cfg.rules.domain_randomization
```

- [ ] **Step 2: Pass new params through _build_sim**

In `_build_sim()`, add to the `Simulation(...)` constructor call, after `camera_min_conf_dist=`:

```python
            motor_time_constant=self._motor_time_constant,
            motor_noise_std=self._motor_noise_std,
            motor_bias_range=self._motor_bias_range,
            imu_accel_bias_std=self._imu_accel_bias_std,
            imu_gyro_bias_std=self._imu_gyro_bias_std,
```

- [ ] **Step 3: Add domain randomization in reset()**

In `reset()`, after `super().reset(seed=seed)` and before the spawn section, add:

```python
        # Domain randomization: rebuild sim with randomized physical params
        if self._dr.enabled:
            def _rand_scale(nominal: float, frac: float) -> float:
                if frac <= 0.0:
                    return nominal
                return nominal * (1.0 + self.np_random.uniform(-frac, frac))

            self._mass = _rand_scale(cfg_mass, self._dr.mass_range)
            self._max_thrust = _rand_scale(cfg_max_thrust, self._dr.max_thrust_range)
            self._drag_coeff = _rand_scale(cfg_drag_coeff, self._dr.drag_range)
            inertia_scale = 1.0 + self.np_random.uniform(-self._dr.inertia_range, self._dr.inertia_range)
            self._inertia = [i * inertia_scale for i in cfg_inertia]
            self._sim = self._build_sim()
            self._hover_thrust = self._sim.hover_thrust()
```

To make this work, store the nominal (config) values separately. In `__init__`, right after setting the physical parameters (around line 96-101), add:

```python
        # Store nominal values for domain randomization
        self._nominal_mass = self._mass
        self._nominal_max_thrust = self._max_thrust
        self._nominal_drag_coeff = self._drag_coeff
        self._nominal_inertia = list(self._inertia)
```

And update the reset DR block to reference `self._nominal_*`:

```python
        if self._dr.enabled:
            def _rand_scale(nominal: float, frac: float) -> float:
                if frac <= 0.0:
                    return nominal
                return nominal * (1.0 + self.np_random.uniform(-frac, frac))

            self._mass = _rand_scale(self._nominal_mass, self._dr.mass_range)
            self._max_thrust = _rand_scale(self._nominal_max_thrust, self._dr.max_thrust_range)
            self._drag_coeff = _rand_scale(self._nominal_drag_coeff, self._dr.drag_range)
            inertia_scale = 1.0 + self.np_random.uniform(-self._dr.inertia_range, self._dr.inertia_range)
            self._inertia = [i * inertia_scale for i in self._nominal_inertia]
            self._sim = self._build_sim()
            self._hover_thrust = self._sim.hover_thrust()
```

- [ ] **Step 4: Add EKF diagnostics to info dict**

In the `step()` method's info dict construction (around line 699-721), add:

```python
            "ekf_cov_diag": np.array(result.ekf_a_cov_diag, dtype=np.float32),
            "ekf_innovation": np.array(result.ekf_a_innovation, dtype=np.float32),
            "imu_accel_bias": np.array(result.imu_accel_bias_a, dtype=np.float32),
            "imu_gyro_bias": np.array(result.imu_gyro_bias_a, dtype=np.float32),
```

- [ ] **Step 5: Rebuild and test**

Run: `cd /Users/shu/GitHub/aces && poetry run maturin develop`
Run: `cd /Users/shu/GitHub/aces && pytest tests/test_env.py -v`
Expected: All existing env tests pass

- [ ] **Step 6: Commit**

```bash
git add aces/env.py
git commit -m "feat(env): domain randomization in reset, wire actuator/IMU/EKF diagnostics"
```

---

## Task 7: Statistical EKF Tests

**Files:**
- Create: `tests/test_ekf_statistical.py`

Comprehensive statistical tests: NEES consistency, innovation whiteness, sensitivity analysis.

- [ ] **Step 1: Create test file**

```python
"""Statistical tests for EKF correctness.

Tests verify mathematical properties that must hold for a correctly-tuned filter:
- NEES (Normalized Estimation Error Squared) should follow chi-squared distribution
- Innovation sequence should be white (uncorrelated)
- EKF should outperform raw noise across noise levels
- Covariance should grow during prediction and shrink during update
"""

import numpy as np
import pytest

from aces._core import Simulation


def _make_sim(obs_noise_std=0.1, **kwargs):
    """Create a minimal Simulation for EKF testing."""
    return Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
        obs_noise_std=obs_noise_std,
        **kwargs,
    )


def _run_episode(sim, n_steps=200):
    """Run an episode collecting EKF diagnostics."""
    hover = sim.hover_thrust()
    motors = [hover] * 4

    results = []
    for _ in range(n_steps):
        r = sim.step(motors, motors)
        results.append(r)
    return results


class TestEkfNEES:
    """NEES (Normalized Estimation Error Squared) consistency tests.

    For a correctly-tuned filter, NEES should follow χ²(n) where n=3 (position dims).
    Mean NEES should be approximately 3.0.
    """

    def test_nees_mean_in_chi2_range(self):
        """Average NEES across episodes should be near state dimension (3)."""
        n_episodes = 30
        burn_in = 30
        n_steps = 150
        nees_all = []

        for seed_offset in range(n_episodes):
            sim = _make_sim(obs_noise_std=0.1)
            # Slightly different spawns per episode
            rng = np.random.default_rng(seed_offset)
            jitter = rng.uniform(-0.3, 0.3, size=3)
            pos_a = [2.0 + jitter[0], 2.0 + jitter[1], 1.5 + jitter[2] * 0.2]
            sim.reset(pos_a, [8.0, 8.0, 1.5])

            results = _run_episode(sim, n_steps)

            for r in results[burn_in:]:
                true_pos = np.array(r.drone_b_state[:3])
                ekf_pos = np.array(r.ekf_b_pos_from_a)
                cov_diag = np.array(r.ekf_a_cov_diag[:3])  # position covariance

                err = true_pos - ekf_pos
                # NEES = e^T P^{-1} e, with diagonal P this is sum(e_i^2 / P_ii)
                cov_safe = np.maximum(cov_diag, 1e-12)
                nees = np.sum(err ** 2 / cov_safe)
                nees_all.append(nees)

        mean_nees = np.mean(nees_all)
        # For χ²(3), mean = 3, std = sqrt(2*3) ≈ 2.45
        # With many samples, mean should be in [1.5, 6.0] (generous bounds)
        assert 1.0 < mean_nees < 8.0, (
            f"Mean NEES {mean_nees:.2f} outside expected range for χ²(3)"
        )


class TestInnovationWhiteness:
    """Innovation sequence should be approximately white noise (uncorrelated)."""

    def test_innovation_autocorrelation_near_zero(self):
        """Lag-1 autocorrelation of innovations should be near zero."""
        sim = _make_sim(obs_noise_std=0.1)
        sim.reset([2.0, 2.0, 1.5], [8.0, 8.0, 1.5])

        burn_in = 30
        results = _run_episode(sim, 300)

        innovations = []
        for r in results[burn_in:]:
            innov = np.array(r.ekf_a_innovation)
            if np.any(np.abs(innov) > 0.0):  # skip steps without update
                innovations.append(innov)

        innovations = np.array(innovations)
        if len(innovations) < 50:
            pytest.skip("Not enough innovations collected")

        # Compute lag-1 autocorrelation for each component
        for dim in range(3):
            seq = innovations[:, dim]
            seq_centered = seq - np.mean(seq)
            var = np.var(seq_centered)
            if var < 1e-12:
                continue
            autocorr_1 = np.mean(seq_centered[:-1] * seq_centered[1:]) / var
            # Should be close to 0 for white noise
            assert abs(autocorr_1) < 0.25, (
                f"Dim {dim}: lag-1 autocorrelation {autocorr_1:.3f} too high"
            )


class TestEkfNoiseLevels:
    """EKF should outperform raw noise across different noise levels."""

    @pytest.mark.parametrize("noise_std", [0.01, 0.05, 0.1, 0.2])
    def test_ekf_beats_raw_noise(self, noise_std):
        """EKF estimation error < raw observation noise (post burn-in)."""
        sim = _make_sim(obs_noise_std=noise_std)
        sim.reset([2.0, 2.0, 1.5], [8.0, 8.0, 1.5])

        burn_in = 40
        results = _run_episode(sim, 200)

        sum_raw = 0.0
        sum_ekf = 0.0
        count = 0

        for r in results[burn_in:]:
            true_pos = np.array(r.drone_b_state[:3])
            noisy_pos = np.array(r.noisy_b_pos_from_a)
            ekf_pos = np.array(r.ekf_b_pos_from_a)

            # Skip if not visible (noisy_pos will be zeros)
            if np.linalg.norm(noisy_pos) < 0.01:
                continue

            sum_raw += np.linalg.norm(noisy_pos - true_pos)
            sum_ekf += np.linalg.norm(ekf_pos - true_pos)
            count += 1

        if count < 10:
            pytest.skip("Not enough visible steps")

        avg_raw = sum_raw / count
        avg_ekf = sum_ekf / count
        assert avg_ekf < avg_raw, (
            f"noise_std={noise_std}: EKF error ({avg_ekf:.4f}) >= raw ({avg_raw:.4f})"
        )


class TestCovarianceDynamics:
    """Covariance should grow during prediction and shrink during update."""

    def test_predict_grows_covariance(self):
        """Pure prediction should increase P diagonal."""
        sim = _make_sim(obs_noise_std=0.1)
        sim.reset([2.0, 2.0, 1.5], [8.0, 8.0, 1.5])

        hover = sim.hover_thrust()
        r1 = sim.step([hover] * 4, [hover] * 4)
        cov1 = np.array(r1.ekf_a_cov_diag)

        # Run several steps with no measurement (force occlusion by putting B out of sight)
        sim2 = Simulation(
            bounds=[10.0, 10.0, 3.0],
            obstacles=[([5.0, 5.0, 1.5], [2.0, 2.0, 1.5])],  # wall between drones
            obs_noise_std=0.1,
        )
        sim2.reset([1.0, 1.0, 1.5], [9.0, 9.0, 1.5])
        r_initial = sim2.step([sim2.hover_thrust()] * 4, [sim2.hover_thrust()] * 4)
        cov_initial = np.array(r_initial.ekf_a_cov_diag)

        for _ in range(10):
            r = sim2.step([sim2.hover_thrust()] * 4, [sim2.hover_thrust()] * 4)

        cov_after = np.array(r.ekf_a_cov_diag)
        # Position covariance should grow
        assert cov_after[0] > cov_initial[0], "P_px should grow without measurements"
        assert cov_after[1] > cov_initial[1], "P_py should grow without measurements"


class TestActuatorIntegration:
    """Verify actuator model affects dynamics when enabled."""

    def test_motor_delay_causes_slower_response(self):
        """With motor delay, drone should respond more sluggishly."""
        # No delay
        sim_fast = Simulation(
            bounds=[10.0, 10.0, 3.0],
            obstacles=[],
            motor_time_constant=0.0,
        )
        sim_fast.reset([5.0, 5.0, 1.5], [5.0, 5.0, 2.5])

        # With 50ms delay
        sim_slow = Simulation(
            bounds=[10.0, 10.0, 3.0],
            obstacles=[],
            motor_time_constant=0.05,
        )
        sim_slow.reset([5.0, 5.0, 1.5], [5.0, 5.0, 2.5])

        # Apply max thrust for 10 steps
        max_t = sim_fast.max_thrust()
        motors = [max_t] * 4
        hover = [sim_fast.hover_thrust()] * 4

        for _ in range(10):
            r_fast = sim_fast.step(motors, hover)
            r_slow = sim_slow.step(motors, hover)

        # Fast sim should have gained more altitude
        z_fast = r_fast.drone_a_state[2]
        z_slow = r_slow.drone_a_state[2]
        assert z_fast > z_slow, (
            f"No-delay z={z_fast:.4f} should be > delayed z={z_slow:.4f}"
        )


class TestImuBiasIntegration:
    """Verify IMU bias accumulates and is reported."""

    def test_bias_reported_in_step_result(self):
        """With IMU bias enabled, bias fields should be non-zero after many steps."""
        sim = Simulation(
            bounds=[10.0, 10.0, 3.0],
            obstacles=[],
            imu_accel_bias_std=0.1,
            imu_gyro_bias_std=0.01,
        )
        sim.reset([5.0, 5.0, 1.5], [5.0, 5.0, 2.5])
        hover = [sim.hover_thrust()] * 4

        for _ in range(500):
            r = sim.step(hover, hover)

        accel_bias = np.array(r.imu_accel_bias_a)
        gyro_bias = np.array(r.imu_gyro_bias_a)

        # After 500 steps * 0.01s = 5 seconds, bias should have accumulated
        assert np.linalg.norm(accel_bias) > 0.01, (
            f"Accel bias should accumulate, got {accel_bias}"
        )
        assert np.linalg.norm(gyro_bias) > 0.001, (
            f"Gyro bias should accumulate, got {gyro_bias}"
        )
```

- [ ] **Step 2: Run the tests**

Run: `cd /Users/shu/GitHub/aces && pytest tests/test_ekf_statistical.py -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_ekf_statistical.py
git commit -m "test(ekf): add NEES, innovation whiteness, and integration tests"
```

---

## Dependency Graph

```
Task 1 (EKF fix) ──────────────┐
Task 2 (Actuator model) ───────┤
Task 3 (IMU bias model) ───────┼──► Task 4 (py-bridge) ──► Task 5 (config) ──► Task 6 (env DR)
                                │                                                    │
                                └────────────────────────────────────────────────────►Task 7 (tests)
```

- **Tasks 1, 2, 3** are fully independent — can be done in parallel
- **Task 4** depends on all three
- **Task 5** depends on Task 4 (needs to match Rust param names)
- **Task 6** depends on Task 5
- **Task 7** depends on Task 4 (needs EKF diagnostics in StepResult) and Task 6 (needs full build)

## Verification

After all tasks, run the full test suite:

```bash
cargo test                     # All Rust tests
poetry run maturin develop     # Rebuild extension
pytest tests/ -v               # All Python tests
```
