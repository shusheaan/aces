# Wire Up PyO3 Bridge + Python Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the fully-implemented Rust crates (sim-core, mppi, estimator) to Python via PyO3/maturin, then build a Gymnasium environment, Rerun visualization, and self-play PPO training loop on top.

**Architecture:** A `Simulation` pyclass in py-bridge holds both drones, the arena, and lock-on trackers. A single `step()` call advances physics (RK4 with substeps), checks collisions/OOB, updates lock-on, and returns a flat `StepResult` struct. Python env.py builds observations and computes rewards from StepResult. An `MppiController` pyclass wraps a new Rust MPPI optimizer for MPPI-vs-MPPI mode. Rerun logs 3D state each step for real-time visualization.

**Tech Stack:** Rust (nalgebra, rayon, pyo3 0.22), Python 3.11+ (maturin, gymnasium, stable-baselines3, rerun-sdk, numpy, toml)

---

## File Structure

```
crates/
  sim-core/src/state.rs         ← MODIFY: add from_array()
  mppi/src/lib.rs               ← MODIFY: add pub mod optimizer
  mppi/src/optimizer.rs          ← CREATE: MPPI optimizer loop
  py-bridge/src/lib.rs           ← REWRITE: full PyO3 bridge
python/aces/
  __init__.py                    ← MODIFY: update exports
  env.py                         ← REWRITE: Gymnasium env using Rust backend
  trainer.py                     ← REWRITE: SB3 self-play PPO
  viz.py                         ← REWRITE: Rerun 3D visualization
scripts/
  run.py                         ← REWRITE: main entry point
tests/
  test_core.py                   ← CREATE: bridge binding tests
  test_env.py                    ← CREATE: Gymnasium env tests
  test_dynamics.py               ← REWRITE: real dynamics tests
```

---

### Task 1: Rust — DroneState::from_array + MPPI Optimizer

**Files:**
- Modify: `crates/sim-core/src/state.rs`
- Create: `crates/mppi/src/optimizer.rs`
- Modify: `crates/mppi/src/lib.rs`

- [ ] **Step 1: Add from_array to DroneState**

Add this method inside the `impl DroneState` block (the non-trait one, after `to_array`) in `crates/sim-core/src/state.rs`:

```rust
    /// Reconstruct from flat array [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz].
    pub fn from_array(arr: &[f64; 13]) -> Self {
        Self {
            position: Vector3::new(arr[0], arr[1], arr[2]),
            velocity: Vector3::new(arr[3], arr[4], arr[5]),
            attitude: UnitQuaternion::from_quaternion(
                Quaternion::new(arr[6], arr[7], arr[8], arr[9]),
            ),
            angular_velocity: Vector3::new(arr[10], arr[11], arr[12]),
        }
    }
```

Also add `Quaternion` to the existing `use nalgebra::` import at the top of the file:

```rust
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
```

- [ ] **Step 2: Add Rust test for from_array round-trip**

Add this test inside the file `crates/sim-core/src/state.rs` (create a `#[cfg(test)]` module at the bottom):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_from_array_round_trip() {
        let state = DroneState::hover_at(Vector3::new(1.0, 2.0, 3.0));
        let arr = state.to_array();
        let reconstructed = DroneState::from_array(&arr);
        let arr2 = reconstructed.to_array();
        for i in 0..13 {
            assert!((arr[i] - arr2[i]).abs() < 1e-12, "mismatch at index {i}");
        }
    }
}
```

- [ ] **Step 3: Run the test**

```bash
cd /Users/shu/GitHub/aces && cargo test -p aces-sim-core test_from_array_round_trip
```

Expected: PASS

- [ ] **Step 4: Create MPPI optimizer**

Create file `crates/mppi/src/optimizer.rs`:

```rust
use crate::cost::{evasion_cost, pursuit_cost, CostWeights};
use crate::rollout::rollout;
use aces_sim_core::dynamics::DroneParams;
use aces_sim_core::environment::Arena;
use aces_sim_core::state::DroneState;
use nalgebra::Vector4;
use rand::Rng;
use rand::distributions::Distribution;
use rand_distr::Normal;
use rayon::prelude::*;

/// Full MPPI optimizer: sample, rollout, cost, softmax-weight, update mean.
pub struct MppiOptimizer {
    pub num_samples: usize,
    pub horizon: usize,
    pub noise_std: f64,
    pub temperature: f64,
    pub params: DroneParams,
    pub arena: Arena,
    pub weights: CostWeights,
    pub dt_ctrl: f64,
    pub substeps: usize,
    mean_controls: Vec<Vector4<f64>>,
}

impl MppiOptimizer {
    pub fn new(
        num_samples: usize,
        horizon: usize,
        noise_std: f64,
        temperature: f64,
        params: DroneParams,
        arena: Arena,
        weights: CostWeights,
        dt_ctrl: f64,
        substeps: usize,
    ) -> Self {
        let hover = params.hover_thrust();
        let hover_vec = Vector4::new(hover, hover, hover, hover);
        Self {
            num_samples,
            horizon,
            noise_std,
            temperature,
            params,
            arena,
            weights,
            dt_ctrl,
            substeps,
            mean_controls: vec![hover_vec; horizon],
        }
    }

    /// Compute optimal action for the current state.
    pub fn compute_action(
        &mut self,
        self_state: &DroneState,
        enemy_state: &DroneState,
        pursuit: bool,
    ) -> Vector4<f64> {
        let max_t = self.params.max_thrust;
        let hover = self.params.hover_thrust();

        // Generate perturbed control sequences in parallel.
        // Each thread gets its own RNG seeded from the main RNG.
        let seeds: Vec<u64> = {
            let mut rng = rand::thread_rng();
            (0..self.num_samples).map(|_| rng.gen()).collect()
        };

        let perturbed_and_costs: Vec<(Vec<Vector4<f64>>, f64)> = seeds
            .par_iter()
            .map(|&seed| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, self.noise_std).unwrap();

                // Generate perturbed sequence: mean + noise, clamped
                let controls: Vec<Vector4<f64>> = (0..self.horizon)
                    .map(|t| {
                        let mean = self.mean_controls[t];
                        Vector4::new(
                            (mean[0] + normal.sample(&mut rng)).clamp(0.0, max_t),
                            (mean[1] + normal.sample(&mut rng)).clamp(0.0, max_t),
                            (mean[2] + normal.sample(&mut rng)).clamp(0.0, max_t),
                            (mean[3] + normal.sample(&mut rng)).clamp(0.0, max_t),
                        )
                    })
                    .collect();

                // Rollout and compute cost
                let states =
                    rollout(self_state, &controls, &self.params, self.dt_ctrl, self.substeps);
                let mut total_cost = 0.0;
                for (t, state) in states[1..].iter().enumerate() {
                    total_cost += if pursuit {
                        pursuit_cost(
                            state,
                            enemy_state,
                            &controls[t],
                            hover,
                            &self.arena,
                            &self.weights,
                        )
                    } else {
                        evasion_cost(
                            state,
                            enemy_state,
                            &controls[t],
                            hover,
                            &self.arena,
                            &self.weights,
                        )
                    };
                }

                (controls, total_cost)
            })
            .collect();

        // Softmax weights
        let min_cost = perturbed_and_costs
            .iter()
            .map(|(_, c)| *c)
            .fold(f64::INFINITY, f64::min);
        let exp_costs: Vec<f64> = perturbed_and_costs
            .iter()
            .map(|(_, c)| (-(c - min_cost) / self.temperature).exp())
            .collect();
        let total_exp: f64 = exp_costs.iter().sum();

        // Weighted average of control sequences
        let mut new_mean = vec![Vector4::zeros(); self.horizon];
        for (k, (controls, _)) in perturbed_and_costs.iter().enumerate() {
            let w = exp_costs[k] / total_exp;
            for t in 0..self.horizon {
                new_mean[t] += controls[t] * w;
            }
        }

        let action = new_mean[0];

        // Warm-start: shift left, append hover
        self.mean_controls = new_mean[1..].to_vec();
        let hover_vec = Vector4::new(hover, hover, hover, hover);
        self.mean_controls.push(hover_vec);

        action
    }

    /// Reset mean controls to hover.
    pub fn reset(&mut self) {
        let hover = self.params.hover_thrust();
        let hover_vec = Vector4::new(hover, hover, hover, hover);
        self.mean_controls = vec![hover_vec; self.horizon];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aces_sim_core::environment::Arena;
    use nalgebra::Vector3;

    #[test]
    fn test_optimizer_returns_valid_action() {
        let params = DroneParams::crazyflie();
        let arena = Arena::warehouse();
        let weights = CostWeights::default();

        let mut opt = MppiOptimizer::new(
            64,  // fewer samples for test speed
            10,  // shorter horizon
            0.03,
            10.0,
            params.clone(),
            arena,
            weights,
            0.01,
            10,
        );

        let attacker = DroneState::hover_at(Vector3::new(2.0, 5.0, 1.5));
        let target = DroneState::hover_at(Vector3::new(8.0, 5.0, 1.5));

        let action = opt.compute_action(&attacker, &target, true);

        // All motor thrusts should be in [0, max_thrust]
        for i in 0..4 {
            assert!(action[i] >= 0.0, "motor {i} negative: {}", action[i]);
            assert!(
                action[i] <= params.max_thrust,
                "motor {i} exceeds max: {}",
                action[i]
            );
        }
    }

    #[test]
    fn test_optimizer_warm_start() {
        let params = DroneParams::crazyflie();
        let arena = Arena::warehouse();
        let weights = CostWeights::default();

        let mut opt = MppiOptimizer::new(32, 5, 0.03, 10.0, params, arena, weights, 0.01, 10);
        let a = DroneState::hover_at(Vector3::new(2.0, 5.0, 1.5));
        let b = DroneState::hover_at(Vector3::new(8.0, 5.0, 1.5));

        // First call sets up the mean
        opt.compute_action(&a, &b, true);
        // Second call uses warm-started mean
        let action2 = opt.compute_action(&a, &b, true);

        // Should still produce valid actions
        for i in 0..4 {
            assert!(action2[i] >= 0.0);
        }
    }
}
```

- [ ] **Step 5: Add `use rand::SeedableRng;` to optimizer.rs imports**

The `StdRng::seed_from_u64` call requires `SeedableRng` in scope. Add to the imports at the top of `crates/mppi/src/optimizer.rs`:

```rust
use rand::SeedableRng;
```

- [ ] **Step 6: Register optimizer module in mppi lib.rs**

Edit `crates/mppi/src/lib.rs` to add the new module:

```rust
pub mod cost;
pub mod optimizer;
pub mod rollout;
pub mod sampler;
```

- [ ] **Step 7: Run all Rust tests**

```bash
cd /Users/shu/GitHub/aces && cargo test
```

Expected: all tests pass including the new `test_from_array_round_trip`, `test_optimizer_returns_valid_action`, and `test_optimizer_warm_start`.

- [ ] **Step 8: Commit**

```bash
cd /Users/shu/GitHub/aces
git add crates/sim-core/src/state.rs crates/mppi/src/optimizer.rs crates/mppi/src/lib.rs
git commit -m "feat: add DroneState::from_array and MPPI optimizer with parallel rollouts"
```

---

### Task 2: Rust — PyO3 Bridge

**Files:**
- Rewrite: `crates/py-bridge/src/lib.rs`

- [ ] **Step 1: Rewrite py-bridge lib.rs**

Replace the entire contents of `crates/py-bridge/src/lib.rs` with:

```rust
use pyo3::prelude::*;

use aces_estimator::ekf::EKF as RustEKF;
use aces_mppi::cost::CostWeights;
use aces_mppi::optimizer::MppiOptimizer;
use aces_sim_core::dynamics::{step_rk4, DroneParams};
use aces_sim_core::environment::{Arena, Obstacle};
use aces_sim_core::lockon::{LockOnParams, LockOnTracker};
use aces_sim_core::state::DroneState;
use nalgebra::{Vector3, Vector4};

// ---------------------------------------------------------------------------
// StepResult — returned from Simulation.step()
// ---------------------------------------------------------------------------

#[pyclass(get_all)]
#[derive(Clone)]
struct StepResult {
    drone_a_state: [f64; 13],
    drone_b_state: [f64; 13],
    drone_a_forward: [f64; 3],
    drone_b_forward: [f64; 3],
    drone_a_euler: [f64; 3],
    drone_b_euler: [f64; 3],
    drone_a_collision: bool,
    drone_a_oob: bool,
    drone_b_collision: bool,
    drone_b_oob: bool,
    lock_a_progress: f64,
    lock_b_progress: f64,
    kill_a: bool,
    kill_b: bool,
    distance: f64,
    nearest_obs_dist_a: f64,
    nearest_obs_dist_b: f64,
}

// ---------------------------------------------------------------------------
// Simulation — holds two drones, arena, lock-on trackers
// ---------------------------------------------------------------------------

#[pyclass]
struct Simulation {
    arena: Arena,
    params: DroneParams,
    drone_a: DroneState,
    drone_b: DroneState,
    lock_a: LockOnTracker,
    lock_b: LockOnTracker,
    dt_ctrl: f64,
    substeps: usize,
}

fn build_arena(bounds: [f64; 3], obstacles: Vec<([f64; 3], [f64; 3])>, drone_radius: f64) -> Arena {
    let mut arena = Arena::new(Vector3::new(bounds[0], bounds[1], bounds[2]));
    arena.drone_radius = drone_radius;
    for (center, half_ext) in obstacles {
        arena.obstacles.push(Obstacle::Box {
            center: Vector3::new(center[0], center[1], center[2]),
            half_extents: Vector3::new(half_ext[0], half_ext[1], half_ext[2]),
        });
    }
    arena
}

fn build_params(
    mass: f64,
    arm_length: f64,
    inertia: [f64; 3],
    max_thrust: f64,
    torque_coeff: f64,
    drag_coeff: f64,
) -> DroneParams {
    DroneParams {
        mass,
        arm_length,
        inertia: Vector3::new(inertia[0], inertia[1], inertia[2]),
        max_thrust,
        torque_coeff,
        drag_coeff,
        gravity: 9.81,
    }
}

fn euler_from_quat(state: &DroneState) -> [f64; 3] {
    let (roll, pitch, yaw) = state.attitude.euler_angles();
    [roll, pitch, yaw]
}

fn v3(a: [f64; 3]) -> Vector3<f64> {
    Vector3::new(a[0], a[1], a[2])
}

#[pymethods]
impl Simulation {
    #[new]
    #[pyo3(signature = (
        bounds,
        obstacles,
        mass = 0.027,
        arm_length = 0.04,
        inertia = [1.4e-5, 1.4e-5, 2.17e-5],
        max_thrust = 0.15,
        torque_coeff = 0.005964,
        drag_coeff = 0.01,
        fov = 1.5707963267948966,
        lock_distance = 2.0,
        lock_duration = 1.5,
        dt_ctrl = 0.01,
        substeps = 10,
        drone_radius = 0.05,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bounds: [f64; 3],
        obstacles: Vec<([f64; 3], [f64; 3])>,
        mass: f64,
        arm_length: f64,
        inertia: [f64; 3],
        max_thrust: f64,
        torque_coeff: f64,
        drag_coeff: f64,
        fov: f64,
        lock_distance: f64,
        lock_duration: f64,
        dt_ctrl: f64,
        substeps: usize,
        drone_radius: f64,
    ) -> Self {
        let arena = build_arena(bounds, obstacles, drone_radius);
        let params = build_params(mass, arm_length, inertia, max_thrust, torque_coeff, drag_coeff);
        let lock_params = LockOnParams {
            fov,
            lock_distance,
            lock_duration,
        };
        Self {
            arena,
            params,
            drone_a: DroneState::default(),
            drone_b: DroneState::default(),
            lock_a: LockOnTracker::new(lock_params.clone()),
            lock_b: LockOnTracker::new(lock_params),
            dt_ctrl,
            substeps,
        }
    }

    /// Reset both drones to hover at given positions. Returns (state_a, state_b).
    fn reset(&mut self, pos_a: [f64; 3], pos_b: [f64; 3]) -> ([f64; 13], [f64; 13]) {
        self.drone_a = DroneState::hover_at(v3(pos_a));
        self.drone_b = DroneState::hover_at(v3(pos_b));
        self.lock_a.reset();
        self.lock_b.reset();
        (self.drone_a.to_array(), self.drone_b.to_array())
    }

    /// Advance one control step. motors_a/b are [f1,f2,f3,f4] in Newtons.
    fn step(&mut self, motors_a: [f64; 4], motors_b: [f64; 4]) -> StepResult {
        let ua = Vector4::new(motors_a[0], motors_a[1], motors_a[2], motors_a[3]);
        let ub = Vector4::new(motors_b[0], motors_b[1], motors_b[2], motors_b[3]);

        // Sub-step integration
        let dt_sim = self.dt_ctrl / self.substeps as f64;
        for _ in 0..self.substeps {
            self.drone_a = step_rk4(&self.drone_a, &ua, &self.params, dt_sim);
            self.drone_b = step_rk4(&self.drone_b, &ub, &self.params, dt_sim);
        }

        // Lock-on updates
        let kill_a =
            self.lock_a
                .update(&self.drone_a, &self.drone_b, &self.arena, self.dt_ctrl);
        let kill_b =
            self.lock_b
                .update(&self.drone_b, &self.drone_a, &self.arena, self.dt_ctrl);

        let fwd_a = self.drone_a.forward();
        let fwd_b = self.drone_b.forward();

        StepResult {
            drone_a_state: self.drone_a.to_array(),
            drone_b_state: self.drone_b.to_array(),
            drone_a_forward: [fwd_a.x, fwd_a.y, fwd_a.z],
            drone_b_forward: [fwd_b.x, fwd_b.y, fwd_b.z],
            drone_a_euler: euler_from_quat(&self.drone_a),
            drone_b_euler: euler_from_quat(&self.drone_b),
            drone_a_collision: self.arena.is_collision(&self.drone_a.position),
            drone_a_oob: self.arena.is_out_of_bounds(&self.drone_a.position),
            drone_b_collision: self.arena.is_collision(&self.drone_b.position),
            drone_b_oob: self.arena.is_out_of_bounds(&self.drone_b.position),
            lock_a_progress: self.lock_a.progress(),
            lock_b_progress: self.lock_b.progress(),
            kill_a,
            kill_b,
            distance: self.drone_a.distance_to(&self.drone_b),
            nearest_obs_dist_a: self.arena.obstacle_sdf(&self.drone_a.position),
            nearest_obs_dist_b: self.arena.obstacle_sdf(&self.drone_b.position),
        }
    }

    /// Hover thrust per motor (N).
    fn hover_thrust(&self) -> f64 {
        self.params.hover_thrust()
    }

    /// Max thrust per motor (N).
    fn max_thrust(&self) -> f64 {
        self.params.max_thrust
    }

    /// Current state arrays.
    fn drone_a_state(&self) -> [f64; 13] {
        self.drone_a.to_array()
    }

    fn drone_b_state(&self) -> [f64; 13] {
        self.drone_b.to_array()
    }

    /// SDF at a point (for debugging / observation).
    fn arena_sdf(&self, point: [f64; 3]) -> f64 {
        self.arena.sdf(&v3(point))
    }
}

// ---------------------------------------------------------------------------
// MppiController — wraps the MPPI optimizer for Python
// ---------------------------------------------------------------------------

#[pyclass]
struct MppiController {
    optimizer: MppiOptimizer,
}

#[pymethods]
impl MppiController {
    #[new]
    #[pyo3(signature = (
        bounds,
        obstacles,
        num_samples = 1024,
        horizon = 50,
        noise_std = 0.03,
        temperature = 10.0,
        mass = 0.027,
        arm_length = 0.04,
        inertia = [1.4e-5, 1.4e-5, 2.17e-5],
        max_thrust = 0.15,
        torque_coeff = 0.005964,
        drag_coeff = 0.01,
        dt_ctrl = 0.01,
        substeps = 10,
        drone_radius = 0.05,
        w_lock = 100.0,
        w_dist = 1.0,
        w_face = 5.0,
        w_vel = 0.1,
        w_ctrl = 0.01,
        w_obs = 1000.0,
        d_safe = 0.3,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bounds: [f64; 3],
        obstacles: Vec<([f64; 3], [f64; 3])>,
        num_samples: usize,
        horizon: usize,
        noise_std: f64,
        temperature: f64,
        mass: f64,
        arm_length: f64,
        inertia: [f64; 3],
        max_thrust: f64,
        torque_coeff: f64,
        drag_coeff: f64,
        dt_ctrl: f64,
        substeps: usize,
        drone_radius: f64,
        w_lock: f64,
        w_dist: f64,
        w_face: f64,
        w_vel: f64,
        w_ctrl: f64,
        w_obs: f64,
        d_safe: f64,
    ) -> Self {
        let arena = build_arena(bounds, obstacles, drone_radius);
        let params = build_params(mass, arm_length, inertia, max_thrust, torque_coeff, drag_coeff);
        let weights = CostWeights {
            w_lock,
            w_dist,
            w_face,
            w_vel,
            w_ctrl,
            w_obs,
            d_safe,
        };
        let optimizer = MppiOptimizer::new(
            num_samples, horizon, noise_std, temperature, params, arena, weights, dt_ctrl, substeps,
        );
        Self { optimizer }
    }

    /// Compute optimal motor thrusts [f1,f2,f3,f4] given 13-element state arrays.
    fn compute_action(
        &mut self,
        self_state: [f64; 13],
        enemy_state: [f64; 13],
        pursuit: bool,
    ) -> [f64; 4] {
        let s = DroneState::from_array(&self_state);
        let e = DroneState::from_array(&enemy_state);
        let a = self.optimizer.compute_action(&s, &e, pursuit);
        [a[0], a[1], a[2], a[3]]
    }

    /// Reset warm-start state.
    fn reset(&mut self) {
        self.optimizer.reset();
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<StepResult>()?;
    m.add_class::<Simulation>()?;
    m.add_class::<MppiController>()?;
    Ok(())
}
```

- [ ] **Step 2: Verify Rust compilation**

```bash
cd /Users/shu/GitHub/aces && cargo check
```

Expected: no errors.

- [ ] **Step 3: Run full Rust test suite**

```bash
cd /Users/shu/GitHub/aces && cargo test
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/shu/GitHub/aces
git add crates/py-bridge/src/lib.rs
git commit -m "feat: PyO3 bridge exposing Simulation, MppiController, StepResult"
```

---

### Task 3: Build Extension + Python Bridge Tests

**Files:**
- Create: `tests/test_core.py`
- Modify: `tests/test_dynamics.py`

- [ ] **Step 1: Build the maturin extension**

```bash
cd /Users/shu/GitHub/aces && .venv/bin/maturin develop --release
```

Expected: builds successfully, installs `aces._core` into the venv.

- [ ] **Step 2: Quick import smoke test**

```bash
cd /Users/shu/GitHub/aces && .venv/bin/python -c "from aces._core import Simulation, MppiController, StepResult; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 3: Write bridge tests**

Replace the contents of `tests/test_dynamics.py` with the following (the old file was just a placeholder):

```python
"""Tests for the Rust PyO3 bridge (aces._core)."""

from aces._core import Simulation, MppiController, StepResult


def test_simulation_creation():
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
    )
    assert sim.hover_thrust() > 0
    assert sim.max_thrust() > sim.hover_thrust()


def test_simulation_reset():
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
    )
    state_a, state_b = sim.reset([1.0, 1.0, 1.5], [9.0, 9.0, 1.5])
    assert len(state_a) == 13
    assert len(state_b) == 13
    assert abs(state_a[0] - 1.0) < 1e-9
    assert abs(state_b[0] - 9.0) < 1e-9


def test_simulation_hover_stationary():
    """At hover thrust, drone should stay approximately still."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
    )
    sim.reset([5.0, 5.0, 1.5], [5.0, 5.0, 1.5])
    hover = sim.hover_thrust()
    motors = [hover, hover, hover, hover]

    result = sim.step(motors, motors)
    pos = result.drone_a_state[:3]
    assert abs(pos[0] - 5.0) < 0.01
    assert abs(pos[1] - 5.0) < 0.01
    assert abs(pos[2] - 1.5) < 0.01


def test_simulation_freefall():
    """Zero thrust should cause descent."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
    )
    sim.reset([5.0, 5.0, 1.5], [5.0, 5.0, 1.5])
    zero = [0.0, 0.0, 0.0, 0.0]

    for _ in range(10):
        result = sim.step(zero, zero)

    pos = result.drone_a_state[:3]
    assert pos[2] < 1.5, "drone should have fallen"


def test_step_result_fields():
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
    )
    sim.reset([1.0, 1.0, 1.5], [9.0, 9.0, 1.5])
    hover = sim.hover_thrust()
    result = sim.step([hover] * 4, [hover] * 4)

    assert isinstance(result, StepResult)
    assert len(result.drone_a_state) == 13
    assert len(result.drone_a_forward) == 3
    assert len(result.drone_a_euler) == 3
    assert isinstance(result.drone_a_collision, bool)
    assert isinstance(result.kill_a, bool)
    assert isinstance(result.distance, float)
    assert isinstance(result.nearest_obs_dist_a, float)
    assert not result.drone_a_collision
    assert not result.drone_a_oob


def test_collision_detection():
    """Drone spawned inside an obstacle should be detected as collision."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
    )
    sim.reset([5.0, 5.0, 1.5], [1.0, 1.0, 1.5])  # drone_a inside pillar
    hover = sim.hover_thrust()
    result = sim.step([hover] * 4, [hover] * 4)
    assert result.drone_a_collision


def test_oob_detection():
    """Drone spawned outside bounds should be detected."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
    )
    sim.reset([-1.0, 5.0, 1.5], [5.0, 5.0, 1.5])
    hover = sim.hover_thrust()
    result = sim.step([hover] * 4, [hover] * 4)
    assert result.drone_a_oob


def test_mppi_controller_creation():
    ctrl = MppiController(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
        num_samples=32,
        horizon=5,
    )
    assert ctrl is not None


def test_mppi_controller_compute_action():
    ctrl = MppiController(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
        num_samples=32,
        horizon=5,
    )
    # Hovering attacker facing target on +X axis
    sim = Simulation(bounds=[10.0, 10.0, 3.0], obstacles=[])
    sim.reset([2.0, 5.0, 1.5], [8.0, 5.0, 1.5])
    state_a = sim.drone_a_state()
    state_b = sim.drone_b_state()

    action = ctrl.compute_action(state_a, state_b, pursuit=True)
    assert len(action) == 4
    for m in action:
        assert 0.0 <= m <= 0.15  # within motor thrust range
```

- [ ] **Step 4: Run bridge tests**

```bash
cd /Users/shu/GitHub/aces && .venv/bin/python -m pytest tests/test_dynamics.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/shu/GitHub/aces
git add tests/test_dynamics.py
git commit -m "test: comprehensive PyO3 bridge tests for Simulation and MppiController"
```

---

### Task 4: Gymnasium Environment

**Files:**
- Rewrite: `python/aces/env.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write env tests**

Create `tests/test_env.py`:

```python
"""Tests for the Gymnasium environment."""

import numpy as np
from aces.env import DroneDogfightEnv


def test_env_creation():
    env = DroneDogfightEnv()
    assert env.observation_space.shape == (18,)
    assert env.action_space.shape == (4,)


def test_env_reset():
    env = DroneDogfightEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (18,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_env_step():
    env = DroneDogfightEnv()
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == (18,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "agent_pos" in info
    assert "opponent_pos" in info


def test_env_episode_runs():
    env = DroneDogfightEnv(max_episode_steps=50)
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    steps = 0
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    assert steps > 0
    assert isinstance(total_reward, float)


def test_env_with_mppi_opponent():
    env = DroneDogfightEnv(opponent="mppi", mppi_samples=32, mppi_horizon=5)
    obs, _ = env.reset(seed=42)
    obs2, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs2.shape == (18,)


def test_env_gymnasium_api():
    """Basic Gymnasium API compliance."""
    env = DroneDogfightEnv(max_episode_steps=10)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/shu/GitHub/aces && .venv/bin/python -m pytest tests/test_env.py -v 2>&1 | head -20
```

Expected: FAIL — the current env.py doesn't use the Rust backend.

- [ ] **Step 3: Implement the environment**

Replace the entire contents of `python/aces/env.py`:

```python
"""Gymnasium environment for ACES 1v1 quadrotor dogfight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import toml

from aces._core import MppiController, Simulation


class DroneDogfightEnv(gym.Env):
    """1v1 quadrotor dogfight environment backed by Rust simulation.

    Observation (18-dim):
        own_velocity(3), own_angular_velocity(3),
        opponent_relative_position(3), opponent_relative_velocity(3),
        own_attitude(roll,pitch,yaw)(3),
        nearest_obstacle_dist(1), lock_progress(1), being_locked_progress(1)

    Action (4-dim, continuous [-1, 1]):
        Mapped to motor thrusts: u_i = hover + action_i * (max_thrust - hover)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_dir: str | None = None,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        opponent: str = "random",
        mppi_samples: int = 1024,
        mppi_horizon: int = 50,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._opponent_mode = opponent

        # Load configs
        if config_dir is None:
            config_dir = str(Path(__file__).parents[2] / "configs")
        cfg_dir = Path(config_dir)
        drone_cfg = toml.load(cfg_dir / "drone.toml")
        arena_cfg = toml.load(cfg_dir / "arena.toml")
        rules_cfg = toml.load(cfg_dir / "rules.toml")

        # Parse arena
        bounds = [
            arena_cfg["bounds"]["x"],
            arena_cfg["bounds"]["y"],
            arena_cfg["bounds"]["z"],
        ]
        obstacles = [
            (obs["center"], obs["half_extents"])
            for obs in arena_cfg.get("obstacles", [])
        ]
        self._bounds = bounds
        self._obstacles = obstacles
        self._spawn_a = arena_cfg["spawn"]["drone_a"]
        self._spawn_b = arena_cfg["spawn"]["drone_b"]
        drone_radius = arena_cfg["drone"]["collision_radius"]

        # Parse drone params
        phys = drone_cfg["physical"]
        inertia_cfg = drone_cfg["inertia"]
        sim_cfg = drone_cfg["simulation"]

        # Create Rust simulation
        self._sim = Simulation(
            bounds=bounds,
            obstacles=obstacles,
            mass=phys["mass"],
            arm_length=phys["arm_length"],
            inertia=[inertia_cfg["ixx"], inertia_cfg["iyy"], inertia_cfg["izz"]],
            max_thrust=phys["max_motor_thrust"],
            torque_coeff=phys["torque_coefficient"],
            drag_coeff=phys["drag_coefficient"],
            fov=np.radians(rules_cfg["lockon"]["fov_degrees"]),
            lock_distance=rules_cfg["lockon"]["lock_distance"],
            lock_duration=rules_cfg["lockon"]["lock_duration"],
            dt_ctrl=sim_cfg["dt_ctrl"],
            substeps=sim_cfg["substeps"],
            drone_radius=drone_radius,
        )

        self._hover = self._sim.hover_thrust()
        self._max_thrust = self._sim.max_thrust()
        self._delta = self._max_thrust - self._hover

        # Reward config
        self._reward_cfg = rules_cfg["reward"]

        # MPPI opponent (optional)
        self._mppi: MppiController | None = None
        if opponent == "mppi":
            mppi_cfg = rules_cfg["mppi"]
            w = mppi_cfg["weights"]
            self._mppi = MppiController(
                bounds=bounds,
                obstacles=obstacles,
                num_samples=mppi_samples,
                horizon=mppi_horizon,
                noise_std=mppi_cfg["noise_std"],
                temperature=mppi_cfg["temperature"],
                mass=phys["mass"],
                arm_length=phys["arm_length"],
                inertia=[inertia_cfg["ixx"], inertia_cfg["iyy"], inertia_cfg["izz"]],
                max_thrust=phys["max_motor_thrust"],
                torque_coeff=phys["torque_coefficient"],
                drag_coeff=phys["drag_coefficient"],
                dt_ctrl=sim_cfg["dt_ctrl"],
                substeps=sim_cfg["substeps"],
                drone_radius=drone_radius,
                w_lock=w["w_lock"],
                w_dist=w["w_dist"],
                w_face=w["w_face"],
                w_vel=w["w_vel"],
                w_ctrl=w["w_ctrl"],
                w_obs=w["w_obs"],
                d_safe=w["d_safe"],
            )

        # Self-play opponent policy (set by trainer)
        self._opponent_policy = None

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._step_count = 0
        self._prev_distance = 0.0
        self._prev_lock_progress = 0.0
        self._rng = np.random.default_rng()

    def set_opponent_policy(self, policy):
        """Set a trained policy as opponent (must have .predict(obs))."""
        self._opponent_policy = policy
        self._opponent_mode = "policy"

    def _action_to_motors(self, action: np.ndarray) -> list[float]:
        """Map [-1, 1]^4 action to motor thrusts [0, max_thrust]."""
        motors = self._hover + np.clip(action, -1.0, 1.0) * self._delta
        return np.clip(motors, 0.0, self._max_thrust).tolist()

    def _build_obs(self, result, perspective: str = "a") -> np.ndarray:
        """Build 18-dim observation from StepResult."""
        if perspective == "a":
            own_state = result.drone_a_state
            opp_state = result.drone_b_state
            own_euler = result.drone_a_euler
            lock_progress = result.lock_a_progress
            being_locked = result.lock_b_progress
            obs_dist = result.nearest_obs_dist_a
        else:
            own_state = result.drone_b_state
            opp_state = result.drone_a_state
            own_euler = result.drone_b_euler
            lock_progress = result.lock_b_progress
            being_locked = result.lock_a_progress
            obs_dist = result.nearest_obs_dist_b

        own_vel = np.array(own_state[3:6])
        own_angvel = np.array(own_state[10:13])
        rel_pos = np.array(opp_state[:3]) - np.array(own_state[:3])
        rel_vel = np.array(opp_state[3:6]) - np.array(own_state[3:6])
        attitude = np.array(own_euler)

        obs = np.concatenate([
            own_vel,
            own_angvel,
            rel_pos,
            rel_vel,
            attitude,
            [obs_dist, lock_progress, being_locked],
        ])
        return obs.astype(np.float32)

    def _compute_reward(self, result, motors: list[float]) -> tuple[float, bool]:
        """Compute reward for drone A. Returns (reward, terminated)."""
        rc = self._reward_cfg

        # Terminal: agent collision or out-of-bounds
        if result.drone_a_collision or result.drone_a_oob:
            return float(rc["collision_penalty"]), True

        # Terminal: opponent killed agent
        if result.kill_b:
            return float(rc["killed_penalty"]), True

        # Terminal: agent killed opponent
        if result.kill_a:
            return float(rc["kill_reward"]), True

        # Terminal: opponent self-destructed
        if result.drone_b_collision or result.drone_b_oob:
            return float(rc["kill_reward"]) * 0.5, True

        # Shaping rewards
        reward = float(rc["survival_bonus"])

        # Lock progress
        delta_lock = result.lock_a_progress - self._prev_lock_progress
        if delta_lock > 0:
            reward += float(rc["lock_progress_reward"]) * delta_lock

        # Approach reward
        delta_dist = self._prev_distance - result.distance
        reward += float(rc["approach_reward"]) * delta_dist

        # Control penalty
        hover_arr = np.full(4, self._hover)
        ctrl_diff = np.array(motors) - hover_arr
        reward -= float(rc["control_penalty"]) * float(np.sum(ctrl_diff**2))

        return reward, False

    def _get_opponent_motors(self, result) -> list[float]:
        """Get opponent motor commands based on opponent mode."""
        if self._opponent_mode == "mppi" and self._mppi is not None:
            state_b = list(result.drone_b_state) if result else list(self._sim.drone_b_state())
            state_a = list(result.drone_a_state) if result else list(self._sim.drone_a_state())
            return list(self._mppi.compute_action(state_b, state_a, pursuit=False))
        elif self._opponent_mode == "policy" and self._opponent_policy is not None:
            opp_obs = self._build_obs(result, perspective="b") if result else np.zeros(18, dtype=np.float32)
            action, _ = self._opponent_policy.predict(opp_obs, deterministic=False)
            return self._action_to_motors(action)
        else:
            # Random opponent
            return self._action_to_motors(self._rng.uniform(-1, 1, size=4))

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Spawn with small random offset
        jitter = 0.5
        pos_a = [
            self._spawn_a[0] + self._rng.uniform(-jitter, jitter),
            self._spawn_a[1] + self._rng.uniform(-jitter, jitter),
            self._spawn_a[2] + self._rng.uniform(-jitter, jitter),
        ]
        pos_b = [
            self._spawn_b[0] + self._rng.uniform(-jitter, jitter),
            self._spawn_b[1] + self._rng.uniform(-jitter, jitter),
            self._spawn_b[2] + self._rng.uniform(-jitter, jitter),
        ]

        self._sim.reset(pos_a, pos_b)
        if self._mppi is not None:
            self._mppi.reset()

        self._step_count = 0

        # Build initial observation by doing a zero-effect step at hover
        hover = [self._hover] * 4
        result = self._sim.step(hover, hover)
        self._prev_distance = result.distance
        self._prev_lock_progress = 0.0

        obs = self._build_obs(result)
        return obs, {"agent_pos": list(result.drone_a_state[:3]),
                      "opponent_pos": list(result.drone_b_state[:3])}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        agent_motors = self._action_to_motors(action)

        # Use previous result for opponent policy input (or None on first step)
        # We need to get opponent motors before stepping
        # Build a temporary result from current state for opponent observation
        hover = [self._hover] * 4
        pre_result = self._sim.step(hover, hover)  # peek state - WRONG, this advances state

        # Actually, we can't peek without advancing. Instead, get states directly.
        # Revert: compute opponent action from stored state, then step.
        # Problem: Simulation doesn't have a "peek" method. Let me use drone_*_state() instead.

        # Workaround: build opponent obs manually from the state arrays
        state_a = self._sim.drone_a_state()
        state_b = self._sim.drone_b_state()

        if self._opponent_mode == "mppi" and self._mppi is not None:
            opp_motors = list(self._mppi.compute_action(list(state_b), list(state_a), pursuit=False))
        elif self._opponent_mode == "policy" and self._opponent_policy is not None:
            # Build a minimal obs for opponent - we need the last step result
            # Use the stored _last_result
            if hasattr(self, "_last_result") and self._last_result is not None:
                opp_obs = self._build_obs(self._last_result, perspective="b")
            else:
                opp_obs = np.zeros(18, dtype=np.float32)
            opp_action, _ = self._opponent_policy.predict(opp_obs, deterministic=False)
            opp_motors = self._action_to_motors(opp_action)
        else:
            opp_motors = self._action_to_motors(self._rng.uniform(-1, 1, size=4).astype(np.float32))

        result = self._sim.step(agent_motors, opp_motors)
        self._last_result = result

        reward, terminated = self._compute_reward(result, agent_motors)

        self._prev_distance = result.distance
        self._prev_lock_progress = result.lock_a_progress

        truncated = self._step_count >= self.max_episode_steps

        obs = self._build_obs(result)
        info = {
            "agent_pos": list(result.drone_a_state[:3]),
            "opponent_pos": list(result.drone_b_state[:3]),
            "distance": result.distance,
            "lock_a_progress": result.lock_a_progress,
            "lock_b_progress": result.lock_b_progress,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info
```

Wait — there's a bug in the above: the `reset` method calls `sim.step(hover, hover)` which advances the sim by one step just to get an initial observation. That's wasteful and changes the initial state. Let me fix this.

Actually, looking at the code more carefully, the issue is that we need an initial observation after reset but `StepResult` only comes from `step()`. The cleanest fix: add a `get_state()` method to `Simulation` that returns the same data structure without advancing. But that requires modifying the Rust bridge again.

Simpler approach: build the initial obs directly from the state arrays returned by `reset()`. We have all the raw data we need.

Let me rewrite the env without this issue:

```python
"""Gymnasium environment for ACES 1v1 quadrotor dogfight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import toml

from aces._core import MppiController, Simulation


class DroneDogfightEnv(gym.Env):
    """1v1 quadrotor dogfight environment backed by Rust simulation.

    Observation (18-dim):
        own_velocity(3), own_angular_velocity(3),
        opponent_relative_position(3), opponent_relative_velocity(3),
        own_attitude(roll,pitch,yaw)(3),
        nearest_obstacle_dist(1), lock_progress(1), being_locked_progress(1)

    Action (4-dim, continuous [-1, 1]):
        Mapped to motor thrusts: u_i = hover + action_i * (max_thrust - hover)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_dir: str | None = None,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        opponent: str = "random",
        mppi_samples: int = 1024,
        mppi_horizon: int = 50,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._opponent_mode = opponent

        # Load configs
        if config_dir is None:
            config_dir = str(Path(__file__).parents[2] / "configs")
        cfg_dir = Path(config_dir)
        drone_cfg = toml.load(cfg_dir / "drone.toml")
        arena_cfg = toml.load(cfg_dir / "arena.toml")
        rules_cfg = toml.load(cfg_dir / "rules.toml")

        # Parse arena
        bounds = [
            arena_cfg["bounds"]["x"],
            arena_cfg["bounds"]["y"],
            arena_cfg["bounds"]["z"],
        ]
        obstacles = [
            (obs["center"], obs["half_extents"])
            for obs in arena_cfg.get("obstacles", [])
        ]
        self._bounds = bounds
        self._obstacles = obstacles
        self._spawn_a = arena_cfg["spawn"]["drone_a"]
        self._spawn_b = arena_cfg["spawn"]["drone_b"]
        drone_radius = arena_cfg["drone"]["collision_radius"]

        # Parse drone params
        phys = drone_cfg["physical"]
        inertia_cfg = drone_cfg["inertia"]
        sim_cfg = drone_cfg["simulation"]

        # Create Rust simulation
        self._sim = Simulation(
            bounds=bounds,
            obstacles=obstacles,
            mass=phys["mass"],
            arm_length=phys["arm_length"],
            inertia=[inertia_cfg["ixx"], inertia_cfg["iyy"], inertia_cfg["izz"]],
            max_thrust=phys["max_motor_thrust"],
            torque_coeff=phys["torque_coefficient"],
            drag_coeff=phys["drag_coefficient"],
            fov=np.radians(rules_cfg["lockon"]["fov_degrees"]),
            lock_distance=rules_cfg["lockon"]["lock_distance"],
            lock_duration=rules_cfg["lockon"]["lock_duration"],
            dt_ctrl=sim_cfg["dt_ctrl"],
            substeps=sim_cfg["substeps"],
            drone_radius=drone_radius,
        )

        self._hover = self._sim.hover_thrust()
        self._max_thrust = self._sim.max_thrust()
        self._delta = self._max_thrust - self._hover

        # Reward config
        self._reward_cfg = rules_cfg["reward"]

        # MPPI opponent (optional)
        self._mppi: MppiController | None = None
        if opponent == "mppi":
            mppi_cfg = rules_cfg["mppi"]
            w = mppi_cfg["weights"]
            self._mppi = MppiController(
                bounds=bounds,
                obstacles=obstacles,
                num_samples=mppi_samples,
                horizon=mppi_horizon,
                noise_std=mppi_cfg["noise_std"],
                temperature=mppi_cfg["temperature"],
                mass=phys["mass"],
                arm_length=phys["arm_length"],
                inertia=[inertia_cfg["ixx"], inertia_cfg["iyy"], inertia_cfg["izz"]],
                max_thrust=phys["max_motor_thrust"],
                torque_coeff=phys["torque_coefficient"],
                drag_coeff=phys["drag_coefficient"],
                dt_ctrl=sim_cfg["dt_ctrl"],
                substeps=sim_cfg["substeps"],
                drone_radius=drone_radius,
                w_lock=w["w_lock"],
                w_dist=w["w_dist"],
                w_face=w["w_face"],
                w_vel=w["w_vel"],
                w_ctrl=w["w_ctrl"],
                w_obs=w["w_obs"],
                d_safe=w["d_safe"],
            )

        # Self-play opponent policy (set by trainer)
        self._opponent_policy = None

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._step_count = 0
        self._prev_distance = 0.0
        self._prev_lock_progress = 0.0
        self._last_result = None
        self._rng = np.random.default_rng()

    def set_opponent_policy(self, policy):
        """Set a trained policy as opponent (must have .predict(obs))."""
        self._opponent_policy = policy
        self._opponent_mode = "policy"

    def _action_to_motors(self, action: np.ndarray) -> list[float]:
        motors = self._hover + np.clip(action, -1.0, 1.0) * self._delta
        return np.clip(motors, 0.0, self._max_thrust).tolist()

    def _obs_from_states(self, state_a, state_b, obs_dist=1.0, lock_prog=0.0, being_locked=0.0):
        """Build observation from raw 13-element state arrays."""
        own_vel = np.array(state_a[3:6])
        own_angvel = np.array(state_a[10:13])
        rel_pos = np.array(state_b[:3]) - np.array(state_a[:3])
        rel_vel = np.array(state_b[3:6]) - np.array(state_a[3:6])
        # Euler angles from quaternion (simplified: extract from quat)
        qw, qx, qy, qz = state_a[6], state_a[7], state_a[8], state_a[9]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        attitude = np.array([roll, pitch, yaw])

        obs = np.concatenate([
            own_vel, own_angvel, rel_pos, rel_vel, attitude,
            [obs_dist, lock_prog, being_locked],
        ])
        return obs.astype(np.float32)

    def _build_obs(self, result, perspective: str = "a") -> np.ndarray:
        if perspective == "a":
            return self._obs_from_states(
                result.drone_a_state, result.drone_b_state,
                result.nearest_obs_dist_a, result.lock_a_progress, result.lock_b_progress,
            )
        else:
            return self._obs_from_states(
                result.drone_b_state, result.drone_a_state,
                result.nearest_obs_dist_b, result.lock_b_progress, result.lock_a_progress,
            )

    def _compute_reward(self, result, motors: list[float]) -> tuple[float, bool]:
        rc = self._reward_cfg
        if result.drone_a_collision or result.drone_a_oob:
            return float(rc["collision_penalty"]), True
        if result.kill_b:
            return float(rc["killed_penalty"]), True
        if result.kill_a:
            return float(rc["kill_reward"]), True
        if result.drone_b_collision or result.drone_b_oob:
            return float(rc["kill_reward"]) * 0.5, True

        reward = float(rc["survival_bonus"])
        delta_lock = result.lock_a_progress - self._prev_lock_progress
        if delta_lock > 0:
            reward += float(rc["lock_progress_reward"]) * delta_lock
        delta_dist = self._prev_distance - result.distance
        reward += float(rc["approach_reward"]) * delta_dist
        hover_arr = np.full(4, self._hover)
        ctrl_diff = np.array(motors) - hover_arr
        reward -= float(rc["control_penalty"]) * float(np.sum(ctrl_diff**2))
        return reward, False

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        jitter = 0.5
        pos_a = [c + self._rng.uniform(-jitter, jitter) for c in self._spawn_a]
        pos_b = [c + self._rng.uniform(-jitter, jitter) for c in self._spawn_b]

        state_a, state_b = self._sim.reset(pos_a, pos_b)
        if self._mppi is not None:
            self._mppi.reset()

        self._step_count = 0
        dist = np.linalg.norm(np.array(state_a[:3]) - np.array(state_b[:3]))
        self._prev_distance = float(dist)
        self._prev_lock_progress = 0.0
        self._last_result = None

        obs_dist = self._sim.arena_sdf(list(state_a[:3]))
        obs = self._obs_from_states(state_a, state_b, obs_dist=obs_dist)
        info = {"agent_pos": list(state_a[:3]), "opponent_pos": list(state_b[:3])}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        agent_motors = self._action_to_motors(action)

        # Opponent action
        state_a = self._sim.drone_a_state()
        state_b = self._sim.drone_b_state()
        if self._opponent_mode == "mppi" and self._mppi is not None:
            opp_motors = list(self._mppi.compute_action(list(state_b), list(state_a), pursuit=False))
        elif self._opponent_mode == "policy" and self._opponent_policy is not None:
            if self._last_result is not None:
                opp_obs = self._build_obs(self._last_result, perspective="b")
            else:
                opp_obs = np.zeros(18, dtype=np.float32)
            opp_action, _ = self._opponent_policy.predict(opp_obs, deterministic=False)
            opp_motors = self._action_to_motors(opp_action)
        else:
            opp_motors = self._action_to_motors(
                self._rng.uniform(-1, 1, size=4).astype(np.float32)
            )

        result = self._sim.step(agent_motors, opp_motors)
        self._last_result = result

        reward, terminated = self._compute_reward(result, agent_motors)
        self._prev_distance = result.distance
        self._prev_lock_progress = result.lock_a_progress
        truncated = self._step_count >= self.max_episode_steps

        obs = self._build_obs(result)
        info = {
            "agent_pos": list(result.drone_a_state[:3]),
            "opponent_pos": list(result.drone_b_state[:3]),
            "distance": result.distance,
            "lock_a_progress": result.lock_a_progress,
            "lock_b_progress": result.lock_b_progress,
            "step": self._step_count,
        }
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 4: Run env tests**

```bash
cd /Users/shu/GitHub/aces && .venv/bin/python -m pytest tests/test_env.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/shu/GitHub/aces
git add python/aces/env.py tests/test_env.py
git commit -m "feat: Gymnasium environment backed by Rust sim-core with MPPI opponent support"
```

---

### Task 5: Rerun 3D Visualization

**Files:**
- Rewrite: `python/aces/viz.py`

- [ ] **Step 1: Write viz smoke test**

Create `tests/test_viz.py`:

```python
"""Tests for Rerun visualization (headless)."""

import numpy as np
import rerun as rr

from aces.viz import AcesVisualizer


def test_visualizer_creation():
    rr.init("test_viz", spawn=False)
    vis = AcesVisualizer.__new__(AcesVisualizer)
    vis.trail_a = []
    vis.trail_b = []
    vis.max_trail = 100
    assert vis is not None


def test_log_step_no_crash():
    """Verify logging doesn't crash (headless, no viewer)."""
    rr.init("test_viz_step", spawn=False)
    vis = AcesVisualizer.__new__(AcesVisualizer)
    vis.trail_a = []
    vis.trail_b = []
    vis.max_trail = 100

    # Simulate a minimal StepResult-like object
    class FakeResult:
        drone_a_state = [1.0, 1.0, 1.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        drone_b_state = [9.0, 9.0, 1.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        drone_a_forward = [1.0, 0.0, 0.0]
        drone_b_forward = [-1.0, 0.0, 0.0]
        lock_a_progress = 0.0
        lock_b_progress = 0.0
        distance = 10.0

    vis.log_step(0, FakeResult())
    vis.log_step(1, FakeResult())
```

- [ ] **Step 2: Implement AcesVisualizer**

Replace the entire contents of `python/aces/viz.py`:

```python
"""Real-time 3D visualization for ACES using Rerun."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
import toml


class AcesVisualizer:
    """Logs simulation state to Rerun for 3D viewing."""

    def __init__(
        self,
        config_dir: str | None = None,
        recording_id: str = "aces",
        max_trail: int = 500,
        spawn: bool = True,
    ):
        self.max_trail = max_trail
        self.trail_a: list[list[float]] = []
        self.trail_b: list[list[float]] = []

        rr.init(recording_id, spawn=spawn)

        # Load arena config and log static geometry
        if config_dir is None:
            config_dir = str(Path(__file__).parents[2] / "configs")
        arena_cfg = toml.load(Path(config_dir) / "arena.toml")
        self._log_arena(arena_cfg)

    def _log_arena(self, arena_cfg: dict):
        """Log static arena geometry."""
        b = arena_cfg["bounds"]
        half = [b["x"] / 2, b["y"] / 2, b["z"] / 2]
        center = half  # origin at corner, so center = half-extents

        # Arena wireframe box
        rr.log(
            "world/arena",
            rr.Boxes3D(centers=[center], half_sizes=[half], colors=[[80, 80, 80, 60]]),
            static=True,
        )

        # Ground plane
        rr.log(
            "world/ground",
            rr.Boxes3D(
                centers=[[half[0], half[1], -0.01]],
                half_sizes=[[half[0], half[1], 0.01]],
                colors=[[40, 100, 40, 100]],
            ),
            static=True,
        )

        # Obstacles
        for i, obs in enumerate(arena_cfg.get("obstacles", [])):
            rr.log(
                f"world/obstacles/{i}",
                rr.Boxes3D(
                    centers=[obs["center"]],
                    half_sizes=[obs["half_extents"]],
                    colors=[[200, 60, 60, 180]],
                ),
                static=True,
            )

    def log_step(self, step: int, result):
        """Log one simulation step to Rerun."""
        rr.set_time_sequence("step", step)

        pos_a = list(result.drone_a_state[:3])
        pos_b = list(result.drone_b_state[:3])
        fwd_a = list(result.drone_a_forward)
        fwd_b = list(result.drone_b_forward)

        # Drone positions
        rr.log("world/drone_a", rr.Points3D([pos_a], radii=[0.05], colors=[[0, 255, 255]]))
        rr.log("world/drone_b", rr.Points3D([pos_b], radii=[0.05], colors=[[255, 165, 0]]))

        # Forward direction arrows
        arrow_scale = 0.3
        rr.log(
            "world/drone_a/heading",
            rr.Arrows3D(
                origins=[pos_a],
                vectors=[[f * arrow_scale for f in fwd_a]],
                colors=[[0, 255, 255]],
            ),
        )
        rr.log(
            "world/drone_b/heading",
            rr.Arrows3D(
                origins=[pos_b],
                vectors=[[f * arrow_scale for f in fwd_b]],
                colors=[[255, 165, 0]],
            ),
        )

        # Trails
        self.trail_a.append(pos_a)
        self.trail_b.append(pos_b)
        if len(self.trail_a) > self.max_trail:
            self.trail_a = self.trail_a[-self.max_trail :]
        if len(self.trail_b) > self.max_trail:
            self.trail_b = self.trail_b[-self.max_trail :]

        if len(self.trail_a) >= 2:
            rr.log(
                "world/drone_a/trail",
                rr.LineStrips3D([self.trail_a], colors=[[0, 200, 200, 120]]),
            )
        if len(self.trail_b) >= 2:
            rr.log(
                "world/drone_b/trail",
                rr.LineStrips3D([self.trail_b], colors=[[200, 130, 0, 120]]),
            )

        # Scalar metrics
        rr.log("metrics/lock_a_progress", rr.Scalar(result.lock_a_progress))
        rr.log("metrics/lock_b_progress", rr.Scalar(result.lock_b_progress))
        rr.log("metrics/distance", rr.Scalar(result.distance))

    def reset(self):
        """Clear trails for a new episode."""
        self.trail_a.clear()
        self.trail_b.clear()
```

- [ ] **Step 3: Run viz tests**

```bash
cd /Users/shu/GitHub/aces && .venv/bin/python -m pytest tests/test_viz.py -v
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/shu/GitHub/aces
git add python/aces/viz.py tests/test_viz.py
git commit -m "feat: Rerun 3D visualization with arena, drones, trails, lock-on metrics"
```

---

### Task 6: Self-Play PPO Training

**Files:**
- Rewrite: `python/aces/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write trainer tests**

Create `tests/test_trainer.py`:

```python
"""Tests for self-play PPO training."""

import pytest
from aces.trainer import SelfPlayTrainer


def test_trainer_creation():
    trainer = SelfPlayTrainer(total_timesteps=512, n_steps=256, max_episode_steps=50)
    assert trainer.model is not None


def test_trainer_short_run():
    trainer = SelfPlayTrainer(
        total_timesteps=512,
        n_steps=256,
        batch_size=64,
        max_episode_steps=50,
    )
    trainer.train()
    assert trainer.model is not None


def test_opponent_updates():
    trainer = SelfPlayTrainer(
        total_timesteps=1024,
        n_steps=256,
        batch_size=64,
        opponent_update_interval=256,
        max_episode_steps=50,
    )
    trainer.train()
    assert trainer.opponent_update_count > 0
```

- [ ] **Step 2: Implement SelfPlayTrainer**

Replace the entire contents of `python/aces/trainer.py`:

```python
"""Self-play PPO training for ACES."""

from __future__ import annotations

import copy
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from aces.env import DroneDogfightEnv


class OpponentUpdateCallback(BaseCallback):
    """Periodically copies current policy to the opponent."""

    def __init__(self, update_interval: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.update_interval = update_interval
        self.update_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_interval < self.model.n_steps:
            state_dict = copy.deepcopy(self.model.policy.state_dict())
            for env in self.training_env.envs:
                unwrapped = env.unwrapped
                if hasattr(unwrapped, "_update_opponent_weights"):
                    unwrapped._update_opponent_weights(state_dict)
            self.update_count += 1
            if self.verbose:
                print(f"[SelfPlay] Opponent updated (#{self.update_count})")
        return True


class StateCallback(BaseCallback):
    """Forwards per-step info to an external callback."""

    def __init__(self, state_callback: Callable | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.state_callback = state_callback

    def _on_step(self) -> bool:
        if self.state_callback is not None:
            for info in self.locals.get("infos", []):
                if "agent_pos" in info:
                    self.state_callback(info)
        return True


class SelfPlayTrainer:
    """Trains a PPO agent via self-play in the ACES environment."""

    def __init__(
        self,
        config_dir: str | None = None,
        total_timesteps: int = 500_000,
        n_steps: int = 2048,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        opponent_update_interval: int = 10000,
        max_episode_steps: int = 1000,
        state_callback: Callable | None = None,
    ):
        self.total_timesteps = total_timesteps
        self.opponent_update_interval = opponent_update_interval
        self.state_callback = state_callback
        self.opponent_update_count = 0

        self.env = DroneDogfightEnv(
            config_dir=config_dir,
            max_episode_steps=max_episode_steps,
            opponent="random",  # starts random, switches to self-play
        )

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            n_epochs=n_epochs,
            verbose=0,
        )

        self._setup_opponent()

    def _setup_opponent(self):
        """Wire opponent policy into the environment."""
        self._opponent_model = PPO("MlpPolicy", self.env, verbose=0)
        opponent_model = self._opponent_model

        def update_weights(state_dict):
            opponent_model.policy.load_state_dict(state_dict)

        self.env._update_opponent_weights = update_weights
        self.env.set_opponent_policy(opponent_model.policy)

    def train(self) -> PPO:
        opp_cb = OpponentUpdateCallback(
            update_interval=self.opponent_update_interval, verbose=1
        )
        state_cb = StateCallback(state_callback=self.state_callback)

        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[opp_cb, state_cb],
        )
        self.opponent_update_count = opp_cb.update_count
        return self.model

    def save(self, path: str = "aces_model"):
        self.model.save(path)

    def load(self, path: str = "aces_model"):
        self.model = PPO.load(path, env=self.env)
        self._setup_opponent()
```

- [ ] **Step 3: Run trainer tests**

```bash
cd /Users/shu/GitHub/aces && .venv/bin/python -m pytest tests/test_trainer.py -v --timeout=120
```

Expected: all 3 tests pass (training runs briefly).

- [ ] **Step 4: Commit**

```bash
cd /Users/shu/GitHub/aces
git add python/aces/trainer.py tests/test_trainer.py
git commit -m "feat: self-play PPO trainer with opponent policy mirroring"
```

---

### Task 7: Integration Script + Smoke Test

**Files:**
- Rewrite: `scripts/run.py`
- Modify: `python/aces/__init__.py`

- [ ] **Step 1: Update __init__.py**

Replace the entire contents of `python/aces/__init__.py`:

```python
"""ACES -- Air Combat Engagement Simulation.

Quadrotor drone dogfight with MPPI control and reinforcement learning.
"""

__version__ = "0.1.0"

from aces.env import DroneDogfightEnv
from aces.trainer import SelfPlayTrainer
from aces.viz import AcesVisualizer
```

- [ ] **Step 2: Rewrite run.py**

Replace the entire contents of `scripts/run.py`:

```python
"""ACES -- Main entry point for training and visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import toml

from aces._core import MppiController, Simulation
from aces.viz import AcesVisualizer


def _load_configs(config_dir: str):
    cfg_dir = Path(config_dir)
    return (
        toml.load(cfg_dir / "drone.toml"),
        toml.load(cfg_dir / "arena.toml"),
        toml.load(cfg_dir / "rules.toml"),
    )


def _build_sim_and_mppi(drone_cfg, arena_cfg, rules_cfg, mppi_samples, mppi_horizon):
    """Build Simulation and two MppiControllers from config."""
    phys = drone_cfg["physical"]
    inertia = drone_cfg["inertia"]
    sim_cfg = drone_cfg["simulation"]
    bounds = [arena_cfg["bounds"]["x"], arena_cfg["bounds"]["y"], arena_cfg["bounds"]["z"]]
    obstacles = [(o["center"], o["half_extents"]) for o in arena_cfg.get("obstacles", [])]
    drone_radius = arena_cfg["drone"]["collision_radius"]

    sim_kwargs = dict(
        bounds=bounds,
        obstacles=obstacles,
        mass=phys["mass"],
        arm_length=phys["arm_length"],
        inertia=[inertia["ixx"], inertia["iyy"], inertia["izz"]],
        max_thrust=phys["max_motor_thrust"],
        torque_coeff=phys["torque_coefficient"],
        drag_coeff=phys["drag_coefficient"],
        fov=np.radians(rules_cfg["lockon"]["fov_degrees"]),
        lock_distance=rules_cfg["lockon"]["lock_distance"],
        lock_duration=rules_cfg["lockon"]["lock_duration"],
        dt_ctrl=sim_cfg["dt_ctrl"],
        substeps=sim_cfg["substeps"],
        drone_radius=drone_radius,
    )

    sim = Simulation(**sim_kwargs)

    mppi_cfg = rules_cfg["mppi"]
    w = mppi_cfg["weights"]
    mppi_kwargs = dict(
        bounds=bounds,
        obstacles=obstacles,
        num_samples=mppi_samples,
        horizon=mppi_horizon,
        noise_std=mppi_cfg["noise_std"],
        temperature=mppi_cfg["temperature"],
        mass=phys["mass"],
        arm_length=phys["arm_length"],
        inertia=[inertia["ixx"], inertia["iyy"], inertia["izz"]],
        max_thrust=phys["max_motor_thrust"],
        torque_coeff=phys["torque_coefficient"],
        drag_coeff=phys["drag_coefficient"],
        dt_ctrl=sim_cfg["dt_ctrl"],
        substeps=sim_cfg["substeps"],
        drone_radius=drone_radius,
        w_lock=w["w_lock"],
        w_dist=w["w_dist"],
        w_face=w["w_face"],
        w_vel=w["w_vel"],
        w_ctrl=w["w_ctrl"],
        w_obs=w["w_obs"],
        d_safe=w["d_safe"],
    )

    pursuer = MppiController(**mppi_kwargs)
    evader = MppiController(**mppi_kwargs)
    return sim, pursuer, evader


def run_mppi_vs_mppi(args):
    """Run MPPI pursuer vs MPPI evader with Rerun visualization."""
    drone_cfg, arena_cfg, rules_cfg = _load_configs(args.config_dir)
    sim, pursuer, evader = _build_sim_and_mppi(
        drone_cfg, arena_cfg, rules_cfg, args.mppi_samples, args.mppi_horizon
    )

    vis = None
    if not args.no_vis:
        vis = AcesVisualizer(config_dir=args.config_dir, recording_id="aces_mppi")

    spawn_a = arena_cfg["spawn"]["drone_a"]
    spawn_b = arena_cfg["spawn"]["drone_b"]
    sim.reset(spawn_a, spawn_b)
    pursuer.reset()
    evader.reset()

    max_steps = args.max_steps
    print(f"[ACES] MPPI vs MPPI — {max_steps} steps")

    for step in range(max_steps):
        state_a = sim.drone_a_state()
        state_b = sim.drone_b_state()

        motors_a = pursuer.compute_action(list(state_a), list(state_b), pursuit=True)
        motors_b = evader.compute_action(list(state_b), list(state_a), pursuit=False)

        result = sim.step(list(motors_a), list(motors_b))

        if vis is not None:
            vis.log_step(step, result)

        # Check terminal
        if result.drone_a_collision or result.drone_a_oob:
            print(f"[ACES] Drone A crashed at step {step}")
            break
        if result.drone_b_collision or result.drone_b_oob:
            print(f"[ACES] Drone B crashed at step {step}")
            break
        if result.kill_a:
            print(f"[ACES] Drone A locked on Drone B at step {step} — KILL!")
            break
        if result.kill_b:
            print(f"[ACES] Drone B locked on Drone A at step {step} — KILL!")
            break

        if step % 100 == 0:
            print(f"  step {step}: dist={result.distance:.2f}m  "
                  f"lock_a={result.lock_a_progress:.2f}  lock_b={result.lock_b_progress:.2f}")

    print("[ACES] Done.")


def run_train(args):
    """Train RL agent with self-play."""
    from aces.trainer import SelfPlayTrainer

    vis = None
    state_callback = None
    if not args.no_vis:
        vis = AcesVisualizer(config_dir=args.config_dir, recording_id="aces_train")

        def state_callback(info):
            vis.log_step(info.get("step", 0), info)

    trainer = SelfPlayTrainer(
        config_dir=args.config_dir,
        total_timesteps=args.timesteps,
        state_callback=state_callback,
    )

    print(f"[ACES] Training for {args.timesteps} timesteps...")
    trainer.train()
    trainer.save(args.save_path)
    print(f"[ACES] Model saved to {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description="ACES: Air Combat Engagement Simulation")
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).parent.parent / "configs"),
        help="Path to configs directory",
    )
    parser.add_argument(
        "--mode",
        choices=["mppi-vs-mppi", "train"],
        default="mppi-vs-mppi",
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--mppi-samples", type=int, default=1024)
    parser.add_argument("--mppi-horizon", type=int, default=50)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--save-path", default="aces_model")
    args = parser.parse_args()

    if args.mode == "mppi-vs-mppi":
        run_mppi_vs_mppi(args)
    elif args.mode == "train":
        run_train(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/shu/GitHub/aces && .venv/bin/python -m pytest tests/ -v --timeout=120
```

Expected: all tests pass across test_dynamics.py, test_env.py, test_viz.py, test_trainer.py.

- [ ] **Step 4: Run MPPI vs MPPI smoke test (headless, few steps)**

```bash
cd /Users/shu/GitHub/aces && .venv/bin/python scripts/run.py --mode mppi-vs-mppi --no-vis --max-steps 50 --mppi-samples 32 --mppi-horizon 5
```

Expected: prints step-by-step distance/lock info, completes without error.

- [ ] **Step 5: Commit**

```bash
cd /Users/shu/GitHub/aces
git add scripts/run.py python/aces/__init__.py
git commit -m "feat: integration script with MPPI-vs-MPPI and training modes"
```

- [ ] **Step 6: Create .gitignore if not already correct**

Ensure `aces/.gitignore` contains:

```
target/
__pycache__/
*.egg-info/
.venv/
dist/
*.so
*.dylib
*.pyd
.pytest_cache/
*.zip
tmpzf126oom/
matplotlib-*/
```

```bash
cd /Users/shu/GitHub/aces && git add .gitignore && git commit -m "chore: update gitignore"
```

---

## Execution Summary

| Task | Component | What |
|------|-----------|------|
| 1 | Rust | MPPI optimizer + DroneState::from_array |
| 2 | Rust | PyO3 bridge (Simulation, MppiController, StepResult) |
| 3 | Build | maturin develop + Python bridge tests |
| 4 | Python | Gymnasium DroneDogfightEnv |
| 5 | Python | Rerun 3D visualization |
| 6 | Python | Self-play PPO training |
| 7 | Python | Integration script + smoke test |

After all tasks complete, run `python scripts/run.py --mode mppi-vs-mppi` to see two MPPI-controlled drones chase each other in 3D.
