# ACES Training Pipeline Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate duplicated code, fix consistency bugs, remove dead code, and build a proper curriculum training pipeline with opponent pool, VecEnv, checkpointing, and TensorBoard for headless server training.

**Architecture:** Config loading unified into `aces/config.py` dataclasses. `CurriculumTrainer` upgraded with TOML-driven phases, Elo-rated opponent pool, SubprocVecEnv parallelism, and checkpoint/resume. Rust side gets MPPI dedup, deterministic particle filter, and wind timing fix.

**Tech Stack:** Python 3.11+, stable-baselines3, gymnasium, PyTorch, TensorBoard, PyO3/maturin, nalgebra, rayon

---

## File Structure

### New Files
- `aces/config.py` — Unified TOML config loading with typed dataclasses
- `aces/opponent_pool.py` — Elo-rated opponent checkpoint pool
- `aces/curriculum.py` — TOML-driven curriculum phase manager
- `configs/curriculum.toml` — Default 5-phase curriculum definition
- `scripts/train_server.py` — Headless server training entry point
- `tests/test_config.py` — Config loading tests
- `tests/test_curriculum.py` — Curriculum + opponent pool tests

### Modified Files
- `aces/__init__.py` — Remove predictor export, add config/curriculum exports
- `aces/env.py` — Use `config.py`, remove `_quat_to_euler` and `_load_configs`, add `set_opponent_weights`
- `aces/trainer.py` — Integrate curriculum, opponent pool, VecEnv, TensorBoard, checkpointing
- `aces/viz.py` — Use `config.py`, replace `toml` with `tomllib`
- `scripts/run.py` — Use `config.py`, remove duplicated `_load_configs` and `_build_sim_and_mppi`
- `crates/mppi/src/optimizer.rs` — Extract shared logic into `compute_action_inner`
- `crates/mppi/src/lib.rs` — Remove `pub mod sampler`
- `crates/mppi/src/cost.rs` — Remove `w_lock`/`w_vel` from `CostWeights`
- `crates/sim-core/src/environment.rs` — Remove `Arena::warehouse()`
- `crates/estimator/src/particle_filter.rs` — Accept `&mut impl Rng` instead of `thread_rng()`
- `crates/py-bridge/src/lib.rs` — Pass simulation RNG to particle filter
- `crates/game/src/simulation.rs` — Wind per-substep instead of per-control-step
- `pyproject.toml` — Add `torch`, remove `toml` dependency

### Deleted Files
- `aces/predictor.py`
- `tests/test_predictor.py`
- `crates/mppi/src/sampler.rs`

---

## Task 1: Unified Config Module

**Files:**
- Create: `aces/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config loading**

```python
# tests/test_config.py
from pathlib import Path
from aces.config import load_configs, AcesConfig

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def test_load_configs_returns_dataclass():
    cfg = load_configs(str(CONFIG_DIR))
    assert isinstance(cfg, AcesConfig)


def test_drone_config_fields():
    cfg = load_configs(str(CONFIG_DIR))
    assert cfg.drone.mass == 0.027
    assert cfg.drone.max_motor_thrust == 0.15
    assert cfg.drone.dt_ctrl == 0.01
    assert cfg.drone.substeps == 10


def test_arena_config_fields():
    cfg = load_configs(str(CONFIG_DIR))
    assert cfg.arena.bounds == [10.0, 10.0, 3.0]
    assert len(cfg.arena.obstacles) == 5
    assert cfg.arena.collision_radius == 0.05


def test_rules_config_fields():
    cfg = load_configs(str(CONFIG_DIR))
    assert cfg.rules.lockon.fov_degrees == 90.0
    assert cfg.rules.lockon.lock_distance == 2.0
    assert cfg.rules.reward.kill_reward == 100.0
    assert cfg.rules.mppi.num_samples == 1024


def test_training_config_fields():
    cfg = load_configs(str(CONFIG_DIR))
    assert cfg.rules.training.total_timesteps == 500_000
    assert cfg.rules.training.learning_rate == 3e-4
    assert cfg.rules.training.n_steps == 2048


def test_load_configs_default_dir():
    cfg = load_configs()
    assert cfg.drone.mass == 0.027


def test_obstacles_parsed_as_tuples():
    cfg = load_configs(str(CONFIG_DIR))
    center, half = cfg.arena.obstacles[0]
    assert len(center) == 3
    assert len(half) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aces.config'`

- [ ] **Step 3: Implement config module**

```python
# aces/config.py
"""Unified TOML configuration loading for ACES."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@dataclass
class DroneConfig:
    mass: float
    arm_length: float
    max_motor_thrust: float
    torque_coefficient: float
    drag_coefficient: float
    gravity: float
    ixx: float
    iyy: float
    izz: float
    dt_sim: float
    dt_ctrl: float
    substeps: int

    @property
    def inertia(self) -> list[float]:
        return [self.ixx, self.iyy, self.izz]


@dataclass
class ArenaConfig:
    bounds: list[float]
    collision_radius: float
    spawn_a: list[float]
    spawn_b: list[float]
    obstacles: list[tuple[list[float], list[float]]]


@dataclass
class LockOnConfig:
    fov_degrees: float
    lock_distance: float
    lock_duration: float

    @property
    def fov_radians(self) -> float:
        return math.radians(self.fov_degrees)


@dataclass
class MppiWeightsConfig:
    w_lock: float
    w_dist: float
    w_face: float
    w_vel: float
    w_ctrl: float
    w_obs: float
    d_safe: float


@dataclass
class MppiRiskConfig:
    wind_theta: float = 0.0
    wind_sigma: float = 0.0
    cvar_alpha: float = 0.0
    cvar_penalty: float = 0.0


@dataclass
class MppiConfig:
    num_samples: int
    horizon: int
    temperature: float
    noise_std: float
    weights: MppiWeightsConfig
    risk: MppiRiskConfig = field(default_factory=MppiRiskConfig)


@dataclass
class NoiseConfig:
    wind_theta: float = 0.0
    wind_mu: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wind_sigma: float = 0.0
    obs_noise_std: float = 0.0


@dataclass
class CameraConfig:
    enabled: bool = False
    width: int = 320
    height: int = 240
    fov_deg: float = 90.0
    max_depth: float = 15.0
    render_hz: float = 30.0
    policy_width: int = 80
    policy_height: int = 60


@dataclass
class DetectionConfig:
    drone_radius: float = 0.05
    min_confidence_distance: float = 5.0


@dataclass
class RewardConfig:
    kill_reward: float
    killed_penalty: float
    collision_penalty: float
    lock_progress_reward: float
    control_penalty: float
    approach_reward: float
    survival_bonus: float
    info_gain_reward: float = 0.0
    lost_contact_penalty: float = 0.0


@dataclass
class TrainingConfig:
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_epochs: int = 10
    opponent_update_interval: int = 10_000
    max_episode_steps: int = 1000


@dataclass
class RulesConfig:
    lockon: LockOnConfig
    mppi: MppiConfig
    noise: NoiseConfig
    camera: CameraConfig
    detection: DetectionConfig
    reward: RewardConfig
    training: TrainingConfig


@dataclass
class AcesConfig:
    drone: DroneConfig
    arena: ArenaConfig
    rules: RulesConfig


def load_configs(config_dir: str | None = None) -> AcesConfig:
    """Load drone, arena, and rules TOML configs into typed dataclasses."""
    cfg = Path(config_dir) if config_dir else _CONFIG_DIR

    with open(cfg / "drone.toml", "rb") as f:
        drone_raw = tomllib.load(f)
    with open(cfg / "arena.toml", "rb") as f:
        arena_raw = tomllib.load(f)
    with open(cfg / "rules.toml", "rb") as f:
        rules_raw = tomllib.load(f)

    phys = drone_raw["physical"]
    inertia = drone_raw["inertia"]
    sim = drone_raw["simulation"]
    drone = DroneConfig(
        mass=phys["mass"],
        arm_length=phys["arm_length"],
        max_motor_thrust=phys["max_motor_thrust"],
        torque_coefficient=phys["torque_coefficient"],
        drag_coefficient=phys["drag_coefficient"],
        gravity=phys.get("gravity", 9.81),
        ixx=inertia["ixx"],
        iyy=inertia["iyy"],
        izz=inertia["izz"],
        dt_sim=sim["dt_sim"],
        dt_ctrl=sim["dt_ctrl"],
        substeps=sim["substeps"],
    )

    b = arena_raw["bounds"]
    spawn = arena_raw["spawn"]
    obstacles = [
        (obs["center"], obs["half_extents"])
        for obs in arena_raw.get("obstacles", [])
    ]
    arena = ArenaConfig(
        bounds=[b["x"], b["y"], b["z"]],
        collision_radius=arena_raw["drone"]["collision_radius"],
        spawn_a=list(spawn["drone_a"]),
        spawn_b=list(spawn["drone_b"]),
        obstacles=obstacles,
    )

    lockon_raw = rules_raw["lockon"]
    mppi_raw = rules_raw["mppi"]
    w = mppi_raw["weights"]
    risk_raw = mppi_raw.get("risk", {})
    noise_raw = rules_raw.get("noise", {})
    camera_raw = rules_raw.get("camera", {})
    detection_raw = rules_raw.get("detection", {})
    reward_raw = rules_raw["reward"]
    training_raw = rules_raw.get("training", {})

    rules = RulesConfig(
        lockon=LockOnConfig(
            fov_degrees=lockon_raw["fov_degrees"],
            lock_distance=lockon_raw["lock_distance"],
            lock_duration=lockon_raw["lock_duration"],
        ),
        mppi=MppiConfig(
            num_samples=mppi_raw["num_samples"],
            horizon=mppi_raw["horizon"],
            temperature=mppi_raw["temperature"],
            noise_std=mppi_raw["noise_std"],
            weights=MppiWeightsConfig(
                w_lock=w["w_lock"], w_dist=w["w_dist"], w_face=w["w_face"],
                w_vel=w["w_vel"], w_ctrl=w["w_ctrl"], w_obs=w["w_obs"],
                d_safe=w["d_safe"],
            ),
            risk=MppiRiskConfig(
                wind_theta=float(risk_raw.get("wind_theta", 0.0)),
                wind_sigma=float(risk_raw.get("wind_sigma", 0.0)),
                cvar_alpha=float(risk_raw.get("cvar_alpha", 0.0)),
                cvar_penalty=float(risk_raw.get("cvar_penalty", 0.0)),
            ),
        ),
        noise=NoiseConfig(
            wind_theta=float(noise_raw.get("wind_theta", 0.0)),
            wind_mu=[float(x) for x in noise_raw.get("wind_mu", [0.0, 0.0, 0.0])],
            wind_sigma=float(noise_raw.get("wind_sigma", 0.0)),
            obs_noise_std=float(noise_raw.get("obs_noise_std", 0.0)),
        ),
        camera=CameraConfig(
            enabled=camera_raw.get("enabled", False),
            width=int(camera_raw.get("width", 320)),
            height=int(camera_raw.get("height", 240)),
            fov_deg=float(camera_raw.get("fov_deg", 90.0)),
            max_depth=float(camera_raw.get("max_depth", 15.0)),
            render_hz=float(camera_raw.get("render_hz", 30.0)),
            policy_width=int(camera_raw.get("policy_width", 80)),
            policy_height=int(camera_raw.get("policy_height", 60)),
        ),
        detection=DetectionConfig(
            drone_radius=float(detection_raw.get("drone_radius", 0.05)),
            min_confidence_distance=float(detection_raw.get("min_confidence_distance", 5.0)),
        ),
        reward=RewardConfig(
            kill_reward=reward_raw["kill_reward"],
            killed_penalty=reward_raw["killed_penalty"],
            collision_penalty=reward_raw["collision_penalty"],
            lock_progress_reward=reward_raw["lock_progress_reward"],
            control_penalty=reward_raw["control_penalty"],
            approach_reward=reward_raw["approach_reward"],
            survival_bonus=reward_raw["survival_bonus"],
            info_gain_reward=float(reward_raw.get("info_gain_reward", 0.0)),
            lost_contact_penalty=float(reward_raw.get("lost_contact_penalty", 0.0)),
        ),
        training=TrainingConfig(**{
            k: v for k, v in training_raw.items()
            if k in TrainingConfig.__dataclass_fields__
        }),
    )

    return AcesConfig(drone=drone, arena=arena, rules=rules)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add aces/config.py tests/test_config.py
git commit -m "feat: unified config module with typed dataclasses"
```

---

## Task 2: Wire Config Module Into Existing Code

**Files:**
- Modify: `aces/env.py` — Replace `_load_configs` with `config.load_configs`, remove `_quat_to_euler`
- Modify: `aces/viz.py` — Replace `import toml` with `config.load_configs`
- Modify: `scripts/run.py` — Replace `_load_configs` and `_build_sim_and_mppi` with config module

- [ ] **Step 1: Update env.py to use config module**

In `aces/env.py`, replace the config loading imports and functions:

1. Remove `import tomllib` / `import tomli` block (lines 12-15)
2. Remove `_CONFIG_DIR` (line 24)
3. Remove `_load_configs()` function (lines 27-36)
4. Remove `_parse_obstacles()` function (lines 39-44)
5. Remove `_quat_to_euler()` static method (lines 331-338)
6. Add `from aces.config import load_configs` at top
7. In `__init__`, replace `drone_cfg, arena_cfg, rules_cfg = _load_configs(config_dir)` + field extraction with:
   ```python
   cfg = load_configs(config_dir)
   ```
   Then use `cfg.drone.mass`, `cfg.arena.bounds`, `cfg.rules.lockon.fov_radians`, etc.
8. In `_build_obs` and `_build_fpv_obs`, replace `self._quat_to_euler(own)` with reading euler angles from `StepResult.drone_a_euler` passed through from the step method.

The key change to `_build_obs` and `_build_fpv_obs` signatures: add `euler: tuple[float, float, float]` parameter, remove the quaternion-to-euler conversion.

In `step()`, pass `result.drone_a_euler` to the obs builders:
```python
roll, pitch, yaw = result.drone_a_euler
```

- [ ] **Step 2: Update viz.py to use config module**

Replace:
```python
import toml
```
With:
```python
from aces.config import load_configs
```

In `__init__`, replace:
```python
arena_cfg = toml.load(Path(config_dir) / "arena.toml")
self._log_arena(arena_cfg)
```
With:
```python
cfg = load_configs(config_dir)
self._log_arena_from_config(cfg.arena)
```

Add new method:
```python
def _log_arena_from_config(self, arena: "ArenaConfig"):
    bx, by, bz = arena.bounds
    half = [bx / 2, by / 2, bz / 2]
    center = half
    rr.log("world/arena", rr.Boxes3D(centers=[center], half_sizes=[half], colors=[[80, 80, 80, 60]]), static=True)
    rr.log("world/ground", rr.Boxes3D(centers=[[half[0], half[1], -0.01]], half_sizes=[[half[0], half[1], 0.01]], colors=[[40, 100, 40, 100]]), static=True)
    for i, (center, half_ext) in enumerate(arena.obstacles):
        rr.log(f"world/obstacles/{i}", rr.Boxes3D(centers=[center], half_sizes=[half_ext], colors=[[200, 60, 60, 180]]), static=True)
```

Remove old `_log_arena` method.

- [ ] **Step 3: Update scripts/run.py**

1. Remove `_load_configs()` (lines 13-27)
2. Remove `_build_sim_and_mppi()` (lines 30-105)
3. Add `from aces.config import load_configs`
4. In `run_mppi_vs_mppi`, build Simulation/MppiController directly from `cfg = load_configs(args.config_dir)` using the dataclass fields.

- [ ] **Step 4: Run all existing tests to verify nothing broke**

Run: `pytest tests/ -v`
Expected: All existing tests pass (some may need minor adjustments for euler parameter)

- [ ] **Step 5: Commit**

```bash
git add aces/env.py aces/viz.py scripts/run.py
git commit -m "refactor: wire unified config module into env, viz, and run"
```

---

## Task 3: Dead Code Cleanup (Python)

**Files:**
- Delete: `aces/predictor.py`
- Delete: `tests/test_predictor.py`
- Modify: `aces/__init__.py`

- [ ] **Step 1: Remove predictor module and test**

```bash
rm aces/predictor.py tests/test_predictor.py
```

- [ ] **Step 2: Verify __init__.py has no predictor imports**

Current `__init__.py` does not import predictor (confirmed), so no change needed.

- [ ] **Step 3: Run tests to confirm nothing depends on predictor**

Run: `pytest tests/ -v`
Expected: All tests pass (test_predictor.py is gone)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove unused predictor module"
```

---

## Task 4: Dead Code Cleanup (Rust) + MPPI Dedup

**Files:**
- Delete: `crates/mppi/src/sampler.rs`
- Modify: `crates/mppi/src/lib.rs` — Remove `pub mod sampler`
- Modify: `crates/mppi/src/cost.rs` — Remove `w_lock`/`w_vel` from `CostWeights`
- Modify: `crates/mppi/src/optimizer.rs` — Extract `compute_action_inner`
- Modify: `crates/py-bridge/src/lib.rs` — Update `CostWeights` construction (remove w_lock/w_vel)
- Modify: `crates/sim-core/src/environment.rs` — Remove `Arena::warehouse()`

- [ ] **Step 1: Delete sampler.rs and remove from lib.rs**

Delete `crates/mppi/src/sampler.rs`.

In `crates/mppi/src/lib.rs`, change:
```rust
pub mod cost;
pub mod optimizer;
pub mod rollout;
pub mod sampler;
```
To:
```rust
pub mod cost;
pub mod optimizer;
pub mod rollout;
```

- [ ] **Step 2: Remove w_lock and w_vel from CostWeights**

In `crates/mppi/src/cost.rs`, change `CostWeights` to:
```rust
pub struct CostWeights {
    pub w_dist: f64,
    pub w_face: f64,
    pub w_ctrl: f64,
    pub w_obs: f64,
    pub d_safe: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            w_dist: 1.0,
            w_face: 5.0,
            w_ctrl: 0.01,
            w_obs: 1000.0,
            d_safe: 0.3,
        }
    }
}
```

In `crates/py-bridge/src/lib.rs`, update `MppiController::new` to remove `w_lock`/`w_vel` parameters from the Python signature and `CostWeights` construction. Keep the Python API parameters for backward compat but just ignore them (or remove them and update scripts/run.py + env.py accordingly).

Better approach: Remove the parameters entirely since we're already changing the Python callers in Task 2. In `MppiController::new` pyo3 signature, remove `w_lock` and `w_vel`. Update the `CostWeights` construction:
```rust
let weights = CostWeights {
    w_dist,
    w_face,
    w_ctrl,
    w_obs,
    d_safe,
};
```

Update `env.py` and `scripts/run.py` to not pass `w_lock`/`w_vel` to `MppiController()`.

- [ ] **Step 3: Extract compute_action_inner in optimizer.rs**

Replace the duplicated logic in `compute_action` and `compute_action_with_belief` with a shared inner function. The key difference is only the cost function.

Add a private method:
```rust
fn compute_action_with_cost_fn<F>(
    &mut self,
    self_state: &DroneState,
    enemy_state: &DroneState,
    cost_fn: F,
) -> Vector4<f64>
where
    F: Fn(&DroneState, &DroneState, &Vector4<f64>, f64, &Arena, &CostWeights) -> f64
        + Send + Sync,
{
    // ... the shared sample-rollout-softmax-warmstart logic (from compute_action)
    // using cost_fn instead of pursuit_cost/evasion_cost
}
```

Then `compute_action` becomes:
```rust
pub fn compute_action(
    &mut self,
    self_state: &DroneState,
    enemy_state: &DroneState,
    pursuit: bool,
) -> Vector4<f64> {
    let cost_fn = if pursuit {
        pursuit_cost as fn(&DroneState, &DroneState, &Vector4<f64>, f64, &Arena, &CostWeights) -> f64
    } else {
        evasion_cost
    };
    self.compute_action_with_cost_fn(self_state, enemy_state, cost_fn)
}
```

And `compute_action_with_belief` becomes:
```rust
pub fn compute_action_with_belief(
    &mut self,
    self_state: &DroneState,
    enemy_state: &DroneState,
    pursuit: bool,
    belief_var: f64,
) -> Vector4<f64> {
    if belief_var < 1e-6 {
        return self.compute_action(self_state, enemy_state, pursuit);
    }
    let cost_fn = if pursuit {
        move |s: &DroneState, e: &DroneState, c: &Vector4<f64>, h: f64, a: &Arena, w: &CostWeights| {
            belief_pursuit_cost(s, e, c, h, a, w, belief_var)
        }
    } else {
        move |s: &DroneState, e: &DroneState, c: &Vector4<f64>, h: f64, a: &Arena, w: &CostWeights| {
            belief_evasion_cost(s, e, c, h, a, w, belief_var)
        }
    };
    self.compute_action_with_cost_fn(self_state, enemy_state, cost_fn)
}
```

Note: Because the belief closures capture `belief_var`, and the function pointer and closure types differ, use a generic `F: Fn(...) -> f64 + Send + Sync` for `compute_action_with_cost_fn`. For `compute_action`, box the function pointer to match. Or better: make `compute_action_with_cost_fn` generic so monomorphization handles both cases.

- [ ] **Step 4: Remove Arena::warehouse()**

In `crates/sim-core/src/environment.rs`, remove the `warehouse()` method (lines 71-86).

Update tests in that file that use `Arena::warehouse()` to construct the arena manually:
```rust
fn test_arena() -> Arena {
    let mut arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
    for (x, y) in [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)] {
        arena.obstacles.push(Obstacle::Box {
            center: Vector3::new(x, y, 1.5),
            half_extents: Vector3::new(0.5, 0.5, 1.5),
        });
    }
    arena
}
```

Also grep for `Arena::warehouse()` usage in optimizer.rs tests and update similarly.

- [ ] **Step 5: Run Rust tests**

Run: `cargo test`
Expected: All Rust tests pass

- [ ] **Step 6: Rebuild py-bridge and run Python tests**

Run: `poetry run maturin develop && pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove dead code (sampler, w_lock/w_vel, warehouse), dedup MPPI optimizer"
```

---

## Task 5: Particle Filter Determinism

**Files:**
- Modify: `crates/estimator/src/particle_filter.rs` — Accept `&mut impl Rng`
- Modify: `crates/py-bridge/src/lib.rs` — Pass `self.rng` to particle filter

- [ ] **Step 1: Update ParticleFilter to accept external RNG**

Change `ParticleFilter::new` to accept `rng: &mut impl Rng`:
```rust
pub fn new(
    initial_position: Vector3<f64>,
    num_particles: usize,
    process_noise_accel: f64,
    measurement_noise: f64,
    rng: &mut impl Rng,
) -> Self {
    let normal = Normal::new(0.0, 0.1).unwrap();
    let particles = (0..num_particles)
        .map(|_| Particle {
            position: initial_position
                + Vector3::new(
                    normal.sample(rng),
                    normal.sample(rng),
                    normal.sample(rng),
                ),
            velocity: Vector3::zeros(),
            weight: 1.0 / num_particles as f64,
        })
        .collect();
    Self {
        particles,
        process_noise_accel,
        measurement_noise,
        bounds: None,
    }
}
```

Update `predict_with_sdf` to accept `rng: &mut impl Rng` instead of `let mut rng = rand::thread_rng();`.

Update `predict` to accept `rng: &mut impl Rng` similarly.

Update `resample` to accept `rng: &mut impl Rng` similarly.

Update `update` to pass rng through to `resample`:
```rust
pub fn update(&mut self, measured_position: &Vector3<f64>, rng: &mut impl Rng) {
    // ... weight computation ...
    self.resample(rng);
}
```

- [ ] **Step 2: Update py-bridge to pass simulation RNG**

In `crates/py-bridge/src/lib.rs`, update all particle filter calls to pass `&mut self.rng`:

In `Simulation::new`:
```rust
let mut pf_a = ParticleFilter::new(Vector3::zeros(), 200, 2.0, ekf_noise, &mut rng);
let mut pf_b = ParticleFilter::new(Vector3::zeros(), 200, 2.0, ekf_noise, &mut rng);
```
(Note: need to create rng before constructing Self, then move it in.)

In `Simulation::reset`:
```rust
let mut pf_a = ParticleFilter::new(v3(pos_b), 200, 2.0, self.obs_noise.std_dev.max(0.1), &mut self.rng);
```

In `Simulation::step`, for predict and update calls:
```rust
self.pf_a.predict_with_sdf(self.dt_ctrl, |p| arena.sdf(p), &mut rng_clone);
```
Note: borrow conflict — `self.rng` is borrowed while `self.pf_a` is also borrowed mutably. Solution: temporarily take the rng out:
```rust
let mut rng = std::mem::replace(&mut self.rng, rand::rngs::StdRng::seed_from_u64(0));
// ... particle filter operations using &mut rng ...
self.rng = rng;
```
Or split the RNG into a separate sub-rng for particle filters. Better: derive a child RNG from the main one:
```rust
let pf_seed: u64 = self.rng.gen();
let mut pf_rng = rand::rngs::StdRng::seed_from_u64(pf_seed);
```

Actually the simplest approach: predict_with_sdf needs `&mut self` on PF and `&mut rng`. The arena borrow is the issue. Use the temporary rng extraction approach:
```rust
{
    let arena = &self.arena;
    let mut rng = std::mem::replace(&mut self.rng, rand::rngs::StdRng::seed_from_u64(0));
    self.pf_a.predict_with_sdf(self.dt_ctrl, |p| arena.sdf(p), &mut rng);
    if a_sees_b {
        self.pf_a.update(&noisy_b, &mut rng);
    }
    self.pf_b.predict_with_sdf(self.dt_ctrl, |p| arena.sdf(p), &mut rng);
    if b_sees_a {
        self.pf_b.update(&noisy_a, &mut rng);
    }
    self.rng = rng;
}
```

- [ ] **Step 3: Update particle filter tests**

Update tests in `particle_filter.rs` to pass a seeded RNG:
```rust
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
let mut pf = ParticleFilter::new(Vector3::new(5.0, 5.0, 1.5), 100, 5.0, 0.1, &mut rng);
```

- [ ] **Step 4: Run tests**

Run: `cargo test`
Run: `poetry run maturin develop && pytest tests/ -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix: make particle filter deterministic via seeded RNG"
```

---

## Task 6: Wind Timing Fix (Game Crate)

**Files:**
- Modify: `crates/game/src/simulation.rs`

- [ ] **Step 1: Move wind.step inside substep loop**

In `sim_step` function, change from:
```rust
// Wind
let wind_force_a = s.wind_a.step(dt_ctrl, &mut s.rng);
let wind_force_b = s.wind_b.step(dt_ctrl, &mut s.rng);

// Physics substeps
for _ in 0..substeps {
    s.state_a = step_rk4(&s.state_a, &motors_a, &s.params, dt_sim, &wind_force_a);
    s.state_b = step_rk4(&s.state_b, &motors_b, &s.params, dt_sim, &wind_force_b);
}
```

To:
```rust
// Physics substeps with per-substep wind (matches py-bridge behavior)
for _ in 0..substeps {
    let wind_force_a = s.wind_a.step(dt_sim, &mut s.rng);
    let wind_force_b = s.wind_b.step(dt_sim, &mut s.rng);
    s.state_a = step_rk4(&s.state_a, &motors_a, &s.params, dt_sim, &wind_force_a);
    s.state_b = step_rk4(&s.state_b, &motors_b, &s.params, dt_sim, &wind_force_b);
}
```

- [ ] **Step 2: Run Rust tests**

Run: `cargo test`
Expected: All pass (game crate has no unit tests for wind, but compilation must succeed)

- [ ] **Step 3: Commit**

```bash
git add crates/game/src/simulation.rs
git commit -m "fix: apply wind per-substep in game crate to match training env"
```

---

## Task 7: Fix pyproject.toml Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add torch, remove toml**

Change:
```toml
[tool.poetry.dependencies]
python = ">=3.11,<3.15"
gymnasium = ">=0.29"
stable-baselines3 = ">=2.0"
numpy = ">=1.24"
rerun-sdk = ">=0.18"
toml = ">=0.10"
maturin = ">=1.5,<2.0"
```
To:
```toml
[tool.poetry.dependencies]
python = ">=3.11,<3.15"
gymnasium = ">=0.29"
stable-baselines3 = ">=2.0"
numpy = ">=1.24"
torch = ">=2.0"
rerun-sdk = ">=0.18"
tensorboard = ">=2.14"
maturin = ">=1.5,<2.0"
```

Removed `toml` (no longer used after viz.py switch to tomllib). Added `torch` (explicit). Added `tensorboard` (for training monitoring).

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "fix: add torch and tensorboard deps, remove unused toml dep"
```

---

## Task 8: Opponent Pool

**Files:**
- Create: `aces/opponent_pool.py`
- Create: `tests/test_opponent_pool.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_opponent_pool.py
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from aces.env import DroneDogfightEnv
from aces.opponent_pool import OpponentPool


def test_pool_creation(tmp_path):
    pool = OpponentPool(pool_dir=tmp_path, max_size=5)
    assert pool.size == 0


def test_pool_add_and_sample(tmp_path):
    env = DroneDogfightEnv(max_episode_steps=10)
    model = PPO("MlpPolicy", env, verbose=0)

    pool = OpponentPool(pool_dir=tmp_path, max_size=5)
    pool.add(model, metadata={"phase": "test", "timestep": 100})
    assert pool.size == 1

    sampled_policy, meta = pool.sample()
    assert sampled_policy is not None
    assert meta["phase"] == "test"


def test_pool_max_size(tmp_path):
    env = DroneDogfightEnv(max_episode_steps=10)
    model = PPO("MlpPolicy", env, verbose=0)

    pool = OpponentPool(pool_dir=tmp_path, max_size=3)
    for i in range(5):
        pool.add(model, metadata={"step": i})
    assert pool.size == 3


def test_pool_elo_update(tmp_path):
    env = DroneDogfightEnv(max_episode_steps=10)
    model = PPO("MlpPolicy", env, verbose=0)

    pool = OpponentPool(pool_dir=tmp_path, max_size=5)
    pool.add(model, metadata={"step": 0})
    opp_id = pool.entries[0].id

    initial_elo = pool.entries[0].elo
    pool.update_elo(agent_won=True, opponent_id=opp_id)
    assert pool.entries[0].elo < initial_elo  # opponent lost, elo decreased


def test_pool_state_dict_roundtrip(tmp_path):
    env = DroneDogfightEnv(max_episode_steps=10)
    model = PPO("MlpPolicy", env, verbose=0)

    pool = OpponentPool(pool_dir=tmp_path, max_size=5)
    pool.add(model, metadata={"step": 0})
    pool.update_elo(agent_won=True, opponent_id=pool.entries[0].id)

    state = pool.state_dict()
    pool2 = OpponentPool(pool_dir=tmp_path, max_size=5)
    pool2.load_state_dict(state)
    assert pool2.size == 1
    assert pool2.entries[0].elo == pool.entries[0].elo
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_opponent_pool.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement opponent pool**

```python
# aces/opponent_pool.py
"""Elo-rated opponent checkpoint pool for self-play training."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


@dataclass
class PoolEntry:
    id: str
    path: str
    elo: float = 1000.0
    metadata: dict = field(default_factory=dict)


class OpponentPool:
    """Manages a pool of opponent checkpoints with Elo ratings."""

    K_FACTOR = 32.0  # Elo K-factor

    def __init__(self, pool_dir: Path | str, max_size: int = 20):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.entries: list[PoolEntry] = []

    @property
    def size(self) -> int:
        return len(self.entries)

    def add(self, model: PPO, metadata: dict | None = None) -> str:
        """Save model checkpoint to pool. Returns entry ID."""
        entry_id = uuid.uuid4().hex[:8]
        path = str(self.pool_dir / f"opponent_{entry_id}")
        model.save(path)

        entry = PoolEntry(
            id=entry_id,
            path=path,
            metadata=metadata or {},
        )
        self.entries.append(entry)

        # Evict lowest-Elo entry if over capacity
        if len(self.entries) > self.max_size:
            self.entries.sort(key=lambda e: e.elo)
            evicted = self.entries.pop(0)
            # Clean up file
            p = Path(evicted.path + ".zip")
            if p.exists():
                p.unlink()

        return entry_id

    def sample(self, env=None) -> tuple:
        """Sample an opponent weighted by Elo proximity to 1000 (prefer challenging opponents).

        Returns (policy, metadata) where policy is a loaded SB3 policy object.
        """
        if not self.entries:
            raise ValueError("Pool is empty")

        # Weight by inverse distance from pool mean Elo (prefer diverse)
        elos = np.array([e.elo for e in self.entries])
        # Softmax on Elo scores (higher Elo = harder opponent = more weight)
        weights = np.exp(elos / 400.0)
        weights /= weights.sum()

        idx = np.random.choice(len(self.entries), p=weights)
        entry = self.entries[idx]
        model = PPO.load(entry.path, env=env)
        return model.policy, entry.metadata

    def update_elo(self, agent_won: bool, opponent_id: str) -> None:
        """Update opponent's Elo after a match."""
        entry = next((e for e in self.entries if e.id == opponent_id), None)
        if entry is None:
            return

        # Simplified Elo: agent is at 1000, opponent at entry.elo
        expected = 1.0 / (1.0 + 10 ** ((1000.0 - entry.elo) / 400.0))
        actual = 0.0 if agent_won else 1.0
        entry.elo += self.K_FACTOR * (actual - expected)

    def state_dict(self) -> dict:
        """Serialize pool state for checkpointing."""
        return {
            "entries": [
                {"id": e.id, "path": e.path, "elo": e.elo, "metadata": e.metadata}
                for e in self.entries
            ]
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore pool from checkpoint."""
        self.entries = [
            PoolEntry(
                id=e["id"],
                path=e["path"],
                elo=e["elo"],
                metadata=e.get("metadata", {}),
            )
            for e in state["entries"]
        ]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_opponent_pool.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add aces/opponent_pool.py tests/test_opponent_pool.py
git commit -m "feat: Elo-rated opponent pool for self-play training"
```

---

## Task 9: Curriculum Manager

**Files:**
- Create: `aces/curriculum.py`
- Create: `configs/curriculum.toml`
- Create: `tests/test_curriculum.py`

- [ ] **Step 1: Create curriculum config**

```toml
# configs/curriculum.toml
# Default curriculum: 5 phases of increasing difficulty

[[phase]]
name = "pursuit_linear"
task = "pursuit_linear"
opponent = "trajectory"
wind_sigma = 0.0
obs_noise_std = 0.0
use_fpv = false
max_timesteps = 200_000
promote_condition = "win_rate>0.3"
promote_window = 100

[[phase]]
name = "pursuit_evasive"
task = "pursuit_evasive"
opponent = "mppi_evasion"
wind_sigma = 0.0
obs_noise_std = 0.0
use_fpv = false
max_timesteps = 300_000
promote_condition = "win_rate>0.3"
promote_window = 200

[[phase]]
name = "search_pursuit"
task = "search_pursuit"
opponent = "mppi_evasion"
wind_sigma = 0.0
obs_noise_std = 0.1
use_fpv = false
max_timesteps = 300_000
promote_condition = "win_rate>0.25"
promote_window = 200

[[phase]]
name = "self_play_noisy"
task = "dogfight"
opponent = "pool"
wind_sigma = 0.3
obs_noise_std = 0.1
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
use_fpv = true
max_timesteps = 5_000_000
promote_condition = "steps"
promote_window = 500
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_curriculum.py
from pathlib import Path
from aces.curriculum import CurriculumManager, Phase, load_curriculum

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def test_load_curriculum():
    phases = load_curriculum(str(CONFIG_DIR / "curriculum.toml"))
    assert len(phases) == 5
    assert phases[0].name == "pursuit_linear"
    assert phases[-1].use_fpv is True


def test_curriculum_manager_initial_phase():
    phases = load_curriculum(str(CONFIG_DIR / "curriculum.toml"))
    mgr = CurriculumManager(phases)
    assert mgr.current_phase().name == "pursuit_linear"
    assert mgr.phase_index == 0


def test_curriculum_promote():
    phases = [
        Phase(name="a", task="dogfight", opponent="random",
              max_timesteps=100, promote_condition="win_rate>0.5",
              promote_window=10),
        Phase(name="b", task="dogfight", opponent="random",
              max_timesteps=100, promote_condition="steps",
              promote_window=10),
    ]
    mgr = CurriculumManager(phases)

    # Not enough data
    assert not mgr.should_promote({"win_rate": 0.6, "episodes": 5})
    # Enough data, condition met
    assert mgr.should_promote({"win_rate": 0.6, "episodes": 15})

    next_phase = mgr.promote()
    assert next_phase.name == "b"
    assert mgr.phase_index == 1


def test_curriculum_promote_last_phase():
    phases = [
        Phase(name="a", task="dogfight", opponent="random",
              max_timesteps=100, promote_condition="steps",
              promote_window=10),
    ]
    mgr = CurriculumManager(phases)
    assert mgr.promote() is None  # already at last phase


def test_curriculum_state_dict():
    phases = load_curriculum(str(CONFIG_DIR / "curriculum.toml"))
    mgr = CurriculumManager(phases)
    mgr.promote()  # advance to phase 1

    state = mgr.state_dict()
    mgr2 = CurriculumManager(phases)
    mgr2.load_state_dict(state)
    assert mgr2.phase_index == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_curriculum.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement curriculum manager**

```python
# aces/curriculum.py
"""TOML-driven curriculum manager for progressive RL training."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class Phase:
    name: str
    task: str = "dogfight"
    opponent: str = "random"
    wind_sigma: float = 0.0
    obs_noise_std: float = 0.0
    use_fpv: bool = False
    max_timesteps: int = 500_000
    promote_condition: str = "steps"  # "steps" | "win_rate>X" | "reward>X"
    promote_window: int = 100
    reward_overrides: dict = field(default_factory=dict)
    ppo_overrides: dict = field(default_factory=dict)


def load_curriculum(path: str | Path) -> list[Phase]:
    """Load curriculum phases from a TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    phases = []
    for p in raw.get("phase", []):
        phases.append(Phase(
            name=p["name"],
            task=p.get("task", "dogfight"),
            opponent=p.get("opponent", "random"),
            wind_sigma=float(p.get("wind_sigma", 0.0)),
            obs_noise_std=float(p.get("obs_noise_std", 0.0)),
            use_fpv=bool(p.get("use_fpv", False)),
            max_timesteps=int(p.get("max_timesteps", 500_000)),
            promote_condition=p.get("promote_condition", "steps"),
            promote_window=int(p.get("promote_window", 100)),
            reward_overrides=p.get("reward_overrides", {}),
            ppo_overrides=p.get("ppo_overrides", {}),
        ))
    return phases


class CurriculumManager:
    """Manages progression through curriculum phases."""

    def __init__(self, phases: list[Phase]):
        self.phases = phases
        self.phase_index = 0

    def current_phase(self) -> Phase:
        return self.phases[self.phase_index]

    def is_last_phase(self) -> bool:
        return self.phase_index >= len(self.phases) - 1

    def should_promote(self, stats: dict) -> bool:
        """Check if current stats satisfy the promotion condition."""
        phase = self.current_phase()

        if self.is_last_phase():
            return False

        # Need enough episodes for statistical significance
        episodes = stats.get("episodes", 0)
        if episodes < phase.promote_window:
            return False

        condition = phase.promote_condition

        if condition == "steps":
            return True  # promote when max_timesteps reached (handled externally)

        # Parse "metric>threshold" conditions
        match = re.match(r"(\w+)([><=]+)([\d.]+)", condition)
        if match:
            metric_name, op, threshold_str = match.groups()
            threshold = float(threshold_str)
            value = stats.get(metric_name, 0.0)

            if ">" in op:
                return value > threshold
            elif "<" in op:
                return value < threshold

        return False

    def promote(self) -> Phase | None:
        """Advance to next phase. Returns new Phase or None if already at end."""
        if self.is_last_phase():
            return None
        self.phase_index += 1
        return self.current_phase()

    def state_dict(self) -> dict:
        return {"phase_index": self.phase_index}

    def load_state_dict(self, state: dict) -> None:
        self.phase_index = state["phase_index"]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_curriculum.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add aces/curriculum.py configs/curriculum.toml tests/test_curriculum.py
git commit -m "feat: TOML-driven curriculum manager with promotion conditions"
```

---

## Task 10: Trainer Overhaul — VecEnv, TensorBoard, Checkpoint, Curriculum Integration

**Files:**
- Modify: `aces/trainer.py` — Major rewrite of `CurriculumTrainer`
- Modify: `aces/env.py` — Add `set_opponent_weights(bytes)` for cross-process support
- Modify: `tests/test_trainer.py` — Update tests for new API

This is the largest task. The core changes:

1. `DroneDogfightEnv.set_opponent_weights(state_dict)` — new method for cross-process opponent updates
2. `CurriculumTrainer` rewritten to use `CurriculumManager`, `OpponentPool`, `SubprocVecEnv`, TensorBoard
3. `CheckpointCallback` for auto-saving
4. `CurriculumCallback` for checking promotion conditions

- [ ] **Step 1: Add set_opponent_weights to env.py**

In `aces/env.py`, add method to `DroneDogfightEnv`:
```python
def set_opponent_weights(self, state_dict: dict) -> None:
    """Update opponent policy weights (works across processes for SubprocVecEnv)."""
    if self._opponent_policy is not None:
        self._opponent_policy.load_state_dict(state_dict)
```

- [ ] **Step 2: Rewrite CurriculumTrainer in trainer.py**

The new `CurriculumTrainer` integrates:
- `CurriculumManager` for phase transitions
- `OpponentPool` for diverse opponents
- `SubprocVecEnv` with `n_envs` parallel environments
- TensorBoard logging via SB3's native support
- Checkpoint/resume

Key API:
```python
class CurriculumTrainer:
    def __init__(
        self,
        curriculum_path: str | None = None,  # path to curriculum.toml
        config_dir: str | None = None,
        n_envs: int = 8,
        save_dir: str = "checkpoints",
        checkpoint_interval: int = 50_000,
        fpv: bool = False,
    ): ...

    def train(self) -> PPO: ...
    def save_checkpoint(self, path: str) -> None: ...
    def load_checkpoint(self, path: str) -> None: ...
```

The `train()` method loops through curriculum phases:
```python
def train(self) -> PPO:
    for phase_idx in range(self.curriculum.phase_index, len(self.curriculum.phases)):
        phase = self.curriculum.current_phase()
        print(f"[ACES] === Phase {phase_idx}: {phase.name} ===")

        # Create VecEnv with phase-specific parameters
        vec_env = self._make_vec_env(phase)

        if self.model is None:
            self.model = self._create_model(vec_env, phase)
        else:
            self.model.set_env(vec_env)

        # Callbacks
        callbacks = self._build_callbacks(phase)

        # Train
        self.model.learn(
            total_timesteps=phase.max_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=phase.name,
        )

        # Add current model to opponent pool
        if phase.opponent == "pool":
            self.pool.add(self.model, {"phase": phase.name})

        # Check promotion
        if not self.curriculum.is_last_phase():
            self.curriculum.promote()

    return self.model
```

For `_make_vec_env`, use `SubprocVecEnv` when n_envs > 1:
```python
def _make_vec_env(self, phase: Phase) -> VecEnv:
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    def make_env(rank: int):
        def _init():
            env = DroneDogfightEnv(
                config_dir=self._config_dir,
                max_episode_steps=self._max_episode_steps,
                task=phase.task,
                wind_sigma=phase.wind_sigma,
                obs_noise_std=phase.obs_noise_std,
                fpv=phase.use_fpv or self._fpv,
            )
            return env
        return _init

    if self._n_envs > 1:
        return SubprocVecEnv([make_env(i) for i in range(self._n_envs)])
    return DummyVecEnv([make_env(0)])
```

For opponent update with VecEnv:
```python
class OpponentUpdateCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_interval < self.model.n_steps:
            state_dict = copy.deepcopy(self.model.policy.state_dict())
            # Works for both DummyVecEnv and SubprocVecEnv
            self.training_env.env_method("set_opponent_weights", state_dict)
        return True
```

For pool opponent sampling:
```python
class PoolOpponentCallback(BaseCallback):
    """Periodically sample a new opponent from the pool."""
    def __init__(self, pool, sample_interval=20000, verbose=0):
        super().__init__(verbose)
        self.pool = pool
        self.sample_interval = sample_interval

    def _on_step(self) -> bool:
        if self.pool.size > 0 and self.num_timesteps % self.sample_interval < self.model.n_steps:
            policy, meta = self.pool.sample()
            state_dict = policy.state_dict()
            self.training_env.env_method("set_opponent_weights", state_dict)
        return True
```

TensorBoard integration:
```python
self.model = PPO(
    policy,
    vec_env,
    tensorboard_log=str(self._log_dir / "tensorboard"),
    **ppo_kwargs,
)
```

Custom metrics callback:
```python
class TensorBoardMetricsCallback(BaseCallback):
    """Log custom metrics to TensorBoard."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._kills = 0
        self._deaths = 0
        self._episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("kill_a"):
                self._kills += 1
            if info.get("kill_b"):
                self._deaths += 1

        for done in self.locals.get("dones", []):
            if done:
                self._episodes += 1

        if self._episodes > 0 and self.num_timesteps % 1000 == 0:
            self.logger.record("custom/kill_rate", self._kills / max(self._episodes, 1))
            self.logger.record("custom/death_rate", self._deaths / max(self._episodes, 1))
            win_rate = self._kills / max(self._kills + self._deaths, 1)
            self.logger.record("custom/win_rate", win_rate)
            self.logger.record("custom/episodes", self._episodes)

        return True
```

Checkpoint callback:
```python
class CheckpointCallback(BaseCallback):
    def __init__(self, save_dir, curriculum, pool, interval=50000, verbose=0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.curriculum = curriculum
        self.pool = pool
        self.interval = interval

    def _on_step(self) -> bool:
        if self.num_timesteps % self.interval < self.model.n_steps:
            phase = self.curriculum.current_phase()
            ckpt_dir = self.save_dir / f"{phase.name}_step_{self.num_timesteps}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(ckpt_dir / "model"))
            import json
            with open(ckpt_dir / "curriculum_state.json", "w") as f:
                json.dump(self.curriculum.state_dict(), f)
            with open(ckpt_dir / "pool_state.json", "w") as f:
                json.dump(self.pool.state_dict(), f)
        return True
```

- [ ] **Step 3: Keep backward-compatible SelfPlayTrainer**

The existing `SelfPlayTrainer` stays as-is for simple single-env self-play. It now reads PPO hyperparams from `rules.toml [training]` by default:

```python
class SelfPlayTrainer:
    def __init__(self, config_dir=None, fpv=False, task="dogfight", **kwargs):
        cfg = load_configs(config_dir)
        tc = cfg.rules.training

        # kwargs override config file values
        self.total_timesteps = kwargs.get("total_timesteps", tc.total_timesteps)
        n_steps = kwargs.get("n_steps", tc.n_steps)
        batch_size = kwargs.get("batch_size", tc.batch_size)
        learning_rate = kwargs.get("learning_rate", tc.learning_rate)
        # ... etc
```

- [ ] **Step 4: Update tests**

Keep existing `test_trainer.py` tests working. Add new test:
```python
def test_curriculum_trainer_with_checkpoint(tmp_path):
    """CurriculumTrainer saves checkpoints during training."""
    from aces.curriculum import Phase, CurriculumManager
    phases = [
        Phase(name="test", task="pursuit_linear", max_timesteps=256,
              promote_condition="steps", promote_window=10),
    ]
    trainer = CurriculumTrainer(
        phases=phases,
        config_dir=None,
        n_envs=1,
        save_dir=str(tmp_path),
        checkpoint_interval=128,
    )
    model = trainer.train()
    assert model is not None
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add aces/env.py aces/trainer.py tests/test_trainer.py
git commit -m "feat: curriculum trainer with VecEnv, TensorBoard, opponent pool, checkpointing"
```

---

## Task 11: Headless Server Training Script

**Files:**
- Create: `scripts/train_server.py`
- Modify: `scripts/run.py` — Update curriculum mode to use new system

- [ ] **Step 1: Create train_server.py**

```python
#!/usr/bin/env python3
"""Headless server training script for ACES.

Usage:
    # Full curriculum from scratch
    python scripts/train_server.py --n-envs 8

    # Resume from checkpoint
    python scripts/train_server.py --resume checkpoints/pursuit_linear_step_100000/

    # Custom curriculum config
    python scripts/train_server.py --curriculum configs/curriculum.toml --n-envs 16
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="ACES Headless Training")
    parser.add_argument("--config-dir", default=str(Path(__file__).parent.parent / "configs"))
    parser.add_argument("--curriculum", default=None, help="Path to curriculum.toml")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--resume", default=None, help="Path to checkpoint dir to resume from")
    parser.add_argument("--fpv", action="store_true")
    parser.add_argument("--save-path", default="aces_model_final")
    args = parser.parse_args()

    from aces.curriculum import load_curriculum, CurriculumManager, Phase
    from aces.trainer import CurriculumTrainer

    # Load curriculum
    curriculum_path = args.curriculum or str(Path(args.config_dir) / "curriculum.toml")
    phases = load_curriculum(curriculum_path)
    print(f"[ACES] Loaded {len(phases)} curriculum phases from {curriculum_path}")
    for i, p in enumerate(phases):
        print(f"  Phase {i}: {p.name} ({p.task}, {p.max_timesteps} steps)")

    # Create trainer
    trainer = CurriculumTrainer(
        phases=phases,
        config_dir=args.config_dir,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        checkpoint_interval=args.checkpoint_interval,
        fpv=args.fpv,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"[ACES] Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Graceful shutdown handler
    def shutdown_handler(signum, frame):
        print("\n[ACES] Caught signal, saving checkpoint...")
        trainer.save_checkpoint(str(Path(args.save_dir) / "interrupted"))
        print(f"[ACES] Checkpoint saved to {args.save_dir}/interrupted/")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Train
    print(f"[ACES] Starting training with {args.n_envs} parallel envs")
    print(f"[ACES] TensorBoard: tensorboard --logdir {args.save_dir}/tensorboard/")
    model = trainer.train()

    # Save final model
    model.save(args.save_path)
    print(f"[ACES] Final model saved to {args.save_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update scripts/run.py curriculum mode**

In `scripts/run.py`, update the `curriculum` mode to use the new system:
```python
elif args.mode == "curriculum":
    from aces.curriculum import load_curriculum
    from aces.trainer import CurriculumTrainer

    curriculum_path = str(Path(args.config_dir) / "curriculum.toml")
    phases = load_curriculum(curriculum_path)

    # Override timesteps if provided via CLI
    ts_parts = [int(x) for x in str(args.timesteps).split(",")]
    for i, phase in enumerate(phases):
        if i < len(ts_parts):
            phase.max_timesteps = ts_parts[i]

    trainer = CurriculumTrainer(
        phases=phases,
        config_dir=args.config_dir,
        n_envs=1,  # single env for CLI mode
        save_dir=args.save_path,
        fpv=args.fpv,
    )
    model = trainer.train()
    model.save(args.save_path + "_final")
    print(f"[ACES] Final model saved to {args.save_path}_final")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_server.py scripts/run.py
git commit -m "feat: headless server training script with graceful shutdown"
```

---

## Task 12: Update __init__.py and Final Integration

**Files:**
- Modify: `aces/__init__.py`
- Run: Full test suite + Rust build

- [ ] **Step 1: Update __init__.py exports**

```python
"""ACES -- Air Combat Engagement Simulation."""

__version__ = "0.1.0"

from aces.config import AcesConfig as AcesConfig
from aces.config import load_configs as load_configs
from aces.curriculum import CurriculumManager as CurriculumManager
from aces.env import DroneDogfightEnv as DroneDogfightEnv
from aces.opponent_pool import OpponentPool as OpponentPool
from aces.policy import CnnImuExtractor as CnnImuExtractor
from aces.trainer import CurriculumTrainer as CurriculumTrainer
from aces.trainer import SelfPlayTrainer as SelfPlayTrainer
from aces.viz import AcesVisualizer as AcesVisualizer
```

- [ ] **Step 2: Full Rust build**

Run: `cargo test`
Expected: All Rust tests pass

- [ ] **Step 3: Rebuild py-bridge**

Run: `poetry run maturin develop`
Expected: Build succeeds

- [ ] **Step 4: Full Python test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add aces/__init__.py
git commit -m "feat: update package exports for new curriculum pipeline"
```

---

## Execution Dependencies

```
Task 1 (config.py)
  └→ Task 2 (wire config into env/viz/run)
       └→ Task 10 (trainer overhaul — needs config, pool, curriculum)
            └→ Task 11 (train_server.py)
                 └→ Task 12 (final integration)

Task 3 (dead code Python) — independent
Task 4 (dead code Rust + MPPI dedup) — independent
Task 5 (particle filter) — independent
Task 6 (wind timing) — independent
Task 7 (pyproject.toml) — independent
Task 8 (opponent pool) — independent, needed by Task 10
Task 9 (curriculum manager) — independent, needed by Task 10
```

Tasks 3-9 can all run in parallel. Tasks 1→2 must be sequential. Task 10 depends on 1, 2, 8, 9. Task 11 depends on 10. Task 12 is final.
