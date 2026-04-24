# Plan: Parallel Simulation — Rayon Batch + WGPU GPU Acceleration

Status: **Phase 1, Phase 2, Phase 3 complete — Phase 4 pending**
Created: 2026-04-23
Last updated: 2026-04-24 (Phase 2 + Phase 3 landed + 7 consistency audits)

User-facing guide: [`docs/gpu-mppi.md`](../docs/gpu-mppi.md).
Session archive: [`docs/2026-04-24-session-archive.md`](../docs/2026-04-24-session-archive.md).

---

## Status (as of 2026-04-24)

- **Phase 1 — Rayon CPU batch**: DONE (commit `c6b1d22` "add batch-sim crate for parallel battle simulation").
- **Phase 2 — WGPU batch MPPI**: DONE. Landed as 11 slices from
  `feature/f32-rk4-validation` (`b6ceb84`) through `feature/gpu-bench`
  (`b0f1082`). GPU pipeline lives in `crates/batch-sim/src/gpu/` with WGSL
  shaders in `.../gpu/shaders/`.
- **Phase 3 — PyO3 integration**: DONE. Landed as 13 slices from
  `feature/pyo3-gpu` (`4278b9e`) through `feature/curriculum-gpu-opt`
  (`fef1050`) which wires `--use-gpu-env` into the curriculum trainer, plus
  follow-up docs / utilities (`feature/gpu-docs` `0de67a6`,
  `feature/training-bench` `c253ac6`, `feature/setup-check` `86728e9`,
  `feature/obs-describe` `652cd30`, `feature/test-conftest` `9f40bef`,
  `feature/denorm-unit-test` `a4f4bd8`, `feature/gpu-wind` `4a541dd`).
- **Post-Phase-3 consistency audits**: 7 audits landed comparing CPU env
  vs batch-sim vs GPU vs Bevy, each finding at least one real divergence
  (`feature/action-consistency` `41aeb05`, `feature/reward-consistency`
  `ee83fd1`, `feature/obs-consistency` `057e5d4`,
  `feature/fix-test-and-game` `b1f403a`, `feature/spawn-audit` `6141a8a`,
  `feature/rules-toml-plumb` `4c6678c`, `feature/wind-plumb` `9a6b716`).
  Full rundown in the session archive (Part 2). Three reward-path
  divergences remain — see archive Part 3.
- **Phase 4 — Full physics on GPU**: NOT STARTED. Section below is the
  original design sketch and is preserved for reference.

### Full slice index (in chronological merge order)

| #  | Phase | Slice                                          | Landed as                         |
|----|-------|------------------------------------------------|-----------------------------------|
|  1 | P2    | f32 RK4 reference + parity tests               | `feature/f32-rk4-validation` (`b6ceb84`) |
|  2 | P2    | f32 Arena / Obstacle SDF                       | `feature/f32-sdf` (`3cfe38a`)           |
|  3 | P2    | GpuBatchMppi buffer allocation                 | `feature/gpu-pipeline-skeleton` (`9d88db1`) |
|  4 | P2    | f32 pursuit / evasion cost                     | `feature/f32-cost` (`3c2f41d`)           |
|  5 | P2    | WGSL helpers + naga validation                 | `feature/wgsl-helpers` (`06b4e26`)       |
|  6 | P2    | `rollout_and_cost` kernel                      | `feature/mppi-rollout-kernel` (`31b87f9`) |
|  7 | P2    | `softmax_reduce` kernel                        | `feature/mppi-softmax-kernel` (`e6c8072`) |
|  8 | P2    | BindGroupLayout + 2 ComputePipelines           | `feature/gpu-pipelines` (`dce9617`)     |
|  9 | P2    | `compute_batch_actions()` dispatch             | `feature/gpu-dispatch` (`6a3e250`)      |
| 10 | P2    | CPU reference + parity test                    | `feature/gpu-cpu-parity` (`52836de`)    |
| 11 | P2    | `bench_gpu_vs_cpu` example                     | `feature/gpu-bench` (`b0f1082`)         |
| 12 | P3    | PyO3 `PyGpuBatchMppi`                          | `feature/pyo3-gpu` (`4278b9e`)          |
| 13 | P3    | `GpuBatchOrchestrator` end-to-end              | `feature/orchestrator-gpu-opt` (`dc7a8cd`) |
| 14 | P3    | PyO3 `GpuVecEnv` + reset                       | `feature/gpu-vec-env` (`bb6aa96`)       |
| 15 | P3    | `step_with_agent_a_actions`                    | `feature/gpu-ppo-mode` (`4b4d70c`)      |
| 16 | P3    | SB3-compatible Python GpuVecEnv                | `feature/gpu-sb3-wrapper` (`98ba353`)   |
| 17 | P3    | GPU PPO end-to-end smoke                       | `feature/gpu-ppo-smoke` (`1be11aa`)     |
| 18 | P3    | `--use-gpu-env` on CurriculumTrainer           | `feature/curriculum-gpu-opt` (`fef1050`) |
| 19 | P3    | GPU MPPI user guide                            | `feature/gpu-docs` (`0de67a6`)          |
| 20 | P3    | PPO training throughput benchmark              | `feature/training-bench` (`c253ac6`)    |
| 21 | P3    | `denormalize_action` pure fn                   | `feature/denorm-unit-test` (`a4f4bd8`)  |
| 22 | P3    | conftest.py fixtures                           | `feature/test-conftest` (`9f40bef`)     |
| 23 | P3    | `check_gpu_setup.sh`                           | `feature/setup-check` (`86728e9`)       |
| 24 | P3    | `obs_layout.describe_obs`                      | `feature/obs-describe` (`652cd30`)      |
| 25 | P3    | Per-drone wind in rollout (binding 11)         | `feature/gpu-wind` (`4a541dd`)          |
| 26 | Audit | Action denormalization alignment               | `feature/action-consistency` (`41aeb05`) |
| 27 | Audit | Reward formula alignment                       | `feature/reward-consistency` (`ee83fd1`) |
| 28 | Audit | Obs[15] combined SDF                           | `feature/obs-consistency` (`057e5d4`)   |
| 29 | Audit | Obs test + Bevy game obstacle_sdf              | `feature/fix-test-and-game` (`b1f403a`) |
| 30 | Audit | SpawnMode + max_steps                          | `feature/spawn-audit` (`6141a8a`)       |
| 31 | Audit | reward_config TOML plumbing                    | `feature/rules-toml-plumb` (`4c6678c`)  |
| 32 | Audit | wind_sigma / wind_theta TOML plumbing          | `feature/wind-plumb` (`9a6b716`)        |

See the session archive for per-audit bug details and the list of
remaining divergences.

---

## Context

ACES is a 1v1 quadrotor dogfight simulator. The training pipeline uses a Python RL loop
(SB3 PPO) calling into Rust physics/MPPI via PyO3. The bottleneck is single-environment
stepping: ~1K env-steps/s. The 6-phase curriculum needs ~8M total steps, taking hours.

Phase 1 (Rayon batch) is implemented in `crates/batch-sim/`. This plan covers the remaining
phases plus architectural details for future agents.

---

## What Exists (Phase 1 — Done)

### Crate: `crates/batch-sim/`

```
src/
├── lib.rs              — module exports, conditionally includes gpu/
├── battle.rs           — BattleState + BattleInfo + StepResult + episode lifecycle
├── orchestrator.rs     — BatchOrchestrator: N parallel MPPI-vs-MPPI dogfights
├── observation.rs      — 21-dim observation builder (matches dogfight.py)
├── reward.rs           — shaped reward (approach, lock, survival, collision, control)
└── gpu/
    ├── mod.rs
    └── adapter.rs      — WGPU hardware probe + compute shader verification
examples/
├── bench_batch.rs      — throughput benchmark (varying battles/samples/horizon)
└── gpu_probe.rs        — hardware capability report
```

### Key types and their locations

| Type | File | Purpose |
|------|------|---------|
| `BattleState` | `battle.rs:71` | Per-battle state: 2 DroneStates, 2 WindModels, 2 LockOnTrackers, step counter |
| `BattleInfo` | `battle.rs:33` | Terminal flags: kill_a/b, collision_a/b, timeout, distance, lock progress, visibility |
| `StepResult` | `battle.rs:58` | Full step output: obs (21-dim each), motors, rewards, done, info |
| `BatchConfig` | `battle.rs:12` | max_steps, dt_ctrl, substeps, wind_sigma, wind_theta |
| `BatchOrchestrator` | `orchestrator.rs:50` | Main struct: Vec<BattleState>, Vec<MppiOptimizer> ×2, Vec<SmallRng> |
| `MppiConfig` | `orchestrator.rs:14` | num_samples, horizon, noise_std, temperature, weights |
| `RewardConfig` | `reward.rs:6` | kill/death/collision/approach/lock/survival/control reward weights |
| `BatchStats` | `orchestrator.rs:36` | Aggregated: steps, episodes, kills, collisions, timeouts, mean reward/distance |

### How `step_all()` works (orchestrator.rs:149)

```
step_all() → Vec<StepResult>

  Phase 1+2 (Rayon parallel over battles):
    par_iter_mut over (battles, optimizers_a, optimizers_b, rngs)
    For each battle:
      1. opt_a.compute_action(state_a, state_b, pursuit=true)   ← Rayon inner parallelism
      2. opt_b.compute_action(state_b, state_a, pursuit=false)  ← Rayon inner parallelism
      3. battle.step_physics(motors_a, motors_b, ...)
         └─ 10 substeps: wind.step() + step_rk4() per drone
         └─ collision check (SDF) + lock-on update + visibility check
      4. build_observation() × 2
      5. compute_reward_a() × 2 (symmetric via swapped BattleInfo)
      → (StepResult, done)

  Phase 3 (sequential — needs RNG for reset):
    For each (result, done):
      Update stats (kills, collisions, timeouts)
      If done: battle.reset() + optimizer.reset()
    → Vec<StepResult>
```

### Benchmark results (Apple Silicon 8 cores, release mode)

```
64 battles × 64 samples × H=10:    3,228 env-steps/s   19.8 ms/tick
16 battles × 16 samples × H=10:   11,480 env-steps/s    1.4 ms/tick
16 battles × 128 samples × H=15:   1,084 env-steps/s   14.8 ms/tick
 1 battle  × 1024 samples × H=50:     42 env-steps/s   24.0 ms/tick (rules.toml full)
```

Bottleneck: MPPI trajectory rollout scales linearly with samples × horizon.
At production settings (1024×50), single-battle MPPI takes ~24ms.

---

## Phase 2: WGPU Batch MPPI

**STATUS: DONE (2026-04-23).** Shipped in `crates/batch-sim/src/gpu/`
(pipeline, shaders, orchestrator) behind the `gpu` cargo feature. See
[`docs/gpu-mppi.md`](../docs/gpu-mppi.md) for the user-facing summary and the
status table of the individual merged slices. The design sketch below is
preserved for historical reference.

### Goal

Replace `MppiOptimizer.compute_action()` Rayon inner loop with GPU compute shaders.
One GPU dispatch computes all MPPI actions for all drones across all battles.

### Architecture

```
CPU side (Rust):                    GPU side (WGSL compute):
───────��─────────                   ────────────────────────
1. Collect states for all drones    
2. Upload to GPU storage buffers    
3. Generate noise (CPU SmallRng)    
4. Upload noise to GPU              
                                    Pass 1: rollout_and_cost
                                      @compute @workgroup_size(1)
                                      dispatch(N_DRONES × N_SAMPLES, 1, 1)
                                      Each thread:
                                        - Read initial state + perturbation
                                        - RK4 rollout: H steps × S substeps
                                        - Accumulate cost (pursuit/evasion + SDF)
                                        - Write cost + control sequence

                                    Pass 2: softmax_reduce
                                      @compute @workgroup_size(256)
                                      dispatch(N_DRONES, 1, 1)
                                      Each workgroup (256 threads):
                                        - Tree-reduction to find min cost
                                        - Compute exp weights
                                        - Weighted average of controls
                                        - Write optimal motors

5. Copy result → staging buffer
6. Map staging buffer (async)
7. Read optimal motors per drone
8. Shift mean_controls (warm-start)
```

### GPU buffer layout

All buffers use f32 (GPU f64 is 16-32x slower where available at all).

```
Binding  Type       Shape                               Size (64 battles)
───────  ─────────  ──────────────────────────────────  ─────────────────
0        storage    states[N_DRONES × 13]               128 × 52 = 6.5 KB
1        storage    enemies[N_DRONES × 13]              6.5 KB
2        storage    mean_ctrls[N_DRONES × H × 4]        128 × 50 × 16 = 100 KB
3        storage    noise[N_DRONES × N × H × 4]         128 × 1024 × 50 × 16 = 100 MB  ← largest
4        storage    costs[N_DRONES × N]                  128 × 1024 × 4 = 512 KB
5        storage    ctrls_out[N_DRONES × N × H × 4]     100 MB
6        storage    result[N_DRONES × H × 4]             100 KB
7        uniform    DroneParams                          32 bytes
8        uniform    CostWeights + Arena bounds           ~256 bytes
9        storage    obstacles[MAX_OBS]                   variable

N_DRONES = n_battles × 2
N = num_samples (1024)
H = horizon (50)
```

**Memory concern**: noise + ctrls_out = ~200MB for 128 drones × 1024 samples × 50 horizon.
Apple Silicon unified memory handles this, discrete GPUs need ≥256MB VRAM.
At 128 samples × 15 horizon (lighter config): ~4MB total — trivial.

### WGSL shader structure

The shader needs these Rust functions ported to f32 WGSL:
- `step_rk4()` from `crates/sim-core/src/dynamics.rs:127` — RK4 integrator
- `state_derivative()` from `crates/sim-core/src/dynamics.rs:77` — 13D state derivative
- `motor_mixing()` from `crates/sim-core/src/dynamics.rs:46` — X-config motor→thrust+torque
- `pursuit_cost()` / `evasion_cost()` from `crates/mppi/src/cost.rs:33,157` — cost functions
- `sdf()` from `crates/sim-core/src/environment.rs:100` — signed distance field
- `boundary_sdf()` + `obstacle_sdf()` — arena geometry

Key f32 considerations:
- Quaternion renormalize after each RK4 step (f32 drifts faster than f64)
- SDF margin: drone_radius=0.05m gives 5cm buffer for f32 precision at dt=0.001
- WGSL lacks `acos()` for angle_to — use dot product directly in cost

### Implementation steps

1. **Create `crates/batch-sim/src/gpu/pipeline.rs`**: GpuBatchMppi struct
   - `new(device, queue, n_drones, n_samples, horizon, params, arena, weights)` — create pipeline + buffers
   - `compute_batch_actions(states, enemies, mean_ctrls) → Vec<[f32; 4]>` — single dispatch
   - Buffer management: persistent GPU buffers, reuse across frames

2. **Create `crates/batch-sim/src/gpu/shaders/mppi.wgsl`**: Compute shader
   - Struct definitions (DroneState, DroneParams, CostWeights, Obstacle)
   - `fn rk4_step(state, motors, params, dt, wind) → DroneState`
   - `fn motor_mixing(motors, params) → (thrust, torque)`
   - `fn box_sdf(p, center, half_extents) → f32`
   - `fn arena_sdf(p) → f32`
   - `fn pursuit_cost(state, enemy, ctrl, hover, weights) → f32`
   - `@compute fn rollout_and_cost()` — main rollout kernel
   - `@compute fn softmax_reduce()` — reduction kernel

3. **Create `crates/batch-sim/src/gpu/batch_mppi.rs`**: High-level integration
   - Wraps pipeline + handles warm-start shift + noise generation
   - Provides same interface as CPU BatchOrchestrator
   - Automatic fallback to Rayon if no GPU detected

4. **Validation**: Run GPU MPPI vs CPU MPPI side-by-side
   - Compare per-trajectory costs (should agree within f32 tolerance)
   - Compare final motor commands (softmax may differ slightly due to f32)
   - Benchmark: GPU vs CPU at various (battles, samples, horizon) configs

### f32 validation plan

Before shipping GPU MPPI, validate f32 RK4 stability:

```rust
// In batch-sim tests:
#[test]
fn test_f32_rk4_matches_f64() {
    // Run identical 50-step rollout with f32 and f64
    // Compare final states — should agree within 1mm position, 0.01 rad attitude
    // at dt_sim = 0.001s, 50 × 10 = 500 steps
}
```

Quaternion norm drift check: after 500 f32 RK4 steps, quat norm should be within 1e-3 of 1.0.
If not, add renormalization every N steps in the shader.

---

## Phase 3: PyO3 Integration

**STATUS: DONE (2026-04-23).** Shipped as `aces._core.GpuBatchMppi`
(standalone planner) and `aces._core.GpuVecEnv` (SB3-compatible VecEnv) in
`crates/py-bridge/`, plus the Python-side `aces.training.gpu_vec_env.GpuVecEnv`
wrapper and the `--use-gpu-env` opt-in on `CurriculumTrainer`. The actual
delivered design deviates from the sketch below in two ways:
(a) the Python-facing class names are `GpuBatchMppi` / `GpuVecEnv` rather than
`BatchVecEnv`, and (b) the current `GpuVecEnv` only supports MPPI-vs-MPPI
opponent semantics (see caveats in `docs/gpu-mppi.md`). The original design
sketch below is preserved for reference.

### Goal

Expose `BatchOrchestrator` to Python so the curriculum trainer can use it as a
drop-in VecEnv replacement, eliminating Python→Rust FFI per-step overhead.

### Architecture

```python
# Python side (replaces SubprocVecEnv + DroneDogfightEnv)
from aces._core import BatchVecEnv

env = BatchVecEnv(
    n_envs=64,
    config_dir="configs/",
    mppi_samples=128,
    mppi_horizon=15,
    wind_sigma=0.3,
    obs_noise_std=0.1,
)

# Gymnasium VecEnv-compatible interface
obs = env.reset()                    # → np.ndarray (64, 21)
obs, rewards, dones, infos = env.step(actions)  # actions from PPO policy
```

### Implementation in `crates/py-bridge/src/lib.rs`

Add a new `#[pyclass]` wrapping `BatchOrchestrator`:

```rust
#[pyclass]
struct BatchVecEnv {
    orchestrator: BatchOrchestrator,
    // For PPO-vs-MPPI mode: agent A uses external actions, agent B uses MPPI
}

#[pymethods]
impl BatchVecEnv {
    #[new]
    fn new(n_envs: usize, config_dir: &str, ...) -> Self { ... }

    /// Step with external actions for agent A, MPPI for agent B.
    /// actions: np.ndarray shape (n_envs, 4) — normalized [-1, 1]
    fn step(&mut self, actions: Vec<[f64; 4]>) -> (Vec<[f64; 21]>, Vec<f64>, Vec<bool>, Vec<PyDict>) { ... }

    /// Step with MPPI for both agents (data collection / expert demo).
    fn step_mppi(&mut self) -> (Vec<[f64; 21]>, Vec<f64>, Vec<bool>, Vec<PyDict>) { ... }

    fn reset(&mut self) -> Vec<[f64; 21]> { ... }
}
```

### Integration modes

The orchestrator needs to support two modes for curriculum phases:

1. **MPPI-vs-MPPI** (`step_mppi`): Both drones use MPPI. Used for benchmarks and expert data.
2. **PPO-vs-MPPI** (`step`): Drone A action from Python PPO policy, drone B from MPPI.
   This requires a new method on `BatchOrchestrator`:

```rust
// In orchestrator.rs, add:
pub fn step_with_agent_a_actions(&mut self, actions_a: &[[f64; 4]]) -> Vec<StepResult> {
    // Same as step_all() but skip opt_a.compute_action(),
    // use provided motors_a instead.
    // opt_b.compute_action() still runs MPPI for evasion.
}
```

### SB3 compatibility

SB3 `PPO.learn()` expects a VecEnv with:
- `observation_space`: Box(21,) or Dict for FPV
- `action_space`: Box(4,) in [-1, 1]
- `step(actions)` → (obs, rewards, dones, infos)
- `reset()` → obs
- `num_envs` property

Wrap `BatchVecEnv` in a Python class that inherits from `gymnasium.vector.VectorEnv`.
The Rust side handles all physics; Python only does PPO forward/backward.

---

## Phase 4: Full GPU Pipeline

### Goal

Move the true physics step (not just MPPI planning rollout) to GPU, eliminating
the CPU physics bottleneck entirely.

### What moves to GPU

| Component | Current (CPU) | GPU version |
|-----------|--------------|-------------|
| MPPI rollout | Phase 2 | Already done |
| True physics (RK4) | `step_rk4()` in Rayon | Compute shader: 128 drones parallel |
| Wind (OU process) | `wind.step()` sequential | GPU PCG hash RNG |
| Collision (SDF) | `arena.is_collision()` | Same SDF functions as MPPI shader |
| Lock-on | `lockon.update()` + LOS check | Compute shader: angle + distance + sphere trace |
| Observation | `build_observation()` | Compute shader: relative position/velocity + euler |
| Reward | `compute_reward_a()` | Compute shader or CPU (cheap) |

### What stays on CPU

- Episode reset (infrequent, needs CPU RNG for spawn positions)
- PPO buffer collection (Python numpy arrays)
- Statistics accumulation

### Pipeline per tick

```
GPU dispatch 1: mppi_rollout (N_DRONES × N_SAMPLES workgroups)
GPU dispatch 2: mppi_reduce  (N_DRONES workgroups)
GPU dispatch 3: physics_step (N_DRONES workgroups)
  - Apply optimal motors
  - RK4 with GPU-generated wind
  - Collision detection
  - Lock-on update + LOS
  - Build observation
  - Compute reward
GPU → CPU copy: (N_DRONES × 21 obs) + (N_DRONES × reward) + (N_DRONES × done)

Total data transfer per tick: ~128 × (21×8 + 8 + 1) ≈ 22 KB  ← negligible
```

### Estimated performance

3 dispatches per tick, each ~0.3ms on Apple M-series = ~1ms total.
64 battles × 100Hz control = 6,400 env-steps per 1ms = **6.4M env-steps/s**.

---

## Curriculum Integration Map

How the batch simulator plugs into each curriculum phase:

```
Phase 0: hover_stabilize
  Agent A: PPO (learning)
  Agent B: none (no opponent)
  Mode: Python env only — batch-sim not needed (single drone, no MPPI)
  File: aces/env/dogfight.py

Phase 1: pursuit_linear
  Agent A: PPO (learning)
  Agent B: trajectory controller (circle/lemniscate)
  Mode: Need step_with_agent_a_actions() — B is trajectory, not MPPI
  Needs: Add trajectory opponent mode to BatchOrchestrator
  File: aces/env/trajectory.py defines the trajectory generators

Phase 2: pursuit_evasive
  Agent A: PPO (learning)
  Agent B: MPPI evasion
  Mode: step_with_agent_a_actions() — PPO vs MPPI evasion ✓
  File: orchestrator.rs (new method)

Phase 3: search_pursuit
  Agent A: PPO (learning)
  Agent B: MPPI evasion (with occlusion / partial observability)
  Mode: step_with_agent_a_actions() — same as Phase 2, + belief state
  Needs: Add EKF/PF belief tracking to BattleState for realistic obs
  File: crates/estimator/src/ekf.rs, particle_filter.rs

Phase 4: self_play_noisy
  Agent A: PPO (learning)
  Agent B: opponent pool / self-play lagged copy
  Mode: step_with_agent_a_actions() — B uses loaded PPO policy
  Needs: Add policy opponent mode (load SB3 weights, batched inference)
  Also: domain randomization (mass, inertia, thrust, drag variance per episode)
  File: aces/training/opponent_pool.py, batched_vec_env.py

Phase 5: fpv_transfer
  Agent A: PPO with CNN+MLP (learning)
  Agent B: opponent pool
  Mode: Needs depth camera rendering — cannot use batch-sim (no sphere tracing on GPU yet)
  Alternative: Phase 4 GPU could add sphere-trace depth rendering (expensive)
  File: crates/sim-core/src/camera.rs

MPPI benchmark (no training):
  Agent A: MPPI pursuit
  Agent B: MPPI evasion
  Mode: step_mppi() — fully self-contained ✓
  File: orchestrator.rs (existing step_all)
```

---

## Key Implementation Details for Future Agents

### MPPI Algorithm (for GPU shader porting)

The MPPI optimizer (`crates/mppi/src/optimizer.rs:177-342`) works as follows:

1. **Generate seeds** for parallel RNG (one u64 per sample)
2. **For each sample k** (parallel over N=1024):
   - Create SmallRng from seed
   - For each timestep t in [0, horizon):
     - Sample perturbation: `ε ~ N(0, noise_std²)`
     - Perturbed control: `u[t] = clamp(mean[t] + ε, 0, max_thrust)`
   - Rollout: for each t, for each substep s: `state = step_rk4(state, u[t], params, dt_sim, wind)`
   - Cost: `Σ_t cost_fn(state[t], enemy, u[t], hover, arena, weights)`
   - If max_penetration > 0: cost += 1e8 (hard collision penalty)
3. **CVaR** (optional): `select_nth_unstable` to find threshold, penalize worst-α fraction
4. **Chance constraint** (optional): Lagrangian penalty on colliding trajectories, dual ascent on λ
5. **Softmax**: `w[k] = exp(-(cost[k] - min_cost) / temperature) / Σ_j exp(...)`
6. **Weighted mean**: `new_mean[t] = Σ_k w[k] · u_k[t]`
7. **Warm-start**: `mean_controls = [new_mean[1], ..., new_mean[H-1], hover]`

### Physics (for GPU shader porting)

`step_rk4()` at `crates/sim-core/src/dynamics.rs:127`:
- Clamp motors to [0, max_thrust]
- 4 stages of RK4: each calls `state_derivative()` which computes:
  - `p_dot = velocity`
  - `v_dot = (R(q)·[0,0,F_total] + gravity + drag + wind) / mass`
  - `q_dot = 0.5 * q * [0, ω]` (quaternion kinematics)
  - `ω_dot = I⁻¹(τ - ω×(Iω))` (Euler equation)
- Motor mixing (X-config): `F_total = Σf_i`, `τ_x = d/√2·(f0-f1-f2+f3)`, etc.
- After RK4: renormalize quaternion

### SDF (for GPU shader porting)

`Arena::sdf()` at `crates/sim-core/src/environment.rs:100`:
- `sdf(p) = min(boundary_sdf(p), obstacle_sdf(p))`
- `boundary_sdf(p) = min(p.x, bounds.x-p.x, p.y, bounds.y-p.y, p.z, bounds.z-p.z)`
- `obstacle_sdf(p) = min over obstacles of obstacle.sdf(p)`
- Box SDF: standard signed box distance (outside norm + inside max clamp)
- Sphere SDF: `||p - center|| - radius`
- `is_collision(p) = sdf(p) < drone_radius` where drone_radius = 0.05m

### Observation vector (21-dim)

Layout defined in `crates/batch-sim/src/observation.rs`:
```
[0:3]    own velocity (world frame)
[3:6]    own angular velocity (body frame)
[6:9]    relative position to opponent (world frame)
[9:12]   relative velocity (world frame)
[12:15]  own attitude (roll, pitch, yaw from quaternion)
[15]     nearest obstacle distance (SDF)
[16]     lock-on progress self→opponent [0,1]
[17]     being-locked progress opponent→self [0,1]
[18]     opponent visible (0 or 1)
[19]     belief uncertainty (0 for MPPI-vs-MPPI)
[20]     time since last seen (0 for always-visible)
```

### Reward (shaped)

From `crates/batch-sim/src/reward.rs`:
- Terminal: kill=+100, killed=-100, collision=-50, opponent_crash=+5, timeout=0
- Shaping: survival_bonus=+0.01, approach=+3.0×Δdistance, lock_progress=+5.0×Δlock, control=-0.01×||u-hover||²
- Task overrides in `configs/rules.toml [task_reward_overrides.*]`

### Platform support

WGPU `Backends::all()` covers:
- macOS (Apple Silicon): Metal backend via `metal` crate
- Linux/Windows + NVIDIA: Vulkan backend
- Linux/Windows + AMD: Vulkan backend
- Windows + Intel/AMD: DX12 backend

The GPU feature is opt-in: `cargo build --features gpu`.
Fallback: Rayon CPU path always available.

GPU probe tool: `cargo run -p aces-batch-sim --example gpu_probe --features gpu`

---

## Open Questions

1. **Noise generation on GPU**: Should we use GPU PCG hash for noise instead of CPU SmallRng?
   Pro: eliminates CPU→GPU noise upload (100MB for full config). Con: quality of GPU RNG.

2. **Warm-start on GPU**: Should mean_controls live persistently on GPU, shifted in a compute pass?
   Pro: avoids CPU→GPU upload every frame. Con: adds complexity to buffer management.

3. **Adaptive MPPI samples**: Could reduce samples when battles are in "easy" states (hovering)
   and increase when in combat. Saves GPU cycles but adds dispatch complexity.

4. **FPV depth rendering on GPU**: Phase 5 needs sphere-trace depth images. Already done
   in `crates/sim-core/src/camera.rs` with Rayon. GPU sphere tracing would unify the pipeline
   but is a significant shader addition.

5. **Multi-GPU**: For NVIDIA clusters, could dispatch different battle subsets to different GPUs.
   WGPU doesn't natively support multi-GPU — would need multiple device instances.
