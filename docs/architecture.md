# ACES — Architecture Map

> **Purpose**: a single top-down map of the project. Read top → bottom for a
> walking tour of the system; jump to any section by following the
> `[[wikilink]]`s. The doc doubles as the root of an Obsidian vault — see
> [[#16. Using this doc as an Obsidian vault]].
>
> **Companion docs**:
> - [`design.md`](design.md) — long-form technical reference
> - [`gpu-mppi.md`](gpu-mppi.md) — GPU MPPI user guide
> - [`2026-04-24-session-archive.md`](2026-04-24-session-archive.md) — GPU MPPI architecture + 7 CPU/GPU consistency audits
> - [`runpod.md`](runpod.md) — cloud setup
> - [`plans/parallel-simulation.md`](../plans/parallel-simulation.md) — Phase-by-phase GPU plan
> - [`plans/curriculum-architecture.md`](../plans/curriculum-architecture.md) — curriculum + crate dependency reference

---

## 0. TL;DR — the project in 90 seconds

ACES is a **1v1 quadrotor dogfight simulator** built around three observations:

1. **Physics must be fast** → all dynamics, MPPI, estimator, perception in **Rust**, with a PyO3 bridge.
2. **RL needs the Python ecosystem** → Gymnasium env, SB3 PPO, curriculum, opponent pool in **Python**.
3. **Training is the bottleneck** → batched CPU MPPI (`batch-sim`) and a full **WGSL GPU MPPI pipeline** (`batch-sim/src/gpu/`) accelerate rollouts ~1000×.

**Combat rule**: keep the opponent inside a 90° forward-facing cone within
2 m and clear line-of-sight for **1.5 s** → kill (no projectiles).

**Current state (2026-04-25)**: GPU MPPI Phases 2 + 3 shipped. 7 CPU/GPU
consistency audits landed. **3+ known divergences remain** (see
[[#11. Fragile boundaries — the correctness map]]). Phase 4 (full GPU
physics) not started.

---

## 1. Stack at a glance

```
┌─────────────────────────────────────────────────────────────────────────┐
│ User                                                                    │
│    ├─ scripts/run.py            interactive runs, training, eval        │
│    ├─ scripts/train_server.py   headless multi-env training             │
│    └─ cargo run -p aces-game    Bevy 3D viewer                          │
└─────────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│ L7  Training — Python                                                   │
│       SB3 PPO  ·  CurriculumTrainer  ·  SelfPlayTrainer  ·  OpponentPool │
│       Callbacks · GpuVecEnv (SB3 wrapper)                                │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│ L6  Environment — Python (Gymnasium)                                    │
│       DroneDogfightEnv (vector 21-dim + FPV dict)                        │
│       NeuralSymbolicEnv (10-step decision interval)                      │
│       reward, obs builder, opponent dispatcher                           │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │  PyO3 (aces._core)
┌───────────────────────────▼─────────────────────────────────────────────┐
│ L5  Bridge — Rust (crates/py-bridge)                                    │
│       Simulation · MppiController · GpuBatchMppi · GpuVecEnv             │
│       StepResult (~110 fields)                                           │
└───────────────┬───────────────────────────┬──────────────────────────────┘
                │                           │
┌───────────────▼─────────────┐  ┌──────────▼──────────────────────────────┐
│ L4  Parallel orchestration  │  │ L3  Estimation                           │
│      crates/batch-sim       │  │      crates/estimator                    │
│        BatchOrchestrator    │  │        EKF (6D, const-vel)               │
│        GpuBatchOrchestrator │  │        ParticleFilter (200, SDF-aware)   │
│        WGSL kernels         │  └──────────┬──────────────────────────────┘
└───────────────┬─────────────┘             │
                │                           │
┌───────────────▼───────────────────────────▼─────────────────────────────┐
│ L2  Control & planning                                                   │
│      crates/mppi: MppiOptimizer (Rayon), pursuit/evasion costs,         │
│        CVaR, chance constraints, belief-weighting                        │
└───────────────┬─────────────────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────────────────┐
│ L1  Simulation core                                                      │
│      crates/sim-core: dynamics(RK4), environment(SDF), lockon,          │
│        collision/LOS, camera(sphere-trace), detection, wind(OU),         │
│        noise, actuator, imu_bias, safety, recorder                       │
└─────────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│ L0  Configuration — TOML                                                 │
│      configs/{drone,arena,rules,curriculum}.toml                         │
└─────────────────────────────────────────────────────────────────────────┘

Side track:
   crates/game (Bevy 3D viewer) — depends on sim-core + mppi only,
   loads exported MLP policy.bin / perception.bin for AI opponents.
```

---

## 2. Repository layout (verified, not as docs claim)

> Drift note: CLAUDE.md says "5 crates, 30 source files". Actual: **6 crates,
> ~48 Rust source files + 3 WGSL shaders**. Update the count when convenient.

```
aces/
├── Cargo.toml                  workspace root, 6 crates
├── pyproject.toml              maturin + poetry, deps + ruff/mypy/pytest
├── CLAUDE.md                   dev conventions
├── README.md
├── .dockerignore
│
├── crates/                     ─────────  Rust workspace  ─────────
│   ├── sim-core/               L1 physics + perception (13 .rs)
│   ├── mppi/                   L2 control (3 .rs)
│   ├── estimator/              L3 estimation (2 .rs)
│   ├── batch-sim/              L4 batched + GPU (10 .rs + gpu/ + 3 .wgsl)
│   ├── py-bridge/              L5 PyO3 bridge (1 .rs)
│   └── game/                   side: Bevy viewer (12 .rs, binary crate)
│
├── aces/                       ─────────  Python package  ─────────
│   ├── config.py               TOML → frozen dataclasses
│   ├── curriculum.py           Phase + CurriculumManager
│   ├── viz.py                  Rerun visualizer
│   ├── env/                    L6 Gymnasium envs
│   │   ├── dogfight.py         DroneDogfightEnv
│   │   ├── ns_env.py           NeuralSymbolicEnv
│   │   ├── obs_layout.py       21-dim slot constants + describe_obs
│   │   └── trajectory.py       Circle / lemniscate / patrol opponents
│   ├── training/               L7 RL training
│   │   ├── self_play.py        SelfPlayTrainer
│   │   ├── curriculum_trainer.py  CurriculumTrainer (--use-gpu-env)
│   │   ├── opponent_pool.py    Elo-rated checkpoint pool
│   │   ├── batched_vec_env.py  Batched opponent inference
│   │   ├── gpu_vec_env.py      SB3 wrapper over aces._core.GpuVecEnv
│   │   ├── callbacks.py        SB3 callbacks (logging, opponent updates)
│   │   ├── evaluate.py         Win-rate / episode metrics
│   │   └── logging.py          Structured logging
│   ├── policy/                 Policy networks + export
│   │   ├── extractors.py       CnnImuExtractor (FPV)
│   │   ├── constrained_ppo.py  LagrangianPPO
│   │   └── export.py           MLP → policy.bin
│   └── perception/             Neural-symbolic stack
│       ├── oracle.py           God Oracle ground-truth labels
│       ├── perception_net.py   Supervised perception MLP
│       └── neural_symbolic.py  Mode selector + MPPI executor
│
├── configs/                    ─────────  TOML configuration  ─────────
│   ├── drone.toml              Crazyflie 2.1 physics
│   ├── arena.toml              10×10×3 m + 5 pillars
│   ├── rules.toml              lockon, mppi, noise, camera, reward, training
│   └── curriculum.toml         6 phases
│
├── scripts/                    ─────────  Entry points  ─────────
│   ├── run.py                  multi-mode CLI
│   ├── train_server.py         headless training daemon
│   ├── train_perception.py     perception NN trainer
│   ├── collect_oracle_data.py  oracle dataset collector
│   ├── experiment_chain.py     hover→pursuit transfer experiment
│   ├── smoke_gpu_ppo.py        GPU PPO smoke test
│   ├── bench_training_throughput.py  PPO throughput benchmark
│   ├── check_gpu_setup.sh      5-stage GPU setup validator
│   ├── pre-commit.sh           lint + tests
│   └── install-hooks.sh
│
├── tests/                      ~150 Rust #[test] + ~183 Python test fns
│   └── conftest.py             core_available, gpu_available fixtures
│
├── docker/                     ─────────  Cloud / containers  ─────────
│   ├── Dockerfile.dev-base     Runpod base + Rust + Node + Vulkan + Claude Code
│   ├── Dockerfile.aces         Thin dev image (clone-on-boot)
│   ├── Dockerfile.train        Multi-stage CPU training image (wheel baked)
│   ├── runpod-start.sh         8-stage Pod bootstrap
│   └── apply-dotfiles.sh       Dotfiles installer
│
└── docs/                       ─────────  Documentation  ─────────
    ├── architecture.md         (this file)
    ├── design.md               long-form reference
    ├── gpu-mppi.md             user guide
    ├── runpod.md               cloud setup
    ├── parallel_architecture.md  parallel sim history
    ├── overnight-report.md     2026-04-23 training-bug session
    └── 2026-04-24-session-archive.md  GPU MPPI + 7 audits
```

---

## 3. Layer details

Each layer has: **what** it does, **where** it lives, **what it depends on**,
and **what crosses its boundary**.

### 3.1 Layer L1 — Simulation core ([[#L1 sim-core]])

**Where**: `crates/sim-core/`, no external Rust deps within the workspace.

**13 source files** = 4 concerns:

| Concern        | Files                                              |
|----------------|----------------------------------------------------|
| State + dyn    | `state.rs` (13-DOF), `dynamics.rs` (RK4, motor mix) |
| World          | `environment.rs` (Arena + SDF), `collision.rs` (LOS), `lockon.rs` |
| Disturbances   | `wind.rs` (OU), `noise.rs` (Gaussian), `actuator.rs` (motor), `imu_bias.rs` |
| Perception     | `camera.rs` (sphere-traced depth), `detection.rs` (geometric)       |
| Engineering    | `safety.rs` (envelope), `recorder.rs` (trajectories)                |

**Public API surface**: every module is `pub mod`, plus a few re-exports
(`SimFrame`, `SimRecorder`, `SafetyEnvelope`, `SafetyStatus`).

**Math invariants** (see [[#10. Math framework]] for derivations):
- 13-DOF state: `[p(3), v(3), q_wxyz(4), ω(3)]`. Quaternion **must be unit
  length** after every step — `step_rk4` renormalizes.
- RK4 at 1 ms (1000 Hz physics), 10 substeps per control tick (100 Hz
  control), camera at 30 Hz.
- All world frames: **Z-up**. Body frame: **+X forward, +Y left, +Z up**.
- SDF convention: `arena.sdf(p) = min(boundary_sdf(p), obstacle_sdf(p))`.
  Collision when `sdf(p) < drone_radius` (0.05 m).

### 3.2 Layer L2 — Control & planning ([[#L2 mppi]])

**Where**: `crates/mppi/`, depends on `sim-core`.

**3 source files**:

- `cost.rs` — pursuit / evasion / belief-weighted costs.
- `optimizer.rs` — `MppiOptimizer`: parallel sampling (Rayon), softmax
  weighting, CVaR risk, chance-constrained Lagrangian, warm-start shift.
- `rollout.rs` — `rollout()` propagates a control sequence through dynamics.

**MPPI parameters** (from `rules.toml [mppi]`):
- `num_samples = 1024`, `horizon = 50`, `temperature = 10.0`,
  `noise_std = 0.03 N`.
- Cost weights: `w_dist=1.0, w_face=5.0, w_ctrl=0.01, w_obs=1000.0,
  d_safe=0.3 m`.
- Risk: `cvar_alpha=0.05, cvar_penalty=10.0`,
  `chance_delta=0.01, λ_lr=0.1, λ_init=100`.

### 3.3 Layer L3 — Estimation ([[#L3 estimator]])

**Where**: `crates/estimator/`, depends on `sim-core`.

- `ekf.rs` — 6D constant-velocity EKF (pos + vel). Predict every control
  tick, update only when opponent visible.
- `particle_filter.rs` — 200 particles, SDF-constrained prediction,
  systematic resampling, used when occluded.

**Switching rule**: EKF when LOS clear, PF when occluded. Belief output
(mean + variance) feeds belief-weighted MPPI (`mppi/cost.rs`).

### 3.4 Layer L4 — Parallel orchestration ([[#L4 batch-sim]])

**Where**: `crates/batch-sim/`, depends on `sim-core` + `mppi`. GPU support
is feature-gated (`--features gpu`).

**Two orchestrators**:

| Orchestrator             | Backend         | File                            |
|--------------------------|-----------------|---------------------------------|
| `BatchOrchestrator`      | Rayon CPU       | `src/orchestrator.rs`           |
| `GpuBatchOrchestrator`   | WGSL compute    | `src/gpu/orchestrator.rs`       |

**Per-battle data**: `BattleState { state_a, state_b, wind_a, wind_b,
lockon_a, lockon_b, step_count, done }`.

**Spawn modes** ([[#11.5 Spawn alignment]]):
`SpawnMode::{FixedFromArena, FixedWithJitter, Random}`. Default is
`FixedWithJitter` to match CPU env's small per-reset variation.

**21-dim observation builder** (`src/observation.rs`) — this is the
canonical layout. Slot reference in `aces/env/obs_layout.py` mirrors it.

**Reward** (`src/reward.rs`) — canonical formula. Three drift cases
remain vs CPU env (see [[#11.2 Reward formula]]).

**f32 reference files** (`src/f32_dynamics.rs`, `src/f32_sdf.rs`,
`src/f32_cost.rs`) — pure CPU f32 ports of the f64 sim-core math, used as
the **GPU parity baseline** (no SIMD, no Rayon, intentionally unoptimized).

#### GPU sub-pipeline (`src/gpu/`)

| File                  | Purpose                                                         |
|-----------------------|-----------------------------------------------------------------|
| `mod.rs`              | Module root, adapter re-export                                  |
| `adapter.rs`          | `wgpu` device probe + minimal compute round-trip                |
| `pipeline.rs`         | `GpuBatchMppi`: 12 bindings, 2 pipelines, dispatch + readback   |
| `orchestrator.rs`     | `GpuBatchOrchestrator`: full episodic loop on GPU MPPI          |
| `shader.rs`           | WGSL concat + naga structural validation at crate-build time    |
| `shaders/mppi_helpers.wgsl`  | f32 RK4 + SDF + quaternion helpers                       |
| `shaders/mppi_rollout.wgsl`  | `rollout_and_cost` @compute @workgroup_size(1)           |
| `shaders/mppi_softmax.wgsl`  | `softmax_reduce` @compute @workgroup_size(256)           |

**Bind group** (12 bindings, see `pipeline.rs`):
`0 states · 1 enemies · 2 mean_ctrls · 3 noise · 4 costs · 5 ctrls_out ·
6 result · 7 DroneParams · 8 CostWeights+arena · 9 obstacles · 10 MppiDims ·
11 wind_per_drone`. Binding 11 added by `feature/gpu-wind`; future
additions need explicit std140 padding.

**Validation** (3 levels):
1. Naga structural validation (no GPU needed) — `gpu/shader.rs` runs at
   crate-build time.
2. CPU↔GPU parity — `tests/gpu_pipeline.rs::test_gpu_matches_cpu_reference_parity`,
   gated on `--features gpu`. Asserts `max_diff < 1e-3`.
3. End-to-end PPO smoke — `tests/test_gpu_ppo_smoke.py`. Auto-skips when
   the `gpu` feature is not built into `aces._core`.

### 3.5 Layer L5 — Python ↔ Rust bridge ([[#L5 py-bridge]])

**Where**: `crates/py-bridge/src/lib.rs`, depends on `sim-core` + `mppi` +
`estimator` + (gpu) `batch-sim`. Module name: `aces._core`.

**PyO3 surface**:

| Class                  | Backed by                                | Used by                              |
|------------------------|------------------------------------------|--------------------------------------|
| `Simulation`           | sim-core + estimator                     | CPU env, scripts/run.py, Bevy game   |
| `StepResult`           | the ~110-field result of `Simulation.step` | CPU env, scripts/run.py            |
| `MppiController`       | `mppi::MppiOptimizer`                    | CPU env opponent, MPPI-vs-MPPI mode  |
| `GpuBatchMppi`         | `batch-sim::gpu::pipeline::GpuBatchMppi` | scripts/smoke, low-level Python use  |
| `GpuVecEnv`            | `batch-sim::gpu::orchestrator::GpuBatchOrchestrator` | `aces.training.gpu_vec_env` |

GPU classes are guarded by `cfg(feature = "gpu")`; build with
`maturin develop --features gpu` to enable.

### 3.6 Layer L6 — Gymnasium environment ([[#L6 env]])

**Where**: `aces/env/`.

- `dogfight.py::DroneDogfightEnv` — Gymnasium env; vector (21-dim) or FPV
  (`Dict{image: (1,60,80), vector: (12,)}`) observation; 4-dim continuous
  action `[-1, 1]` denormalized to `motor = hover + a * (max_thrust - hover)`.
- `obs_layout.py` — observation slot constants + `describe_obs(obs)`
  helper. **This is the canonical Python-side reference for the 21-dim
  layout** (mirror of `crates/batch-sim/src/observation.rs`).
- `trajectory.py` — circle / lemniscate / patrol generators with PD
  position controller for Phase-1 opponents.
- `ns_env.py` — wraps `DroneDogfightEnv` for hierarchical
  neural-symbolic policies (10-step decision interval).

### 3.7 Layer L7 — Training ([[#L7 training]])

**Where**: `aces/training/`.

| Component                     | File                       | Role                                              |
|-------------------------------|----------------------------|---------------------------------------------------|
| SelfPlayTrainer               | `self_play.py`             | single-env PPO + lagged opponent                  |
| CurriculumTrainer             | `curriculum_trainer.py`    | multi-phase, weight transfer, `--use-gpu-env`     |
| OpponentPool                  | `opponent_pool.py`         | Elo-rated (K=32, max 20 ckpts)                    |
| BatchedOpponentVecEnv         | `batched_vec_env.py`       | one batched policy.predict() per step             |
| GpuVecEnv (Python wrapper)    | `gpu_vec_env.py`           | SB3 VecEnv around `aces._core.GpuVecEnv`          |
| Callbacks                     | `callbacks.py`             | logging, opponent update, promotion check         |
| evaluate                      | `evaluate.py`              | win-rate / episode metrics                        |
| logging                       | `logging.py`               | structured run dirs (training.log, episodes.csv)  |

### 3.8 Layer L8 — Policy + perception ([[#L8 policy]])

**Where**: `aces/policy/` + `aces/perception/`.

- `policy/extractors.py::CnnImuExtractor` — Conv2d(1→32→64→64) +
  Linear(12→64), concat → 192-dim, used by SB3 `MultiInputPolicy` for FPV.
- `policy/constrained_ppo.py::LagrangianPPO` — PPO with Lagrangian dual
  ascent on a constraint cost (used for collision-rate constraint).
- `policy/export.py` — `export_mlp_policy(model, path)` writes binary
  format `[u32 num_layers, [u32 rows, u32 cols, f32[r×c] W, f32[r] b]×N]`
  consumed by `crates/game/src/policy.rs` and `weights.rs`.
- `perception/oracle.py::GodOracle` — ground-truth semantic labels.
- `perception/perception_net.py` — supervised MLP `21→64→64→9`.
- `perception/neural_symbolic.py` — mode selector NN + MPPI executor.

### 3.9 Bevy game (side track) ([[#Bevy game]])

**Where**: `crates/game/`, depends on `sim-core` + `mppi` only — **no
Python at runtime**.

12 source files — main pieces:
- `simulation.rs` — fixed-update at 100 Hz calling sim-core.
- `policy.rs` + `weights.rs` — load `policy.bin` for AI opponent.
- `perception.rs` + `fsm.rs` — neural-symbolic FSM if `perception.bin`
  present.
- `arena.rs` / `drone.rs` / `camera.rs` / `hud.rs` / `marker.rs` /
  `input.rs` — rendering, controls, HUD.
- Coordinate transform: `sim (x,y,z) → Bevy (x,z,y)`. Quaternion mapping
  in `simulation.rs`.

**Runtime**: `cargo run -p aces-game --release`.

### 3.10 Layer L0 — Configuration ([[#L0 configs]])

See [[#9. Configuration map]] for full TOML-by-TOML reference.

---

## 4. Data flow — single CPU training step

```
SB3 PPO.collect_rollouts()
   │
   │ actions ∈ [-1, 1], shape (n_envs, 4)
   ▼
DroneDogfightEnv.step(action)         (aces/env/dogfight.py)
   │
   │ a) denormalize: motor = hover + a * (max_thrust - hover)        (Audit 1)
   │ b) compute opponent action: trajectory / MPPI / lagged policy
   ▼
aces._core.Simulation.step(motors_a, motors_b)         (PyO3 → Rust)
   │
   ├─ 10 × RK4 substeps with OU wind                   (sim-core/dynamics.rs)
   ├─ collision check (SDF, drone_radius=0.05m)        (sim-core/environment.rs)
   ├─ lockon update (FOV ∧ range ∧ LOS, accumulate)    (sim-core/lockon.rs)
   ├─ EKF predict (always) + update (if visible)       (estimator/ekf.rs)
   ├─ Particle filter (if occluded)                    (estimator/particle_filter.rs)
   ├─ camera render (if 30 Hz tick)                    (sim-core/camera.rs)
   └─ geometric detection                              (sim-core/detection.rs)
   │
   │ → StepResult (~110 fields)                         (py-bridge/src/lib.rs)
   ▼
DroneDogfightEnv builds obs (21-dim or FPV dict)       (env/dogfight.py + obs_layout.py)
   │
   │ uses arena.sdf() for slot [15]                    (Audit 3)
   ▼
DroneDogfightEnv.compute_reward                         (env/dogfight.py)
   │
   │ canonical formula matches batch-sim/src/reward.rs (Audit 2)
   │ but: 3 known divergences remain                   (see §11)
   ▼
return (obs, reward, terminated, truncated, info)
   │
   ▼
PPO learner buffer
```

## 5. Data flow — single GPU batched training step

```
SB3 PPO.collect_rollouts()                       (n_envs = N)
   │
   │ actions ∈ [-1, 1], shape (N, 4)
   ▼
aces.training.GpuVecEnv.step(actions)            (gpu_vec_env.py)
   │
   │ denormalize_action: motor = hover + a * (max - hover)        (Audit 1)
   │ Note: matches CPU env. Pure-fn unit test at
   │       tests/test_denormalize_action.py
   ▼
aces._core.GpuVecEnv.step_with_agent_a_actions(motors_a)   (PyO3)
   │
   ▼
GpuBatchOrchestrator::step                       (batch-sim/src/gpu/orchestrator.rs)
   │
   │ 1. pack_and_dispatch_gpu_mppi:                            ────────────► GPU
   │      - upload state/enemy/mean_ctrl/noise/wind_per_drone
   │      - dispatch rollout_and_cost(N_DRONES × N_SAMPLES)
   │                                              one workgroup per (drone, sample)
   │                                              produces costs[d,s] + ctrls_out[d,s,t,c]
   │      - dispatch softmax_reduce(N_DRONES)     workgroup_size=256
   │                                              produces result[d, t, c]
   │      - readback via staging buffer           ◄────────────── GPU
   │
   │ 2. for each battle (Rayon CPU):
   │      - apply motors_a[i] (PPO) + motors_b[i, 0] (MPPI)
   │      - step_physics: 10 × RK4 substeps with OU wind
   │      - lockon update + collision + visibility
   │      - build_observation (21-dim, matches CPU env)
   │      - compute_reward_a (matches CPU env up to known drift)
   │
   │ 3. warm-start shift:
   │      mean_ctrls[:, 1:] = result[:, 1:]
   │      mean_ctrls[:, H-1] = hover
   │
   │ → (obs (N,21), reward (N,), done (N,), info)
   ▼
SB3 PPO buffer
```

**Caveat — GPU-only opponent semantics**: `GpuVecEnv` only does
**MPPI-vs-MPPI** semantics. Curriculum phases configured with other
opponent types (trajectory follower, hover, opponent pool) **silently
fall back** to MPPI-vs-MPPI under `--use-gpu-env`. Trainer logs a warning.
See `docs/gpu-mppi.md` Caveats §1.

## 6. Data flow — MPPI action computation (single tick)

```
mean_controls (warm-start, shape H×4)
   │
   ▼
For each sample k ∈ [0, N=1024)         (par_iter, Rayon)
   │
   │ 1. seed = thread_rng().gen()
   │ 2. for t ∈ [0, H=50):
   │      ε ~ N(0, σ² = 0.03²)
   │      u_k[t] = clamp(mean[t] + ε, 0, max_thrust)
   │ 3. rollout: H steps × 10 substeps × RK4   (with sampled wind, see CVaR)
   │ 4. cost_k = Σ_t cost_fn(state[t], enemy, u_k[t], hover, arena, weights)
   │ 5. if max_penetration > 0: cost_k += 1e8     (hard collision penalty)
   │
   ▼
[optional] CVaR: select_nth_unstable for worst-α threshold
            → penalize fraction
[optional] chance constraint: λ * P̂(collision) — dual ascent on λ
   │
   ▼
softmax: w_k = exp(-(c_k - c_min) / T) / Σ_j exp(...)
   │
   ▼
new_mean[t] = Σ_k w_k · u_k[t]                      (per-time-step)
   │
   ▼
return new_mean[0]; shift mean_controls left, append hover (warm-start)
```

GPU version: same algorithm, two compute kernels. See [[#3.4 Layer L4 — Parallel orchestration]].

## 7. Curriculum data flow

```
CurriculumTrainer.train()                   (aces/training/curriculum_trainer.py)
   │
   │ for phase ∈ phases (ordered by curriculum.toml):
   │
   ├─ build VecEnv from phase params:
   │    - task name → opponent type (trajectory / mppi / pool)
   │    - wind_sigma, obs_noise_std
   │    - motor delay/noise/bias, IMU bias
   │    - use_fpv → CnnImuExtractor
   │    - if --use-gpu-env: build GpuVecEnv (MPPI-vs-MPPI fallback for non-MPPI opps)
   │
   ├─ load PPO model from previous phase's checkpoint
   │    - VecNormalize stats (obs/reward running mean+var) carried over
   │
   ├─ install callbacks:
   │    - TrainingStatsCallback (kills, deaths, rewards)
   │    - EpisodeLoggerCallback (CSV per episode)
   │    - TensorBoardMetricsCallback (win_rate, kill_rate)
   │    - PromotionCheckCallback (promote_condition each 5k)
   │    - VecOpponentUpdateCallback (copy weights to opponent envs)
   │    - PoolOpponentCallback (resample from Elo pool)
   │
   ├─ model.learn(max_timesteps_phase, callback=composite)
   │
   ├─ on promotion (or max_timesteps):
   │    - save model + VecNormalize stats
   │    - add to OpponentPool
   │
   └─ next phase
```

**Promotion conditions** parsed by `curriculum.py`:
- `"steps"` — always True (timestep budget is the gate).
- `"win_rate>0.30"` — regex `(\w+)>([\d.]+)`. Requires
  `stats["episodes"] >= promote_window`.
- See `_CONDITION_RE` in `aces/curriculum.py`.

---

## 8. Entry points

```
scripts/run.py
   --mode mppi-vs-mppi     interactive (default)         imports Simulation, MppiController
   --mode train            single-task PPO               imports SelfPlayTrainer
   --mode curriculum       multi-phase PPO               imports CurriculumTrainer
   --mode evaluate         win-rate evaluation           imports evaluate.evaluate
   --mode export           export MLP → policy.bin       imports policy.export

   Common flags:
     --fpv                 enable depth camera + CNN
     --no-noise            zero wind + obs noise
     --no-vis              skip Rerun viewer
     --use-gpu-env         use GpuVecEnv backend
     --gpu-mppi-samples N  default 128
     --gpu-mppi-horizon N  default 15
     --n-envs N            default 8
     --task <hover|pursuit_linear|pursuit_evasive|search_pursuit|dogfight>

scripts/train_server.py
   Headless training daemon.
   --n-envs N              default 8
   --resume <path>         resume from checkpoint dir
   --checkpoint-interval   default 50000
   SIGINT/SIGTERM handlers checkpoint before exit.

scripts/smoke_gpu_ppo.py
   poetry run python scripts/smoke_gpu_ppo.py --timesteps 256 --n-envs 4
   Sanity check: PPO actually learns against GpuVecEnv.

scripts/bench_training_throughput.py
   PPO throughput benchmark CPU vs GPU env, config sweeps.

scripts/check_gpu_setup.sh
   5-stage validator (see §13.4).

cargo run -p aces-game --release
   Bevy 3D viewer. Loads policy.bin / perception.bin if present.
```

---

## 9. Configuration map

All tunables live in `configs/*.toml`. **Audit pattern from
2026-04-23/24**: when a config knob isn't taking effect, walk this chain
and look for silent defaults at every layer:

```
configs/X.toml
   │  toml.load
   ▼
aces/config.py dataclass (frozen)
   │  __init__ kwargs / attribute access
   ▼
aces/env/dogfight.py    or    aces/training/gpu_vec_env.py
   │  PyO3 kwargs
   ▼
crates/py-bridge/src/lib.rs  (PyGpuVecEnv, Simulation, …)
   │  struct field assignment
   ▼
Rust BatchConfig / RewardConfig / WindConfig / DroneParams
   │  uniform buffer write
   ▼
WGSL uniform binding
   │
   ▼  used in shader
```

Audits 1, 6, 7 were **silent defaults** at one of these boundaries.

### 9.1 `drone.toml` — Crazyflie 2.1

`[physical]`: mass=0.027 kg · arm_length=0.04 m · max_motor_thrust=0.15 N
· torque_coefficient=0.005964 · drag_coefficient=0.01 · gravity=9.81.
`[inertia]`: ixx=iyy=1.4e-5 · izz=2.17e-5.
`[simulation]`: dt_sim=0.001 (1000 Hz) · dt_ctrl=0.01 (100 Hz) · substeps=10.

### 9.2 `arena.toml`

`[bounds]`: 10×10×3 m. `[drone]`: collision_radius=0.05 m.
`[spawn]`: drone_a=(1,1,1.5), drone_b=(9,9,1.5).
`[[obstacles]]`: 5 × box, half_extents (0.5, 0.5, 1.5), centers
(2,2), (2,8), (5,5), (8,2), (8,8) at z=1.5.

### 9.3 `rules.toml` — the workhorse

| Section | Key params |
|---------|-----------|
| `[lockon]`            | fov=90°, lock_distance=2.0 m, lock_duration=1.5 s |
| `[mppi]`              | num_samples=1024, horizon=50, T=10.0, σ=0.03 N |
| `[mppi.weights]`      | w_dist=1.0, w_face=5.0, w_ctrl=0.01, w_obs=1000, d_safe=0.3 |
| `[mppi.risk]`         | wind_θ=2.0, wind_σ=0.0, cvar_α=0.05, cvar_pen=10, cc_δ=0.01, cc_λ_lr=0.1, cc_λ_init=100 |
| `[noise]`             | wind_θ=2.0, wind_σ=0.3 N, obs_noise_std=0.1, motor_τ=0, motor_σ=0, motor_bias=0, imu_accel/gyro_bias=0 |
| `[domain_randomization]` | enabled=false (mass/inertia/thrust/drag ranges) |
| `[camera]`            | enabled=false, 320×240 @ 90°, 30 Hz, max_depth=15 m, policy 80×60 |
| `[detection]`         | drone_radius=0.05, min_confidence_distance=5.0 |
| `[reward]`            | kill=+100, killed=−100, collision=−50, lock_progress=5, control=0.01, approach=0.05, survival=0.01, info_gain=0.02, lost_contact=0.005 |
| `[task_reward_overrides.<task>]` | per-task overrides (pursuit_linear, pursuit_evasive, search_pursuit, dogfight, search_destroy, hover) |
| `[training]`          | total_timesteps=500k, lr=3e-4, batch=64, n_steps=2048, γ=0.99, gae_λ=0.95, clip=0.2, n_epochs=10, opponent_update_interval=10000, max_episode_steps=1000 |

### 9.4 `curriculum.toml` — the 6 phases

| # | Phase            | Task            | Opponent     | Wind | Obs noise | FPV | Steps | Promote |
|---|------------------|-----------------|--------------|------|-----------|-----|-------|---------|
| 0 | hover_stabilize  | hover           | none         | 0    | 0         | no  | 100k  | reward>−5.0 (window 50) |
| 1 | pursuit_linear   | pursuit_linear  | trajectory   | 0    | 0         | no  | 200k  | win_rate>0.30 (100) |
| 2 | pursuit_evasive  | pursuit_evasive | mppi_evasion | 0    | 0         | no  | 300k  | win_rate>0.30 (200) |
| 3 | search_pursuit   | search_pursuit  | mppi_evasion | 0    | 0.1       | no  | 300k  | win_rate>0.25 (200) |
| 4 | self_play_noisy  | dogfight        | pool         | 0.3  | 0.1       | no  | 2M    | win_rate>0.55 (500) |
| 5 | fpv_transfer     | dogfight        | pool         | 0.3  | 0.1       | yes | 5M    | steps (500) |

---

## 10. Math framework

### 10.1 13-DOF rigid-body dynamics

State `x = [p, v, q, ω]` with `p ∈ ℝ³` (world), `v ∈ ℝ³` (world), `q ∈ S³`
(unit quaternion, `wxyz` storage), `ω ∈ ℝ³` (body).

```
ṗ = v
v̇ = [0, 0, -g] + (1/m) R(q) [0, 0, F_total] + (1/m) F_drag + F_wind/m
q̇ = ½ q ⊗ [0, ω]                            (Hamilton product, normalize each step)
I ω̇ = τ - ω × (I ω)                          (Euler equation)
```

X-config motor mixing (`crates/sim-core/src/dynamics.rs::motor_mixing`):
```
F_total = f1 + f2 + f3 + f4
τ_x = (d/√2) ( f1 - f2 - f3 + f4)
τ_y = (d/√2) ( f1 + f2 - f3 - f4)
τ_z = c_τ   (-f1 + f2 - f3 + f4)
```

Integrator: explicit RK4 with `dt = 0.001 s`, 10 substeps per control tick.

### 10.2 SDF + collision

`Arena::sdf(p) = min(boundary_sdf(p), obstacle_sdf(p))`:
- `boundary_sdf(p) = min(p.x, B.x − p.x, p.y, B.y − p.y, p.z, B.z − p.z)`
- `obstacle_sdf(p) = min over obstacles of obstacle.sdf(p)`
- Box SDF: `q = |p − c| − h`, `outside = ‖max(q,0)‖`,
  `inside = min(max(q.x, q.y, q.z), 0)`, `sdf = outside + inside`.
- Collision: `sdf(p) < drone_radius (0.05 m)`.

### 10.3 Lock-on rule

```
d   = p_B − p_A
f_A = R(q_A) · [1, 0, 0]                (body +X)
θ   = arccos((d · f_A) / ‖d‖)
lock_holds = θ < FOV/2  ∧  ‖d‖ < D_lock  ∧  LOS_clear(A,B)
timer accumulates while lock_holds, resets on any break
kill when timer ≥ T_lock
```

### 10.4 OU wind disturbance

```
dF = θ (μ − F) dt + σ dW       θ=2.0, μ=0, σ_default=0.3 N
```

GPU rollout uses **constant wind across horizon** (current OU realization,
held). True physics step still applies stochastic OU. See
[[#11.6 GPU rollout wind approximation]].

### 10.5 MPPI

```
sample N controls:    u_k[t] ~ clamp(mean[t] + N(0, σ²), 0, max_thrust)
rollout:              x_k[t+1] = step_rk4(x_k[t], u_k[t], wind_k)
cost:                 c_k = Σ_t cost_fn(x_k[t], enemy, u_k[t]) + 1e8 · 𝟙[collision]
softmax weight:       w_k = exp(-(c_k - c_min)/T) / Σ_j exp(...)
update:               new_mean[t] = Σ_k w_k u_k[t]
warm start:           mean = [new_mean[1], …, new_mean[H-1], hover]
```

CVaR variant: penalize the worst-α fraction. Chance-constrained variant:
add Lagrangian `λ · P̂(collision)` with online dual ascent.

### 10.6 EKF (6D, const-velocity)

```
F = [I  dt·I; 0  I]                       (6×6)
Q = q_a · [dt³/3 · I  dt²/2 · I;          q_a = 4.0 (accel spectral density)
           dt²/2 · I  dt    · I]
H = [I  0]                                (position-only obs)
predict: x ← F x;  P ← F P Fᵀ + Q
update (visible only): K = P Hᵀ (H P Hᵀ + R)⁻¹; x ← x + K(z − Hx); P ← (I − KH) P (Joseph form)
```

### 10.7 Particle filter (200 particles, SDF-aware)

Predict: const-vel + accel noise + reject samples with `arena.sdf(p) < 0`.
Weight: `w_i ∝ exp(−‖p_i − z‖² / (2σ²))`.
Resample: systematic resampling.

### 10.8 21-dim observation layout

(Mirror of `crates/batch-sim/src/observation.rs` and
`aces/env/obs_layout.py`.)

| Slot   | Field                               | Frame          |
|--------|-------------------------------------|----------------|
| 0:3    | own velocity                        | world          |
| 3:6    | own angular velocity                | body           |
| 6:9    | relative position to opponent       | world          |
| 9:12   | relative velocity                   | world          |
| 12:15  | own attitude (roll, pitch, yaw)     | from quaternion |
| 15     | nearest-obstacle SDF (combined!)    | scalar (Audit 3) |
| 16     | lock_progress self → opp [0,1]      | scalar         |
| 17     | being_locked progress opp → self    | scalar         |
| 18     | opponent visible (0/1)              | scalar         |
| 19     | belief uncertainty (0 in MPPI-only) | scalar         |
| 20     | time since last seen                | scalar         |

### 10.9 Reward (canonical, `crates/batch-sim/src/reward.rs`)

```
Terminal:
   +kill_reward        if kill_a
   −killed_penalty     if kill_b
   −collision_penalty  if collision_a
   +opp_crash_reward   if collision_b ∧ ¬collision_a   (Audit 2)
   0                   if truncation                    (CPU drift, see §11)

Shaping (per step):
   +survival_bonus
   +lock_progress_reward   · max(lock_new_a − lock_prev_a, 0)   (Audit 2: clamp to [0,∞))
   +approach_reward        · max(d_prev − d_new, 0)
   −control_penalty        · ‖u − hover‖²
   +info_gain_reward       · max(belief_var_prev − belief_var_new, 0)
   −lost_contact_penalty   · time_since_seen
```

---

## 11. Fragile boundaries — the correctness map

This is the section to read **before** any cloud run. The same logic
exists in **four** places — they must agree.

```
configs/*.toml
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│            aces/env/dogfight.py        (CPU env, Gymnasium)             │
├────────────────────────────────────────────────────────────────────────┤
│            crates/batch-sim/src/       (CPU Rayon batch — canonical)    │
├────────────────────────────────────────────────────────────────────────┤
│            crates/batch-sim/src/gpu/   (GPU MPPI + WGSL kernels)        │
├────────────────────────────────────────────────────────────────────────┤
│            crates/py-bridge/src/lib.rs::Simulation  (single-drone PyO3) │
├────────────────────────────────────────────────────────────────────────┤
│            crates/game/src/simulation.rs  (Bevy interactive)            │
└────────────────────────────────────────────────────────────────────────┘
```

### 11.1 Action denormalization — **AUDITED ✓** (`feature/action-consistency`)

CPU + GPU now agree: `motor = hover + a · (max − hover)`. Pure-fn unit
test: `tests/test_denormalize_action.py`. Round-trip parity test:
`tests/test_action_normalization_consistency.py`.

### 11.2 Reward formula — **PARTIAL ⚠** (`feature/reward-consistency`)

Aligned: lock-progress delta clamped to `[0, ∞)`; opp-crash gated on
`¬collision_a`. Test: `tests/test_reward_consistency.py` (11 cases).

**Remaining drift**:
1. **Terminal priority** when `kill_a ∧ collision_a` same step: CPU emits
   −50 (collision first), Rust emits +100 (kill first). Recommendation:
   align both to **kill-first** (lock-on completion is the goal terminal).
2. **Truncation reward**: CPU runs full shaping on truncation tick; Rust
   short-circuits to 0. Tiny per-step drift; only matters for
   reward-curve baselines.

### 11.3 Observation slot [15] — **AUDITED ✓** (`feature/obs-consistency`)

CPU + Bevy now use combined `arena.sdf() = min(boundary, obstacle)`.
Test: `tests/test_observation_consistency.py`.

### 11.4 Test precondition + Bevy SDF — **AUDITED ✓** (`feature/fix-test-and-game`)

Observation-consistency test now pins `obs_noise_std = 0` and gates on
LOS visibility. Bevy `simulation.rs` uses `arena.sdf()` for HUD wall
warning.

### 11.5 Spawn alignment — **AUDITED ✓** (`feature/spawn-audit`)

`SpawnMode::{FixedFromArena, FixedWithJitter, Random}`. Default matches
CPU env. `BatchConfig::max_steps` now honored (was hardcoded 1000).

### 11.6 GPU rollout wind approximation — **PENDING ⚠**

GPU rollout uses **constant wind across horizon** (current OU
realization). CPU MPPI samples OU per step. Effect: CVaR risk MPPI
collapses to deterministic on GPU. Fix needs GPU-side PCG hash RNG +
in-shader OU sampling.

### 11.7 Reward TOML plumbing — **AUDITED ✓** (`feature/rules-toml-plumb`)

`GpuVecEnv` now loads `[reward]` from `rules.toml`. Test:
`tests/test_gpu_reward_plumbing.py`.

### 11.8 Wind TOML plumbing — **AUDITED ✓** (`feature/wind-plumb`)

`wind_sigma` + `wind_theta` now read from `[noise]`. Test:
`tests/test_gpu_wind_plumbing.py`.

### 11.9 Per-task reward overrides → GpuVecEnv — **PENDING ⚠**

`[task_reward_overrides.<task>]` is read by CPU env but **not** plumbed
to GpuVecEnv. GPU curriculum phases always use base `[reward]`. Fix
pattern: extend Audit 6 with a per-phase merge layer.

### 11.10 batch-sim missing CPU noise features — **PENDING ⚠⚠**

CPU env applies `obs_noise_std`, motor delay (first-order), motor noise +
bias, IMU random-walk bias (`crates/sim-core/src/{noise,actuator,imu_bias}.rs`).
**`crates/batch-sim/` does none of this.** Phases 3–5 will train under
*less* noise on GPU than on CPU. Fix is non-trivial: port crates and add
WGSL code where applicable.

### 11.11 Quaternion convention

`DroneState` stores `(qw, qx, qy, qz)` (`wxyz`). GPU side requires
**unit norm on input** — `compute_batch_actions` does **not**
renormalize on unpack. Orchestrator enforces via post-RK4 normalization.
**Drift risk**: anyone calling `GpuBatchMppi` directly from Python must
pass unit quaternions or get silent wrong rollouts. Documented in
`docs/gpu-mppi.md` Caveat 3.

### 11.12 Euler-angle convention — **AUDITED 2026-04-25, MINOR DRIFT ⚠**

Two independent implementations exist:
- **nalgebra `euler_angles()`** — used by CPU env path
  (`crates/py-bridge/src/lib.rs:202-205`) and Bevy
  (`crates/game/src/policy.rs:73`).
- **Hand-rolled** in `crates/batch-sim/src/observation.rs:60-83`.

Both produce **intrinsic ZYX / extrinsic XYZ Tait-Bryan** angles —
algebraically equivalent for unit quaternions, but the **gimbal-lock
branch differs**: nalgebra returns `yaw=0` and a special-cased `roll`
when `|sin(pitch)| ≥ 1`; batch-sim returns the unclamped formula. At
pitch ≈ ±90° the obs slots `[12]` (roll) and `[14]` (yaw) can disagree
between CPU env and batch-sim/GpuVecEnv.

GPU rollout never computes Euler (cost uses quaternions directly via
`quat_rotate`), so no third divergence point.

**Severity: Low.** Drones unlikely to hit pitch ≈ ±π/2 in normal
training, and PPO learns around the discontinuity. Add a parametric
regression test now (cheap), but does not block cloud run.

### 11.13 Lock-on parameter plumbing — **AUDITED 2026-04-25, BUG FOUND ⚠⚠⚠**

Two stacked silent defaults on the GPU path:
1. `crates/batch-sim/src/gpu/orchestrator.rs:124` —
   `let lockon_params = LockOnParams::default();` (hardcoded
   2.0 m / 1.5 s / 45° FOV).
2. `crates/batch-sim/src/gpu/shaders/mppi_rollout.wgsl:178` —
   `let fov_half: f32 = 0.7853981633974483; // PI / 4` (hardcoded
   literal in the cost shader).
3. `crates/py-bridge/src/lib.rs:1206-1228` (`PyGpuVecEnv::new`) takes
   **no `fov_degrees / lock_distance / lock_duration` kwargs**, so even if
   the orchestrator is fixed, Python users have no override.

**Severity: HIGH.** Same failure mode as Audits 6 + 7 — every GPU
curriculum phase ignores `[lockon]` from `rules.toml`. Fix: extend
`PyGpuVecEnv::new` with `fov_degrees / lock_distance / lock_duration`
kwargs (defaults from TOML), pipe through `GpuBatchOrchestrator::new`,
push as a uniform alongside `dims` / `weights`, replace the WGSL literal
with a uniform read. Add `tests/test_gpu_lockon_plumbing.py` modeled on
`test_gpu_reward_plumbing.py`.

### 11.14 EKF / belief inside batch-sim — **MISSING**

`BattleState` has no EKF / particle filter. Obs slots `[19]`
(belief_uncertainty) and `[20]` (time_since_seen) are **always 0**
under batch-sim/GPU. Phase 3 (`search_pursuit`) curriculum needs these
for realistic training signal — currently the GPU path silently drops
this feature.

### 11.15 Motor / IMU noise on GPU — see [[#11.10]]

### 11.16 CPU vs GPU precision

GPU is f32, CPU is f64. Drift bounded by parity test
(`max_diff < 1e-3`). Quaternion norm renormalize after every RK4 step in
the WGSL helper.

### 11.17 Hard collision penalty divergence — **NEW ⚠⚠⚠ CRITICAL**

CPU MPPI (`crates/mppi/src/optimizer.rs:222-260`) tracks
`max_penetration` across **every RK4 sub-step** and adds `+1e8` to the
sample's cost if any sub-step penetrates an obstacle.

GPU rollout (`crates/batch-sim/src/gpu/shaders/mppi_rollout.wgsl:283-294`)
checks SDF **only at the end of each control tick** (after `substeps`
RK4 steps), and adds `+1e6` (six orders of magnitude smaller).

Two real consequences:
1. **Tunneling**: a trajectory that passes *through* a pillar between
   SDF samples (10 ms × motor commands ≈ 10 cm of traversal at 100 Hz)
   gets caught by CPU but **silently accepted** by GPU.
2. **Magnitude mismatch matters once CVaR-style risk MPPI lands on
   GPU** — the relative weighting of "almost-colliding" vs "actually
   colliding" trajectories diverges by 100x.

**Severity: Critical for self-play safety.** Fix: track
`max_penetration` inside the substep loop in `rollout_and_cost`, apply
`+1e8` once at the end. Bump SDF magic constant from `1e6` to `1e8`.

### 11.18 `[mppi.weights]` not plumbed to GPU — **NEW ⚠⚠⚠ CRITICAL**

`crates/batch-sim/src/gpu/orchestrator.rs:126-138` hardcodes
`CostWeightsGpu::new(1.0, 5.0, 0.01, 1000.0, 0.3, ...)`. Today these
match `[mppi.weights]` defaults so no observed numerical divergence —
**but** `aces/training/gpu_vec_env.py` plumbs `[reward]` and `[noise]`
only, and `PyGpuVecEnv::new` has **no kwarg** for cost weights. Any
tuning of `[mppi.weights]` (e.g. raising `w_face` to 10.0) is
**silently ignored on GPU**. Same pattern as Audits 6 + 7.

**Severity: Critical.** Fix identical to Audit 6: load
`[mppi.weights]` in `gpu_vec_env.py`, add kwargs to PyGpuVecEnv, plumb
through to `GpuBatchOrchestrator::new`. Add
`tests/test_gpu_cost_weights_plumbing.py`.

### 11.19 GPU MPPI substeps / dt_sim not wired — **NEW ⚠⚠ HIGH**

`GpuBatchMppi::new` hardcodes `MppiDims::new(..., **10**, ..., **0.001**)`
(`crates/batch-sim/src/gpu/pipeline.rs:521-528`). `update_dims` exists
(line 619) but is **never called** from `GpuBatchOrchestrator`.
`BatchConfig::substeps` and `BatchConfig::dt_ctrl` flow only to physics
step, not to MPPI rollout.

Today defaults align (both 10 substeps × 0.001 s). If a user sets
`substeps=20` for higher physics fidelity, **MPPI rollout silently keeps
10 substeps × 0.001 s** while physics runs 20 × 0.0005 s. MPPI plans
against subtly wrong dynamics.

**Severity: High** when anyone tunes `substeps`. Fix: call
`pipeline.update_dims(...)` in `GpuBatchOrchestrator::new` after
`GpuBatchMppi::new`.

### 11.20 MPPI per-sample parity gap — **NEW ⚠ HIGH**

`crates/batch-sim/tests/gpu_pipeline.rs::test_gpu_matches_cpu_reference_parity`
reads back **only the final reduced action**. Bindings 4 (`costs[d, s]`)
and 5 (`ctrls_out[d, s, t, c]`) are never read back or compared. Two
paths can diverge per-sample yet softmax-average to the same final
action (e.g. an off-by-one in noise indexing where two samples swap
perturbations).

**Severity: High** for any future shader edit. Fix: extend
`compute_batch_actions` (debug-only) to optionally return
`(costs_readback, ctrls_out_readback)`; assert per-sample parity in the
test.

### 11.21 RNG seeding incompatibility — **NEW ⚠ MED**

CPU MPPI seeds per-sample from `rand::thread_rng()` (non-deterministic).
GPU orchestrator holds one `noise_rng: SmallRng` seeded from a master
seed. The two paths can never produce bit-exact noise even with the
same seed:
- CPU draws per-sample (sample-major); GPU draws serially (drone-major).
- CPU re-runs are non-reproducible.

Implication: **you can never diff CPU MPPI vs GPU MPPI traces from a
real training run** — different RNG ↔ different noise ↔ different
trajectories. Fine for parity tests (which pre-generate noise and pass
the same buffer to both) but blocks any field debugging.

**Severity: Med.** Fix: make CPU MPPI instance-seeded.

### 11.22 RL training: callback opponent-init is dead — **NEW ⚠⚠⚠ CRITICAL**

`VecOpponentUpdateCallback._on_training_start`
(`aces/training/callbacks.py:73`) checks
`hasattr(env, "set_opponent_policy")` where `env` is the
`VecNormalize`-wrapped training env. **`VecNormalize` does NOT forward
arbitrary attributes** — it only forwards via `env_method`. So the
branch is silently dead, the opponent policy is **never initialized**,
and the first opponent inference call inside
`BatchedOpponentVecEnv._inject_opponent_actions` crashes on
`self._opponent_policy.predict(...)` with `None`.

**Severity: Critical** for the planned cloud run with
`opponent="policy"` and `n_envs > 1`. Fix: walk the wrapper chain to
find the `BatchedOpponentVecEnv`, or expose `set_opponent_policy` via
`VecNormalize.__getattr__`.

### 11.23 RL training: `should_promote("steps")` short-circuits phases — **NEW ⚠⚠⚠ CRITICAL**

`aces/curriculum.py:117-118` returns `True` for `condition == "steps"`.
`PromotionCheckCallback._on_step`
(`aces/training/curriculum_trainer.py:69-72`) treats that as
"promote now" and sets `model.stop_training = True`. Result: **every
phase with `promote_condition="steps"` (the default!) gets aborted at
~5 000 steps** instead of running to its `max_timesteps`.

The intended semantics: `"steps"` means "let `model.learn(timesteps)`
run to completion." Either `should_promote` should return `False` for
`"steps"`, or `PromotionCheckCallback` should skip itself when the
condition is `"steps"`.

**Severity: Critical.** This is the single highest-impact bug found.
Phase 5 (5 M steps) would terminate at 5 k.

### 11.24 RL training: per-task reward overrides dropped under `--use-gpu-env` — **CONFIRMED ⚠⚠⚠ HIGH**

(Was [[#11.9]] — now verified by review.) `aces/training/gpu_vec_env.py`
reads only base `[reward]`. CPU env (`aces/env/dogfight.py:74`) merges
`[task_reward_overrides.<task>]`. Hover and pursuit reward fixes from
the overnight session live in `task_reward_overrides` and are
**silently discarded** under `--use-gpu-env`. The hover phase's
perverse incentive (overnight bug #3) is back on GPU.

### 11.25 RL training: `info` dict field-name mismatch CPU vs GPU env — **NEW ⚠⚠ HIGH**

| CPU env emits        | GPU env emits         | Callback reads        | Result                     |
|----------------------|-----------------------|-----------------------|----------------------------|
| `collision`          | `collision_a` / `_b`  | `info.get("collision")` | always False on GPU       |
| `truncated`          | `timeout`             | `info.get("truncated")` | always False on GPU       |
| `lock_a_progress`    | `lock_a`              | `info.get("lock_a_progress")` | always 0 on GPU      |

Sites: `aces/env/dogfight.py:862, 886, 890` (CPU);
`crates/py-bridge/src/lib.rs:1022, 1027-1029` (GPU bridge);
`aces/training/callbacks.py:401`, etc. (consumers).

**Severity: High** for GPU runs. Reproduces the post-fix
crash-misclassification from the overnight session — every GPU episode
will show `crash=0`, `timeout=0`. Fix: in
`crates/py-bridge/src/lib.rs::pack_step_results`, also emit
`info["collision"] = collision_a`, `info["truncated"] = timeout`,
`info["lock_a_progress"] = lock_a`.

### 11.26 RL training: `n_envs=1` + `opponent="policy"` crashes — **NEW ⚠⚠ HIGH**

`CurriculumTrainer` skips `BatchedOpponentVecEnv` when `n_envs == 1`
(`curriculum_trainer.py:351-362`). Callbacks then call
`env.env_method("set_opponent_weights", ...)` which falls through to
`DroneDogfightEnv` — **which exposes `_update_opponent_weights`, not
`set_opponent_weights`**. Raises `AttributeError` mid-training.

**Severity: High** if anyone debugs locally with `--n-envs 1`. Fix:
route `n_envs == 1` through `BatchedOpponentVecEnv` too, or add a shim
on `DroneDogfightEnv`.

### 11.27 RL training: `VecNormalize` stats round-trip via pickle — **NEW ⚠ MED**

`curriculum_trainer.py:339, 365-375, 448-453` pickles
`{obs_rms, ret_rms}` and re-assigns onto a freshly-constructed
`VecNormalize`. Other config (`clip_obs`, `epsilon`, `gamma`) does not
round-trip. If a user mixes CPU phases (`norm_reward=True`) and GPU
phases (`norm_reward=False`), the carried-over `ret_rms` is
incompatible.

**Severity: Med.** Fix: use `VecNormalize.save()` /
`VecNormalize.load(path, venv)` (the official API).

### 11.28 RL training: promote-condition regex doesn't tolerate whitespace — **NEW ⚠ LOW**

`aces/curriculum.py:16` regex `^(\w+)>([\d.]+)$` rejects
`"win_rate > 0.30"`. Also `[\d.]+` accepts `"0..30"` and crashes on
`float()`. Fix: `^(\w+)\s*>\s*(\d+(?:\.\d+)?)$`.

### Summary table

> Updated 2026-04-25 after the post-architecture review pass. New rows
> 17-28 came from three reviewer agents.

| #  | Boundary                                          | Status            | Severity                              |
|----|---------------------------------------------------|-------------------|---------------------------------------|
| 1  | Action denormalization                            | ✓ audited         | High (12% thrust error)               |
| 2  | Reward formula                                    | ⚠ partial         | Med (terminal cases)                  |
| 3  | Obs slot [15] (SDF)                               | ✓ audited         | High (wall avoidance)                 |
| 4  | Bevy obstacle SDF                                 | ✓ audited         | Low (HUD only)                        |
| 5  | Spawn / max_steps                                 | ✓ audited         | High (curriculum mismatch)            |
| 6  | Reward config plumbing                            | ✓ audited         | High                                  |
| 7  | Wind config plumbing                              | ✓ audited         | Med                                   |
| 8  | Per-task reward overrides → GPU                   | ⚠ pending         | **High for GPU curriculum** (= #24)   |
| 9  | batch-sim noise modules                           | ⚠ pending         | High for phases 3-5                   |
| 10 | GPU rollout OU wind                               | ⚠ pending         | Med (CVaR collapses on GPU)           |
| 11 | Quaternion convention (orchestrator path safe)    | ✓ documented      | Critical if direct API misused        |
| 12 | Euler-angle convention                            | ✓ audited 2026-04-25 | Low (gimbal-lock branch differs)   |
| 13 | Lock-on params plumbing                           | ⚠ **bug found 2026-04-25** | **High** (= #11.13)          |
| 14 | EKF in batch-sim                                  | ⚠ missing         | High for Phase 3                      |
| 15 | f32 vs f64 drift                                  | ✓ tested          | Low (under tolerance)                 |
| 16 | f32 quaternion drift over long horizons (GPU)     | ⚠ undertested     | Low                                   |
| 17 | Hard collision penalty (CPU 1e8 every-substep / GPU 1e6 end-of-tick) | ⚠ **NEW** | **CRITICAL** (tunneling, CVaR magnitude) |
| 18 | `[mppi.weights]` not plumbed to GPU               | ⚠ **NEW**         | **CRITICAL** if anyone tunes weights  |
| 19 | GPU MPPI `substeps` / `dt_sim` not wired          | ⚠ **NEW**         | High if `substeps` ≠ default          |
| 20 | MPPI per-sample CPU↔GPU parity gap (test)         | ⚠ **NEW**         | High before next shader edit          |
| 21 | RNG seeding CPU↔GPU incompatibility               | ⚠ **NEW**         | Med (blocks field debugging)          |
| 22 | Callback opponent-init dead through `VecNormalize`| ⚠ **NEW**         | **CRITICAL** for `opponent="policy"`  |
| 23 | `should_promote("steps")` short-circuits phases   | ⚠ **NEW**         | **CRITICAL** — phases die at 5k steps |
| 24 | Per-task reward overrides dropped under GPU       | ⚠ **NEW** (= #8 confirmed) | **High** — overnight fixes lost |
| 25 | `info` dict field-name mismatch CPU vs GPU        | ⚠ **NEW**         | High (CSV always says crash=0)        |
| 26 | `n_envs=1` + `opponent="policy"` crashes          | ⚠ **NEW**         | High for local debug                  |
| 27 | `VecNormalize` stats round-trip via pickle        | ⚠ **NEW**         | Med (mixed CPU/GPU phases break)      |
| 28 | Promote-condition regex (whitespace)              | ⚠ **NEW**         | Low (cosmetic)                        |

### Recommended fix order before the cloud run

**Block the cloud run on these (CRITICAL)**:
1. **#23** — fix `should_promote("steps")` (1-line change in
   `aces/curriculum.py`).
2. **#22** — fix `VecOpponentUpdateCallback` to walk the wrapper chain
   (~5 lines in `aces/training/callbacks.py`).
3. **#17** — fix collision penalty in WGSL (in-loop tracking, bump
   constant to 1e8). Tens of lines in `mppi_rollout.wgsl`.
4. **#18** — plumb `[mppi.weights]` through `GpuVecEnv` →
   `PyGpuVecEnv` → orchestrator. Same pattern as Audits 6 + 7.

**Strongly recommended before GPU curriculum runs (HIGH)**:
5. **#24 / #8** — plumb per-task reward overrides through `GpuVecEnv`.
   Without this every GPU phase loses the overnight reward fixes.
6. **#13** — plumb `[lockon]` params through GpuVecEnv + WGSL. Same
   pattern as Audits 6 + 7.
7. **#25** — emit `collision`, `truncated`, `lock_a_progress` keys in
   the GPU bridge `pack_step_results`.
8. **#19** — wire `update_dims` in `GpuBatchOrchestrator::new`.

**Defense in depth (worth doing this session)**:
9. **#26** — add `set_opponent_weights` shim on `DroneDogfightEnv` (or
   route `n_envs=1` through `BatchedOpponentVecEnv`).
10. **#28** — tighten promote-condition regex.
11. **#20** — extend GPU parity test to compare per-sample `costs` and
    `ctrls_out`.

**Defer (acceptable for the next run)**:
12. **#9** — port batch-sim noise modules.
13. **#10** — GPU OU wind sampling.
14. **#11** — WGSL `quat_normalize` defensive call (today's caller path
    is safe).
15. **#12** — add Euler convention test.
16. **#21** — make CPU MPPI instance-seeded.
17. **#27** — switch to `VecNormalize.save/load`.

---

## 12. Test taxonomy

### 12.1 Categories

| Category               | Question it answers                                           |
|------------------------|---------------------------------------------------------------|
| `unit-physics`         | Does dynamics / SDF / quaternion behave correctly in isolation? |
| `unit-mppi`            | Do cost functions and the optimizer return sensible values?   |
| `unit-estimator`       | Does EKF / PF satisfy statistical invariants (NEES, whiteness)? |
| `parity-cpu-vs-batch`  | Does batch-sim match the CPU env's outputs?                    |
| `parity-cpu-vs-gpu`    | Does GPU MPPI match the CPU reference within tolerance?        |
| `config-plumbing`      | Does every TOML knob actually reach the consumer?              |
| `env-correctness`      | Does the Gymnasium env return the right obs / reward / done?   |
| `training`             | Do trainer / opponent pool / callback machinery behave?        |
| `viz`                  | Does Rerun visualization construct without errors?             |
| `e2e-smoke`            | Does PPO actually .learn() against the env?                    |

### 12.2 Rust suites (per crate, `cargo test --workspace`)

| Crate      | #tests | Notable files                                         |
|------------|-------:|-------------------------------------------------------|
| sim-core   | ~57 | `dynamics.rs (7)`, `safety.rs (8)`, `detection.rs (6)`, `camera.rs (5)`, `actuator.rs (5)`, `wind.rs (3)`, `lockon.rs (3)`, etc. |
| mppi       |  5  | `optimizer.rs`                                        |
| estimator  |  8  | `ekf.rs (5)`, `particle_filter.rs (3)`               |
| batch-sim  | ~54 | `f32_*.rs (17)`, `battle.rs (7)`, `observation.rs (3)`, `reward.rs (3)`, `orchestrator.rs (4)`, `tests/gpu_pipeline.rs (14, --features gpu)`, `gpu/{shader, orchestrator, pipeline}.rs (23, --features gpu)` |
| game       |  3  | `fsm.rs`                                              |
| py-bridge  |  0  | (covered indirectly through Python tests)             |

### 12.3 Python suites (`pytest tests/ -v`)

| File                                         | Category               | # |
|----------------------------------------------|------------------------|--:|
| `test_dynamics.py`                           | unit-physics           | 19 |
| `test_ekf_statistical.py`                    | unit-estimator         | 6 classes (~30) |
| `test_env.py`                                | env-correctness        | 22 |
| `test_config.py`                             | config-plumbing        | 9 |
| `test_obs_layout.py`                         | env-correctness        | 13 |
| `test_observation_consistency.py`            | parity-cpu-vs-batch    | 6 |
| `test_reward_consistency.py`                 | env-correctness        | 11 |
| `test_action_normalization_consistency.py`   | env-correctness        | 5 |
| `test_denormalize_action.py`                 | env-correctness        | 9 |
| `test_fsm.py`                                | env-correctness        | 12 |
| `test_trajectory.py`                         | unit-physics           | 8 |
| `test_trainer.py`                            | training               | 12 |
| `test_opponent_pool.py`                      | training               | 6 |
| `test_god_oracle.py`                         | training               | 11 |
| `test_perception.py`                         | unit-estimator         | 6 |
| `test_curriculum.py`                         | training               | 7 |
| `test_curriculum_gpu_opt.py` *(gpu)*         | parity-cpu-vs-gpu      | 3 |
| `test_gpu_vec_env.py` *(gpu)*                | parity-cpu-vs-gpu      | 5 |
| `test_gpu_reward_plumbing.py` *(gpu)*        | parity-cpu-vs-gpu      | 4 |
| `test_gpu_wind_plumbing.py` *(gpu)*          | parity-cpu-vs-gpu      | 4 |
| `test_gpu_ppo_smoke.py` *(gpu)*              | e2e-smoke              | 2 |
| `test_viz.py`                                | viz                    | 2 |
| `test_neurosymbolic_integration.py`          | other                  | 2 |
| `test_training_bench.py`                     | other (benchmark)      | 3 |

`tests/conftest.py` provides session-scope fixtures `core_available`
(does `aces._core` import?) and `gpu_available` (does `GpuVecEnv(n_envs=1,...)`
construct?). All GPU tests skip gracefully when the feature is not built.

### 12.4 Coverage gaps to close

> These are listed in priority order. None block the next cloud run on
> their own, but each is a real exposure.

1. **EKF / PF inside the full sim loop** — `test_ekf_statistical.py`
   tests filters in isolation. Add an integration test that asserts
   filter convergence as `Simulation.step` runs across an episode.
2. **Per-sample MPPI parity** (CPU vs GPU) — current parity test
   compares only the final reduced action. Add a test that compares
   `costs[d, s]` and `ctrls_out[d, s, t, c]` across the two paths to
   catch divergent rollouts that happen to softmax-average alike.
3. **Lock-on + targeting parity** across CPU / batch-sim / GPU / Bevy —
   no test today.
4. **Config → Simulation end-to-end** — `test_config.py` validates TOML
   parsing and `test_env.py` validates env semantics, but no test loads
   a real config and asserts that `Simulation.drone_a_state` reflects
   `cfg.drone.mass`, etc.
5. **EKF + PPO** — does PPO converge with `obs_noise_std > 0`? The smoke
   test runs noise-free.
6. **Euler-angle convention** ([[#11.12]]) — single parametric test.
7. **Lock-on param plumbing** ([[#11.13]]) — analogous to Audits 6/7.

---

## 13. GPU + cloud pipeline

### 13.1 GPU MPPI — runtime stack

```
SB3 PPO
   │  Python
   ▼
aces.training.GpuVecEnv                 (SB3 VecEnv — gpu_vec_env.py)
   │
   ▼
aces._core.GpuVecEnv                    (PyO3, py-bridge/src/lib.rs)
   │
   ▼
GpuBatchOrchestrator                    (batch-sim/src/gpu/orchestrator.rs)
   │  buffer uploads + 2 dispatches per side + physics step
   ▼
GpuBatchMppi pipeline                   (batch-sim/src/gpu/pipeline.rs)
   │  wgpu 23 / naga 23
   ▼
WGSL kernels                            (batch-sim/src/gpu/shaders/*.wgsl)
   │
   ▼
GPU (Metal / Vulkan / DX12)
```

### 13.2 Hardware support

| Platform       | Backend         |
|----------------|-----------------|
| macOS Apple Si | Metal           |
| Linux NVIDIA   | Vulkan          |
| Linux AMD      | Vulkan          |
| Windows        | DX12 or Vulkan  |

Probe: `cargo run -p aces-batch-sim --features gpu --example gpu_probe`.

### 13.3 Docker — the 3-image strategy

```
Dockerfile.dev-base       (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)
   │  Rust + Node 25 + Vulkan + Poetry + Maturin + Claude Code + tmux/zsh
   │
   ├──► Dockerfile.aces        thin dev image
   │       ENV ACES_PROJECT_DIR=/workspace/aces
   │           ACES_BOOTSTRAP_PROJECT=1
   │           POETRY_VIRTUALENVS_IN_PROJECT=true
   │       no code baked in — clone-on-boot via runpod-start.sh
   │
   └──► Dockerfile.train       multi-stage CPU training image
           Stage 1: build aces wheel (release, --interpreter python3.12)
           Stage 2: ubuntu:24.04 + Python 3.12 + PyTorch CPU + ML stack
           ENTRYPOINT python scripts/run.py
```

When to use which:
- **dev-base** — internal layer, rarely pushed alone.
- **aces** — primary remote dev box (Runpod). `ssh in → claude → work
  like local`.
- **train** — headless batch jobs. Wheel is baked, no Poetry overhead.

### 13.4 Runpod boot flow (`docker/runpod-start.sh`)

```
ENTRYPOINT runpod-start.sh
   │
   ▼
ensure_persistent_home          symlink ~/.claude, ~/.config, ~/.ssh,
                                 ~/.gitconfig, ~/.git-credentials,
                                 ~/.bash_history, ~/.zsh_history
                                 → /workspace/.dev-home
   │
   ▼
clone_dotfiles_if_requested     ACES_DOTFILES_REPO → ACES_DOTFILES_DIR
   │
   ▼
apply_dotfiles_if_present       apply-dotfiles.sh merges configs + .config + .local/bin
   │
   ▼
bootstrap_project_from_git      clone ACES_REPO_URL → ACES_PROJECT_DIR
       OR
   bootstrap_project_from_image  cp -a IMAGE_SOURCE_DIR → ACES_PROJECT_DIR
   │
   ▼
checkout_requested_ref          ACES_REPO_REF
   │
   ▼
sync_project_repo               git pull --ff-only (if AUTO_PULL=1, clean tree)
   │
   ▼
bootstrap_project_env           poetry install --with dev
                                 + maturin develop --release --features gpu
                                 + check_gpu_setup.sh (if RUN_GPU_CHECK=1)
                                 marker tied to git commit → re-runs after pull
   │
   ▼
exec "$@"  /  /start.sh  /  sleep infinity
```

### 13.5 `scripts/check_gpu_setup.sh` — 5 stages

| Stage | Check | PASS/FAIL/SKIP |
|------:|-------|----------------|
| 1     | `cargo`, `rustc` present                       | hard FAIL on missing |
| 2     | `cargo run --example gpu_probe --features gpu` | SKIP if no adapter, FAIL on build |
| 3     | `cargo test -p aces-batch-sim --features gpu`  | FAIL on test failure |
| 4     | `python -c "import aces._core"`                | SKIP if no poetry, FAIL on missing extension |
| 5     | `pytest test_denormalize_action test_gpu_vec_env test_gpu_ppo_smoke` | FAIL on test failure |

Exits 0 if all PASS or SKIP, 1 on any FAIL.

### 13.6 Cloud gotchas

- **linux/amd64 platform pin** — Runpod GPU pods are amd64 only.
- **Vulkan packages** (`libvulkan1`, `vulkan-tools`) — required for wgpu
  backend on Linux NVIDIA; baked into `Dockerfile.dev-base`.
- **Poetry venv-in-project** — `POETRY_VIRTUALENVS_IN_PROJECT=true`
  prevents Poetry from modifying the Runpod base image's system Python.
  If bootstrap fails uninstalling `pyparsing` etc., you forgot this.
- **GPU feature must be passed to maturin** — `maturin develop --features gpu`,
  otherwise `aces._core.GpuVecEnv` won't exist.
- **Stale bootstrap marker** — set `ACES_FORCE_BOOTSTRAP=1` to force
  re-build after a manual code change.
- **Persistent state** — `/workspace/.dev-home` survives Pod restarts;
  Claude Code login + git config + SSH keys persist.

---

## 14. Roadmap

```
Level 0 → Level 1 → Level 2 → Level 3 → Level 4 → Level 5 → Level 6 → Level 7
Core sim  RL policy  Uncertainty Info-asym  FPV vis  Curriculum Bevy viz Hardware
   ✓         ✓          ✓          ✓          ✓          ✓         ✓     planned

GPU MPPI:        Phase 1 ✓ (CPU Rayon)
                 Phase 2 ✓ (WGSL compute)
                 Phase 3 ✓ (PyO3 + SB3)
                 Phase 4 ✗ (full physics on GPU)
```

**Next slices** (priority order, from `2026-04-24-session-archive.md`
Part 6):
1. Per-task reward overrides → GpuVecEnv ([[#11.9]]).
2. Terminal priority + truncation reward semantics ([[#11.2]]).
3. Port batch-sim noise modules (`obs_noise`, `actuator`, `imu_bias`)
   ([[#11.10]]).
4. GPU RNG for OU wind in rollout ([[#11.6]]).
5. Audit Euler convention ([[#11.12]]) and lock-on plumbing
   ([[#11.13]]).

---

## 15. Pre-cloud-run checklist

Before kicking off a long cloud training run, **do these in order**:

### 15.1 Hard blockers — fix first (CRITICAL bugs from §11)

- [ ] **#23** Fix `should_promote("steps")` short-circuit
      (`aces/curriculum.py`).
- [ ] **#22** Fix `VecOpponentUpdateCallback` wrapper-chain walk
      (`aces/training/callbacks.py`).
- [ ] **#17** Fix GPU collision penalty mid-rollout tracking + magnitude
      (`crates/batch-sim/src/gpu/shaders/mppi_rollout.wgsl`).
- [ ] **#18** Plumb `[mppi.weights]` through to GPU orchestrator (same
      pattern as Audits 6 + 7).

### 15.2 Strongly recommended (HIGH bugs from §11)

- [ ] **#24** Plumb per-task reward overrides through `GpuVecEnv`.
- [ ] **#13** Plumb `[lockon]` params through `GpuVecEnv` + replace WGSL
      literal with uniform read.
- [ ] **#25** Emit `collision`, `truncated`, `lock_a_progress` in GPU
      bridge `pack_step_results`.
- [ ] **#19** Call `pipeline.update_dims(...)` in
      `GpuBatchOrchestrator::new`.
- [ ] **#26** Either route `n_envs=1` through `BatchedOpponentVecEnv` or
      add a shim on `DroneDogfightEnv`.

### 15.3 Verification — run AFTER fixes

- [ ] `cargo test --workspace` — Rust tests pass.
- [ ] `cargo test -p aces-batch-sim --features gpu` — GPU tests pass on
      the target hardware.
- [ ] `poetry run maturin develop --release --features gpu` — extension
      built with GPU.
- [ ] `poetry run pytest tests/ -v` — Python tests pass (GPU tests run,
      not skip).
- [ ] `bash scripts/check_gpu_setup.sh` — all 5 stages PASS or SKIP.
- [ ] `poetry run python scripts/smoke_gpu_ppo.py --timesteps 256 --n-envs 4`
      — PPO actually learns 256 steps without NaN.
- [ ] `poetry run python scripts/bench_training_throughput.py` — measure
      real-world env-steps/s on the target GPU.
- [ ] **Add new tests** that pin the four CRITICAL fixes — they should
      all fail before the fix and pass after.
- [ ] **Re-read [[#11. Fragile boundaries — the correctness map]]** —
      every ⚠ entry is a known silent-bug class.

### 15.4 Sanity gates before the launch command

- [ ] **Cross-check the curriculum config** — for each phase, are the
      effective opponent / noise / reward / lockon what you expect?
      Note: under `--use-gpu-env`, non-MPPI phases fall back to
      MPPI-vs-MPPI (Caveat §1 in `gpu-mppi.md`).
- [ ] **Decide CPU vs GPU env** — until #24 is fixed, the CPU env
      (`SubprocVecEnv` + `n_envs=8+`) gives correct curriculum
      semantics; the GPU env trades correctness for throughput.
- [ ] **Pin the config snapshot** — copy `configs/*.toml` into the run
      output dir so you know what you actually trained against.
- [ ] **Commit + tag** the config snapshot + git SHA you launched from.

---

## 16. Using this doc as an Obsidian vault

### 16.1 Why Obsidian helps here

The natural unit of this codebase is the **subsystem**, and the
interesting questions are usually "what crosses this boundary?". A graph
view makes those edges visible.

Obsidian is *better than a flat doc* for three things:
1. **Bidirectional navigation**: see who else mentions
   `BatchOrchestrator` without grepping.
2. **Incremental refactor planning**: when you decide to redesign a
   layer, the graph shows everything that needs to move with it.
3. **Mental warm-up**: "remind me what this project does" → open vault,
   look at graph, click into the subsystem you forgot.

But Obsidian is *worse than the current doc* for "give me a 90-second
top-down overview", because the graph view is naturally non-linear. So:
**use both — this doc is the reading-order spine; Obsidian gives the
navigation graph**.

### 16.2 Suggested vault layout

Initialize an Obsidian vault rooted at `docs/`. Each subsystem becomes
its own note. Wikilinks below already use that convention.

```
docs/                                        ← Obsidian vault root
├── architecture.md                          ← (this) the spine
├── design.md                                ← long-form reference (linked from spine)
├── gpu-mppi.md                              ← user guide
├── runpod.md                                ← cloud setup
├── 2026-04-24-session-archive.md            ← archives
├── overnight-report.md
├── parallel_architecture.md
└── nodes/                                   ← optional: one note per subsystem
    ├── L1-sim-core.md
    ├── L2-mppi.md
    ├── L3-estimator.md
    ├── L4-batch-sim.md
    ├── L4-gpu-pipeline.md
    ├── L5-py-bridge.md
    ├── L6-env.md
    ├── L7-training.md
    ├── L8-policy.md
    ├── Bevy-game.md
    ├── L0-configs.md
    ├── obs-layout-21dim.md
    ├── reward-formula.md
    ├── lockon-rule.md
    ├── ekf-vs-pf.md
    ├── audit-1-action-denorm.md
    ├── audit-2-reward.md
    ├── …
    ├── audit-7-wind-plumbing.md
    ├── pending-1-task-reward-overrides.md
    ├── pending-2-batch-sim-noise.md
    ├── pending-3-gpu-ou-wind.md
    ├── pending-4-euler-convention.md
    └── pending-5-lockon-plumbing.md
```

Each `nodes/*.md` should be **one screen** of: purpose, key files, what
crosses its boundary, links to related nodes. Treat the existing audit
sections in `2026-04-24-session-archive.md` as templates.

### 16.3 Recommended Obsidian setup

- **Core plugins**: Graph view, Backlinks, Outgoing links, Tags pane.
- **Community plugins**: Excalidraw (sketch dataflows next to nodes),
  Dataview (auto-list every "pending" audit by tag).
- **Tags to use** consistently:
  - `#layer/L1` … `#layer/L8` — assign each node to its layer.
  - `#audit/done` / `#audit/pending` — audit status.
  - `#config-knob` — anything that flows from TOML.
  - `#correctness-risk` — fragile boundary.
  - `#gpu` — GPU-specific.
- **Filters**: graph-view filter `#correctness-risk` to see only the
  fragile boundaries; filter `#layer/L4` to see batch-sim's graph
  neighborhood.
- **Local graph**: open any node, ⌘P → "Local graph view" → see all
  direct neighbors. Best one-click "what touches this?" view.

### 16.4 Conversion script (optional)

If you actually want the `nodes/` skeleton, the wikilinks in §3 already
name them. A 30-line Python script could extract every `[[#…]]` heading
and stub out a file per node — happy to write it on request.

### 16.5 What stays in this doc vs what moves to nodes

| In this doc (`architecture.md`)              | In a `nodes/*.md`                   |
|----------------------------------------------|-------------------------------------|
| Top-down map, layer overview, data flows     | Subsystem deep dive                 |
| Math framework summary                       | Per-formula derivations             |
| Config map, fragile boundaries summary table | One node per audit / pending issue  |
| Pre-cloud-run checklist                      | (n/a)                               |

Rule of thumb: if the answer to "where do I look this up?" is "everyone
needs this", keep it in this doc. If it's "only when I'm working on X",
it belongs in a node.

---

## 17. References

- [`design.md`](design.md) — long-form technical reference (sections 1–14).
- [`gpu-mppi.md`](gpu-mppi.md) — GPU MPPI user guide.
- [`runpod.md`](runpod.md) — Runpod / Docker cloud setup.
- [`2026-04-24-session-archive.md`](2026-04-24-session-archive.md) — GPU
  MPPI architecture + 7 consistency audits + remaining divergences.
- [`overnight-report.md`](overnight-report.md) — 2026-04-23 training-bug
  session (hover reward, pursuit reward, callback dedup).
- [`parallel_architecture.md`](parallel_architecture.md) — early
  parallel-sim design (preserved for context).
- [`plans/parallel-simulation.md`](../plans/parallel-simulation.md) —
  Phase 1–4 GPU plan + slice index.
- [`plans/curriculum-architecture.md`](../plans/curriculum-architecture.md)
  — curriculum + crate dependency reference.
- [`CLAUDE.md`](../CLAUDE.md) — dev conventions.
