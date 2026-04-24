# ACES Session Archive — 2026-04-23/24

Snapshot of a concentrated ~24-hour work session that landed GPU MPPI end-to-end
plus seven CPU-vs-batch consistency audits. Written so a future contributor
(human or AI) can reconstruct the state of the codebase, understand why the
individual pieces are shaped the way they are, and continue the remaining
follow-up work without re-discovering the context.

## Overview

- **Duration**: ~24 hours (2026-04-23 afternoon → 2026-04-24 afternoon).
- **Commits**: 94 total on `main`, 35 of which are merges from feature
  branches. See Part 5 for the ordered index.
- **Workstreams**:
  1. **GPU MPPI pipeline** (Phase 2 + Phase 3 of
     [`plans/parallel-simulation.md`](../plans/parallel-simulation.md)): full
     WGSL compute-shader MPPI + orchestrator + PyO3 bindings + SB3 VecEnv +
     curriculum integration.
  2. **CPU-vs-batch consistency audits**: seven targeted audits, each of
     which compared `aces/env/dogfight.py` (CPU env), `crates/batch-sim/src/`
     (batch + GPU), `crates/py-bridge/src/lib.rs::Simulation` (single-drone
     PyO3), and `crates/game/src/simulation.rs` (Bevy). Every audit found at
     least one real divergence; each became a `feature/*-consistency` or
     `feature/*-audit` / `*-plumb` branch.
- **Tests**: ~149 Rust `#[test]` functions (~73 default pass cleanly via
  `cargo test --workspace`; additional tests gated on `--features gpu`) +
  25 Python test modules with ~187 test functions, all passing or skipping
  cleanly when the GPU feature is absent.

---

## Part 1 — GPU MPPI Architecture (Phase 2 + Phase 3)

The GPU MPPI subsystem replaces the Rayon-parallel CPU MPPI path for the
compute-heavy curriculum phases. It computes `N_DRONES * N_SAMPLES` rollouts
in a single GPU dispatch, then a second dispatch performs the softmax
weighted-average reduction per drone. For RL training, a Python SB3 VecEnv
wrapper drives the whole `2 * N_BATTLES` population from a single PPO policy.

### 1.1 Crate / file layout

```
crates/batch-sim/src/
├── battle.rs              BattleState, BatchConfig, BattleInfo, StepResult, SpawnMode
├── observation.rs         21-dim observation (unchanged from Phase 1)
├── reward.rs              RewardConfig + shaped reward (batch-sim canonical formula)
├── orchestrator.rs        CPU Rayon BatchOrchestrator (Phase 1)
├── f32_dynamics.rs        f32 CPU reference for RK4 + motor mixing (GPU parity baseline)
├── f32_sdf.rs             f32 CPU reference for Arena / Obstacle SDF
├── f32_cost.rs            f32 pursuit / evasion cost functions
└── gpu/
    ├── mod.rs             Module entry + adapter probe
    ├── adapter.rs         wgpu adapter probe + compute-shader round trip
    ├── shader.rs          WGSL concat + naga structural validation
    ├── pipeline.rs        GpuBatchMppi — standalone MPPI planner (12 bindings, 2 pipelines)
    ├── orchestrator.rs    GpuBatchOrchestrator — full episodic sim loop on GPU MPPI
    └── shaders/
        ├── mppi_helpers.wgsl   f32 RK4 + SDF + quaternion helpers
        ├── mppi_rollout.wgsl   rollout_and_cost @compute kernel
        └── mppi_softmax.wgsl   softmax_reduce @compute kernel
```

PyO3 bindings in `crates/py-bridge/src/lib.rs`:

- `PyGpuBatchMppi`  — wraps `GpuBatchMppi` (`compute_batch_actions`)
- `PyGpuVecEnv`     — wraps `GpuBatchOrchestrator` (`step`, `reset`,
                      `step_with_agent_a_actions`)

Python layer:

- `aces/training/gpu_vec_env.py::GpuVecEnv` — SB3-compatible VecEnv wrapper.
- `aces/env/obs_layout.py` — observation-index constants + `describe_obs`
  helper; used by consistency tests.

### 1.2 Data flow (SB3 → GPU → SB3)

```
Python                                   Rust (GpuBatchOrchestrator)             GPU
------                                   ---------------------------             ---
PPO.learn() loop
  |
  | actions in [-1, 1], shape (N, 4)
  v
GpuVecEnv.step(actions)
  |  denormalize_action(a)               step_with_agent_a_actions(motors_a)
  |    motors = hover + a * (max - hover)   |
  |                                         |  pack_and_dispatch_gpu_mppi(state, enemy, noise, wind)
  |                                         |    build MppiDims + wind uniforms
  |                                         |    upload states, enemies, mean_ctrls, noise, wind_per_drone
  |                                         |    dispatch rollout_and_cost(N_DRONES * N_SAMPLES)
  |                                         |                                   -> per-sample cost + ctrls
  |                                         |    dispatch softmax_reduce(N_DRONES) @wg 256
  |                                         |                                   -> per-drone H*4 result
  |                                         |    readback via staging buffer
  |                                         |
  |                                         | for each battle (Rayon):
  |                                         |   apply motors_a (PPO) + motors_b[0] (MPPI) via step_physics
  |                                         |   lock-on update, LOS, collision
  |                                         |   observation + reward build
  |                                         |
  |                                         | warm-start shift: mean_ctrls[:, 1:] = result[:, 1:]; mean_ctrls[:, H-1] = hover
  |                                         |
  |<--- obs (N, 21), reward (N,), done (N,), info (dict)
  v
PPO rollout buffer
```

Ownership invariants:

- GPU state (device / queue / pipelines / persistent buffers) lives inside
  `GpuBatchMppi`; `GpuBatchOrchestrator` owns it.
- Each `step` uploads only the *changing* buffers (states, enemies, noise,
  wind, mean_ctrls); the obstacle buffer + drone params + arena bounds are
  static uniforms written at construction.
- Per-drone wind vectors come from `BattleState::wind_{a,b}.force` — the
  rollout uses the *current* OU realization, held constant over the
  horizon.

### 1.3 WGSL compute kernels

`mppi_rollout.wgsl` — `rollout_and_cost`
  - `@compute @workgroup_size(1)` — one workgroup per (drone, sample) pair.
  - Dispatch: `[N_DRONES * N_SAMPLES, 1, 1]`.
  - Per-thread: unpack state + enemy, for each `t in [0, H)`:
    - perturbed control: `u = clamp(mean_ctrls[t] + noise[t], 0, max_thrust)`
    - rollout: `substeps` × `step_rk4_f32` with constant per-drone wind
    - accumulate `cost_t = pursuit_cost_f32(state, enemy, u, weights, arena)`
      (or evasion, by drone parity)
  - Writes: `costs[d, s]`, `ctrls_out[d, s, t, c]`.

`mppi_softmax.wgsl` — `softmax_reduce`
  - `@compute @workgroup_size(256)` — one workgroup per drone, 256 threads
    collaborating via shared-memory reduction.
  - Dispatch: `[N_DRONES, 1, 1]`.
  - Three passes: find `min_cost`, compute `sum_w = Σ exp(-(c-min)/T)`,
    compute per-time-step weighted mean of controls.
  - Writes `result[d, t, c]`.

Constants live in the `MppiDims` uniform (`n_drones`, `n_samples`, `horizon`,
`substeps`, `n_obstacles`, `dt_sim`, `temperature`), so shader size does not
depend on config.

### 1.4 Bind group layout (12 bindings)

Shared by both kernels.

| Binding | Kind    | Shape                                 | Access    | Contents                          |
|--------:|---------|---------------------------------------|-----------|-----------------------------------|
| 0       | storage | `states[N_DRONES × 13]`               | read      | per-drone initial state (pos,vel,quat_xyzw,angvel) |
| 1       | storage | `enemies[N_DRONES × 13]`              | read      | paired opponent state per drone   |
| 2       | storage | `mean_ctrls[N_DRONES × H × 4]`        | read      | warm-start nominal plan           |
| 3       | storage | `noise[N_DRONES × N × H × 4]`         | read      | CPU-generated Gaussian perturbations |
| 4       | storage | `costs[N_DRONES × N]`                 | read_write| per-sample accumulated cost       |
| 5       | storage | `ctrls_out[N_DRONES × N × H × 4]`     | read_write| perturbed-clamped controls        |
| 6       | storage | `result[N_DRONES × H × 4]`            | read_write| softmax-weighted optimal plan     |
| 7       | uniform | `DroneParamsGpu` (48 B)               | read      | mass, arm, max_thrust, inertia    |
| 8       | uniform | `CostWeightsGpu + arena_bounds` (48 B)| read      | w_dist, w_face, w_ctrl, w_obs, arena |
| 9       | storage | `obstacles[MAX_OBSTACLES (32)]`       | read      | kind tag + center + extents       |
| 10      | uniform | `MppiDims` (32 B)                     | read      | dispatch dims + dt + temperature  |
| 11      | storage | `wind_per_drone[N_DRONES × 4]`        | read      | current OU wind vector per drone  |

Binding 11 was added by `feature/gpu-wind` after the initial 11-binding
design. Several comments still said "11 bindings" and were fixed in a
follow-up (`cf8034c`). Future additions (e.g. per-task reward overrides)
should be aware that WGSL std140 layout requires explicit padding and 16-B
alignment on `vec3<f32>` members.

### 1.5 Validation path

Three independent levels of validation ship with the pipeline:

1. **Static shader validation** (`crates/batch-sim/src/gpu/shader.rs`): the
   three WGSL files are concatenated and fed to `naga` at crate-build time
   for structural + type checks. Runs without a GPU adapter.
2. **GPU ↔ CPU parity** (`crates/batch-sim/tests/gpu_pipeline.rs`): the
   `compute_batch_actions_cpu_reference` function replicates the shader
   math in plain Rust (intentionally unoptimized) and asserts
   `max_diff < 1e-3` on a full end-to-end run. Gated on `cargo test
   --features gpu` — skipped silently when no adapter is present.
3. **End-to-end PPO smoke** (`tests/test_gpu_ppo_smoke.py`): a 256-step PPO
   `.learn()` against the GPU VecEnv to catch interface / dtype / NaN
   regressions. Auto-skips when `aces._core` has no `gpu` feature compiled
   in (the Python extension must be built with
   `maturin develop --features gpu`).

All Python GPU tests follow the same skip-gracefully pattern, so CI
machines without GPUs do not fail.

---

## Part 2 — Consistency Audits (7 audits, 7+ bugs found)

Each audit compared a specific piece of logic across up to four code paths
and landed a fix plus regression tests. The consistent pattern: **dispatch a
fresh worker to compare files line-by-line, let it report divergences,
dispatch a second fresh reviewer to verify + run tests, then merge.**

### Audit 1: Action denormalization (`feature/action-consistency`, merge `41aeb05`)

**Scope**: how `actions ∈ [-1, 1]` become motor thrusts.

**Divergence**:
- CPU env (`aces/env/dogfight.py`): `motor = hover + action * (max_thrust - hover)`  → hover-centered; `a=0 ⇒ motor = hover_thrust ≈ 0.0662 N`.
- Old GPU wrapper: `motor = (action + 1) / 2 * max_thrust`  → symmetric-scaling; `a=0 ⇒ motor = 0.075 N`.

At `a=0` the two paths produced a ~12 % thrust mismatch, which at 100 Hz
over a short episode is enough to drift pursuit trajectories.

**Fix**: GPU path switched to hover-centered convention
(`aces/training/gpu_vec_env.py::denormalize_action`). Added:
- `tests/test_denormalize_action.py` — pure-function unit tests.
- `tests/test_action_normalization_consistency.py` — round-trip equality
  test vs CPU env.

### Audit 2: Reward formula (`feature/reward-consistency`, merge `ee83fd1`)

**Scope**: the shaped-reward formula applied at each step.

**Divergence**:
- CPU env `reward` clamped lock-progress delta to `[0, 1]` without also
  clamping against 0 (could emit negative deltas on timer reset).
- CPU env had no fallback when `opponent_crash` fired but `collision_a` was
  already set for the current drone — it double-counted.

**Fix**: `aces/env/dogfight.py` aligned to the `crates/batch-sim/src/reward.rs`
canonical formula: clamp `lock_delta = max(lock_new - lock_prev, 0)` and
use `opponent_crash_reward` only when `collision_a` is false. Added 364-line
`tests/test_reward_consistency.py` covering every terminal + shaping
combination.

**Remaining divergence** (documented in Part 3): terminal priority when
`kill_a` *and* `collision_a` both fire the same step; and truncation
reward semantics.

### Audit 3: Observation slot [15] (`feature/obs-consistency`, merge `057e5d4`)

**Scope**: obs[15] = nearest-obstacle SDF.

**Divergence**:
- CPU env: `obs[15] = obstacle_sdf(pos)` (obstacles only, ignores boundary).
- batch-sim: `obs[15] = arena.sdf(pos) = min(boundary_sdf, obstacle_sdf)`.

Near the arena wall, CPU env reported "no obstacle near" while batch-sim
said "wall 10 cm away" — a meaningful discrepancy for wall-avoidance
learning.

**Fix**: CPU env now uses combined `arena.sdf()` (via a small PyO3 addition
in `crates/py-bridge/src/lib.rs`). 380-line
`tests/test_observation_consistency.py` diffs every slot of a 21-dim
observation across CPU / batch-sim / GPU.

### Audit 4: Test precondition + Bevy `obstacle_sdf` (`feature/fix-test-and-game`, merge `b1f403a`)

**Scope**: the observation-consistency test itself + Bevy game rendering.

**Findings**:
1. The new `test_observation_consistency.py` was passing noise-dependent
   observations to a deterministic comparator; fixed by pinning
   `obs_noise_std = 0` and gating comparisons on LOS visibility.
2. The Bevy game (`crates/game/src/simulation.rs`) called its own
   single-obstacle distance helper, not the combined `arena.sdf()`. Fixed
   so the Bevy HUD "wall warning" matches training-time observation.

### Audit 5: Spawn initialization (`feature/spawn-audit`, merge `6141a8a`)

**Scope**: initial drone state at `reset()`.

**Divergence**:
- CPU env reset: `DroneState::hover_at(spawn_a)` — identity quaternion,
  zero velocity, fixed spawn point.
- batch-sim reset: identity quaternion + zero velocity, but randomised XY
  jitter over the arena.

Training curricula built around deterministic Phase-0 hover are
incompatible with random spawns. Also: `BattleState::step_physics` had a
hardcoded `max_steps = 1000` ignoring `BatchConfig::max_steps`.

**Fix**: introduced `SpawnMode::{FixedFromArena, FixedWithJitter, Random}`.
Default for new code is `FixedWithJitter` with small σ to match the CPU
env's light per-reset variation, while `Random` is available for domain
randomization. `step_physics` now honours `BatchConfig::max_steps`.

### Audit 6: Reward config plumbing (`feature/rules-toml-plumb`, merge `4c6678c`)

**Scope**: does `configs/rules.toml [reward]` actually reach the GPU env?

**Divergence**: `GpuVecEnv` was constructing its orchestrator with
`RewardConfig::default()` — i.e. the Rust `Default` impl, not the TOML
file. Any reward tuning done in `rules.toml` was silently ignored during
GPU training.

**Fix**:
- `aces/training/gpu_vec_env.py` now loads `configs/rules.toml`, builds
  a `reward_config: dict`, and passes it to `PyGpuVecEnv`.
- `PyGpuVecEnv` accepts a `reward_config` kwarg and translates it into a
  `RewardConfig` before constructing the orchestrator.
- `tests/test_gpu_reward_plumbing.py` pins that each scalar in
  `[reward]` propagates correctly.

### Audit 7: Wind config plumbing (`feature/wind-plumb`, merge `9a6b716`)

**Scope**: do `[noise] wind_sigma` / `wind_theta` reach the GPU wind
uniforms?

**Divergence**:
- `wind_sigma` was a constructor arg to `GpuVecEnv` but defaulted to a
  hardcoded literal, not the config value.
- `wind_theta` was hardcoded to `2.0` inside `PyGpuVecEnv::new` and never
  exposed at all.

**Fix**: `GpuVecEnv` reads `rules.toml [noise]` and passes both scalars
through. `PyGpuVecEnv` accepts them as kwargs and forwards to
`BatchConfig`. `tests/test_gpu_wind_plumbing.py` asserts round-trip
parity.

---

## Part 3 — Known Remaining Divergences

Not yet fixed. Each has a pointer to the relevant code and an expected
fix pattern.

1. **Terminal priority ordering**
   - Site: `aces/env/dogfight.py` reward branch vs
     `crates/batch-sim/src/reward.rs::compute_reward_a`.
   - Case: when both `kill_a` and `collision_a` fire on the same step,
     CPU emits `collision_penalty (-50)` (checks collision first), Rust
     emits `kill_reward (+100)` (checks kill first).
   - Frequency: low in practice (requires lock-on completion in the exact
     step you hit an obstacle) but non-zero.
   - Fix pattern: pick one ordering in the design doc and align both
     paths. Recommend kill-first because a locked-on kill is the "goal"
     terminal.

2. **Timeout / truncation semantics**
   - Site: CPU env runs full shaping reward on the truncation step; Rust
     short-circuits terminal reward to `0.0` for timeout.
   - Effect: tiny per-step reward drift on the last tick of a clipped
     episode. Only matters for baselines on precise reward curves.
   - Fix pattern: agree to short-circuit to 0 on truncation, remove the
     CPU-env shaping block. The test suite in `test_reward_consistency.py`
     already has a placeholder for this case.

3. **Per-task reward overrides** (Task DD follow-up)
   - Site: `configs/rules.toml [task_reward_overrides.<task>]` is read by
     CPU env but not by `GpuVecEnv`.
   - Effect: GPU curriculum phases always use the default `[reward]`
     block — pursuit-specific tuning (e.g. `approach_reward = 5.0` for
     `pursuit_linear`) is lost.
   - Fix pattern: extend `reward_config` plumbing (Audit 6) with a
     per-phase override layer. Need to decide how `GpuVecEnv` knows its
     task name — add an explicit `task_name` kwarg, look it up in the
     overrides table, shallow-merge over the base config before passing
     down.

4. **batch-sim missing CPU noise features**
   - Site: `aces/env/dogfight.py` applies `obs_noise_std`, motor delay
     (first-order), motor noise + bias, and IMU random-walk bias (all
     from `crates/sim-core/src/{noise,actuator,imu_bias}.rs`). batch-sim
     does none of this.
   - Effect: curriculum phases 3-5 (which rely on noise for robustness
     and for particle-filter training signal) would be less noisy under
     GPU training than CPU.
   - Fix pattern: port each noise module into `crates/batch-sim/src/`
     (new files, not just config plumbing) and into the GPU rollout (new
     WGSL code for the deterministic ones, CPU application for
     per-observation noise). Significant work, not in scope for these
     audits.

5. **MPPI rollout wind approximation**
   - Site: `crates/batch-sim/src/gpu/pipeline.rs` + `mppi_rollout.wgsl`.
   - Current behaviour: per-drone wind uploaded each tick (Audit
     `feature/gpu-wind`) but held constant across the rollout horizon.
     CPU MPPI (`crates/mppi/src/optimizer.rs::rollout_with_wind`) samples
     OU process per step.
   - Effect: CVaR-style risk-aware MPPI (`[mppi.risk]` in `rules.toml`)
     is not exercised on GPU — the risk term collapses to deterministic.
   - Fix pattern: add a GPU RNG (PCG hash is standard, single `u32` state
     derived from `sample_id, drone_id, step`) and sample OU wind inline
     in the rollout kernel. Noise correlation needs care — OU is
     auto-correlated, independent Gaussian per step is *not* what we
     want.

---

## Part 4 — Audit Methodology

How to find similar bugs in the remaining code paths.

### 4.1 Pattern-match targets

These four implementations of "the same logic" must agree:

- `aces/env/dogfight.py`                        — CPU Gymnasium env.
- `crates/batch-sim/src/` (Phase 1 Rayon path)  — the canonical Rust.
- `crates/batch-sim/src/gpu/` + WGSL            — GPU compute path.
- `crates/py-bridge/src/lib.rs::Simulation`     — single-drone PyO3, used
                                                   by `scripts/run.py` non-batch
                                                   training and by the Bevy game.
- `crates/game/src/simulation.rs`               — Bevy interactive viewer,
                                                   sometimes has its own helpers
                                                   (see Audit 4).

When a parameter is read from `configs/*.toml`, audit every layer it passes
through:

```
configs/rules.toml
  → Python config.py dataclass
    → Python env / GpuVecEnv kwargs
      → PyGpuVecEnv kwargs (PyO3)
        → Rust BatchConfig / RewardConfig / WindConfig fields
          → GPU uniform struct fields
            → WGSL uniform usage
```

A typical bug is a silent default at one of these boundaries (Audits 1,
6, 7 were all defaults; Audits 2, 3, 4, 5 were formula / logic mismatches).

### 4.2 High-leverage diagnostic questions

- Where does each code path read `X` from? (config, hardcoded, passed-in,
  thread-local, env var)
- Does the formula match byte-for-byte? Write both sides out long-hand.
- For every boundary crossing, is there an explicit test asserting
  parity? If not, add one.
- Is the test itself deterministic? (Audit 4 caught a case where the
  *test* had nondeterministic noise.)

### 4.3 Reproducible workflow used this session

1. Dispatch a fresh worker agent ("compare how `X` is computed in
   `aces/env/dogfight.py` vs `crates/batch-sim/src/`, report line-by-line
   and ignore anything I've told you from prior turns").
2. Worker emits a divergence report + specific fix proposal.
3. Dispatch a fresh reviewer agent to double-check + run tests.
4. Implement the fix in a branch `feature/<audit-name>`, land with a new
   test pinning the alignment, merge.
5. Re-dispatch both agents on adjacent code to find the next divergence.

This pattern found seven bugs in seven attempts. The remaining
divergences in Part 3 were identified by the same agents but deferred to
scope the work per merge.

---

## Part 5 — Git History Index

All 35 merges into `main` during the session, in chronological order.
Commit SHAs are on the merge commit; the underlying work is in the merged
branch.

### Baseline (pre-GPU) cleanup

| SHA       | Description                                                             |
|-----------|-------------------------------------------------------------------------|
| `8fe4b95` | overnight experiment branch — training fixes + logging + experiments    |
| `4e11ed0` | reorganize Python package into subpackages (`env/`, `training/`, ...)   |
| `949e750` | remove backward-compat shims, all imports use new paths                 |

### Phase 2 — WGPU batch MPPI

| SHA       | Description                                                             |
|-----------|-------------------------------------------------------------------------|
| `b6ceb84` | `feature/f32-rk4-validation` — f32 RK4 reference + parity tests         |
| `3cfe38a` | `feature/f32-sdf` — f32 Arena / Obstacle SDF + parity                   |
| `9d88db1` | `feature/gpu-pipeline-skeleton` — GpuBatchMppi buffer allocation        |
| `3c2f41d` | `feature/f32-cost` — f32 pursuit / evasion cost + parity                |
| `06b4e26` | `feature/wgsl-helpers` — WGSL helpers + naga structural validation      |
| `31b87f9` | `feature/mppi-rollout-kernel` — `rollout_and_cost` kernel + MppiDims    |
| `e6c8072` | `feature/mppi-softmax-kernel` — `softmax_reduce` kernel + temperature   |
| `dce9617` | `feature/gpu-pipelines` — BindGroupLayout + two compute pipelines        |
| `6a3e250` | `feature/gpu-dispatch` — `compute_batch_actions()` dispatch + readback   |
| `52836de` | `feature/gpu-cpu-parity` — CPU reference + end-to-end parity test        |
| `b0f1082` | `feature/gpu-bench` — `bench_gpu_vs_cpu` example                         |

### Phase 3 — PyO3 + SB3 integration

| SHA       | Description                                                             |
|-----------|-------------------------------------------------------------------------|
| `4278b9e` | `feature/pyo3-gpu` — PyO3 `PyGpuBatchMppi`                              |
| `dc7a8cd` | `feature/orchestrator-gpu-opt` — `GpuBatchOrchestrator` end-to-end       |
| `bb6aa96` | `feature/gpu-vec-env` — PyO3 `GpuVecEnv` + `reset`                       |
| `4b4d70c` | `feature/gpu-ppo-mode` — `step_with_agent_a_actions` (PPO-vs-MPPI)       |
| `98ba353` | `feature/gpu-sb3-wrapper` — Python `GpuVecEnv` SB3 wrapper               |
| `1be11aa` | `feature/gpu-ppo-smoke` — end-to-end PPO smoke test                      |
| `fef1050` | `feature/curriculum-gpu-opt` — `--use-gpu-env` on `CurriculumTrainer`    |
| `0de67a6` | `feature/gpu-docs` — `docs/gpu-mppi.md` user guide                       |
| `c253ac6` | `feature/training-bench` — PPO throughput benchmark script               |
| `a4f4bd8` | `feature/denorm-unit-test` — `denormalize_action` pure fn + tests        |
| `9f40bef` | `feature/test-conftest` — consolidate pytest fixtures                     |
| `86728e9` | `feature/setup-check` — `scripts/check_gpu_setup.sh`                     |
| `652cd30` | `feature/obs-describe` — `obs_layout` describe helper                     |
| `4a541dd` | `feature/gpu-wind` — per-drone wind in rollout (binding 11)              |

### Consistency audits

| SHA       | Audit                                                                   |
|-----------|-------------------------------------------------------------------------|
| `41aeb05` | `feature/action-consistency` — action denormalization                   |
| `ee83fd1` | `feature/reward-consistency` — reward formula                           |
| `057e5d4` | `feature/obs-consistency` — observation slot [15]                       |
| `b1f403a` | `feature/fix-test-and-game` — obs test precondition + Bevy SDF          |
| `6141a8a` | `feature/spawn-audit` — SpawnMode + max_steps                           |
| `4c6678c` | `feature/rules-toml-plumb` — reward_config plumbing                     |
| `9a6b716` | `feature/wind-plumb` — wind_sigma / wind_theta plumbing                 |

---

## Part 6 — How to Resume

### Verify current state

```bash
# Rust workspace
cargo test --workspace
cargo test -p aces-batch-sim --features gpu

# Python — ensure extension is built first
poetry run maturin develop --features gpu
poetry run pytest tests/ -v

# End-to-end GPU setup (idempotent, prints OK / FAIL / SKIP per stage)
bash scripts/check_gpu_setup.sh
```

### Suggested next slices (priority order)

1. **Per-task reward overrides** (Part 3, #3) — continuation of Audit 6.
   Small: add `task_name` kwarg to `GpuVecEnv`, look up overrides, merge.
2. **Terminal priority + timeout semantics** (Part 3, #1, #2) —
   reward-consistency follow-ups; already have tests ready to flip.
3. **batch-sim `obs_noise` / actuator / IMU port** (Part 3, #4) — biggest
   remaining gap for phase 3-5 training parity. Port CPU modules, wire
   into `BatchConfig`, add parity tests.
4. **GPU RNG for OU wind in rollout** (Part 3, #5) — unlocks CVaR risk
   MPPI on GPU.
5. **Further audit candidates**:
   - Initial-velocity / jitter variance (CPU uses fixed zero; batch-sim
     `SpawnMode::FixedWithJitter` adds small σ — what's the target?).
   - Lock-on parameters: `fov_degrees` / `lock_distance` / `lock_duration`
     — every path reads from `rules.toml`, but is there a default
     somewhere? Same audit pattern as Audit 6/7.
   - Euler-angle convention: observation slot `[12:15]` is
     `(roll, pitch, yaw)` — is the convention (XYZ? ZYX?) identical
     across CPU / batch-sim / GPU? The WGSL code computes euler from
     quaternion independently.

### Useful references

- User-facing guide: [`docs/gpu-mppi.md`](gpu-mppi.md)
- Technical reference: [`docs/design.md`](design.md)
- Plans: [`plans/parallel-simulation.md`](../plans/parallel-simulation.md),
  [`plans/curriculum-architecture.md`](../plans/curriculum-architecture.md)
- Observation layout helper: `aces/env/obs_layout.py` (use
  `describe_obs(obs)` when debugging).
