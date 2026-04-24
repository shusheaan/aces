# Plan: Curriculum Training Architecture — Full System Reference

Status: **Active reference document**
Created: 2026-04-23

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          ACES TRAINING SYSTEM                            │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ Curriculum Pipeline (Python: aces/training/curriculum_trainer.py)    │ │
│  │                                                                     │ │
│  │ Phase 0      Phase 1         Phase 2          Phase 3               │ │
│  │ hover    → pursuit_linear → pursuit_evasive → search_pursuit        │ │
│  │ (no opp)   (trajectory)    (mppi_evasion)    (mppi_evasion)         │ │
│  │                                                                     │ │
│  │ Phase 4              Phase 5                                        │ │
│  │ self_play_noisy   → fpv_transfer                                    │ │
│  │ (Elo pool)          (Elo pool + FPV CNN)                            │ │
│  └──────────┬───────���──────────────────────────────────────────────────┘ │
│             │                                                            │
│  ┌──────────▼──────────────────────────────────────────────────────────┐ │
│  │ RL Training (Python: SB3 PPO)                                       │ │
│  │                                                                     │ │
│  │  DogfightEnv (Gymnasium) ←→ VecNormalize ←→ PPO (Actor-Critic MLP) │ │
│  │  or: BatchVecEnv (Rust)                                             │ │
│  └──────────┬──────────────────────────────────────────────────────────┘ │
│             │ PyO3 FFI                                                   │
│  ┌──────────▼──────────────────────────────────────────────────────────┐ │
│  │ Rust Simulation Layer                                               │ │
│  │                                                                     │ │
│  │  sim-core: dynamics(RK4), environment(SDF), collision(LOS),         │ │
│  │            lockon, camera(depth), detection, wind(OU), noise,       │ │
│  │            actuator, imu_bias, safety                               │ │
│  │  mppi:     optimizer(Rayon par_iter), rollout, cost                  │ │
│  │  estimator: ekf, particle_filter                                    │ │
│  │  batch-sim: orchestrator(N parallel), battle, observation, reward   │ │
│  │  py-bridge: PyO3 Simulation + MppiController + (future) BatchVecEnv│ │
│  │  game:      Bevy 3D visualizer + NN inference + FSM                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Crate Dependency Graph

```
sim-core          (no deps — pure physics)
  ↑
mppi              (depends on sim-core — trajectory optimization)
  ↑
estimator         (depends on sim-core — EKF + particle filter)
  ↑
batch-sim         (depends on sim-core + mppi — parallel orchestration)
  ↑
py-bridge         (depends on sim-core + mppi + estimator — Python FFI)

game              (depends on sim-core + mppi — Bevy standalone)
```

---

## Physics Engine — `crates/sim-core/`

### State representation

`state.rs` — `DroneState`: 13-dimensional

```
position:         Vector3<f64>      [px, py, pz]         world frame, meters
velocity:         Vector3<f64>      [vx, vy, vz]         world frame, m/s
attitude:         UnitQuaternion    [qw, qx, qy, qz]    body→world rotation
angular_velocity: Vector3<f64>      [wx, wy, wz]         body frame, rad/s
```

Key methods: `forward()`, `distance_to()`, `angle_to()`, `to_array()`, `from_array()`, `hover_at()`

### Dynamics

`dynamics.rs` — Crazyflie 2.1 parameters:
- mass=0.027kg, arm=0.04m, max_thrust=0.15N/motor, hover=0.066N/motor
- `state_derivative(state, motors, params, external_force)` → (ṗ, v̇, q̇, ω̇)
- `step_rk4(state, motors, params, dt, wind)` → new state
- X-configuration motor mixing: `motor_mixing(motors)` → (total_thrust, torque_vec)
- dt_ctrl=0.01s (100Hz control), substeps=10 (1ms physics), dt_sim=0.001s

### Environment / SDF

`environment.rs` — Arena + obstacles:
- `Arena { bounds: Vector3, obstacles: Vec<Obstacle>, drone_radius: 0.05 }`
- Obstacle types: Box, Sphere, Cylinder — each with `.sdf(p) → f64`
- `arena.sdf(p)` = min(boundary_sdf, obstacle_sdf) — combined signed distance
- `arena.is_collision(p)` = sdf(p) < drone_radius
- Default: 10×10×3m warehouse, 5 pillar obstacles (1×1×3m boxes)

### Lock-on system

`lockon.rs` — `LockOnTracker`:
- Params: fov=90°, distance=2.0m, duration=1.5s
- `update(attacker, target, arena, dt)` → bool (kill)
- Conditions: in FOV ∧ in range ∧ line-of-sight clear
- Timer accumulates while all hold, resets on any break

### Wind

`wind.rs` — Ornstein-Uhlenbeck: `dw = θ(μ - w)dt + σdW`
- theta=2.0, mu=[0,0,0], sigma=0.3N
- `WindModel::step(dt, rng)` → Vector3 force
- Can be disabled (`WindModel::disabled()`) for early curriculum phases

### Collision / line-of-sight

`collision.rs` — Sphere tracing through obstacle SDF:
- `check_line_of_sight(arena, origin, target)` → Visible|Occluded
- Used by lock-on and visibility checks

### Camera

`camera.rs` — Pinhole depth rendering via sphere tracing:
- 320×240 at 90° FOV, max_depth=15m
- Row-parallel via Rayon
- Policy downsamples to 80×60 for CNN input
- Only active in FPV mode (Phase 5)

### Noise models

- `noise.rs`: Gaussian observation noise on position (σ=0.1m)
- `actuator.rs`: First-order motor delay + multiplicative noise + per-motor bias
- `imu_bias.rs`: Random walk on accel/gyro biases
- All disabled by default, enabled progressively in curriculum phases 4-5

---

## MPPI Optimizer — `crates/mppi/`

### Algorithm

`optimizer.rs:177-342` — `compute_action_with_cost_fn()`:

```
Input: self_state, enemy_state, cost_fn
Output: optimal motors Vector4<f64>

1. Seeds: generate N u64 seeds from thread_rng
2. Par_iter over N samples (Rayon):
   a. SmallRng::seed_from_u64(seed)
   b. Sample perturbations: u[t] = clamp(mean[t] + N(0,σ²), 0, max_thrust)
   c. Rollout: H steps × S substeps of step_rk4
   d. Cost: Σ_t cost_fn(state[t], enemy, u[t], hover, arena, weights)
   e. Hard collision: if penetration > 0, cost += 1e8
3. CVaR (optional): penalize worst-α fraction
4. Chance constraint (optional): Lagrangian on collision probability
5. Softmax: w[k] = exp(-(c[k] - c_min) / T) / Σ exp(...)
6. Weighted mean: new_mean[t] = Σ w[k] · u_k[t]
7. Return new_mean[0], shift left for warm-start
```

### Cost functions

`cost.rs`:
- `pursuit_cost`: w_dist·d² + w_face·(1-cos θ) + w_ctrl·||u-hover||² + obstacle_penalty
- `evasion_cost`: penalize proximity < 3m + being in enemy FOV + same ctrl/obs terms
- `belief_pursuit_cost`: scales opponent-relative terms by confidence = 1/(1+belief_var)
- `belief_evasion_cost`: same confidence scaling + increased d_safe under uncertainty
- Weights: w_dist=1.0, w_face=5.0, w_ctrl=0.01, w_obs=1000.0, d_safe=0.3m

### Risk-aware features

- `RiskConfig`: wind sampling (OU) + CVaR (α=0.05, penalty=10.0)
- `ChanceConstraintConfig`: P(collision) ≤ δ (1%) via online Lagrangian (λ_lr=0.1, λ_init=100)
- `rollout_with_wind()`: each sample gets independent wind realization, tracks max_penetration

---

## Estimator — `crates/estimator/`

### EKF

`ekf.rs` — 6-state (pos + vel), constant-velocity model:
- Predict: F·x + noise, P = F·P·Fᵀ + Q(dt)
- Update: Kalman gain via Joseph form, position-only observation H=[I₃ 0]
- q_a = 4.0 (accel spectral density), R = σ²·I₃

### Particle filter

`particle_filter.rs` — 200 particles:
- Each: position + velocity + weight
- Predict: constant velocity + acceleration noise + SDF rejection
- Update: weight ∝ exp(-||p - measurement||² / 2σ²) + systematic resampling
- Used when opponent is occluded (line-of-sight blocked)

---

## Python Training Layer — `aces/`

### Environment

`aces/env/dogfight.py` — `DroneDogfightEnv(gymnasium.Env)`:
- Observation: 21-dim vector or FPV dict {image: (1,60,80), vector: (12,)}
- Action: Box(4,) in [-1,1] → motors = hover + action × (max_thrust - hover)
- Calls Rust `Simulation.step(motors_a, motors_b)` via PyO3
- Reward: terminal (kill/death/collision) + shaping (approach, lock, survival, control)
- Task-specific overrides in `configs/rules.toml [task_reward_overrides.*]`

### Curriculum

`aces/curriculum.py` — `CurriculumManager`:
- Loads `configs/curriculum.toml` → list of `Phase` objects
- `should_promote(stats)`: check if "win_rate>X" or "reward>X" or "steps" condition met
- `promote()`: advance to next phase

### Curriculum trainer

`aces/training/curriculum_trainer.py` — `CurriculumTrainer`:
- For each phase:
  1. Create VecEnv with phase params (wind, noise, opponent type)
  2. Reuse PPO model from previous phase (weights carry over)
  3. Setup callbacks (stats, logging, promotion check, opponent update)
  4. `model.learn(max_timesteps, callbacks=...)`
  5. Save VecNormalize stats (obs/reward running mean/var) for next phase
  6. Save model checkpoint, add to opponent pool

### Self-play

`aces/training/self_play.py` — `SelfPlayTrainer`:
- Single env, opponent = lagged copy of agent policy
- OpponentUpdateCallback: copy weights every 10K steps

### Opponent pool

`aces/training/opponent_pool.py` — `OpponentPool`:
- Max 20 checkpoints, Elo-rated (K=32)
- Sampling: softmax(elo / 400)
- PoolOpponentCallback: resample from pool every 20K steps

### Batched opponent inference

`aces/training/batched_vec_env.py` — `BatchedOpponentVecEnv`:
- Wraps SubprocVecEnv
- Before each step: collect opponent obs from all N envs → single batched policy.predict()
- Sets opponent actions back via env.set_next_opponent_action()

### Callbacks

`aces/training/callbacks.py`:
- `TrainingStatsCallback`: per-episode stats (kills, deaths, rewards)
- `EpisodeLoggerCallback`: CSV export per episode
- `TensorBoardMetricsCallback`: SB3 logger (win_rate, kill_rate, mean_reward)
- `WindowSummaryCallback`: print convergence every 10K steps
- `PromotionCheckCallback`: check promotion condition every 5K steps
- `VecOpponentUpdateCallback`: copy policy to opponent in VecEnv
- `PoolOpponentCallback`: resample from Elo pool

### Policy architecture

- `aces/policy/extractors.py`: CnnImuExtractor for FPV — Conv2d(1→32→64→64) + Linear(12→64), concat → 192-dim
- `aces/policy/constrained_ppo.py`: Lagrangian PPO — reward' = reward - λ·cost, λ updated via dual ascent
- `aces/policy/export.py`: SB3 MLP → binary (little-endian f32: num_layers, [rows, cols, weights, biases]×N)

### Perception (neural-symbolic)

- `aces/perception/oracle.py`: God Oracle — computes ground truth semantic labels (threat, opportunity, collision_risk, etc.)
- `aces/perception/perception_net.py`: Supervised MLP (21→64→64→8) trained on oracle labels
- `aces/perception/neural_symbolic.py`: Mode Selector NN (21→32→32→4 modes + 5 params) + MPPI executor
- `aces/env/ns_env.py`: Wraps dogfight env with 10-step decision interval

---

## Configuration Files

### `configs/drone.toml`
Crazyflie physics: mass, arm_length, inertia, max_thrust, dt_ctrl=0.01, substeps=10

### `configs/arena.toml`
10×10×3m bounds, 5 box obstacles, spawn (1,1,1.5)/(9,9,1.5), collision_radius=0.05

### `configs/rules.toml`
- `[lockon]`: fov=90°, distance=2m, duration=1.5s
- `[mppi]`: 1024 samples, 50 horizon, T=10, σ=0.03
- `[mppi.weights]`: w_dist=1, w_face=5, w_ctrl=0.01, w_obs=1000, d_safe=0.3
- `[mppi.risk]`: wind θ/σ, CVaR α=0.05, chance δ=0.01
- `[noise]`: wind (OU), obs_noise, actuator (delay, noise, bias), IMU bias
- `[training]`: lr=3e-4, batch=64, n_steps=2048, γ=0.99, λ_GAE=0.95, clip=0.2, epochs=10
- `[reward]`: kill=100, death=-100, collision=-50, lock=5, approach=0.05, survival=0.01
- `[task_reward_overrides.*]`: per-task overrides (pursuit_linear: approach=5.0, etc.)

### `configs/curriculum.toml`
6 phases: hover → pursuit_linear → pursuit_evasive → search_pursuit → self_play_noisy → fpv_transfer
Each: name, task, opponent, wind/noise params, max_timesteps, promote_condition, promote_window

---

## Curriculum Phases — Detailed

### Phase 0: hover_stabilize
- **Task**: hover (no opponent)
- **Opponent**: none
- **Reward**: survival-dominant: 1.0 - 0.1·angular_vel - 0.1·position_drift - 0.001·control
- **Promote**: reward > -5.0 over 50 episodes
- **Trains**: basic attitude/altitude hold (~50K steps to converge)
- **Known issue**: old reward was perverse (penalized survival). Fixed 2026-04-23.

### Phase 1: pursuit_linear
- **Task**: pursuit_linear (opponent flies straight/circle)
- **Opponent**: trajectory controller (trajectory.py)
- **Reward**: approach_reward=5.0 (override), kill=100
- **Promote**: win_rate > 0.3 over 100 episodes
- **Trains**: fly toward target, face opponent, close distance
- **Known issue**: approach signal still weak vs opponent_crash_reward=50. See memory.

### Phase 2: pursuit_evasive
- **Task**: pursuit_evasive (opponent actively evades)
- **Opponent**: MPPI evasion controller
- **Reward**: approach_reward=3.0 (override)
- **Promote**: win_rate > 0.3 over 200 episodes
- **Trains**: predict evasive maneuvers, use obstacles for cut-off

### Phase 3: search_pursuit
- **Task**: search_pursuit (opponent starts hidden / uses occlusion)
- **Opponent**: MPPI evasion
- **Noise**: obs_noise_std=0.1
- **Reward**: info_gain=0.1, lost_contact_penalty=0.02 (overrides)
- **Promote**: win_rate > 0.25 over 200 episodes
- **Trains**: search for occluded opponent, active perception, info-theoretic exploration

### Phase 4: self_play_noisy
- **Task**: dogfight (full combat)
- **Opponent**: Elo pool (self-play)
- **Noise**: wind_sigma=0.3, obs_noise=0.1, motor (τ=0.02, σ=0.05, bias=0.03), IMU bias
- **Reward**: approach=3.0, opponent_crash=5.0 (overrides)
- **Promote**: win_rate > 0.55 over 500 episodes
- **Trains**: generalize across diverse opponents, robustness to noise/wind

### Phase 5: fpv_transfer
- **Task**: dogfight with FPV observation
- **Opponent**: Elo pool
- **Observation**: dict{image: (1,60,80), vector: (12,)} — no privileged opponent info
- **Policy**: CNN+MLP (CnnImuExtractor → 192-dim → PPO head)
- **Promote**: steps (final phase, runs until timestep budget exhausted)
- **Trains**: visual perception, depth→action mapping without ground truth state

---

## Deployment Pipeline

```
Training (Python)                    Export                  Runtime (Rust)
──────���──────────                   ────────                ──────────────
SB3 PPO model.zip                   
  └─ policy_net weights             
  └─ action_net weights    ──────→  export.py ──→ policy.bin ──→ game/policy.rs
                                                                   matmul→Tanh→matmul

Perception NN (PyTorch)    ──────→  export.py ──→ perception.bin ──→ game/perception.rs
                                                                       matmul→Tanh→matmul

                                                  Bevy game (game/simulation.rs):
                                                    If perception.bin: FSM + MPPI
                                                    Elif policy.bin: Direct NN
                                                    Else: Pure MPPI
```

Binary format: `u32 num_layers, per layer: u32 rows, u32 cols, f32[rows×cols] weights, f32[rows] biases`

---

## Parallel Simulation — Summary Reference

See `plans/parallel-simulation.md` for full details.

### Phase 1 (Done): `crates/batch-sim/` — Rayon BatchOrchestrator
- N concurrent MPPI-vs-MPPI dogfights
- Nested Rayon: outer (battles) × inner (MPPI samples)
- 14 unit tests passing, benchmark shows 3.2K steps/s at 64 battles

### Phase 2 (DONE 2026-04-24): WGPU compute shaders for MPPI rollout
- Single GPU dispatch for all drones' MPPI (12 bindings, 2 kernels)
- f32 WGSL shader porting RK4 + SDF + cost + softmax reduction
- Feature-gated: `--features gpu`
- User guide: `docs/gpu-mppi.md`

### Phase 3 (DONE 2026-04-24): PyO3 GpuVecEnv for SB3 integration
- `step_with_agent_a_actions()` — PPO agent vs MPPI opponent
- `aces._core.GpuVecEnv` PyO3 class + `aces.training.GpuVecEnv` SB3 wrapper
- `--use-gpu-env` opt-in on `CurriculumTrainer`
- Known limitation: only MPPI-vs-MPPI opponent semantics supported (other
  phases warn + fall back)

### Phase 4 (Pending): Full GPU physics pipeline
- All simulation on GPU, CPU only does PPO + episode management

---

## Known Issues & Technical Debt

1. **Pursuit reward signal too weak** (Phase 1 curriculum): approach_reward=0.05 default is
   dominated by opponent_crash_reward=50. Override helps but agent still slow to learn approach.
   Options: increase approach_reward further, add distance milestone bonuses, or stabilize
   trajectory opponent to reduce crashes.

2. **Curriculum promote_condition gap**: Phase 0→1 uses reward threshold (easy), Phase 1→2
   uses win_rate > 0.3 (hard jump). Consider intermediate metric like mean_min_distance.

3. **FPV transfer shares no weights**: Phase 4 (MLP) → Phase 5 (CNN+MLP) is effectively
   training from scratch since observation space changes. Consider progressive feature dropout
   or teacher-student distillation.

4. **Opponent pool cold start**: Phase 4 starts with pool containing only weak Phase 0-3 models.
   Consider seeding pool with MPPI-strength opponents.

5. **batch-sim doesn't support belief state**: BattleState has no EKF/PF, so observation
   fields 19-20 (belief_uncertainty, time_since_seen) are always 0. Phase 3 curriculum
   needs this for realistic search_pursuit training.

### 2026-04-24 updates

See [`docs/2026-04-24-session-archive.md`](../docs/2026-04-24-session-archive.md)
for full detail. Summary of what changed in the seven consistency audits:

6. **GpuVecEnv now plumbs rules.toml** (Audit 6 / `feature/rules-toml-plumb`).
   Was: the GPU training path silently used `RewardConfig::default()` — any
   `[reward]` tuning in `rules.toml` was ignored. Now: GpuVecEnv loads the
   file and passes it through to Rust.

7. **GpuVecEnv now plumbs wind config** (Audit 7 / `feature/wind-plumb`).
   Was: `wind_sigma` defaulted to a literal; `wind_theta` was hardcoded
   `2.0`. Now: both read from `rules.toml [noise]`.

8. **Spawn initialization fixed** (Audit 5 / `feature/spawn-audit`).
   Introduced `SpawnMode::{FixedFromArena, FixedWithJitter, Random}`.
   Default matches CPU env (fixed + small jitter); `Random` available for
   domain randomization in later phases. `BatchConfig::max_steps` now
   honoured (was hardcoded 1000).

9. **Reward-formula divergences fixed** (Audits 2-4). Three remaining:
   - Terminal priority when `kill_a && collision_a` fire same step (Rust:
     kill-first, CPU: collision-first).
   - Timeout / truncation reward (Rust short-circuits to 0; CPU runs
     shaping).
   - Per-task reward overrides (`[task_reward_overrides.*]`) not plumbed
     through GpuVecEnv — a future DD follow-up, analogous to Audit 6 but
     with a per-phase merge layer.

10. **Action denormalization aligned** (Audit 1 / `feature/action-consistency`).
    GPU VecEnv now uses hover-centered convention
    `motor = hover + a * (max - hover)` to match CPU env. The previous
    symmetric-scaling formula caused ~12 % thrust mismatch at `a = 0`.

11. **Observation slot [15] aligned** (Audits 3, 4). CPU env + Bevy game
    now both use combined `arena.sdf() = min(boundary, obstacle)` to match
    batch-sim. Previously CPU returned obstacle-only SDF, missing
    wall-proximity signal.

12. **batch-sim still misses CPU noise features**: `obs_noise_std`, motor
    delay (first-order), motor noise + bias, IMU random-walk bias —
    `aces/env/dogfight.py` applies these via `crates/sim-core/src/{noise,
    actuator, imu_bias}.rs`, but `crates/batch-sim/` does not. Would need
    new crate code, not just config plumbing. Impacts phases 3-5 where
    noise is part of the curriculum.
