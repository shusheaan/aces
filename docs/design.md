# ACES — Air Combat Engagement Simulation

1v1 quadrotor drone dogfight: Rust physics core + Python RL/viz layer + Bevy 3D game.

Two drones fight in a 10m x 10m x 3m arena with obstacles. Each has a forward-facing camera. Lock-on = keep opponent in FOV cone at close range for 1.5 seconds. Whoever locks on first wins.

---

## 1. Core Design Decisions

These are the foundational choices that shape every downstream decision. Changing any of them requires significant rework. They should be evaluated and locked in early.

### 1.1 Combat Mechanic: FOV Lock-On

**Decision**: Kill = continuous visual lock-on, not projectile hit.

```
Lock-on conditions (all must hold simultaneously):
  1. Opponent within camera FOV cone (90 deg)
  2. Opponent within D_lock (2.0 m)
  3. Line-of-sight not occluded by obstacles
  4. Above conditions held continuously for T_lock (1.5 s)
  → Kill confirmed

Lock-on math:
  d = pos_B - pos_A                        (relative position)
  f_A = R(q_A) * [1, 0, 0]                (nose direction)
  theta = arccos( (d . f_A) / ||d|| )     (off-boresight angle)
  Lock condition: theta < FOV/2 AND ||d|| < D_lock AND LOS clear
  Timer: accumulates while condition holds, resets on any interruption
```

**Why this matters**: This forces the agent to learn aggressive maneuvering (pursuit/evasion), obstacle exploitation (break LOS to reset timer), and spatial awareness (facing angle matters). A projectile model would reduce the problem to aiming, which is simpler but less interesting for RL.

**Tuning levers**: `fov_degrees`, `lock_distance`, `lock_duration` in `configs/rules.toml`. Tighter FOV = harder kills, longer duration = more defensive play.

### 1.2 Drone Platform: Crazyflie 2.1 Parameters

**Decision**: Use Crazyflie 2.1 physical parameters as reference.

```
Mass:           m = 0.027 kg
Arm length:     d = 0.04 m
Inertia:        Ixx = Iyy = 1.4e-5, Izz = 2.17e-5 kg*m^2
Max thrust:     0.15 N per motor (total ~0.6N, thrust/weight ~ 2.2:1)
Torque coeff:   c_tau = 0.005964
Drag coeff:     k_drag = 0.01
Hover thrust:   f_hover = mg/4 ~ 0.066 N
```

**Why this matters**: Thrust/weight ratio of 2.2:1 limits aggressiveness. Maneuvers are constrained. For wilder dogfights, scale up thrust in config but note this diverges from real hardware. All params in `configs/drone.toml`.

**Future hardware**: Option A = Crazyflie + AI deck (~$600 for two), Option B = custom Jetson Nano drone (~$2000 for two). System identification needed for either.

### 1.3 Architecture: Rust Core + Python RL + Bevy Game

**Decision**: Three-layer hybrid architecture.

```
Layer 1 — Rust (performance-critical):
  Physics, SDF, collision, lock-on, camera rendering, detection,
  MPPI controller, EKF, particle filter, wind/noise
  → Exposed to Python via PyO3/maturin (aces._core)
  → Used directly by Bevy game crate

Layer 2 — Python (ML ecosystem):
  Gymnasium env, PPO self-play, CNN+MLP policy, curriculum trainer,
  opponent pool, trajectory generation, Rerun visualization
  → Orchestrates training and evaluation

Layer 3 — Bevy game (interactive visualization):
  3D arena rendering, keyboard/gamepad input, orbit camera,
  HUD overlay, FPV split-screen, NN policy loading
  → Standalone binary, no Python dependency at runtime
```

**Why this matters**: Rust gives 1000 Hz physics without Python GIL bottleneck. Python gives access to SB3/PyTorch ecosystem. Bevy gives a distributable interactive demo. The PyO3 bridge is the critical integration point.

### 1.4 Observation Design: Vector vs FPV

**Decision**: Two observation modes with shared action space.

**Vector mode** (21-dim) — full state information:
```
own_velocity(3), own_angular_velocity(3),
opponent_relative_position(3), opponent_relative_velocity(3),
own_attitude(3), nearest_obstacle_distance(1),
lock_progress(1), being_locked_progress(1),
opponent_visible(1), belief_uncertainty(1), time_since_last_seen(1)
```

**FPV mode** (Dict) — camera-only spatial awareness:
```python
{
    "image": Box(0.0, 1.0, shape=(1, 60, 80)),  # normalized depth, 4x downsampled
    "vector": Box(-inf, inf, shape=(12,)),        # IMU-only, no opponent info
}
```

FPV vector (12-dim): velocity(3), angular_velocity(3), attitude(3), lock_progress(1), being_locked(1), nearest_obstacle(1). Opponent info must be inferred from the depth image.

**Why this matters**: Vector mode trains fast and works for policy export to Bevy. FPV mode is closer to real hardware but needs CNN and trains much slower. Curriculum training can bridge: train vector first, transfer to FPV last.

### 1.5 Training Strategy: Curriculum + Self-Play

**Decision**: 5-phase curriculum with opponent pool, not end-to-end training.

| Phase | Task | Opponent | Objective |
|-------|------|----------|-----------|
| 1 | `pursuit_linear` | PD on trajectory | Basic flight + pursuit + lock-on |
| 2 | `pursuit_evasive` | MPPI evasion | Adversarial pursuit |
| 3 | `search_pursuit` | MPPI evasion, occluded spawn | Search + info gain + pursuit |
| 4 | `dogfight` | Self-play + Elo pool | Full attack + defense |
| 5 | `fpv_transfer` | Self-play + Elo pool | Vision-only (CNN policy) |

**Why this matters**: Direct dogfight training from scratch doesn't converge — the agent never experiences lock-on rewards. Curriculum ensures each skill is learned before composing. Elo-rated opponent pool in phases 4-5 prevents strategy cycling. Weight transfer works because all phases share the same observation space.

**Tuning levers**: `configs/curriculum.toml` defines phase ordering, timestep budgets, promotion conditions (win rate thresholds), and per-phase overrides (wind, noise, FPV).

### 1.6 Arena Design

**Decision**: 10m x 10m x 3m enclosed arena with 5 floor-to-ceiling pillars.

```
Obstacles (Box pillars, 1m x 1m x 3m each):
  ┌──────────────────────┐
  │                      │
  │  ██          ██      │   Spawn A: (1, 1, 1.5)
  │                      │   Spawn B: (9, 9, 1.5)
  │       ██             │
  │                      │   ██ = 1m x 1m pillar
  │  ██          ██      │
  │                      │
  └──────────────────────┘
```

**Why this matters**: Pillars create occlusion opportunities (break lock-on LOS, force search behavior) and collision hazards (punishment for reckless flight). Arena size vs drone speed determines engagement pace. All defined in `configs/arena.toml`.

### 1.7 Timing Hierarchy

```
1000 Hz — Physics step (RK4 dynamics + wind + collision)
 100 Hz — Control step (policy inference → motor commands)
  30 Hz — Camera render (sphere-trace depth + geometric detection)
```

10 RK4 substeps per control step. Camera renders ~every 3.3 control steps; between renders, last frame is reused.

**Why this matters**: Physics must be fast enough for stable RK4 integration of a 27g quadrotor. Control at 100 Hz matches real flight controller rates. Camera at 30 Hz is typical for real cameras and keeps ray-tracing cost manageable.

---

## 2. Architecture

```
crates/ (Rust, via PyO3)             aces/ (Python package)
┌──────────────────────────┐        ┌──────────────────────────┐
│ sim-core                 │        │ env.py                   │
│   dynamics (13-DOF RK4)  │        │   Gymnasium environment  │
│   environment (SDF)      │        │   vector / FPV obs modes │
│   collision + lock-on    │        │   curriculum task params  │
│   camera (depth render)  │◄──────►│ trainer.py               │
│   detection (geometric)  │  PyO3  │   Self-play PPO          │
│   wind + noise           │        │   CurriculumTrainer      │
│   safety + recorder      │        │ curriculum.py            │
├──────────────────────────┤        │   TOML-driven phases     │
│ mppi                     │        │ trajectory.py            │
│   parallel MPPI + CVaR   │        │   circle/lemniscate/     │
│   belief-weighted costs  │        │   patrol for curriculum  │
├──────────────────────────┤        │ policy.py                │
│ estimator                │        │   CNN+MLP for depth imgs │
│   EKF + particle filter  │        │ opponent_pool.py         │
├──────────────────────────┤        │   Elo-rated self-play    │
│ game (Bevy)              │        │ export.py                │
│   3D visualization       │        │   MLP weight → binary    │
│   keyboard/gamepad input │        │ config.py                │
│   orbit camera + HUD     │        │   Typed TOML loading     │
│   NN policy inference    │        │ viz.py                   │
└──────────────────────────┘        │   Rerun 3D viewer        │
                                    └──────────────────────────┘
```

### Layer Diagram

```
┌──────────────────────────────────────────────────────┐
│                    Decision Layer                      │
│  MPPI Control (pursuit/evasion costs)                 │
│  RL Policy (PPO self-play, curriculum)                │
│  Causal Transformer (trajectory prediction)           │
├──────────────────────────────────────────────────────┤
│                  State Estimation Layer                │
│  EKF (position+velocity when visible)                 │
│  Particle Filter (belief under occlusion, 200 ptcls)  │
│  Belief-MPPI (confidence-weighted planning)           │
├──────────────────────────────────────────────────────┤
│                    Perception Layer                    │
│  Line-of-Sight (SDF sphere tracing)                   │
│  Depth Camera (320x240 @ 30Hz, sphere-traced)         │
│  Geometric Detection (analytical projection + bbox)   │
├──────────────────────────────────────────────────────┤
│                     Control Layer                      │
│  Motor-level control (4-dim thrust)                   │
│  MPPI trajectory optimization (rayon parallel, 1024)  │
├──────────────────────────────────────────────────────┤
│                 Simulation Environment                 │
│  13-DOF Quadrotor Dynamics (RK4 @ 1000 Hz)           │
│  3D Arena + SDF Collision Detection                   │
│  Lock-on Rule Engine + Kill Confirmation              │
│  Wind (Ornstein-Uhlenbeck) + Obs Noise (Gaussian)    │
└──────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
aces/
├── Cargo.toml                      # Workspace root (5 crates)
├── pyproject.toml                  # Python: maturin + poetry
├── Dockerfile                      # Multi-stage headless training image
├── CLAUDE.md                       # Dev conventions
│
├── crates/                          # ── Rust workspace (5 crates) ──
│   ├── sim-core/                    #   Core simulation
│   │   └── src/
│   │       ├── lib.rs               #     Module registry
│   │       ├── dynamics.rs          #     13-DOF quadrotor, RK4 integrator
│   │       ├── state.rs             #     DroneState (pos, vel, quat, angvel)
│   │       ├── environment.rs       #     Arena + SDF obstacles (Box/Sphere/Cylinder)
│   │       ├── collision.rs         #     SDF collision + line-of-sight (sphere tracing)
│   │       ├── lockon.rs            #     FOV cone + lock-on timer + kill
│   │       ├── wind.rs              #     Ornstein-Uhlenbeck stochastic wind
│   │       ├── noise.rs             #     Gaussian observation noise
│   │       ├── camera.rs            #     Pinhole camera, sphere-traced depth rendering
│   │       ├── detection.rs         #     Geometric opponent detection + bounding box
│   │       ├── safety.rs            #     Safety constraints
│   │       └── recorder.rs          #     State recording
│   │
│   ├── mppi/                        #   MPPI controller
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── rollout.rs           #     Parallel trajectory simulation
│   │       ├── cost.rs              #     Pursuit/evasion + belief-weighted costs
│   │       └── optimizer.rs         #     Standard + risk-aware + belief MPPI
│   │
│   ├── estimator/                   #   State estimation
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── ekf.rs               #     Extended Kalman Filter (6D)
│   │       └── particle_filter.rs   #     Particle filter with SDF constraints
│   │
│   ├── py-bridge/                   #   PyO3 bindings → aces._core
│   │   └── src/lib.rs               #     Simulation + MppiController + StepResult
│   │
│   └── game/                        #   Bevy 3D interactive visualizer
│       └── src/
│           ├── main.rs              #     App setup, plugin registration
│           ├── config.rs            #     Load arena/drone/rules TOML
│           ├── arena.rs             #     3D meshes: floor, walls, pillars, lighting
│           ├── drone.rs             #     Drone entities: cross mesh, heading arrow, trail
│           ├── input.rs             #     Keyboard + Xbox gamepad → DroneCommand
│           ├── simulation.rs        #     sim-core @ 100Hz FixedUpdate, MPPI AI
│           ├── camera.rs            #     3 cameras: god-view + 2 FPV split-screen
│           ├── hud.rs               #     UI overlay: lock-on, telemetry, wall warning
│           ├── marker.rs            #     Enemy diamond marker (3D → screen projected)
│           └── policy.rs            #     Load exported MLP weights, run inference
│
├── aces/                            # ── Python package ──
│   ├── __init__.py
│   ├── config.py                    #   Typed TOML config loading (dataclasses)
│   ├── env.py                       #   Gymnasium environment (vector + FPV modes)
│   ├── trainer.py                   #   Self-play PPO (MLP + CNN policies)
│   ├── curriculum.py                #   Phase definitions + CurriculumManager
│   ├── opponent_pool.py             #   Elo-rated opponent pool for self-play
│   ├── trajectory.py                #   Circle/lemniscate/patrol for curriculum
│   ├── policy.py                    #   CnnImuExtractor (CNN+MLP for FPV)
│   ├── export.py                    #   MLP weight → binary for Bevy
│   └── viz.py                       #   Rerun 3D + depth image visualization
│
├── configs/                         # ── TOML configuration ──
│   ├── drone.toml                   #   Crazyflie 2.1 physical parameters
│   ├── arena.toml                   #   Warehouse: 10x10x3m, 5 pillars
│   ├── rules.toml                   #   Lock-on, MPPI, noise, camera, rewards
│   └── curriculum.toml              #   5-phase curriculum definition
│
├── scripts/
│   ├── run.py                       #   CLI: mppi-vs-mppi / train / evaluate / export / curriculum
│   ├── train_server.py              #   Headless server training (VecEnv, TensorBoard)
│   ├── install-hooks.sh             #   Git pre-commit hook installer
│   └── pre-commit.sh               #   Lint + test checks
│
├── tests/                           # ── 142 tests (57 Rust + 85 Python) ──
│   ├── test_config.py
│   ├── test_curriculum.py
│   ├── test_dynamics.py
│   ├── test_env.py
│   ├── test_opponent_pool.py
│   ├── test_trainer.py
│   ├── test_trajectory.py
│   └── test_viz.py
│
└── docs/
    └── design.md                    #   ← This file
```

**Total: ~30 Rust source files + 10 Python modules, 142 tests.**

---

## 4. Simulation Core

### 4.1 Quadrotor Dynamics (13-DOF)

State vector (13 elements):
```
x = [px, py, pz,         -- Position (world frame)
     vx, vy, vz,         -- Velocity (world frame)
     qw, qx, qy, qz,    -- Attitude quaternion (||q||=1)
     wx, wy, wz]         -- Angular velocity (body frame)
```

Quaternion representation: no gimbal lock during aggressive maneuvers.

Equations of motion:
```
p_dot = v
v_dot = [0, 0, -g] + (1/m) * R(q) * [0, 0, F_total] + (1/m) * F_drag + F_wind
q_dot = 0.5 * q ⊗ [0, wx, wy, wz]    (normalize each step)
I * w_dot = tau - w x (I * w)          (gyroscopic term)
```

Control input: `u = [f1, f2, f3, f4]` — four motor thrusts in [0, f_max] Newtons.

Motor mixing (X-configuration):
```
[F_total]   [  1          1          1          1     ] [f1]
[tau_x  ] = [  d/sqrt2   -d/sqrt2   -d/sqrt2    d/sqrt2 ] [f2]
[tau_y  ]   [  d/sqrt2    d/sqrt2   -d/sqrt2   -d/sqrt2 ] [f3]
[tau_z  ]   [  c_tau     -c_tau      c_tau     -c_tau   ] [f4]
```

Integration: RK4 at 0.001s (1000 Hz), 10 substeps per control cycle.

### 4.2 3D Environment (SDF)

Signed Distance Field for collision and ray tracing:
- **Primitives**: Box, Sphere, Cylinder (analytical SDF formulas)
- **Arena**: 10m x 10m x 3m with boundary walls
- **Default scene**: 5 floor-to-ceiling box pillars (1m x 1m x 3m)
- **Collision**: `SDF(p) < collision_radius (0.05m)` → eliminated
- **Out-of-bounds**: `boundary_sdf(p) < 0` → eliminated

### 4.3 Wind and Noise

**Wind**: Ornstein-Uhlenbeck stochastic process as external force:
```
dF = theta * (mu - F) * dt + sigma * dW
Default: theta=2.0, mu=[0,0,0], sigma=0.3 N
```

**Observation noise**: Gaussian on position observations: `N(0, sigma^2)`, default sigma=0.1m. Applied only when opponent is visible.

---

## 5. Control & Planning (MPPI)

Model Predictive Path Integral — parallel trajectory optimizer:
```
Samples:      K = 1024 trajectories
Horizon:      N = 50 steps (0.5s)
Temperature:  lambda = 10.0
Noise std:    sigma = 0.03 N per motor
```

### Cost Functions

**Pursuit**: distance^2 + facing angle + control smoothness + obstacle avoidance
**Evasion**: maintain distance + exit enemy FOV + control + obstacles
**Belief-weighted**: scales opponent-relative terms by `confidence = 1 / (1 + belief_var)`

### Risk-Aware MPPI (CVaR)

1. Roll out trajectories with sampled wind disturbance
2. Sort costs, find (1-alpha) percentile threshold
3. Add `penalty * (cost - threshold)` to worst-alpha fraction
4. Config: `cvar_alpha=0.05` (worst 5%), `cvar_penalty=10.0`

Implementation: rayon parallel sampling, warm-start with shifted mean.

---

## 6. State Estimation

### 6.1 Extended Kalman Filter

6D state (position + velocity) per drone tracking its opponent:
- **Predict**: constant-velocity model every control step
- **Update**: only when opponent is visible (noisy observation)

### 6.2 Particle Filter (Belief State)

When opponent is occluded, tracks belief about opponent position:
- 200 particles, SDF-constrained prediction
- Update: resample toward observation when visible
- Output: mean position + variance
- Switches: EKF when visible (variance=0), particle filter when occluded

---

## 7. Perception

### 7.1 Camera System

Pinhole camera fixed to drone body frame (+X forward, +Y left, +Z up):
```
Resolution:     320 x 240 px
FOV:            90 deg horizontal
Focal length:   fx = fy = 160 px
Principal point: (160, 120)
Max depth:      15.0 m
Render rate:    30 Hz
```

Depth rendering via sphere tracing against arena SDF + opponent sphere. Parallelized over rows with rayon.

### 7.2 Geometric Detection

Analytical projection of opponent into camera frame:
1. Transform to camera frame, check in front of camera
2. Project to pixel coordinates
3. Verify line-of-sight (not occluded)
4. Compute bounding box from angular extent
5. Confidence: `clamp(1.0 - distance / 5.0, 0, 1)`

Output: `Detection { detected, bbox, confidence, depth, pixel_center }`

### 7.3 Line-of-Sight

Sphere tracing from observer to target through arena SDF. If ray hits obstacle before reaching target → `Occluded`. Used for: lock-on validation, EKF update gating, belief state switching.

---

## 8. RL Training

### 8.1 Gymnasium Environment

Action space: 4-dim continuous [-1, 1] → mapped to motor thrusts.

Two observation modes (see Section 1.4).

### 8.2 Reward Function

```
Terminal:
  +100   locked on opponent
  -100   locked on by opponent
   -50   collision with obstacle/wall
   +50   opponent crashes

Shaping (per step):
  +0.01  survival bonus
  +0.1   * lock_progress_delta
  +0.05  * distance_decrease
  -0.01  * control_cost
  +0.02  * belief_variance_decrease     (info-theoretic)
  -0.005 * time_since_last_seen         (lost contact penalty)
```

### 8.3 Curriculum Training

5 phases defined in `configs/curriculum.toml`:

| Phase | Task | Opponent | Wind | Noise | FPV | Max Steps |
|-------|------|----------|------|-------|-----|-----------|
| 1 | `pursuit_linear` | PD trajectory | 0 | 0 | no | 200K |
| 2 | `pursuit_evasive` | MPPI evasion | 0 | 0 | no | 300K |
| 3 | `search_pursuit` | MPPI evasion | 0 | 0.1 | no | 300K |
| 4 | `dogfight` | Elo pool | 0.3 | 0.1 | no | 2M |
| 5 | `fpv_transfer` | Elo pool | 0.3 | 0.1 | yes | 5M |

Each phase loads the previous phase's weights. Promotion based on win rate thresholds.

**Trajectory opponents** (Phase 1): circle, lemniscate, patrol paths with PD position controller.

**Opponent pool** (Phases 4-5): Elo-rated checkpoint pool. Periodic model snapshots are added to the pool. Opponents sampled proportional to Elo-proximity.

### 8.4 CNN + MLP Policy (FPV)

```
Depth image (1x60x80)
  -> Conv2d(1, 32, 8, stride=4) -> ReLU     # 32x14x19
  -> Conv2d(32, 64, 4, stride=2) -> ReLU    # 64x6x9
  -> Conv2d(64, 64, 3, stride=1) -> ReLU    # 64x4x7
  -> Flatten -> Linear(1792, 128) -> ReLU    # 128-dim

IMU vector (12)
  -> Linear(12, 64) -> ReLU                  # 64-dim

Concat(128, 64) = 192
  -> PPO policy head -> 4 (motor deltas)
  -> PPO value head  -> 1 (state value)
```

Implemented as `CnnImuExtractor(BaseFeaturesExtractor)` for SB3 `MultiInputPolicy`.

---

## 9. Bevy Interactive Visualizer

Standalone binary: `cargo run -p aces-game --release`

### Screen Layout
```
┌─────────────────────────────────────────────┐
│             Main Camera (65%)               │
│     God-view / Follow-cam (Tab to toggle)   │
│     HUD: lock-on %, telemetry, wall warn    │
├──────────────────┬──────────────────────────┤
│  FPV Drone A     │  FPV Drone B             │
│  (cyan)          │  (orange)                │
│       35% height, 50% width each            │
└──────────────────┴──────────────────────────┘
```

### Controls

| Keyboard | Gamepad | Function |
|----------|---------|----------|
| W/S | Left Stick Y | Pitch |
| A/D | Left Stick X | Roll |
| Q/E | Right Stick X | Yaw |
| Shift/Ctrl | RT/LT | Throttle |
| Space | Y | Switch active drone |
| Tab | B | Toggle camera mode |
| P | Start | Pause/resume |
| R | Back | Reset |

### Policy Loading

Trained MLP policies exported via `aces/export.py` as flat binary (little-endian f32 weights). The Bevy game crate loads them at runtime for AI opponent inference. Vector-mode only (21 → 64 → 64 → 4, Tanh activations).

### Coordinate Transform

sim-core uses Z-up `(x, y, z)`. Bevy uses Y-up. Mapping: `sim (x, y, z) → Bevy (x, z, y)`. Quaternion: `(qw, qi, qj, qk) → Quat::from_xyzw(qi, qk, qj, qw)`.

---

## 10. Data Flow

### Simulation Step

```
Python env.step(action)
  -> map action [-1,1] to motor thrusts [0, max_thrust]
  -> compute opponent action (trajectory / MPPI / self-play policy)
  -> Rust Simulation.step(motors_a, motors_b)
       ├── 10x RK4 physics substeps (with wind)
       ├── Lock-on update (FOV + distance + timer + LOS)
       ├── EKF predict (always) + update (if visible)
       ├── Particle filter predict (SDF-constrained) + update (if visible)
       ├── Camera render (if 30Hz interval elapsed)
       │     ├── Sphere-trace depth map (rayon parallel)
       │     └── Geometric detection
       └── Return StepResult
  -> build observation (vector 21-dim OR FPV dict)
  -> compute reward (terminal + shaping + info-theoretic)
  -> return (obs, reward, terminated, truncated, info)
```

### StepResult Fields

| Category | Fields |
|----------|--------|
| State | drone_{a,b}_state(13), forward(3), euler(3) |
| Collision | drone_{a,b}_collision, drone_{a,b}_oob |
| Lock-on | lock_{a,b}_progress, kill_{a,b} |
| Distance | distance, nearest_obs_dist_{a,b} |
| Noise | noisy_pos(3), wind_force(3) per drone |
| EKF | ekf_pos(3), ekf_vel(3) per observer |
| Visibility | {a,b}_sees_{b,a}, time_since_saw |
| Belief | belief_pos(3), belief_var per observer |
| Camera | depth_image, camera_rendered per drone |
| Detection | detected, bbox, confidence, depth, pixel_center |

---

## 11. Configuration

All parameters in `configs/*.toml`. No hardcoded constants.

**drone.toml**: mass, arm length, inertia, max thrust, torque/drag coefficients, dt_sim/dt_ctrl/substeps.

**arena.toml**: bounds (10x10x3), spawn positions, obstacle definitions (center + half_extents), collision radius.

**rules.toml**:
- `[lockon]`: fov_degrees, lock_distance, lock_duration
- `[mppi]`: num_samples, horizon, temperature, noise_std
- `[mppi.weights]`: w_dist, w_face, w_ctrl, w_obs, d_safe
- `[mppi.risk]`: wind_theta/sigma, cvar_alpha/penalty
- `[noise]`: wind OU params, obs_noise_std
- `[camera]`: enabled, width, height, fov_deg, render_hz, max_depth, policy resolution
- `[detection]`: drone_radius, min_confidence_distance
- `[training]`: total_timesteps, learning_rate, batch_size, n_steps, gamma, etc.
- `[reward]`: kill/death/collision values, shaping weights, info-theoretic weights

**curriculum.toml**: `[[phase]]` array with name, task, opponent, noise overrides, FPV flag, max_timesteps, promote_condition.

---

## 12. Commands

```bash
# Build
poetry install
poetry run maturin develop            # Rust extension (debug)
poetry run maturin develop --release  # Rust extension (optimized)

# Test
cargo test                            # 57 Rust tests
pytest tests/ -v                      # 85 Python tests

# Run (MPPI vs MPPI)
python scripts/run.py                                # default
python scripts/run.py --fpv                          # with camera viz
python scripts/run.py --no-noise                     # disable wind + noise

# Train
python scripts/run.py --mode train --timesteps 500000 --save-path aces_model --no-vis
python scripts/run.py --mode train --fpv --save-path aces_fpv --no-vis
python scripts/run.py --mode train --task pursuit_linear --timesteps 200000 --no-vis

# Curriculum
python scripts/run.py --mode curriculum --timesteps 200000,300000,300000,500000

# Headless server training (parallel envs, TensorBoard)
python scripts/train_server.py --n-envs 8

# Evaluate
python scripts/run.py --mode evaluate --model-path models/stage5 --opponent mppi --n-episodes 100

# Export (for Bevy)
python scripts/run.py --mode export --model-path models/stage5 --save-path policy.bin

# Bevy interactive visualizer
cargo run -p aces-game --release
```

### Docker (Headless Training)

```bash
docker build -t aces-train .

# Single-run
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/logs:/app/logs \
    aces-train --mode train --timesteps 500000 --save-path models/aces --no-vis

# Curriculum
docker run --rm -it -v $(pwd)/models:/app/models -v $(pwd)/logs:/app/logs \
    --entrypoint bash aces-train
# then: python scripts/run.py --mode curriculum ...
```

---

## 13. Tech Stack

| Component | Language | Libraries |
|-----------|----------|-----------|
| Quadrotor dynamics | Rust | nalgebra |
| SDF environment | Rust | nalgebra |
| MPPI controller | Rust | rayon, rand |
| State estimation | Rust | nalgebra |
| Camera rendering | Rust | rayon |
| Geometric detection | Rust | nalgebra |
| PyO3 bridge | Rust | pyo3 0.25, maturin |
| Bevy game | Rust | bevy 0.15, serde, toml |
| RL training | Python | stable-baselines3, gymnasium |
| FPV policy | Python | PyTorch |
| Visualization | Python | rerun-sdk |
| Config | TOML | toml/tomllib |
| Build | | maturin + poetry |

---

## 14. Roadmap

```
Level 0  ──>  Level 1  ──>  Level 2  ──>  Level 3  ──>  Level 4
Core Sim      RL Policy     Uncertainty    Info Asym     FPV Vision
   done          done           done          done          done

──>  Level 5  ──>  Level 6  ──>  Level 7
     Curriculum     Bevy Viz      Hardware
       done           done        planned
```

### Level 7 — Hardware (Planned)

- **Option A** (Crazyflie 2.1 + AI deck): ~$600. Ground station computes, radio link.
- **Option B** (Custom Jetson Nano): ~$2000. Onboard compute, PX4/ROS2 stack.
- System identification pipeline
- Sim-to-real transfer (domain randomization, parameter identification)
- Safety infrastructure (geofencing, emergency stop)

### Future Enhancements

- Photorealistic rendering (MuJoCo): RGB+depth, YOLOv8-nano detection, RGBD observation
- Multi-agent (>2 drones)
- Adaptive curriculum (auto-promote on win rate)
- Multiplayer / network play
- Recording / replay system
