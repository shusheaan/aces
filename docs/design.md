# ACES — Air Combat Engagement Simulation

1v1 quadrotor drone dogfight simulation with progressive complexity: Rust physics core + Python RL/visualization layer.

Two drones fight in a 3D arena with obstacles. Each drone has a forward-facing camera and must pursue, evade, and lock onto the opponent using first-person vision. Lock-on = keep opponent in FOV cone at close range for 1.5 seconds.

## Architecture Overview

```
crates/ (Rust, via PyO3)             aces/ (Python package)
┌──────────────────────────┐        ┌──────────────────────────┐
│ sim-core                 │        │ env.py                   │
│   dynamics (13-DOF RK4)  │        │   Gymnasium environment  │
│   environment (SDF)      │        │   vector / FPV obs modes │
│   collision + lock-on    │        │   curriculum task param   │
│   camera (depth render)  │◄──────►│ trainer.py               │
│   detection (geometric)  │  PyO3  │   Self-play PPO          │
│   wind + noise           │        │   CurriculumTrainer      │
├──────────────────────────┤        │ trajectory.py            │
│ mppi                     │        │   circle/lemniscate/     │
│   parallel MPPI + CVaR   │        │   patrol for curriculum  │
│   belief-weighted costs  │        │ policy.py                │
├──────────────────────────┤        │   CNN+MLP for depth imgs │
│ estimator                │        │ export.py                │
│   EKF + particle filter  │        │   MLP weight → binary    │
├──────────────────────────┤        │ predictor.py             │
│ game (Bevy)              │        │   Causal Transformer     │
│   3D visualization       │        │ viz.py                   │
│   keyboard/gamepad input │        │   Rerun 3D viewer        │
│   orbit camera + HUD     │        └──────────────────────────┘
│   NN policy inference    │
└──────────────────────────┘
```

## Quick Start

```bash
poetry install
poetry run maturin develop          # build Rust extension
pytest tests/                       # 88 tests (38 Rust + 50 Python)
python scripts/run.py               # launch MPPI-vs-MPPI sim
python scripts/run.py --mode train  # train RL agent
python scripts/run.py --fpv         # first-person vision mode
cargo run -p aces-game --release    # Bevy 3D interactive visualizer
```

## Levels

| Level | Name | Status | What |
|-------|------|--------|------|
| 0 | Core Sim | Done | 13-DOF dynamics, SDF arena, collision, lock-on, MPPI |
| 1 | RL Strategy | Done | Gymnasium env, PPO self-play, reward shaping |
| 2 | Uncertainty | Done | Wind (OU), observation noise, EKF, CVaR-MPPI |
| 3 | Info Asymmetry | Done | Line-of-sight, particle filter, Belief-MPPI, Transformer predictor |
| 4 | FPV Vision | Done | Depth camera (30Hz), geometric detection, CNN policy |
| 5 | Curriculum | Done | pursuit_linear → pursuit_evasive → search_pursuit → dogfight |
| 6 | Bevy Viz | Done | 3D interactive viz, orbit camera, HUD, NN policy loading |
| 7 | Hardware | Planned | Crazyflie / Jetson deployment |

## Tech Stack

- **Rust**: nalgebra, rayon, rand, pyo3 0.25
- **Python**: gymnasium, stable-baselines3, pytorch, rerun-sdk, numpy
- **Build**: maturin + poetry
- **Config**: TOML (`configs/`)

---

## 1. Project Overview

Two quadrotor drones engage in a 1v1 dogfight within a bounded 3D arena with obstacles. Each drone has a forward-facing camera and must use first-person perspective to perceive the environment and opponent — pursuing, evading, and locking on.

The project uses a hybrid **Rust + Python** architecture: Rust handles all performance-critical simulation (physics, collision, rendering), Python handles ML training, visualization, and orchestration. Everything connects through PyO3/maturin.

### 1.1 Design Principles

- **First-person perspective first** — no god-view; simulate real onboard sensor constraints
- **Progressive complexity** — 6 independent levels (0-5), each deliverable on its own
- **Rust-first** — core simulation in Rust for performance; Python only where the ML ecosystem requires it
- **Build first, integrate later** — custom simulator to understand principles before PX4/ROS2

### 1.2 Combat Rules

```
Each drone is equipped with:
  - 1 forward-fixed camera (FOV = 90 deg, resolution 320x240)
  - Camera is fixed to nose direction, rotates with drone attitude

Lock-on conditions:
  1. Opponent is within own camera FOV cone
  2. Opponent distance <= D_lock (default 2.0 m)
  3. Conditions 1 and 2 satisfied continuously for T_lock seconds (default 1.5 s)
  -> Kill confirmed

Elimination conditions:
  - Locked on by opponent (killed)
  - Collision with obstacle or ground
  - Flying out of bounds

Symmetric game: both sides are simultaneously pursuer and evader.
Whoever locks on first wins.
```

Lock-on detection math:

```
Let A = judging party, B = target

Relative position:  d = pos_B - pos_A
Nose direction:     f_A = R(q_A) * [1, 0, 0]

Angle:              theta = arccos( (d . f_A) / ||d|| )

Lock condition:     theta < FOV/2  AND  ||d|| < D_lock
Timer:              lock_timer += dt   if condition continuously satisfied
                    lock_timer = 0     if condition interrupted
Kill:               lock_timer >= T_lock
```

---

## 2. Architecture

```
+---------------------------------------------------------------+
|                      Decision Layer                            |
|   Level 0: MPPI Control (pursuit/evasion cost functions)       |
|   Level 1: RL Policy (PPO self-play via stable-baselines3)     |
|   Level 3: Behavior Prediction (causal Transformer)            |
+---------------------------------------------------------------+
|                    State Estimation Layer                       |
|   Level 2: EKF (position+velocity tracking)                    |
|   Level 2: Particle Filter (belief under occlusion)            |
|   Level 3: Belief-MPPI (confidence-weighted planning)          |
+---------------------------------------------------------------+
|                      Perception Layer                           |
|   Level 3: Line-of-Sight Occlusion (sphere tracing)            |
|   Level 4: Depth Camera (SDF sphere-traced rendering, 30 Hz)   |
|   Level 4: Geometric Detection (analytical projection)          |
+---------------------------------------------------------------+
|                       Control Layer                             |
|   Level 0: Motor-level control (4-dim thrust commands)          |
|   Level 0: MPPI trajectory optimization (rayon parallel)        |
+---------------------------------------------------------------+
|                    Simulation Environment                       |
|   Level 0: 13-DOF Quadrotor Dynamics (RK4 @ 1000 Hz)          |
|   Level 0: 3D Arena + SDF Collision Detection                  |
|   Level 0: Lock-on Rule Engine                                  |
|   Level 2: Wind Disturbance (Ornstein-Uhlenbeck)               |
|   Level 2: Observation Noise (Gaussian)                         |
+---------------------------------------------------------------+
```

### 2.1 Tech Stack

| Module | Language | Libraries | Rationale |
|--------|----------|-----------|-----------|
| Quadrotor dynamics | Rust | nalgebra | Performance, low-level control |
| SDF environment | Rust | nalgebra | Analytical collision, ray tracing |
| MPPI controller | Rust | rayon, rand | Parallel trajectory sampling |
| State estimation | Rust | nalgebra | Real-time EKF and particle filter |
| Camera rendering | Rust | rayon | Parallel sphere-traced depth images |
| Detection | Rust | nalgebra | Geometric projection |
| Rust-Python bridge | Rust | pyo3 0.25, maturin | Expose Rust core to Python |
| RL training | Python | stable-baselines3, gymnasium | PPO self-play |
| FPV policy | Python | PyTorch | CNN+MLP hybrid for depth images |
| Behavior prediction | Python | PyTorch | Causal Transformer |
| 3D visualization | Python | rerun-sdk | Robotics-native viewer |
| Configuration | TOML | toml/tomllib | Human-readable config |

### 2.2 Timing Hierarchy

```
1000 Hz — Physics step (dynamics + wind + collision)
 100 Hz — Control step (policy inference -> motor commands)
  30 Hz — Camera render (sphere-trace depth, run detection)
```

10 RK4 substeps per control step. Camera renders ~every 3.3 control steps; between renders, last frame is reused.

---

## 3. Project Structure

```
aces/
├── Cargo.toml                      # Workspace root (4 crates)
├── pyproject.toml                   # Python project via maturin + poetry
├── CLAUDE.md                        # Dev conventions and quick start
│
├── crates/                          # ── Rust workspace ──
│   ├── sim-core/                    #   Core simulation (1507 lines)
│   │   └── src/
│   │       ├── lib.rs               #     Module registry
│   │       ├── dynamics.rs          #     13-DOF quadrotor, RK4 integrator
│   │       ├── state.rs             #     DroneState (pos, vel, quat, angvel)
│   │       ├── environment.rs       #     Arena + SDF obstacles (Box/Sphere/Cylinder)
│   │       ├── collision.rs         #     SDF collision + line-of-sight (sphere tracing)
│   │       ├── lockon.rs            #     FOV cone + lock-on timer + kill confirmation
│   │       ├── wind.rs              #     Ornstein-Uhlenbeck stochastic wind
│   │       ├── noise.rs             #     Gaussian observation noise
│   │       ├── camera.rs            #     Pinhole camera, sphere-traced depth rendering
│   │       └── detection.rs         #     Geometric opponent detection + bounding box
│   │
│   ├── mppi/                        #   MPPI controller (837 lines)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── sampler.rs           #     Control sequence sampling
│   │       ├── rollout.rs           #     Parallel trajectory simulation
│   │       ├── cost.rs              #     Pursuit/evasion + belief-weighted costs
│   │       └── optimizer.rs         #     Full optimizer: standard + risk-aware + belief
│   │
│   ├── estimator/                   #   State estimation (502 lines)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── ekf.rs              #     Extended Kalman Filter (6D: pos + vel)
│   │       └── particle_filter.rs  #     Particle filter with SDF constraints
│   │
│   └── py-bridge/                   #   PyO3 bindings (798 lines)
│       └── src/
│           └── lib.rs               #     Simulation + MppiController + StepResult
│
├── aces/                            # ── Python package ──
│   ├── __init__.py                  #   Public API exports
│   ├── env.py                       #   Gymnasium environment (vector + FPV modes)
│   ├── trainer.py                   #   Self-play PPO (MLP + CNN policies)
│   ├── curriculum.py                #   CurriculumTrainer (staged training pipeline)
│   ├── policy.py                    #   CnnImuExtractor (CNN+MLP for depth images)
│   ├── export.py                    #   MLP weight → binary for Bevy inference
│   ├── trajectory.py                #   Circle/lemniscate/patrol for curriculum
│   ├── opponent_pool.py             #   Opponent pool for self-play diversity
│   ├── config.py                    #   TOML config loading
│   └── viz.py                       #   Rerun 3D visualization + depth display
│
├── configs/                         # ── TOML configuration ──
│   ├── drone.toml                   #   Crazyflie 2.1 physical parameters
│   ├── arena.toml                   #   Warehouse scene: 10m x 10m x 3m, 5 pillars
│   └── rules.toml                   #   Lock-on, MPPI weights, noise, camera, rewards
│
├── scripts/
│   └── run.py                       #   CLI entry point (mppi-vs-mppi / train / evaluate)
│
├── tests/                           # ── Test suite (50 Python + 38 Rust = 88 tests) ──
│   ├── test_dynamics.py             #   Rust bridge: hover, freefall, collision, EKF, belief
│   ├── test_env.py                  #   Gym env: vector + FPV obs, rewards, camera timing
│   ├── test_trainer.py              #   PPO training: MLP + FPV, self-play, evaluation
│   ├── test_predictor.py            #   Transformer: forward, predict, train convergence
│   └── test_viz.py                  #   Rerun: creation, logging
│
└── docs/
    ├── design.md                    #   ← This file (consolidated reference)
    └── archive/                     #   Historical planning documents
```

**Total codebase: ~5100 lines of Rust + Python source code, 88 tests.**

---

## 4. Level-by-Level Specification

### Level 0 — Core Simulation + Control

**Status: Complete**

#### 4.0.1 Quadrotor Dynamics (13-DOF)

State vector:
```
x = [px, py, pz,         -- Position (world frame, 3)
     vx, vy, vz,         -- Velocity (world frame, 3)
     qw, qx, qy, qz,    -- Attitude quaternion (4, ||q||=1)
     wx, wy, wz]         -- Angular velocity (body frame, 3)
```

Quaternion chosen over Euler angles: no gimbal lock during aggressive maneuvers.

Control input: `u = [f1, f2, f3, f4]` — four motor thrusts in [0, f_max] Newtons.

Equations of motion:
```
p_dot = v
v_dot = [0, 0, -g] + (1/m) * R(q) * [0, 0, F_total] + (1/m) * F_drag + F_wind
q_dot = 0.5 * q ⊗ [0, wx, wy, wz]          (normalize each step)
I * w_dot = tau - w × (I * w)                (gyroscopic term)
```

Motor mixing matrix (X-configuration):
```
[F_total]   [  1           1           1           1    ] [f1]
[tau_x  ] = [  d/√2      -d/√2      -d/√2       d/√2  ] [f2]
[tau_y  ]   [  d/√2       d/√2      -d/√2      -d/√2  ] [f3]
[tau_z  ]   [  c_tau     -c_tau      c_tau      -c_tau  ] [f4]
```

Physical parameters (Crazyflie 2.1):
```
Mass:           m = 0.027 kg
Arm length:     d = 0.04 m
Inertia:        Ixx = Iyy = 1.4e-5, Izz = 2.17e-5 kg*m^2
Max thrust:     f_max = 0.15 N per motor
Torque coeff:   c_tau = 0.005964
Drag coeff:     k_drag = 0.01
Hover thrust:   f_hover = mg/4 ≈ 0.066 N
```

Integration: RK4 at dt_sim = 0.001s (1000 Hz), 10 substeps per control cycle.

#### 4.0.2 3D Environment (SDF-based)

Signed Distance Field for collision and ray tracing:
- **Primitives**: Box, Sphere, Cylinder (analytical SDF formulas)
- **Arena**: 10m x 10m x 3m with boundary walls
- **Default scene** ("Warehouse"): 5 floor-to-ceiling pillars (1m x 1m x 3m)
- **Collision**: `SDF(p) < drone_radius (0.05m)` → eliminated
- **Out-of-bounds**: `boundary_sdf(p) < 0` → eliminated

#### 4.0.3 Lock-on System

- FOV cone check (90 deg) + distance check (2.0m) + continuous timer (1.5s)
- Visibility verification via line-of-sight sphere tracing
- Reset timer if any condition interrupted

#### 4.0.4 MPPI Controller

Model Predictive Path Integral — parallel trajectory optimizer:
```
Samples:      K = 1024 trajectories
Horizon:      N = 50 steps (0.5s)
Temperature:  λ = 10.0
Noise std:    σ = 0.03 N per motor
```

Cost functions:
- **Pursuit**: distance² + facing angle + control smoothness + obstacle avoidance
- **Evasion**: maintain distance + exit enemy FOV + control + obstacles
- **Belief-weighted** (Level 3): scales opponent-relative terms by confidence

Implementation: rayon parallel sampling, warm-start with shifted mean.

#### 4.0.5 Visualization

Rerun 3D viewer: drone positions, heading arrows, trails, obstacles, lock-on progress, belief state markers, particle clouds, depth images, detection overlays.

---

### Level 1 — Strategy Learning (RL)

**Status: Complete**

#### 4.1.1 Gymnasium Environment

Two observation modes:

**Vector mode** (21-dim):
```
own_velocity(3), own_angular_velocity(3),
opponent_relative_position(3), opponent_relative_velocity(3),
own_attitude(roll, pitch, yaw)(3),
nearest_obstacle_distance(1), lock_progress(1),
being_locked_progress(1),
opponent_visible(1), belief_uncertainty(1), time_since_last_seen(1)
```

**FPV mode** (Dict):
```python
{
    "image": Box(0.0, 1.0, shape=(1, 60, 80)),    # normalized depth image
    "vector": Box(-inf, inf, shape=(12,)),          # proprioceptive only
}
```

Action space: 4-dim continuous [-1, 1] → mapped to motor thrusts.

#### 4.1.2 Reward Function

```
Terminal:
  +100   successfully locked on opponent
  -100   locked on by opponent
   -50   collision with obstacle/wall
   +50   opponent crashes

Shaping (per step):
  +0.01  survival bonus
  +0.1   * lock_progress_delta
  +0.05  * distance_decrease
  -0.01  * control_cost
  +0.02  * belief_variance_decrease     (info-theoretic, Level 3)
  -0.005 * time_since_last_seen         (lost contact penalty, Level 3)
```

#### 4.1.3 Self-Play Training

PPO via stable-baselines3:
- Opponent policy periodically mirrors agent weights (configurable interval)
- MLP policy for vector mode, `MultiInputPolicy` + `CnnImuExtractor` for FPV
- Training stats: episode rewards, kill/death rates, mean lengths
- Evaluation: model vs MPPI or random opponent over N episodes

---

### Level 2 — Uncertainty Handling

**Status: Complete**

#### 4.2.1 Wind Disturbance

Ornstein-Uhlenbeck stochastic process applied as external force:
```
dF = θ(μ - F)dt + σ dW
```
Default: θ=2.0 (reversion rate), μ=[0,0,0], σ=0.3 N.

#### 4.2.2 Observation Noise

Gaussian noise on position observations: N(0, σ²), default σ=0.1m.
Only applied when opponent is visible.

#### 4.2.3 Extended Kalman Filter

6D state estimation (position + velocity) per drone tracking its opponent:
- **Predict**: constant-velocity model every control step
- **Update**: only when opponent is visible (via noisy observation)

#### 4.2.4 Risk-Aware MPPI (CVaR)

Conditional Value-at-Risk filtering on MPPI trajectory costs:
1. Roll out trajectories with sampled wind disturbance
2. Sort costs, find (1-α) percentile threshold
3. Add `penalty * (cost - threshold)` to worst-α fraction
4. Penalized costs feed into softmax weighting

Config: `cvar_alpha=0.05` (worst 5%), `cvar_penalty=10.0`.

---

### Level 3 — Information Asymmetry

**Status: Complete**

#### 4.3.1 Line-of-Sight Occlusion

Sphere tracing from observer to target through arena SDF:
- If ray hits obstacle before reaching target → `Visibility::Occluded`
- Used in: lock-on (must see to lock), EKF (no update when occluded), belief state

#### 4.3.2 Particle Filter (Belief State)

When opponent is occluded, tracks belief about opponent position:
- 200 particles, SDF-constrained prediction (particles can't enter obstacles)
- Update: resample toward observation when visible
- Output: mean position + position variance
- Switches: EKF when visible (variance=0), particle filter when occluded

#### 4.3.3 Belief-MPPI

Confidence-weighted cost functions for planning under uncertainty:
```
confidence = 1.0 / (1.0 + belief_variance)
```
- Opponent-relative costs (distance, facing) scaled by confidence
- Obstacle avoidance margin increased: `d_safe * (1 + √variance)`
- When variance is high → conservative, obstacle-avoiding behavior
- When variance is low → normal pursuit/evasion

Exposed as `MppiController.compute_action_with_belief(state, enemy, pursuit, belief_var)`.

#### 4.3.4 Trajectory Prediction (Causal Transformer)

Predicts opponent future positions from observation history:

```
Architecture:
  Input: (seq_len, 6) — relative position + velocity history
  -> Linear projection to d_model=64
  -> Sinusoidal positional encoding
  -> Causal Transformer encoder (2 layers, 4 heads, Pre-LN)
  -> Last token -> MLP -> (pred_steps, 3) future position offsets

Training:
  - Collect rollout data from environment
  - Sliding window subsequences (length 20, stride 10)
  - MSE loss on future positions
  - AdamW optimizer
```

Online usage: `OpponentPredictor` maintains sliding window buffer, returns predictions after ≥2 observations.

#### 4.3.5 Information-Theoretic Rewards

Encourage exploration and maintaining visual contact:
- `info_gain_reward`: reward for reducing belief uncertainty (belief_var decreases)
- `lost_contact_penalty`: per-second penalty for time without visual contact

---

### Level 4 — First-Person Vision

**Status: Complete**

#### 4.4.1 Camera System

Pinhole camera model fixed to drone body frame (+X forward, +Y left, +Z up):
```
Resolution:       320 x 240 pixels
FOV:              90 degrees horizontal
Focal length:     fx = fy = 160 px
Principal point:  (cx, cy) = (160, 120)
Max depth:        15.0 m
Render rate:      30 Hz
```

Ray generation per pixel (u, v):
```
direction_body = normalize([1.0, -(u - cx) / fx, -(v - cy) / fy])
direction_world = R(q_drone) * direction_body
```

#### 4.4.2 Depth Rendering (Sphere Tracing)

Reuses the arena SDF infrastructure. Per ray:
1. Start at drone position
2. Evaluate combined SDF: arena geometry + opponent (sphere, r=0.05m)
3. Advance by max(SDF, epsilon) along ray
4. Terminate at hit (SDF < 0.001m) or max_depth (15m)
5. Record hit distance as depth value

Opponent naturally occluded by closer obstacles. Parallelized over image rows with rayon.

Output: `CameraFrame { depth: Vec<f32>, width, height, timestamp }`

#### 4.4.3 Geometric Detection

Analytical projection of opponent into camera frame (no CNN):
1. Transform opponent position to camera frame
2. Check if in front of camera (d_cam.x > 0)
3. Project to pixel: `u = fx * (-d_cam.y / d_cam.x) + cx`
4. Check image bounds
5. Verify line-of-sight (not occluded)
6. Compute bounding box from angular extent at distance
7. Confidence: `clamp(1.0 - distance / 5.0, 0, 1)`

Output: `Detection { detected, bbox, confidence, depth, pixel_center }`

#### 4.4.4 FPV Observation Space

```python
observation_space = Dict({
    "image": Box(0.0, 1.0, shape=(1, 60, 80)),   # 4x downsampled, normalized
    "vector": Box(-inf, inf, shape=(12,)),         # IMU-only, no opponent info
})
```

Vector (12-dim): velocity(3), angular_velocity(3), attitude(3), lock_progress(1), being_locked(1), nearest_obstacle(1).

Removed from vector (must be inferred from camera): opponent relative position/velocity, visibility, belief.

Image processing: 320x240 → 80x60 via 4x4 average pooling → normalize by max_depth.

#### 4.4.5 CNN + MLP Policy Network

```
Depth image (1x60x80)
  -> Conv2d(1, 32, 8, stride=4) -> ReLU    # -> 32x14x19
  -> Conv2d(32, 64, 4, stride=2) -> ReLU   # -> 64x6x9
  -> Conv2d(64, 64, 3, stride=1) -> ReLU   # -> 64x4x7
  -> Flatten -> Linear(1792, 128) -> ReLU   # -> 128-dim

IMU vector (12)
  -> Linear(12, 64) -> ReLU                 # -> 64-dim

Concat(128, 64) = 192
  -> PPO policy head: Linear -> ... -> 4    # motor deltas
  -> PPO value head:  Linear -> ... -> 1    # state value
```

Implemented as `CnnImuExtractor(BaseFeaturesExtractor)` for stable-baselines3.

#### 4.4.6 Future: Photorealistic Rendering

When adding MuJoCo:
1. Mirror arena geometry in MJCF XML
2. Replace sphere tracer with `mj_render` (RGB + depth)
3. Image channels: 1 → 4 (RGBD)
4. Replace geometric detection with YOLOv8-nano CNN
5. `Detection` interface stays the same

Rust sphere tracer remains available as fast mode / fallback.

---

### Level 5 — Real Hardware (Planned)

**Status: Not implemented**

- **Option A** (low cost): Crazyflie 2.1 + AI deck
- **Option B** (high performance): Custom Jetson Nano drone
- System identification pipeline
- Sim-to-real transfer validation
- ROS2 integration for Option B

---

## 5. Data Flow

### 5.1 Simulation Step

```
Python env.step(action)
  -> map action [-1,1] to motor thrusts [0, max_thrust]
  -> compute opponent action (random / MPPI / self-play policy)
  -> Rust Simulation.step(motors_a, motors_b)
       ├── 10x RK4 physics substeps (with wind)
       ├── Lock-on update (FOV + distance + timer + visibility)
       ├── EKF predict (always) + update (if visible)
       ├── Particle filter predict (SDF-constrained) + update (if visible)
       ├── Camera render (if 30Hz interval elapsed)
       │     ├── Sphere-trace depth map (rayon parallel)
       │     └── Geometric detection
       └── Return StepResult (73+ fields)
  -> build observation (vector 21-dim OR FPV dict)
  -> compute reward (terminal + shaping + info-theoretic)
  -> return (obs, reward, terminated, truncated, info)
```

### 5.2 StepResult Fields

| Category | Fields |
|----------|--------|
| State | drone_{a,b}_state(13), forward(3), euler(3) |
| Collision | drone_{a,b}_collision, drone_{a,b}_oob |
| Lock-on | lock_{a,b}_progress, kill_{a,b} |
| Distance | distance, nearest_obs_dist_{a,b} |
| Noise | noisy_{b,a}_pos_from_{a,b}(3), wind_force_{a,b}(3) |
| EKF | ekf_{b,a}_pos_from_{a,b}(3), ekf_{b,a}_vel_from_{a,b}(3) |
| Visibility | {a,b}_sees_{b,a}, time_since_{a,b}_saw_{b,a} |
| Belief | belief_{b,a}_pos_from_{a,b}(3), belief_{b,a}_var_from_{a,b} |
| Camera | depth_image_{a,b}, camera_rendered_{a,b} |
| Detection | det_{a,b}_{detected, bbox, confidence, depth, pixel_center} |

---

## 6. Configuration

All parameters live in `configs/*.toml`. No hardcoded constants.

**drone.toml**: mass, arm length, inertia, max thrust, torque/drag coefficients, simulation dt/substeps.

**arena.toml**: bounds (10x10x3), spawn positions, obstacle definitions (center + half_extents), drone collision radius.

**rules.toml**:
- `[lockon]`: fov_degrees, lock_distance, lock_duration
- `[mppi]`: num_samples, horizon, temperature, noise_std
- `[mppi.weights]`: w_lock, w_dist, w_face, w_vel, w_ctrl, w_obs, d_safe
- `[mppi.risk]`: wind_theta/sigma, cvar_alpha/penalty
- `[training]`: total_timesteps, learning_rate, batch_size, n_steps, gamma, etc.
- `[noise]`: wind OU params, obs_noise_std
- `[camera]`: width, height, fov_deg, render_hz, max_depth, policy resolution
- `[detection]`: drone_radius, min_confidence_distance
- `[reward]`: kill/death/collision values, shaping weights, info-theoretic weights

---

## 7. Commands

```bash
# Build
poetry install                        # Python dependencies
poetry run maturin develop            # Build Rust extension (debug)
poetry run maturin develop --release  # Build Rust extension (optimized)

# Test
cargo test                            # 38 Rust unit tests
poetry run pytest tests/ -v           # 50 Python tests

# Run
python scripts/run.py                                    # MPPI vs MPPI (default)
python scripts/run.py --fpv                              # MPPI vs MPPI with camera viz
python scripts/run.py --no-noise                         # Disable wind + obs noise

# Train
python scripts/run.py --mode train --timesteps 500000 --save-path aces_model --no-vis
python scripts/run.py --mode train --timesteps 500000 --fpv --save-path aces_fpv --no-vis
python scripts/run.py --mode train --task pursuit_linear --timesteps 200000 --no-vis

# Curriculum (full pipeline)
python scripts/run.py --mode curriculum --timesteps 200000,300000,300000,500000

# Evaluate
python scripts/run.py --mode evaluate --model-path models/stage5 --opponent mppi --n-episodes 100 --no-vis

# Export (for Bevy game)
python scripts/run.py --mode export --model-path models/stage5 --save-path policy.bin

# Bevy interactive visualizer
cargo run -p aces-game --release
```

---

## 8. Training Guide

### 8.1 Single-Run Self-Play

```bash
python scripts/run.py --mode train --timesteps 500000 --save-path aces_model --no-vis
python scripts/run.py --mode train --timesteps 500000 --fpv --save-path aces_fpv --no-vis
```

### 8.2 Curriculum Training (4 Stages)

Train incrementally — each stage loads the previous stage's weights. All stages share the same 21-dim (or FPV Dict) observation space so weights transfer directly.

| Stage | Task | Opponent | Objective | Suggested steps |
|-------|------|----------|-----------|-----------------|
| 2 | `pursuit_linear` | PD controller on circle/lemniscate/patrol | Basic flight + pursuit + lock-on | 200K |
| 3 | `pursuit_evasive` | MPPI evasion mode | Adversarial pursuit | 300K |
| 4 | `search_pursuit` | MPPI evasion, occluded spawn | Search + info gain + pursuit | 300K |
| 5 | `dogfight` | Self-play / MPPI pursuit | Full attack + defense | 500K |

```bash
python scripts/run.py --mode train --task pursuit_linear  --timesteps 200000 --save-path models/stage2 --no-vis
python scripts/run.py --mode train --task pursuit_evasive --timesteps 300000 --save-path models/stage3 --no-vis
python scripts/run.py --mode train --task search_pursuit  --timesteps 300000 --save-path models/stage4 --no-vis
python scripts/run.py --mode train --task dogfight        --timesteps 500000 --save-path models/stage5 --no-vis
```

### 8.3 Determining Training Completion

1. **Timestep budget**: `model.learn(total_timesteps=N)` returns when done.
2. **Episode logs**: check `logs/train_*/episodes.csv` (columns: episode, timestep, reward, length, kill, death, crash, lock_progress, distance).
3. **Post-stage evaluation**: run `--mode evaluate` after each stage and check win_rate.

Suggested per-stage completion thresholds:

| Stage | Target |
|-------|--------|
| 2 — pursuit_linear | win_rate > 80% vs PD opponent |
| 3 — pursuit_evasive | win_rate > 50% vs MPPI evasion |
| 4 — search_pursuit | win_rate > 40% vs MPPI evasion (occluded) |
| 5 — dogfight | win_rate > 30% vs MPPI pursuit |

### 8.4 Evaluation

```bash
python scripts/run.py --mode evaluate --model-path models/stage5 --opponent mppi --n-episodes 100 --no-vis
```

### 8.5 Policy Export (for Bevy Game)

```bash
python scripts/run.py --mode export --model-path models/stage5 --save-path policy.bin
```

Exports MLP weights to a flat binary that the Bevy game crate loads at runtime. Vector-mode only (CNN policy not supported).

---

## 9. Docker (Headless Server Training)

A multi-stage Dockerfile compiles the Rust extension in release mode, installs CPU-only PyTorch, and skips the Bevy game crate.

### 9.1 Build

```bash
docker build -t aces-train .
```

### 9.2 Train

```bash
# Single-run self-play
docker run --rm \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    aces-train --mode train --timesteps 500000 \
        --save-path models/aces_model --no-vis

# FPV mode with CNN policy
docker run --rm \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    aces-train --mode train --timesteps 500000 \
        --fpv --save-path models/aces_fpv --no-vis
```

### 9.3 Curriculum Training (Interactive)

```bash
docker run --rm -it \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    --entrypoint bash \
    aces-train
```

Then inside the container:

```bash
python scripts/run.py --mode train --task pursuit_linear  --timesteps 200000 --save-path models/stage2 --no-vis
python scripts/run.py --mode evaluate --model-path models/stage2 --opponent mppi --n-episodes 50 --no-vis
# check win_rate, then proceed to next stage
python scripts/run.py --mode train --task pursuit_evasive --timesteps 300000 --save-path models/stage3 --no-vis
python scripts/run.py --mode train --task search_pursuit  --timesteps 300000 --save-path models/stage4 --no-vis
python scripts/run.py --mode train --task dogfight        --timesteps 500000 --save-path models/stage5 --no-vis
```

### 9.4 Evaluate & Export

```bash
# Evaluate
docker run --rm \
    -v $(pwd)/models:/app/models \
    aces-train --mode evaluate \
        --model-path models/stage5 --opponent mppi \
        --n-episodes 100 --no-vis

# Export
docker run --rm \
    -v $(pwd)/models:/app/models \
    aces-train --mode export \
        --model-path models/stage5 --save-path models/policy.bin
```

### 9.5 Volume Mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `$(pwd)/models` | `/app/models` | Saved models and exported policies |
| `$(pwd)/logs` | `/app/logs` | Episode CSVs for offline analysis |

Models and logs persist on the host across container restarts.

---

## 10. Roadmap

```
Level 0  ──>  Level 1  ──>  Level 2  ──>  Level 3  ──>  Level 4  ──>  Level 5
Core Sim      RL Policy     Uncertainty    Info Asym     FPV Vision    Hardware
   ✅            ✅             ✅             ✅            ✅            ⬜

Each level is independently deliverable.
Stop at any level and still have a complete project.
```
