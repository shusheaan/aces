# Parallel Simulation Architecture

Batch MPPI-vs-MPPI self-play with Rayon CPU parallelism and WGPU GPU acceleration.

---

## 1. Motivation

Current training bottleneck: single-environment stepping at ~1K env-steps/s (CPU MPPI).
With 8 SubprocVecEnv workers, throughput reaches ~8K env-steps/s — still slow for the
multi-million-step curriculum. The physics and MPPI are already in Rust; what's missing is
**batching multiple independent battles** and optionally offloading trajectory rollouts to GPU.

**Target**: 64+ concurrent dogfights at >100K env-steps/s (Rayon) or >1M env-steps/s (WGPU).

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│              BatchOrchestrator (Rust)                      │
│              manages N concurrent dogfights                │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Phase 1: MPPI Action Selection                     │  │
│  │                                                    │  │
│  │  Rayon: par_iter over N battles × 2 drones         │  │
│  │  Each drone: MppiOptimizer.compute_action()        │  │
│  │    └─ 1024 sample trajectories (Rayon inner)       │  │
│  │    └─ 50-step horizon × 10 substeps RK4            │  │
│  │    └─ Softmax weighting → optimal motors           │  │
│  │                                                    │  │
│  │  GPU alternative: single dispatch for all drones   │  │
│  │    └─ N×2×1024 parallel rollouts on WGPU           │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Phase 2: Physics Step (Rayon)                      │  │
│  │                                                    │  │
│  │  For each battle (parallel):                       │  │
│  │    1. Apply selected motors to both drones         │  │
│  │    2. RK4 physics (10 substeps at 1ms)             │  │
│  │    3. Wind disturbance (Ornstein-Uhlenbeck)        │  │
│  │    4. Collision detection (SDF)                     │  │
│  │    5. Lock-on tracking (FOV + range + LOS)         │  │
│  │    6. Terminal condition check                      │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Phase 3: Observation & Episode Management          │  │
│  │                                                    │  │
│  │  Build 21-dim observation vectors                  │  │
│  │  Compute shaped rewards (approach, lock, survival) │  │
│  │  Reset terminated battles to random spawns         │  │
│  │  Return Vec<StepResult> to Python                  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Data Structures

### BattleState

```rust
pub struct BattleState {
    state_a: DroneState,        // 13-dim: pos, vel, quat, angvel
    state_b: DroneState,
    wind_a: WindModel,          // OU process per drone
    wind_b: WindModel,
    lockon_a: LockOnTracker,    // A targeting B
    lockon_b: LockOnTracker,    // B targeting A
    step_count: u32,            // episode step counter
    done: bool,                 // episode terminated
}
```

### StepResult

```rust
pub struct StepResult {
    obs_a: [f64; 21],          // observation for drone A
    obs_b: [f64; 21],          // observation for drone B
    motors_a: [f64; 4],        // MPPI-selected actions
    motors_b: [f64; 4],
    reward_a: f64,             // shaped reward
    reward_b: f64,
    done: bool,
    info: BattleInfo,          // kill/collision/timeout flags
}
```

### BatchOrchestrator

```rust
pub struct BatchOrchestrator {
    battles: Vec<BattleState>,
    optimizers_a: Vec<MppiOptimizer>,  // one per battle
    optimizers_b: Vec<MppiOptimizer>,
    arena: Arena,
    params: DroneParams,
    config: BatchConfig,
}
```

---

## 4. Parallelism Strategy

### Phase 1: Rayon (CPU)

Two-level parallelism via Rayon's work-stealing pool:

- **Outer level**: `par_iter` over N battles
- **Inner level**: Each MppiOptimizer uses `par_iter` over 1024 samples

Rayon handles nested parallelism correctly — the work-stealing scheduler
distributes work items across all available cores without oversubscription.

For N=64 battles × 2 drones × 1024 samples = 131,072 work items total.
On 8 cores, each core processes ~16,384 trajectories.

### Phase 2: WGPU (GPU, optional)

Replace inner MPPI sampling+rollout with GPU compute shaders:

```
Dispatch (N_DRONES * N_SAMPLES, 1, 1)  // 131,072 workgroups

Per workgroup (1 thread):
  1. Read initial state for this drone
  2. Read mean_controls + noise perturbation
  3. RK4 rollout: 50 steps × 10 substeps = 500 integrations
  4. Accumulate cost (pursuit/evasion + obstacle SDF)
  5. Write cost to output buffer

Second dispatch (N_DRONES, 1, 1):  // 128 workgroups
  1. Softmax reduction over 1024 costs
  2. Weighted average of control sequences
  3. Write optimal motors to output buffer
```

### GPU Memory Layout

```
Storage buffers (persistent, reused every frame):
  states:       N_DRONES × 13 × f32 = 128 × 52 bytes
  enemies:      N_DRONES × 13 × f32
  mean_ctrls:   N_DRONES × HORIZON × 4 × f32
  noise:        N_DRONES × N_SAMPLES × HORIZON × 4 × f32  // largest: ~100MB for 128×1024×50
  costs:        N_DRONES × N_SAMPLES × f32
  ctrls_out:    N_DRONES × N_SAMPLES × HORIZON × 4 × f32
  result:       N_DRONES × HORIZON × 4 × f32

Uniform buffers (updated once):
  params:       DroneParams struct
  arena:        bounds + obstacles array
  weights:      CostWeights struct
```

---

## 5. Precision: f32 vs f64

The GPU compute path uses f32 (GPU f64 is not universally available and 16-32x slower
where it is). Validation shows f32 is adequate for the MPPI planning rollout:

- dt_sim = 0.001s: small enough for f32 RK4 stability
- Quaternion norm drift: correctable with periodic renormalization
- SDF distances: 0.05m drone radius provides sufficient margin
- MPPI is inherently noise-tolerant (sampling + softmax averaging)

The true physics step (Phase 2) remains in f64 on CPU for accuracy.

---

## 6. Platform Compatibility

| Platform | WGPU Backend | Shader Language | Status |
|----------|-------------|-----------------|--------|
| macOS (Apple Silicon) | Metal | WGSL → MSL | Primary target |
| Linux + NVIDIA | Vulkan | WGSL → SPIR-V | Supported |
| Linux + AMD | Vulkan | WGSL → SPIR-V | Supported |
| Windows + NVIDIA/AMD | Vulkan/DX12 | WGSL → SPIR-V/DXIL | Supported |

WGPU abstracts all backends behind a single API. The WGSL shader compiles to
platform-native shading languages at runtime via the Naga compiler.

---

## 7. Implementation Phases

### Phase 1: Rayon BatchOrchestrator (this PR)

- `crates/batch-sim/` new crate
- `BattleState`, `StepResult`, `BatchOrchestrator`
- Rayon parallel `step_all()` with nested MPPI parallelism
- Observation builder (21-dim vector)
- Reward computation (shaped: approach, lock, survival, collision)
- Episode management (reset on terminal)
- Unit tests + benchmark vs serial baseline

### Phase 2: WGPU Batch MPPI

- `GpuBatchMppi` struct wrapping wgpu device/queue/pipelines
- WGSL compute shader: RK4 dynamics + SDF cost + softmax reduction
- Async readback via staging buffer
- Feature-gated: `cargo build --features gpu`
- Benchmark: GPU vs Rayon throughput comparison

### Phase 3: PyO3 Integration

- Expose `BatchOrchestrator` to Python via `crates/py-bridge/`
- VecEnv-compatible interface for stable-baselines3
- Replace SubprocVecEnv with direct Rust batch stepping

### Phase 4: Full GPU Pipeline

- Physics step on GPU (RK4 + collision + lock-on)
- Observation construction on GPU
- CPU only handles: PPO buffer collection, policy updates, episode reset

---

## 8. Performance Projections

| Configuration | Env-steps/s | Relative |
|---------------|-------------|----------|
| Current (1 env, CPU MPPI) | ~1K | 1x |
| Current (8 SubprocVecEnv) | ~8K | 8x |
| Phase 1: 64 battles Rayon | ~100-200K | 100-200x |
| Phase 2: 64 battles WGPU | ~1-2M | 1000-2000x |
| Phase 4: Full GPU | ~5-10M | 5000-10000x |

Phase 1 alone makes the full 8M-step curriculum trainable in minutes
instead of hours. Phase 2+ enables rapid hyperparameter sweeps.

---

## 9. Crate Structure

```
crates/batch-sim/
├── Cargo.toml
└── src/
    ├── lib.rs              # Public API
    ├── battle.rs           # BattleState, BattleInfo, episode management
    ├── orchestrator.rs     # BatchOrchestrator, step_all()
    ├── observation.rs      # 21-dim observation builder
    ├── reward.rs           # Shaped reward computation
    └── gpu/                # Feature-gated GPU acceleration
        ├── mod.rs
        ├── pipeline.rs     # WGPU compute pipeline setup
        ├── adapter.rs      # Hardware detection & capability reporting
        └── shaders/
            └── mppi.wgsl   # MPPI rollout + cost + reduction shaders
```

---

## 10. Integration with Curriculum

The batch orchestrator replaces the Python environment loop for MPPI-vs-MPPI phases.
For RL training phases (PPO policy vs MPPI), the orchestrator provides observations
and rewards while an external policy selects drone A's actions:

```
Curriculum Phase        Agent A         Agent B         Orchestrator Role
──────────────────────────────────────────────────────────────────────
hover                   PPO             none            N/A (single drone)
pursuit_linear          PPO             trajectory      step + obs + reward
pursuit_evasive         PPO             MPPI evasion    step + obs + reward
search_pursuit          PPO             MPPI evasion    step + obs + reward
self_play_noisy         PPO             MPPI/pool       step + obs + reward
fpv_transfer            PPO (CNN)       pool            step + obs + reward (FPV)
MPPI benchmark          MPPI            MPPI            full self-contained
```

For PPO training: Python calls `orchestrator.step_with_actions(agent_a_actions)`
where agent A's motors come from the policy network, and agent B's come from MPPI.
