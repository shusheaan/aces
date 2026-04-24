# GPU MPPI — User Guide

## Purpose

The GPU MPPI subsystem runs many parallel quadrotor dogfight simulations on the
GPU so that RL training (SB3 PPO) collects rollouts much faster than the
single-env CPU path. Both the MPPI planner (sample + rollout + softmax) and the
full battle orchestration (physics step, collision, lock-on, observation,
reward) live on the GPU; Python only does the PPO forward/backward pass. This
replaces the Rayon-parallel CPU batch backend for the compute-heavy curriculum
phases.

## Architecture overview

```
SB3 PPO (policy NN)
    |  actions in [-1, 1]
    v
aces.training.GpuVecEnv  (Python SB3 VecEnv wrapper)
    |  denormalized motor thrusts in [0, max_thrust]
    v
aces._core.GpuVecEnv     (PyO3 class, numpy I/O)
    |
    v
GpuBatchOrchestrator     (crates/batch-sim/src/gpu/orchestrator.rs)
    |  buffer uploads + 2 compute dispatches per side + physics step
    v
Rust GPU pipeline         (crates/batch-sim/src/gpu/pipeline.rs)
    |  wgpu 23 / naga 23
    v
WGSL compute shaders      (crates/batch-sim/src/gpu/shaders/)
    |
    v
GPU (Metal / Vulkan / DX12)
```

MPPI is a sampling-based trajectory optimizer. Each tick it generates `N`
noisy control sequences, rolls each one forward `H` timesteps through the full
rigid-body dynamics, scores each rollout against the pursuit / evasion cost,
and returns a softmax-weighted average as the optimal control. The inner loop
is embarrassingly parallel — every sample is independent — which maps naturally
to GPU workgroups. The ACES implementation dispatches `n_drones * n_samples`
rollout threads, then a second kernel (`workgroup_size=256`) performs the
softmax reduction per drone.

## Requirements

- Rust toolchain (stable) and `cargo`
- A GPU adapter with compute support:
  - Apple Silicon via Metal
  - NVIDIA via Vulkan
  - AMD via Vulkan
  - Intel via DX12 (Windows) or Vulkan (Linux)
- Python 3.10+ with `poetry` for the training side
- `maturin` for building the PyO3 extension (`poetry install` brings it in)

The `gpu` feature is opt-in on both `aces-batch-sim` and `aces-py-bridge`. If
no GPU adapter is detected at runtime, construction fails with a clear error
rather than silently falling back.

## Build instructions

Rust-only, with the GPU feature:

```bash
cargo build -p aces-batch-sim --features gpu --release
```

Run the Rust GPU test suite:

```bash
cargo test -p aces-batch-sim --features gpu --release
```

Python extension with GPU (builds `aces._core.GpuBatchMppi` and
`aces._core.GpuVecEnv`):

```bash
poetry install
poetry run maturin develop --features gpu
```

End-to-end PPO smoke test:

```bash
poetry run python scripts/smoke_gpu_ppo.py --timesteps 256 --n-envs 4
```

Full curriculum training on the GPU-backed env:

```bash
poetry run python scripts/run.py --mode curriculum --use-gpu-env --n-envs 16
```

## Hardware check

Before benchmarking or training, confirm a GPU adapter is visible:

```bash
cargo run -p aces-batch-sim --features gpu --example gpu_probe
```

This prints the adapter name, backend (Metal / Vulkan / DX12), device limits,
and a minimal compute-shader round-trip.

## Usage examples

### A. Standalone batched MPPI planner (Python)

`GpuBatchMppi` is the low-level planner — it consumes numpy arrays of state,
enemy state, mean control, and noise, and returns the optimal motor sequence
per drone. Useful for offline rollouts, MPC experiments, or swapping MPPI into
an existing Python loop without the full VecEnv.

```python
from aces._core import GpuBatchMppi
import numpy as np

p = GpuBatchMppi(n_drones=8, n_samples=128, horizon=15)

states = np.zeros((8, 13), dtype=np.float32)
states[:, 9] = 1.0  # identity quaternion (xyzw layout, w last)
# ... fill in positions [0:3], velocity [3:6], angular velocity [10:13] ...

enemies = np.zeros_like(states)         # paired opponent per drone
mean_ctrls = np.full((8, 15, 4), 0.0662, dtype=np.float32)  # hover thrust
noise = np.random.randn(8, 128, 15, 4).astype(np.float32) * 0.03

result = p.compute_batch_actions(states, enemies, mean_ctrls, noise)
# result.shape == (8, 15, 4) — optimal motor thrust sequence per drone
```

The quaternion portion of each state must be unit-length. The GPU kernel does
not renormalize on unpack; drifted inputs silently produce wrong rollouts. See
the `compute_batch_actions` doc-comment in `crates/batch-sim/src/gpu/pipeline.rs`.

### B. Full RL training env (SB3 VecEnv)

```python
from aces.training.gpu_vec_env import GpuVecEnv
from stable_baselines3 import PPO

env = GpuVecEnv(n_envs=16, mppi_samples=128, mppi_horizon=15)
model = PPO("MlpPolicy", env, n_steps=128, batch_size=64)
model.learn(total_timesteps=100_000)
```

Agent A is driven by the PPO policy; agent B is driven by GPU MPPI. Actions
from SB3 are in `[-1, 1]` — `GpuVecEnv` denormalizes them to motor thrusts in
`[0, max_thrust]` before passing them into the Rust side.

## Performance

A single-threaded CPU reference path (`compute_batch_actions_cpu_reference`)
exists for parity tests only; it is deliberately not optimized. Measured on an
Apple M-series laptop (your hardware will differ — run the benchmark):

| Config (drones x samples x horizon) | CPU reference   | GPU (expected)       |
|-------------------------------------|-----------------|----------------------|
| 4 x 32 x 10                         | ~2-3 ms / tick  | sub-ms               |
| 128 x 256 x 30                      | ~1.2 s / tick   | ~1-10 ms             |

GPU speedup at larger configs is typically **100-1000x** vs the CPU reference,
and the gap grows with `n_samples * horizon * n_drones`. Measure on your own
hardware:

```bash
cargo run -p aces-batch-sim --features gpu --example bench_gpu_vs_cpu --release
```

## Caveats / known limitations

1. **Opponent types are restricted.** `GpuVecEnv` only provides
   MPPI-vs-MPPI dogfight semantics. Curriculum phases configured with other
   opponent types (trajectory follower, hover, opponent-pool self-play) are
   **ignored** when `--use-gpu-env` is passed; the trainer logs a warning and
   falls back to MPPI-vs-MPPI for that phase.
2. **f32 precision.** The GPU path uses f32 throughout, whereas the CPU path
   uses f64. The drift is negligible over typical rollout lengths and is
   verified end-to-end by `test_gpu_matches_cpu_reference_parity` in
   `crates/batch-sim/tests/gpu_pipeline.rs`.
3. **Unit-quaternion input contract.** `compute_batch_actions` requires unit
   quaternions in the 13-dim state; the kernel does not renormalize on unpack.
   The orchestrator enforces this via its RK4 post-step normalization.
4. **Per-call staging buffer.** The GPU readback path allocates a staging
   buffer per dispatch. Fine for RL training (few-hundred-Hz control loop); a
   known concern if you ever want to use this for on-drone high-frequency
   MPC control.
5. **CPU-side warm-start shift.** The GPU returns the full `H`-step optimal
   sequence; the CPU side shifts it left by one and appends a hover control
   before the next tick.
6. **Wind in GPU path**: `wind_sigma` has no effect on the GPU MPPI *rollout*
   planner (wind is zero in the compute shaders). The true physics step (applied
   via CPU RK4 in the orchestrator) still applies OU-process wind noise if
   `wind_sigma > 0`. Porting wind to the rollout shaders is future work.

## Testing

| Suite                                                                | Command                                                  |
|----------------------------------------------------------------------|----------------------------------------------------------|
| Rust GPU unit + integration (68 tests; lib + `tests/gpu_pipeline.rs`) | `cargo test -p aces-batch-sim --features gpu`           |
| Python wrapper                                                        | `pytest tests/test_gpu_vec_env.py`                       |
| End-to-end PPO smoke                                                  | `pytest tests/test_gpu_ppo_smoke.py`                     |
| Curriculum integration                                                | `pytest tests/test_curriculum_gpu_opt.py`                |

All Python GPU tests gracefully skip when `aces._core` lacks the `gpu` feature,
so they are safe to run on CI machines without a GPU adapter.

## Status summary

Shipped slices in the GPU MPPI initiative (each merged via its own feature
branch; see `git log --oneline` for SHAs):

| #  | Slice                                             | Landed as                                |
|----|---------------------------------------------------|------------------------------------------|
| 1  | f32 RK4 reference dynamics + parity tests         | `feature/f32-rk4-validation`             |
| 2  | f32 SDF reference (Arena/Obstacle) + parity       | `feature/f32-sdf`                        |
| 3  | f32 pursuit / evasion cost + parity tests         | `feature/f32-cost`                       |
| 4  | GPU pipeline skeleton (buffers + uniform upload)  | `feature/gpu-pipeline-skeleton`          |
| 5  | WGSL helpers (RK4 / SDF / quaternion)             | `feature/wgsl-helpers`                   |
| 6  | `rollout_and_cost` @compute kernel                | `feature/mppi-rollout-kernel`            |
| 7  | `softmax_reduce` @compute kernel                  | `feature/mppi-softmax-kernel`            |
| 8  | BindGroupLayout + 2 ComputePipelines              | `feature/gpu-pipelines`                  |
| 9  | `compute_batch_actions()` GPU dispatch + readback | `feature/gpu-dispatch`                   |
| 10 | CPU reference + GPU-vs-CPU parity test            | `feature/gpu-cpu-parity`                 |
| 11 | `bench_gpu_vs_cpu` example                        | `feature/gpu-bench`                      |
| 12 | PyO3 `GpuBatchMppi` binding                       | `feature/pyo3-gpu`                       |
| 13 | `GpuBatchOrchestrator` end-to-end simulation loop | `feature/orchestrator-gpu-opt`           |
| 14 | PyO3 `GpuVecEnv` + `GpuBatchOrchestrator::reset`  | `feature/gpu-vec-env`                    |
| 15 | `step_with_agent_a_actions` (PPO-vs-MPPI mode)    | `feature/gpu-ppo-mode`                   |
| 16 | SB3-compatible Python `GpuVecEnv` wrapper         | `feature/gpu-sb3-wrapper`                |
| 17 | GPU PPO end-to-end training smoke test            | `feature/gpu-ppo-smoke`                  |
| 18 | `--use-gpu-env` opt-in for `CurriculumTrainer`    | `feature/curriculum-gpu-opt`             |

Phase 2 (WGPU batch MPPI) and Phase 3 (PyO3 integration) from
[`plans/parallel-simulation.md`](../plans/parallel-simulation.md) are both
complete. Phase 4 (full physics on GPU) remains open.
