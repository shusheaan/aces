# ACES — Air Combat Engagement Simulation

1v1 quadrotor drone dogfight simulation: Rust physics core + Python RL/viz layer + Bevy 3D game.

Full technical reference: [`docs/design.md`](docs/design.md)

## Project Structure

```
aces/
├── crates/                     # Rust workspace (5 crates, 30 source files)
│   ├── sim-core/               #   Dynamics, SDF, collision, lock-on, camera, detection, wind, noise
│   ├── mppi/                   #   MPPI controller (standard + CVaR + belief-weighted)
│   ├── estimator/              #   EKF + particle filter
│   ├── py-bridge/              #   PyO3 bindings -> aces._core
│   ├── batch-sim/              #   Parallel battles (Rayon) + GPU MPPI
│   │   ├── src/battle.rs       #     BattleState, BatchConfig, SpawnMode
│   │   ├── src/observation.rs  #     21-dim obs builder (matches CPU env)
│   │   ├── src/reward.rs       #     Canonical shaped reward (matches CPU env)
│   │   ├── src/f32_{dynamics,sdf,cost}.rs  # f32 CPU reference (GPU parity baseline)
│   │   └── src/gpu/            #     GPU MPPI: pipeline, orchestrator, WGSL shaders
│   └── game/                   #   Bevy 3D interactive visualizer + NN policy loading
├── aces/                       # Python package (subpackages)
│   ├── config.py               #   Typed TOML config loading
│   ├── curriculum.py           #   Phase definitions + CurriculumManager
│   ├── viz.py                  #   Rerun 3D + depth image visualization
│   ├── env/                    #   Environment subpackage
│   │   ├── dogfight.py         #     Gymnasium environment (vector 21-dim / FPV dict)
│   │   ├── obs_layout.py       #     Observation-index constants + describe_obs helper
│   │   ├── ns_env.py           #     Neural-symbolic environment wrapper
│   │   └── trajectory.py       #     Circle/lemniscate/patrol for curriculum
│   ├── training/               #   Training subpackage
│   │   ├── self_play.py        #     Self-play PPO trainer
│   │   ├── curriculum_trainer.py #   Curriculum-based multi-phase trainer (supports --use-gpu-env)
│   │   ├── callbacks.py        #     SB3 training callbacks
│   │   ├── evaluate.py         #     Model evaluation utilities
│   │   ├── gpu_vec_env.py      #     SB3 VecEnv wrapper over aces._core.GpuVecEnv
│   │   ├── opponent_pool.py    #     Elo-rated opponent pool
│   │   ├── batched_vec_env.py  #     Batched opponent inference VecEnv
│   │   └── logging.py          #     Structured logging + run metadata
│   ├── policy/                 #   Policy subpackage
│   │   ├── extractors.py       #     CnnImuExtractor for FPV depth images
│   │   ├── constrained_ppo.py  #     Lagrangian PPO for constraint handling
│   │   └── export.py           #     MLP weight -> binary for Bevy inference
│   └── perception/             #   Perception subpackage
│       ├── neural_symbolic.py  #     Neural-symbolic MPPI policy
│       ├── oracle.py           #     God Oracle ground truth labels
│       └── perception_net.py   #     Supervised perception network
├── configs/                    # TOML configs (drone, arena, rules, curriculum)
├── scripts/                    # run.py, train_server.py, check_gpu_setup.sh, pre-commit hooks
├── tests/                      # ~149 Rust + ~187 Python test fns (25 modules)
└── docs/
    ├── design.md                           # Consolidated technical reference
    ├── gpu-mppi.md                         # GPU MPPI user guide
    └── 2026-04-24-session-archive.md       # GPU MPPI architecture + consistency-audit findings
```

## Quick Start

```bash
poetry install
poetry run maturin develop              # build Rust extension (debug)
poetry run maturin develop --features gpu  # build with GPU MPPI support
cargo test --workspace                  # Rust unit tests (all crates)
cargo test -p aces-batch-sim --features gpu  # GPU feature tests
pytest tests/ -v                        # Python tests (GPU tests auto-skip if no feature)
python scripts/run.py                   # launch MPPI-vs-MPPI simulation
python scripts/run.py --fpv             # with first-person depth cameras
cargo run -p aces-game --release        # Bevy 3D interactive visualizer
bash scripts/check_gpu_setup.sh         # end-to-end GPU setup validator
```

## Key Commands

```bash
cargo check                     # type-check all Rust crates
cargo test                      # Rust tests (dynamics, SDF, camera, detection, MPPI)
pytest tests/ -v                # Python tests (env, trainer, curriculum, viz)
python scripts/run.py --mode train --fpv --timesteps 500000   # train FPV agent
python scripts/run.py --mode train --task pursuit_linear --timesteps 200000  # single curriculum stage
python scripts/run.py --mode curriculum --timesteps 200000,300000,300000,500000  # full curriculum
poetry run python scripts/run.py --mode curriculum --use-gpu-env --n-envs 16  # GPU-batched curriculum training (see docs/gpu-mppi.md)
python scripts/run.py --mode export --model-path aces_model --save-path policy   # export for Bevy
python scripts/run.py --mode evaluate --model-path aces_model # evaluate
python scripts/train_server.py --n-envs 8  # headless server training
```

## Conventions

- Rust code in `crates/`, Python in `aces/`, interop via `crates/py-bridge/`
- Physics: 1000 Hz, control: 100 Hz, camera: 30 Hz
- All parameters in `configs/*.toml`, never hardcoded
- Linting: `ruff` (Python), `cargo clippy` (Rust)
- Commit messages: one-line summary, no signatures or co-author trailers

## References

- [`docs/design.md`](docs/design.md) — consolidated technical reference
- [`docs/gpu-mppi.md`](docs/gpu-mppi.md) — GPU MPPI user guide (build,
  examples, caveats)
- [`docs/2026-04-24-session-archive.md`](docs/2026-04-24-session-archive.md)
  — GPU MPPI (Phase 2 + 3) architecture + 7 CPU/GPU consistency audits
  + remaining divergences + resume guide
- [`plans/parallel-simulation.md`](plans/parallel-simulation.md) — GPU
  MPPI slice index + Phase 4 sketch
- [`plans/curriculum-architecture.md`](plans/curriculum-architecture.md) —
  curriculum + crate dependency reference
