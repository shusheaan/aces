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
│   └── game/                   #   Bevy 3D interactive visualizer + NN policy loading
├── aces/                       # Python package (10 modules)
│   ├── env.py                  #   Gymnasium environment (vector 21-dim / FPV dict)
│   ├── trainer.py              #   Self-play PPO (MLP + CNN policies)
│   ├── curriculum.py           #   Phase definitions + CurriculumManager
│   ├── opponent_pool.py        #   Elo-rated opponent pool for self-play
│   ├── trajectory.py           #   Circle/lemniscate/patrol for curriculum
│   ├── policy.py               #   CnnImuExtractor for FPV depth images
│   ├── export.py               #   MLP weight -> binary for Bevy inference
│   ├── config.py               #   Typed TOML config loading
│   └── viz.py                  #   Rerun 3D + depth image visualization
├── configs/                    # TOML configs (drone, arena, rules, curriculum)
├── scripts/                    # run.py, train_server.py, pre-commit hooks
├── tests/                      # 142 tests (57 Rust + 85 Python)
└── docs/design.md              # Consolidated technical reference
```

## Quick Start

```bash
poetry install
poetry run maturin develop      # build Rust extension (debug)
cargo test                      # 57 Rust unit tests
pytest tests/ -v                # 85 Python tests
python scripts/run.py           # launch MPPI-vs-MPPI simulation
python scripts/run.py --fpv     # with first-person depth cameras
cargo run -p aces-game --release  # Bevy 3D interactive visualizer
```

## Key Commands

```bash
cargo check                     # type-check all Rust crates
cargo test                      # Rust tests (dynamics, SDF, camera, detection, MPPI)
pytest tests/ -v                # Python tests (env, trainer, curriculum, viz)
python scripts/run.py --mode train --fpv --timesteps 500000   # train FPV agent
python scripts/run.py --mode train --task pursuit_linear --timesteps 200000  # single curriculum stage
python scripts/run.py --mode curriculum --timesteps 200000,300000,300000,500000  # full curriculum
python scripts/run.py --mode export --model-path aces_model --save-path policy   # export for Bevy
python scripts/run.py --mode evaluate --model-path aces_model # evaluate
python scripts/train_server.py --n-envs 8  # headless server training
```

## Conventions

- Rust code in `crates/`, Python in `aces/`, interop via `crates/py-bridge/`
- Physics: 1000 Hz, control: 100 Hz, camera: 30 Hz
- All parameters in `configs/*.toml`, never hardcoded
- Linting: `ruff` (Python), `cargo clippy` (Rust)
