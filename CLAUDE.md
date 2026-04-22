# ACES — Air Combat Engagement Simulation

1v1 quadrotor drone dogfight simulation: Rust physics core + Python RL/viz layer.

Full technical reference: [`docs/design.md`](docs/design.md)

## Project Structure

```
aces/
├── crates/                     # Rust workspace (4 crates, ~3600 lines)
│   ├── sim-core/               #   Dynamics, SDF environment, collision, lock-on,
│   │                           #   camera rendering, geometric detection, wind, noise
│   ├── mppi/                   #   MPPI controller (standard + CVaR + belief-weighted)
│   ├── estimator/              #   EKF + particle filter
│   └── py-bridge/              #   PyO3 bindings -> aces._core
├── aces/                       # Python package (~1500 lines)
│   ├── env.py                  #   Gymnasium environment (vector 21-dim / FPV dict)
│   ├── trainer.py              #   Self-play PPO (MLP + CNN policies)
│   ├── policy.py               #   CnnImuExtractor for FPV depth images
│   ├── predictor.py            #   Causal Transformer trajectory prediction
│   └── viz.py                  #   Rerun 3D + depth image visualization
├── configs/                    # TOML configs (drone, arena, rules)
├── scripts/run.py              # CLI: mppi-vs-mppi / train / evaluate (--fpv)
├── tests/                      # 88 tests (38 Rust + 50 Python)
└── docs/design.md              # Consolidated technical reference
```

## Quick Start

```bash
poetry install
poetry run maturin develop      # build Rust extension (debug)
cargo test                      # 38 Rust unit tests
pytest tests/ -v                # 50 Python tests
python scripts/run.py           # launch MPPI-vs-MPPI simulation
python scripts/run.py --fpv     # with first-person depth cameras
```

## Key Commands

```bash
cargo check                     # type-check all Rust crates
cargo test                      # Rust tests (dynamics, SDF, camera, detection, MPPI)
pytest tests/ -v                # Python tests (env, trainer, predictor, viz)
python scripts/run.py --mode train --fpv --timesteps 500000   # train FPV agent
python scripts/run.py --mode train --task pursuit_linear --timesteps 200000  # single curriculum stage
python scripts/run.py --mode curriculum --timesteps 200000,300000,300000,500000  # full curriculum
python scripts/run.py --mode export --model-path aces_model --save-path policy   # export for Bevy
python scripts/run.py --mode evaluate --model-path aces_model # evaluate
```

## Conventions

- Rust code in `crates/`, Python in `aces/`, interop via `crates/py-bridge/`
- Physics: 1000 Hz, control: 100 Hz, camera: 30 Hz
- All parameters in `configs/*.toml`, never hardcoded
- Linting: `ruff` (Python), `cargo clippy` (Rust)
