"""Benchmark PPO training throughput on GpuVecEnv vs CPU VecEnv.

Runs a few hundred PPO timesteps at each config, measures wall-clock time,
reports steps/sec. Not a convergence benchmark - purely throughput.

Usage:
    python scripts/bench_training_throughput.py [--configs small medium ...]
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    name: str
    n_envs: int
    mppi_samples: int
    mppi_horizon: int
    total_timesteps: int


CONFIGS = [
    Config("small", 4, 32, 10, 512),
    Config("medium", 16, 64, 15, 1024),
    Config("large", 64, 128, 20, 2048),
]


def gpu_available() -> bool:
    try:
        from aces._core import GpuVecEnv  # noqa: F401

        return True
    except ImportError:
        return False


def time_gpu(cfg: Config) -> Optional[float]:
    """Return timesteps/sec or None if unavailable."""
    try:
        from stable_baselines3 import PPO

        from aces.training.gpu_vec_env import GpuVecEnv
    except ImportError:
        return None

    try:
        env = GpuVecEnv(
            n_envs=cfg.n_envs,
            mppi_samples=cfg.mppi_samples,
            mppi_horizon=cfg.mppi_horizon,
        )
    except RuntimeError:
        return None

    try:
        n_steps = max(32, cfg.total_timesteps // cfg.n_envs // 4)
        batch_size = min(64, n_steps * cfg.n_envs)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=2,
            verbose=0,
            device="cpu",
        )

        # Warm-up: 1 short learn() so JIT/graph builds, caches warm up
        model.learn(total_timesteps=n_steps * cfg.n_envs, progress_bar=False)

        # Timed run
        t0 = time.time()
        model.learn(total_timesteps=cfg.total_timesteps, progress_bar=False)
        elapsed = time.time() - t0

        return cfg.total_timesteps / elapsed
    finally:
        env.close()


def format_row(cfg: Config, gpu_rate: Optional[float]) -> str:
    gpu_str = f"{gpu_rate:>10.1f}" if gpu_rate is not None else f"{'n/a':>10}"
    return (
        f"{cfg.name:<8} n_envs={cfg.n_envs:<4} "
        f"samples={cfg.mppi_samples:<4} horizon={cfg.mppi_horizon:<3} "
        f"timesteps={cfg.total_timesteps:<5} | GPU: {gpu_str} steps/s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO training throughput benchmark")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["small", "medium"],
        help="Config names to run (default: small medium). 'large' only if GPU.",
    )
    args = parser.parse_args()

    if not gpu_available():
        print(
            "aces._core.GpuVecEnv not importable - build with: "
            "poetry run maturin develop --features gpu"
        )
        print("Falling back to configs that don't require GPU (none currently).")
        return 1

    selected = [c for c in CONFIGS if c.name in args.configs]
    if not selected:
        print(f"No matching configs for {args.configs}")
        return 1

    print("=== PPO Training Throughput Benchmark ===")
    print()
    print("Measuring PPO learn() steps/s (with ~1 warm-up learn before timed run).")
    print()

    for cfg in selected:
        try:
            gpu_rate = time_gpu(cfg)
        except Exception:
            traceback.print_exc()
            gpu_rate = None
        print(format_row(cfg, gpu_rate))

    return 0


if __name__ == "__main__":
    sys.exit(main())
