"""Smoke test: run SB3 PPO on GpuVecEnv for a few hundred timesteps.

Purpose: prove the end-to-end GPU-accelerated RL training pipeline works.
Not a performance benchmark, not a real training run — just that PPO
can ingest obs/act/reward from GpuVecEnv without crashing.

Usage:
    python scripts/smoke_gpu_ppo.py [--timesteps N] [--n-envs K]

Exit code:
    0 if training completed successfully
    1 if any stage failed (including GPU unavailable)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU PPO smoke test")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=512,
        help="Total PPO training timesteps (default: 512)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel GPU envs (default: 4)",
    )
    parser.add_argument(
        "--mppi-samples",
        type=int,
        default=32,
        help="MPPI samples per drone (default: 32 — small for smoke)",
    )
    parser.add_argument(
        "--mppi-horizon",
        type=int,
        default=10,
        help="MPPI horizon (default: 10 — small for smoke)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    try:
        from aces.training.gpu_vec_env import GpuVecEnv
    except ImportError as e:
        print(f"GpuVecEnv import failed: {e}", file=sys.stderr)
        print(
            "Rebuild with: poetry run maturin develop --features gpu",
            file=sys.stderr,
        )
        return 1

    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        print(f"stable-baselines3 not installed: {e}", file=sys.stderr)
        return 1

    print("=== GPU PPO Smoke Test ===")
    print(
        f"n_envs={args.n_envs}, timesteps={args.timesteps}, "
        f"mppi=(N={args.mppi_samples}, H={args.mppi_horizon})"
    )

    try:
        t0 = time.time()
        env = GpuVecEnv(
            n_envs=args.n_envs,
            mppi_samples=args.mppi_samples,
            mppi_horizon=args.mppi_horizon,
        )
        t_env = time.time() - t0
        print(f"GpuVecEnv initialized in {t_env:.2f}s")
    except RuntimeError as e:
        print(f"GpuVecEnv init failed (likely no GPU): {e}", file=sys.stderr)
        return 1

    try:
        # Tiny network + batch for smoke-speed
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=32,  # rollout length per env
            batch_size=32,
            n_epochs=2,
            gamma=0.99,
            verbose=0,
            device="cpu",  # policy NN runs on CPU; GPU is for sim
        )

        t0 = time.time()
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        t_train = time.time() - t0

        print(
            f"Training completed: {args.timesteps} timesteps in {t_train:.2f}s "
            f"({args.timesteps / t_train:.1f} steps/s)"
        )

        # Smoke validation: final policy produces finite actions on a reset obs
        import numpy as np

        obs = env.reset()
        assert isinstance(obs, np.ndarray), f"expected ndarray obs, got {type(obs)}"
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (args.n_envs, 4), f"action shape {action.shape}"
        assert np.all(np.isfinite(action)), "non-finite action from trained policy"

        print(f"Final policy produces finite actions shape {action.shape}")

        env.close()
        return 0

    except Exception:
        print("TRAINING FAILED:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
