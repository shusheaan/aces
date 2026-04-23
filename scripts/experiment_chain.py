"""Chain hover -> pursuit_linear training experiment.

Loads a hover-trained model and fine-tunes on pursuit_linear task.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aces.env import DroneDogfightEnv
from aces.training.logging import (
    create_run_dir,
    save_config_snapshot,
    save_run_metadata,
    setup_logging,
)
from aces.training import (
    EpisodeLoggerCallback,
    OpponentUpdateCallback,
    TrainingStatsCallback,
)

logger = logging.getLogger("aces.experiment")


def run_chain_experiment(
    hover_model_path: str,
    pursuit_timesteps: int = 50000,
    save_path: str | None = None,
):
    """Load hover model, fine-tune on pursuit_linear."""
    log_dir = create_run_dir(prefix="chain_hover_pursuit")
    setup_logging(log_dir=log_dir)
    save_config_snapshot(log_dir)
    save_run_metadata(
        log_dir,
        task="pursuit_linear",
        timesteps=pursuit_timesteps,
        extra={"pretrained_from": hover_model_path, "experiment": "hover_chain"},
    )

    # Create pursuit_linear env
    env = DroneDogfightEnv(
        task="pursuit_linear",
        opponent="random",
        wind_sigma=0.0,
        obs_noise_std=0.0,
    )

    # Load hover model and set new env
    logger.info("Loading hover model from %s", hover_model_path)
    model = PPO.load(hover_model_path, env=env)
    logger.info("Fine-tuning on pursuit_linear for %d steps", pursuit_timesteps)

    opp_cb = OpponentUpdateCallback(update_interval=10000, verbose=1)
    stats_cb = TrainingStatsCallback()
    logger_cb = EpisodeLoggerCallback(log_dir=str(log_dir), verbose=1)

    model.learn(
        total_timesteps=pursuit_timesteps,
        callback=[opp_cb, stats_cb, logger_cb],
    )

    stats = stats_cb.summary()
    logger.info("Training complete: %s", stats)

    if save_path:
        model.save(save_path)
        logger.info("Model saved to %s", save_path)

    # Print summary
    csv_path = log_dir / "episodes.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    print(f"\n=== Chain Experiment Results ({n} episodes) ===")

    for start in range(0, n, max(n // 5, 1)):
        w = rows[start : start + max(n // 5, 1)]
        if not w:
            break
        lengths = [int(r["length"]) for r in w]
        rewards = [float(r["reward"]) for r in w]
        crashes = sum(1 for r in w if r["crash"] == "1")
        kills = sum(1 for r in w if r["kill"] == "1")
        dists = [float(r["distance"]) for r in w]
        locks = [float(r["lock_progress"]) for r in w]
        print(
            f"  Ep {start + 1:4d}-{start + len(w):4d}: "
            f"len={np.mean(lengths):.0f} reward={np.mean(rewards):.1f} "
            f"crash={crashes}/{len(w)} kill={kills}/{len(w)} "
            f"dist={np.mean(dists):.1f} lock={np.mean(locks):.3f}"
        )

    return model, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hover-model", required=True)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    run_chain_experiment(
        args.hover_model,
        pursuit_timesteps=args.timesteps,
        save_path=args.save_path,
    )
