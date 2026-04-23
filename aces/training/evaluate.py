"""Evaluation utilities for trained ACES PPO models."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from aces.env.dogfight import DroneDogfightEnv


def evaluate(
    model_path: str,
    config_dir: str | None = None,
    n_episodes: int = 100,
    max_episode_steps: int = 1000,
    opponent: str = "mppi",
    wind_sigma: float | None = None,
    obs_noise_std: float | None = None,
    mppi_samples: int = 256,
    mppi_horizon: int = 20,
    fpv: bool = False,
) -> dict:
    """Evaluate a trained model against an opponent.

    Returns a dict with win_rate, avg_kill_time, avg_survival_time, etc.
    """
    env = DroneDogfightEnv(
        config_dir=config_dir,
        max_episode_steps=max_episode_steps,
        opponent=opponent,
        mppi_samples=mppi_samples,
        mppi_horizon=mppi_horizon,
        wind_sigma=wind_sigma,
        obs_noise_std=obs_noise_std,
        fpv=fpv,
    )

    model = PPO.load(model_path, env=env)

    wins = 0
    losses = 0
    crashes = 0
    timeouts = 0
    kill_times: list[int] = []
    survival_times: list[int] = []
    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        for step in range(max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if terminated:
                if info.get("kill_a", False):
                    wins += 1
                    kill_times.append(step + 1)
                elif info.get("kill_b", False):
                    losses += 1
                else:
                    crashes += 1
                survival_times.append(step + 1)
                break

            if truncated:
                timeouts += 1
                survival_times.append(step + 1)
                break

        episode_rewards.append(ep_reward)

    total = wins + losses + crashes + timeouts
    return {
        "n_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "crashes": crashes,
        "timeouts": timeouts,
        "win_rate": wins / max(total, 1),
        "loss_rate": losses / max(total, 1),
        "crash_rate": crashes / max(total, 1),
        "avg_kill_time": float(np.mean(kill_times)) if kill_times else float("nan"),
        "avg_survival_time": float(np.mean(survival_times))
        if survival_times
        else float("nan"),
        "mean_reward": float(np.mean(episode_rewards)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick(*values):
    """Return the first non-None value from the argument list."""
    for v in values:
        if v is not None:
            return v
    return values[-1]
