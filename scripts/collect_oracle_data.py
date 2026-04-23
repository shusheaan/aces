#!/usr/bin/env python3
"""Collect (observation, god_oracle_label) pairs from MPPI-vs-MPPI episodes.

Usage:
    poetry run python scripts/collect_oracle_data.py --episodes 500 --output data/oracle_data.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aces.env import DroneDogfightEnv
from aces.god_oracle import GodOracle, extract_oracle_inputs


def collect(n_episodes: int, output_path: str) -> None:
    env = DroneDogfightEnv(task="dogfight", opponent="mppi", fpv=False)
    oracle = GodOracle()

    all_obs: list[np.ndarray] = []
    all_continuous: list[np.ndarray] = []
    all_intent: list[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            inputs = extract_oracle_inputs(obs, info)
            labels = oracle.compute(**inputs)

            all_obs.append(obs.copy())
            all_continuous.append(
                np.array(
                    [
                        labels["threat"],
                        labels["opportunity"],
                        labels["collision_risk"],
                        labels["uncertainty"],
                        labels["opponent_distance"],
                    ],
                    dtype=np.float32,
                )
            )
            all_intent.append(int(labels["opponent_intent"]))

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{n_episodes} — {len(all_obs)} samples collected")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.array(all_obs, dtype=np.float32),
        continuous_labels=np.array(all_continuous, dtype=np.float32),
        intent_labels=np.array(all_intent, dtype=np.int64),
    )
    print(f"[ACES] Saved {len(all_obs)} samples -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect oracle training data")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/oracle_data.npz")
    args = parser.parse_args()
    collect(args.episodes, args.output)
