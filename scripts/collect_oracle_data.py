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
from aces.god_oracle import GodOracle


def collect(n_episodes: int, output_path: str) -> None:
    env = DroneDogfightEnv(
        task="dogfight",
        opponent="mppi",
        fpv=False,
    )
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

            rel_pos = obs[6:9]
            own_vel = obs[0:3]
            opp_vel = obs[9:12]
            rel_dist = float(np.linalg.norm(rel_pos))

            if rel_dist > 0.01:
                direction_to_me = -rel_pos / rel_dist
                opp_vel_norm = max(float(np.linalg.norm(opp_vel)), 0.01)
                own_vel_norm = max(float(np.linalg.norm(own_vel)), 0.01)
                opponent_facing_me = float(
                    np.dot(opp_vel / opp_vel_norm, direction_to_me)
                )
                i_face_opponent = float(
                    np.dot(own_vel / own_vel_norm, rel_pos / rel_dist)
                )
                closing_speed = -float(np.dot(opp_vel, direction_to_me))
            else:
                opponent_facing_me = 0.0
                i_face_opponent = 0.0
                closing_speed = 0.0

            labels = oracle.compute(
                lock_b_progress=info.get("being_locked_progress", obs[17]),
                distance=info.get("distance", rel_dist),
                opponent_facing_me=max(0.0, opponent_facing_me),
                lock_a_progress=info.get("lock_progress", obs[16]),
                a_sees_b=bool(obs[18] > 0.5),
                i_face_opponent=max(0.0, i_face_opponent),
                nearest_obs_dist=info.get("nearest_obs_dist", obs[15]),
                speed=float(np.linalg.norm(own_vel)),
                belief_var=info.get("belief_var", obs[19]),
                opponent_closing_speed=closing_speed,
            )

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
            total = len(all_obs)
            print(f"Episode {ep + 1}/{n_episodes} — {total} samples collected")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.array(all_obs, dtype=np.float32),
        continuous_labels=np.array(all_continuous, dtype=np.float32),
        intent_labels=np.array(all_intent, dtype=np.int64),
    )
    print(f"[ACES] Saved {len(all_obs)} samples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect oracle training data")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/oracle_data.npz")
    args = parser.parse_args()
    collect(args.episodes, args.output)
