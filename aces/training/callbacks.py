"""Callback classes for ACES PPO training."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger("aces.trainer")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class OpponentUpdateCallback(BaseCallback):
    """Periodically copies current policy to the opponent."""

    def __init__(self, update_interval: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.update_interval = update_interval
        self.update_count = 0
        self._last_update_window: int = 0

    def _on_training_start(self) -> None:
        self._last_update_window = self.num_timesteps // self.update_interval

    def _on_step(self) -> bool:
        window = self.num_timesteps // self.update_interval
        if window > self._last_update_window:
            self._last_update_window = window
            state_dict = copy.deepcopy(self.model.policy.state_dict())
            env = self.training_env
            if hasattr(env, "envs"):
                for e in env.envs:  # type: ignore[attr-defined]
                    unwrapped = e.unwrapped
                    if hasattr(unwrapped, "_update_opponent_weights"):
                        unwrapped._update_opponent_weights(state_dict)
            else:
                env.env_method("_update_opponent_weights", state_dict)
            self.update_count += 1
            if self.verbose:
                logger.info(
                    "Opponent updated (#%d) at step %d",
                    self.update_count,
                    self.num_timesteps,
                )
        return True


class VecOpponentUpdateCallback(BaseCallback):
    """Periodically copies current policy to opponents in a VecEnv.

    Works with both plain VecEnv (broadcasts to workers) and
    BatchedOpponentVecEnv (updates the batched policy directly).
    """

    def __init__(self, update_interval: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.update_interval = update_interval
        self.update_count = 0
        self._last_update_window: int = 0

    def _on_training_start(self) -> None:
        self._last_update_window = self.num_timesteps // self.update_interval
        # If wrapped with BatchedOpponentVecEnv, initialize opponent policy
        env = self.training_env
        if hasattr(env, "set_opponent_policy") and not env.has_opponent:  # type: ignore[attr-defined]
            # Clone the current policy for the opponent
            opponent = copy.deepcopy(self.model.policy)
            opponent.set_training_mode(False)
            env.set_opponent_policy(opponent)

    def _on_step(self) -> bool:
        window = self.num_timesteps // self.update_interval
        if window > self._last_update_window:
            self._last_update_window = window
            state_dict = copy.deepcopy(self.model.policy.state_dict())
            env = self.training_env
            if hasattr(env, "set_opponent_weights"):
                env.set_opponent_weights(state_dict)
            else:
                env.env_method("set_opponent_weights", state_dict)
            self.update_count += 1
            if self.verbose:
                logger.info(
                    "VecEnv opponent updated (#%d) at step %d",
                    self.update_count,
                    self.num_timesteps,
                )
        return True


class PoolOpponentCallback(BaseCallback):
    """Periodically samples an opponent from the OpponentPool.

    Works with both plain VecEnv and BatchedOpponentVecEnv.
    """

    def __init__(self, pool, sample_interval: int = 20000, verbose: int = 0):
        super().__init__(verbose)
        self._pool = pool
        self._sample_interval = sample_interval
        self._last_sample_window: int = 0

    def _on_training_start(self) -> None:
        self._last_sample_window = self.num_timesteps // self._sample_interval
        # Initialize batched opponent if wrapper is present
        env = self.training_env
        if (
            hasattr(env, "set_opponent_policy")
            and not env.has_opponent  # type: ignore[attr-defined]
            and self._pool.size > 0
        ):
            policy, _ = self._pool.sample()
            env.set_opponent_policy(policy)

    def _on_step(self) -> bool:
        window = self.num_timesteps // self._sample_interval
        if self._pool.size > 0 and window > self._last_sample_window:
            self._last_sample_window = window
            policy, meta = self._pool.sample()
            state_dict = copy.deepcopy(policy.state_dict())
            env = self.training_env
            if hasattr(env, "set_opponent_weights"):
                env.set_opponent_weights(state_dict)
            else:
                env.env_method("set_opponent_weights", state_dict)
            if self.verbose:
                logger.info("Pool sampled opponent: %s", meta)
        return True


class StateCallback(BaseCallback):
    """Forwards per-step info to an external callback."""

    def __init__(self, state_callback: Callable | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.state_callback = state_callback

    def _on_step(self) -> bool:
        if self.state_callback is not None:
            for info in self.locals.get("infos", []):
                if "agent_pos" in info:
                    self.state_callback(info)
        return True


class TrainingStatsCallback(BaseCallback):
    """Collects training statistics per episode."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.kills: int = 0
        self.deaths: int = 0
        self.collisions: int = 0
        self._current_rewards: np.ndarray = np.array([0.0])
        self._current_lengths: np.ndarray = np.array([0])

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._current_rewards = np.zeros(n, dtype=np.float64)
        self._current_lengths = np.zeros(n, dtype=np.int64)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.zeros(len(self._current_rewards)))
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, info in enumerate(infos):
            self._current_rewards[i] += float(rewards[i])
            self._current_lengths[i] += 1

            if info.get("kill_a", False):
                self.kills += 1
            if info.get("kill_b", False):
                self.deaths += 1

        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(float(self._current_rewards[i]))
                self.episode_lengths.append(int(self._current_lengths[i]))
                self._current_rewards[i] = 0.0
                self._current_lengths[i] = 0
        return True

    def summary(self) -> dict:
        """Return summary statistics."""
        n = len(self.episode_rewards)
        if n == 0:
            return {"episodes": 0}
        return {
            "episodes": n,
            "mean_reward": float(np.mean(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "kills": self.kills,
            "deaths": self.deaths,
            "kill_rate": self.kills / max(n, 1),
        }


class TensorBoardMetricsCallback(BaseCallback):
    """Logs custom ACES metrics to TensorBoard via SB3's logger."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_rewards: list[float] = []
        self._ep_current_rewards: np.ndarray = np.array([0.0])
        self._kills = 0
        self._total_episodes = 0
        self._last_log_window: int = 0

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_current_rewards = np.zeros(n, dtype=np.float64)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.zeros(len(self._ep_current_rewards)))
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            self._ep_current_rewards[i] += float(rewards[i])
            if info.get("kill_a", False):
                self._kills += 1

        for i, done in enumerate(dones):
            if done:
                self._total_episodes += 1
                self._ep_rewards.append(float(self._ep_current_rewards[i]))
                self._ep_current_rewards[i] = 0.0

        # Log every 1000 steps (window-based dedup)
        window = self.num_timesteps // 1000
        if window > self._last_log_window and self._total_episodes > 0:
            self._last_log_window = window
            n = max(len(self._ep_rewards), 1)
            win_rate = self._kills / max(self._total_episodes, 1)
            self.logger.record("aces/win_rate", win_rate)
            self.logger.record("aces/kill_rate", self._kills / max(n, 1))
            self.logger.record("aces/episodes", self._total_episodes)
            if self._ep_rewards:
                self.logger.record(
                    "aces/mean_reward", float(np.mean(self._ep_rewards[-100:]))
                )

        return True


class WindowSummaryCallback(BaseCallback):
    """Prints convergence diagnostics every ``interval`` steps.

    Compares the current window to the previous window so you can see
    at a glance whether training is converging during a long run.
    """

    def __init__(self, interval: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self._interval = interval
        self._last_window: int = 0
        self._ep_rewards: list[float] = []
        self._ep_distances: list[float] = []
        self._ep_crashes: int = 0
        self._ep_kills: int = 0
        self._ep_timeouts: int = 0
        self._ep_count: int = 0
        self._ep_current_rewards: np.ndarray = np.array([0.0])
        # Previous window stats for comparison
        self._prev_dist: float | None = None
        self._prev_reward: float | None = None
        self._prev_crash_rate: float | None = None

    def _on_training_start(self) -> None:
        self._last_window = self.num_timesteps // self._interval
        n = self.training_env.num_envs
        self._ep_current_rewards = np.zeros(n, dtype=np.float64)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.zeros(len(self._ep_current_rewards)))
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            self._ep_current_rewards[i] += float(rewards[i])
            if info.get("kill_a", False):
                self._ep_kills += 1

        for i, done in enumerate(dones):
            if done:
                self._ep_count += 1
                self._ep_rewards.append(float(self._ep_current_rewards[i]))
                self._ep_current_rewards[i] = 0.0
                info = infos[i] if i < len(infos) else {}
                self._ep_distances.append(info.get("distance", 0.0))
                if info.get("collision", False):
                    self._ep_crashes += 1
                if info.get("truncated", False):
                    self._ep_timeouts += 1

        window = self.num_timesteps // self._interval
        if window > self._last_window and self._ep_count > 0:
            self._last_window = window
            n = max(self._ep_count, 1)
            mean_r = float(np.mean(self._ep_rewards)) if self._ep_rewards else 0.0
            mean_d = float(np.mean(self._ep_distances)) if self._ep_distances else 0.0
            min_d = float(np.min(self._ep_distances)) if self._ep_distances else 0.0
            crash_pct = self._ep_crashes / n * 100

            # Delta vs previous window
            dr = (
                f" ({mean_r - self._prev_reward:+.1f})"
                if self._prev_reward is not None
                else ""
            )
            dd = (
                f" ({mean_d - self._prev_dist:+.1f}m)"
                if self._prev_dist is not None
                else ""
            )

            logger.info(
                "[%dk] %d eps | reward=%.1f%s | dist=%.1f%s (min=%.1f) | crash=%.0f%% | kill=%d | timeout=%d",
                self.num_timesteps // 1000,
                n,
                mean_r,
                dr,
                mean_d,
                dd,
                min_d,
                crash_pct,
                self._ep_kills,
                self._ep_timeouts,
            )

            self._prev_dist = mean_d
            self._prev_reward = mean_r
            self._prev_crash_rate = crash_pct

            # Reset window counters
            self._ep_rewards.clear()
            self._ep_distances.clear()
            self._ep_crashes = 0
            self._ep_kills = 0
            self._ep_timeouts = 0
            self._ep_count = 0

        return True


class EpisodeLoggerCallback(BaseCallback):
    """Writes per-episode stats to CSV for offline analysis.

    Output: ``<log_dir>/episodes.csv`` with columns:
        episode, timestep, reward, length, kill, death, crash, timeout, lock_progress, distance
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self._csv_file = None
        self._ep_rewards: np.ndarray = np.array([0.0])
        self._ep_lengths: np.ndarray = np.array([0])
        self._ep_count = 0

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_rewards = np.zeros(n, dtype=np.float64)
        self._ep_lengths = np.zeros(n, dtype=np.int64)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.log_dir / "episodes.csv", "w")  # type: ignore[assignment]
        self._csv_file.write(  # type: ignore[attr-defined]
            "episode,timestep,reward,length,kill,death,crash,timeout,lock_progress,distance\n"
        )

    def __del__(self) -> None:
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.zeros(len(self._ep_rewards)))
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i in range(len(self._ep_rewards)):
            self._ep_rewards[i] += float(rewards[i])
            self._ep_lengths[i] += 1

        for i, done in enumerate(dones):
            if done:
                self._ep_count += 1
                info = infos[i] if i < len(infos) else {}
                kill = 1 if info.get("kill_a", False) else 0
                death = 1 if info.get("kill_b", False) else 0
                crash = 1 if info.get("collision", False) else 0
                timeout = 1 if info.get("truncated", False) else 0
                lock_p = info.get("lock_a_progress", 0.0)
                dist = info.get("distance", 0.0)

                self._csv_file.write(  # type: ignore[attr-defined]
                    f"{self._ep_count},{self.num_timesteps},"
                    f"{self._ep_rewards[i]:.4f},{self._ep_lengths[i]},"
                    f"{kill},{death},{crash},{timeout},{lock_p:.4f},{dist:.4f}\n"
                )
                self._csv_file.flush()  # type: ignore[attr-defined]

                if self.verbose and self._ep_count % 50 == 0:
                    logger.info(
                        "Ep %d: reward=%.2f len=%d kill=%d death=%d crash=%d timeout=%d dist=%.2f",
                        self._ep_count,
                        self._ep_rewards[i],
                        self._ep_lengths[i],
                        kill,
                        death,
                        crash,
                        timeout,
                        dist,
                    )

                self._ep_rewards[i] = 0.0
                self._ep_lengths[i] = 0

        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()


class CheckpointResumeCallback(BaseCallback):
    """Periodically saves model + curriculum/pool state for resume."""

    def __init__(
        self,
        save_fn: Callable[[str], None],
        checkpoint_dir: str,
        interval: int = 50_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._save_fn = save_fn
        self._checkpoint_dir = checkpoint_dir
        self._interval = interval
        self._last_ckpt_window: int = 0

    def _on_training_start(self) -> None:
        self._last_ckpt_window = self.num_timesteps // self._interval

    def _on_step(self) -> bool:
        window = self.num_timesteps // self._interval
        if window > self._last_ckpt_window:
            self._last_ckpt_window = window
            ckpt_path = f"{self._checkpoint_dir}/step_{self.num_timesteps}"
            self._save_fn(ckpt_path)
            if self.verbose:
                logger.info("Checkpoint saved at step %d", self.num_timesteps)
        return True
