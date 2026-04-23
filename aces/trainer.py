"""Self-play PPO training for ACES."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from aces.config import load_configs
from aces.env import DroneDogfightEnv
from aces.logging_config import (
    create_run_dir,
    save_config_snapshot,
    save_run_metadata,
    setup_logging,
)

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
            for env in self.training_env.envs:  # type: ignore[attr-defined]
                unwrapped = env.unwrapped
                if hasattr(unwrapped, "_update_opponent_weights"):
                    unwrapped._update_opponent_weights(state_dict)
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
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self._current_reward += self.locals.get("rewards", [0.0])[0]
            self._current_length += 1

            if info.get("kill_a", False):
                self.kills += 1
            if info.get("kill_b", False):
                self.deaths += 1

        # Check for episode end
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self.episode_rewards.append(self._current_reward)
                self.episode_lengths.append(self._current_length)
                self._current_reward = 0.0
                self._current_length = 0
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
        self._ep_current_reward = 0.0
        self._kills = 0
        self._total_episodes = 0
        self._last_log_window: int = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        self._ep_current_reward += float(rewards[0])

        for info in self.locals.get("infos", []):
            if info.get("kill_a", False):
                self._kills += 1

        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self._total_episodes += 1
                self._ep_rewards.append(self._ep_current_reward)
                self._ep_current_reward = 0.0

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
        self._ep_current_reward: float = 0.0
        # Previous window stats for comparison
        self._prev_dist: float | None = None
        self._prev_reward: float | None = None
        self._prev_crash_rate: float | None = None

    def _on_training_start(self) -> None:
        self._last_window = self.num_timesteps // self._interval

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        self._ep_current_reward += float(rewards[0])

        for info in self.locals.get("infos", []):
            if info.get("kill_a", False):
                self._ep_kills += 1

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [{}])
        if dones[0]:
            self._ep_count += 1
            self._ep_rewards.append(self._ep_current_reward)
            self._ep_current_reward = 0.0
            info = infos[0] if infos else {}
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
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_count = 0

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.log_dir / "episodes.csv", "w")  # type: ignore[assignment]
        self._csv_file.write(  # type: ignore[attr-defined]
            "episode,timestep,reward,length,kill,death,crash,timeout,lock_progress,distance\n"
        )

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals.get("infos", [{}])

        self._ep_reward += float(rewards[0])
        self._ep_length += 1

        if dones[0]:
            self._ep_count += 1
            info = infos[0] if infos else {}
            kill = 1 if info.get("kill_a", False) else 0
            death = 1 if info.get("kill_b", False) else 0
            crash = 1 if info.get("collision", False) else 0
            timeout = 1 if info.get("truncated", False) else 0
            lock_p = info.get("lock_a_progress", 0.0)
            dist = info.get("distance", 0.0)

            self._csv_file.write(  # type: ignore[attr-defined]
                f"{self._ep_count},{self.num_timesteps},"
                f"{self._ep_reward:.4f},{self._ep_length},"
                f"{kill},{death},{crash},{timeout},{lock_p:.4f},{dist:.4f}\n"
            )
            self._csv_file.flush()  # type: ignore[attr-defined]

            if self.verbose and self._ep_count % 50 == 0:
                logger.info(
                    "Ep %d: reward=%.2f len=%d kill=%d death=%d crash=%d timeout=%d dist=%.2f",
                    self._ep_count,
                    self._ep_reward,
                    self._ep_length,
                    kill,
                    death,
                    crash,
                    timeout,
                    dist,
                )

            self._ep_reward = 0.0
            self._ep_length = 0

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


# ---------------------------------------------------------------------------
# SelfPlayTrainer
# ---------------------------------------------------------------------------


class SelfPlayTrainer:
    """Trains a PPO agent via self-play in the ACES environment."""

    def __init__(
        self,
        config_dir: str | None = None,
        total_timesteps: int | None = None,
        n_steps: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        gamma: float | None = None,
        gae_lambda: float | None = None,
        clip_range: float | None = None,
        n_epochs: int | None = None,
        opponent_update_interval: int | None = None,
        max_episode_steps: int | None = None,
        state_callback: Callable | None = None,
        wind_sigma: float | None = None,
        obs_noise_std: float | None = None,
        fpv: bool = False,
        task: str = "dogfight",
        device: str = "auto",
        **kwargs,
    ):
        # Load config defaults from rules.toml
        cfg = load_configs(config_dir)
        tc = cfg.rules.training

        # Resolve parameters: explicit arg > kwargs > config file
        self.total_timesteps = _pick(
            total_timesteps, kwargs.get("total_timesteps"), tc.total_timesteps
        )
        _n_steps = _pick(n_steps, kwargs.get("n_steps"), tc.n_steps)
        _batch_size = _pick(batch_size, kwargs.get("batch_size"), tc.batch_size)
        _learning_rate = _pick(
            learning_rate, kwargs.get("learning_rate"), tc.learning_rate
        )
        _gamma = _pick(gamma, kwargs.get("gamma"), tc.gamma)
        _gae_lambda = _pick(gae_lambda, kwargs.get("gae_lambda"), tc.gae_lambda)
        _clip_range = _pick(clip_range, kwargs.get("clip_range"), tc.clip_range)
        _n_epochs = _pick(n_epochs, kwargs.get("n_epochs"), tc.n_epochs)
        _opponent_update_interval = _pick(
            opponent_update_interval,
            kwargs.get("opponent_update_interval"),
            tc.opponent_update_interval,
        )
        _max_episode_steps = _pick(
            max_episode_steps,
            kwargs.get("max_episode_steps"),
            tc.max_episode_steps,
        )

        self.opponent_update_interval = _opponent_update_interval
        self.state_callback = state_callback
        self.opponent_update_count = 0
        self.stats: dict = {}
        self._fpv = fpv
        self._config_dir = config_dir
        self._task = task
        self._wind_sigma = wind_sigma
        self._obs_noise_std = obs_noise_std

        self.env = DroneDogfightEnv(
            config_dir=config_dir,
            max_episode_steps=_max_episode_steps,
            opponent="random",
            task=task,
            wind_sigma=wind_sigma,
            obs_noise_std=obs_noise_std,
            fpv=fpv,
        )

        # Wrap with VecNormalize for observation normalization
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        vec_env = DummyVecEnv([lambda: self.env])
        self._vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )

        policy, policy_kwargs = self._resolve_policy()
        self.model = PPO(
            policy,
            self._vec_env,
            learning_rate=_learning_rate,
            n_steps=_n_steps,
            batch_size=_batch_size,
            gamma=_gamma,
            gae_lambda=_gae_lambda,
            clip_range=_clip_range,
            n_epochs=_n_epochs,
            verbose=0,
            policy_kwargs=policy_kwargs,
            device=device,
        )

        self._setup_opponent()

    def _resolve_policy(self) -> tuple[str, dict | None]:
        """Return (policy_name, policy_kwargs) based on observation mode."""
        if self._fpv:
            from aces.policy import CnnImuExtractor

            return "MultiInputPolicy", {
                "features_extractor_class": CnnImuExtractor,
                "features_extractor_kwargs": {"features_dim": 192},
            }
        return "MlpPolicy", None

    def _setup_opponent(self):
        """Wire opponent policy into the environment."""
        policy, policy_kwargs = self._resolve_policy()
        kwargs = {"verbose": 0}
        if policy_kwargs:
            kwargs["policy_kwargs"] = policy_kwargs
        # Opponent uses the raw env (not normalized) for its own PPO instance
        self._opponent_model = PPO(policy, self.env, **kwargs)
        opponent_model = self._opponent_model

        def update_weights(state_dict):
            opponent_model.policy.load_state_dict(state_dict)

        self.env._update_opponent_weights = update_weights
        self.env.set_opponent_policy(opponent_model.policy)

    def train(self) -> PPO:
        log_dir = create_run_dir(prefix="train")
        setup_logging(log_dir=log_dir)
        save_config_snapshot(log_dir, self._config_dir)
        save_run_metadata(
            log_dir,
            task=self._task,
            timesteps=self.total_timesteps,
            fpv=self._fpv,
            wind_sigma=self._wind_sigma,
            obs_noise_std=self._obs_noise_std,
        )

        opp_cb = OpponentUpdateCallback(
            update_interval=self.opponent_update_interval, verbose=1
        )
        state_cb = StateCallback(state_callback=self.state_callback)
        stats_cb = TrainingStatsCallback()
        logger_cb = EpisodeLoggerCallback(log_dir=str(log_dir), verbose=1)
        window_cb = WindowSummaryCallback(interval=10_000)

        logger.info(
            "Training started: %d timesteps, task=%s, fpv=%s",
            self.total_timesteps,
            self._task,
            self._fpv,
        )
        logger.info("Episode logs -> %s/episodes.csv", log_dir)

        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[opp_cb, state_cb, stats_cb, logger_cb, window_cb],
        )
        self.opponent_update_count = opp_cb.update_count
        self.stats = stats_cb.summary()

        logger.info("Training complete: %s", self.stats)
        return self.model

    def save(self, path: str = "aces_model"):
        self.model.save(path)
        if hasattr(self, "_vec_env"):
            self._vec_env.save(path + "_vecnorm.pkl")

    def load(self, path: str = "aces_model"):
        from stable_baselines3.common.vec_env import VecNormalize

        vecnorm_path = path + "_vecnorm.pkl"
        if Path(vecnorm_path).exists() and hasattr(self, "_vec_env"):
            self._vec_env = VecNormalize.load(vecnorm_path, self._vec_env.venv)
            logger.info("Loaded VecNormalize stats from %s", vecnorm_path)
        self.model = PPO.load(path, env=self._vec_env)
        self._setup_opponent()


# ---------------------------------------------------------------------------
# CurriculumTrainer
# ---------------------------------------------------------------------------


class CurriculumTrainer:
    """Trains PPO through sequential curriculum stages.

    Each stage uses the same observation space but different opponent behavior
    and reward weights. Weights transfer between stages via model.set_env().

    Supports two construction patterns:

    **Legacy (stages)**: list of dicts with ``"task"`` and ``"timesteps"`` keys.
    **New (phases)**: list of :class:`~aces.curriculum.Phase` objects with
    VecEnv parallelization, TensorBoard logging, checkpoint/resume, and
    integration with :class:`~aces.curriculum.CurriculumManager` and
    :class:`~aces.opponent_pool.OpponentPool`.
    """

    def __init__(
        self,
        stages: list[dict] | None = None,
        phases: list | None = None,
        config_dir: str | None = None,
        n_envs: int = 1,
        save_dir: str | None = None,
        checkpoint_interval: int = 50_000,
        fpv: bool = False,
        n_steps: int = 2048,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        wind_sigma: float | None = None,
        obs_noise_std: float | None = None,
        pool_dir: str | None = None,
        pool_max_size: int = 20,
        device: str = "auto",
    ):
        self._config_dir = config_dir
        self._fpv = fpv
        self._save_dir = Path(save_dir) if save_dir else None
        self._n_envs = n_envs
        self._checkpoint_interval = checkpoint_interval
        self._ppo_kwargs = dict(
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            n_epochs=n_epochs,
            verbose=0,
            device=device,
        )
        self._wind_sigma = wind_sigma
        self._obs_noise_std = obs_noise_std

        # Backward-compat: convert stages dicts -> Phase objects
        if phases is not None:
            self._phases = phases
        elif stages is not None:
            self._phases = self._stages_to_phases(stages)
        else:
            raise ValueError("Either 'stages' or 'phases' must be provided")

        # Keep stages reference for backward compat (stage_stats indexing)
        self.stages = (
            stages
            if stages is not None
            else [{"task": p.task, "timesteps": p.max_timesteps} for p in self._phases]
        )
        self.stage_stats: list[dict] = []

        # CurriculumManager
        from aces.curriculum import CurriculumManager

        self.curriculum = CurriculumManager(self._phases)

        # OpponentPool (optional)
        self.pool = None
        if pool_dir is not None:
            from aces.opponent_pool import OpponentPool

            self.pool = OpponentPool(pool_dir, max_size=pool_max_size)

        # Model (created lazily during train)
        self.model: PPO | None = None

    @staticmethod
    def _stages_to_phases(stages: list[dict]) -> list:
        """Convert legacy stage dicts to Phase objects."""
        from aces.curriculum import Phase

        phases = []
        for i, stage in enumerate(stages):
            phases.append(
                Phase(
                    name=f"stage{i}_{stage['task']}",
                    task=stage["task"],
                    max_timesteps=stage["timesteps"],
                )
            )
        return phases

    def _make_env(
        self, task: str, opponent: str = "random", phase=None
    ) -> DroneDogfightEnv:
        """Create a single environment (backward-compat helper)."""
        kwargs: dict = {
            "config_dir": self._config_dir,
            "max_episode_steps": 1000,
            "task": task,
            "opponent": opponent,
            "wind_sigma": self._wind_sigma,
            "obs_noise_std": self._obs_noise_std,
            "fpv": self._fpv,
        }
        if phase is not None:
            kwargs["wind_sigma"] = phase.wind_sigma
            kwargs["obs_noise_std"] = phase.obs_noise_std
            for key in (
                "motor_time_constant",
                "motor_noise_std",
                "motor_bias_range",
                "imu_accel_bias_std",
                "imu_gyro_bias_std",
            ):
                val = getattr(phase, key, None)
                if val is not None:
                    kwargs[key] = val
        return DroneDogfightEnv(**kwargs)

    def _make_vec_env(self, phase_or_task):
        """Create a VecEnv for the given phase or task string.

        Uses SubprocVecEnv when ``n_envs > 1``, otherwise DummyVecEnv.
        """
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        # Extract parameters from phase or use defaults
        if hasattr(phase_or_task, "task"):
            phase = phase_or_task
            task = phase.task
            opponent = getattr(phase, "opponent", "random")
            wind = phase.wind_sigma
            noise = phase.obs_noise_std
            motor_tc = getattr(phase, "motor_time_constant", None)
            motor_ns = getattr(phase, "motor_noise_std", None)
            motor_br = getattr(phase, "motor_bias_range", None)
            imu_acc = getattr(phase, "imu_accel_bias_std", None)
            imu_gyr = getattr(phase, "imu_gyro_bias_std", None)
        else:
            task = phase_or_task
            opponent = "random"
            wind = self._wind_sigma
            noise = self._obs_noise_std
            motor_tc = None
            motor_ns = None
            motor_br = None
            imu_acc = None
            imu_gyr = None

        def make_env(rank):
            def _init():
                return DroneDogfightEnv(
                    config_dir=self._config_dir,
                    max_episode_steps=1000,
                    task=task,
                    opponent=opponent,
                    wind_sigma=wind,
                    obs_noise_std=noise,
                    fpv=self._fpv,
                    motor_time_constant=motor_tc,
                    motor_noise_std=motor_ns,
                    motor_bias_range=motor_br,
                    imu_accel_bias_std=imu_acc,
                    imu_gyro_bias_std=imu_gyr,
                )

            return _init

        if self._n_envs > 1:
            vec_env = SubprocVecEnv([make_env(i) for i in range(self._n_envs)])
        else:
            vec_env = DummyVecEnv([make_env(0)])

        # Wrap with batched opponent inference for policy/pool opponents
        if opponent in ("policy", "pool"):
            from aces.batched_vec_env import BatchedOpponentVecEnv

            vec_env = BatchedOpponentVecEnv(vec_env)

        # Normalize observations and rewards for stable learning
        from stable_baselines3.common.vec_env import VecNormalize

        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        return vec_env

    def _resolve_policy(self) -> tuple[str, dict | None]:
        if self._fpv:
            from aces.policy import CnnImuExtractor

            return "MultiInputPolicy", {
                "features_extractor_class": CnnImuExtractor,
                "features_extractor_kwargs": {"features_dim": 192},
            }
        return "MlpPolicy", None

    def train(self) -> PPO:
        """Run curriculum training through all phases.

        Returns the trained PPO model.
        """
        log_dir = (
            Path("logs") / f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        # Only enable TensorBoard logging if tensorboard is installed
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401

            tb_log_dir: str | None = str(log_dir / "tensorboard")
        except ImportError:
            tb_log_dir = None

        # Resume from current phase_index
        start_idx = self.curriculum.phase_index

        for i in range(start_idx, len(self._phases)):
            # Ensure curriculum points to the right phase
            self.curriculum.phase_index = i
            phase = self._phases[i]
            task = phase.task
            timesteps = phase.max_timesteps

            logger.info("=== Phase %d -- %s (%d steps) ===", i, phase.name, timesteps)

            # Create env with VecNormalize wrapping
            if self._n_envs > 1:
                env = self._make_vec_env(phase)
            else:
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

                raw_env = self._make_env(
                    task, opponent=getattr(phase, "opponent", "random"), phase=phase
                )
                env = VecNormalize(
                    DummyVecEnv([lambda: raw_env]),
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                )

            if self.model is None:
                policy_name, policy_kwargs = self._resolve_policy()
                kwargs = {**self._ppo_kwargs}
                if policy_kwargs:
                    kwargs["policy_kwargs"] = policy_kwargs
                self.model = PPO(
                    policy_name,
                    env,
                    tensorboard_log=tb_log_dir,
                    **kwargs,  # type: ignore[arg-type]
                )
            else:
                self.model.set_env(env)

            # Build callbacks
            stage_log = log_dir / f"stage{i}_{task}"
            stats_cb = TrainingStatsCallback()
            logger_cb = EpisodeLoggerCallback(log_dir=str(stage_log), verbose=1)
            tb_cb = TensorBoardMetricsCallback()
            window_cb = WindowSummaryCallback(interval=10_000)
            callbacks: list[BaseCallback] = [stats_cb, logger_cb, tb_cb, window_cb]

            # Self-play opponent update (for "policy" opponent mode)
            if hasattr(phase, "opponent") and phase.opponent == "policy":
                opp_cb = VecOpponentUpdateCallback(update_interval=10000, verbose=1)
                callbacks.append(opp_cb)

            # Pool opponent sampling
            if (
                self.pool is not None
                and hasattr(phase, "opponent")
                and phase.opponent == "pool"
            ):
                pool_cb = PoolOpponentCallback(
                    pool=self.pool, sample_interval=20000, verbose=1
                )
                callbacks.append(pool_cb)

            # Checkpoint callback
            if self._checkpoint_interval and self._save_dir:
                ckpt_cb = CheckpointResumeCallback(
                    save_fn=self.save_checkpoint,
                    checkpoint_dir=str(self._save_dir / "checkpoints"),
                    interval=self._checkpoint_interval,
                    verbose=1,
                )
                callbacks.append(ckpt_cb)

            self.model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
            )

            self.stage_stats.append(stats_cb.summary())

            # Save stage model
            if self._save_dir:
                self._save_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self._save_dir / f"stage{i}_{task}"))

            # Add to pool if phase uses pool opponent
            if (
                self.pool is not None
                and hasattr(phase, "opponent")
                and phase.opponent == "pool"
            ):
                self.pool.add(self.model, {"phase": phase.name})

            logger.info("Phase %d done: %s", i, stats_cb.summary())

            # Close VecEnv if we created one
            if self._n_envs > 1:
                env.close()

            # Promote to next phase
            if not self.curriculum.is_last_phase():
                self.curriculum.promote()

        return self.model  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model, curriculum state, and pool state to a directory."""
        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save(str(ckpt_dir / "model"))
        if self.curriculum:
            with open(ckpt_dir / "curriculum_state.json", "w") as f:
                json.dump(self.curriculum.state_dict(), f)
        if self.pool:
            with open(ckpt_dir / "pool_state.json", "w") as f:
                json.dump(self.pool.state_dict(), f)

    def load_checkpoint(self, path: str) -> None:
        """Restore model, curriculum state, and pool state from a directory."""
        ckpt_dir = Path(path)
        if (ckpt_dir / "model.zip").exists():
            self.model = PPO.load(str(ckpt_dir / "model"))
        if (ckpt_dir / "curriculum_state.json").exists():
            with open(ckpt_dir / "curriculum_state.json") as f:
                self.curriculum.load_state_dict(json.load(f))
        if self.pool and (ckpt_dir / "pool_state.json").exists():
            with open(ckpt_dir / "pool_state.json") as f:
                self.pool.load_state_dict(json.load(f))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


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
