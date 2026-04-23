"""Self-play PPO trainer for ACES."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO

from aces.config import load_configs
from aces.env.dogfight import DroneDogfightEnv
from aces.training.callbacks import (
    EpisodeLoggerCallback,
    OpponentUpdateCallback,
    StateCallback,
    TrainingStatsCallback,
    WindowSummaryCallback,
)
from aces.training.logging import (
    create_run_dir,
    save_config_snapshot,
    save_run_metadata,
    setup_logging,
)

logger = logging.getLogger("aces.trainer")


def _pick(*values):
    """Return the first non-None value, or the last value."""
    for v in values:
        if v is not None:
            return v
    return values[-1]


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
        seed: int | None = None,
        **kwargs,
    ):
        cfg = load_configs(config_dir)
        tc = cfg.rules.training

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
        self._seed = seed

        self.env = DroneDogfightEnv(
            config_dir=config_dir,
            max_episode_steps=_max_episode_steps,
            opponent="random",
            task=task,
            wind_sigma=wind_sigma,
            obs_noise_std=obs_noise_std,
            fpv=fpv,
        )

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
            seed=seed,
        )

        self._setup_opponent()

    def _resolve_policy(self) -> tuple[str, dict | None]:
        from aces.training import resolve_policy

        return resolve_policy(self._fpv)

    def _setup_opponent(self):
        """Wire opponent policy into the environment."""
        policy, policy_kwargs = self._resolve_policy()
        kwargs = {"verbose": 0}
        if policy_kwargs:
            kwargs["policy_kwargs"] = policy_kwargs
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
