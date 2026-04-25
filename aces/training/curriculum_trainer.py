"""Curriculum-based PPO trainer for ACES."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from aces.env.dogfight import DroneDogfightEnv
from aces.training.callbacks import (
    CheckpointResumeCallback,
    EpisodeLoggerCallback,
    PoolOpponentCallback,
    TensorBoardMetricsCallback,
    TrainingStatsCallback,
    VecOpponentUpdateCallback,
    WindowSummaryCallback,
)

logger = logging.getLogger("aces.trainer")


class PromotionCheckCallback(BaseCallback):
    """Checks curriculum promotion conditions during training."""

    def __init__(self, curriculum, check_interval: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self._curriculum = curriculum
        self._check_interval = check_interval
        self._last_check_window: int = 0
        self._ep_rewards: list[float] = []
        self._recent_wins: list[bool] = []
        self._ep_current_rewards: np.ndarray = np.array([0.0])
        self.promoted = False

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_current_rewards = np.zeros(n, dtype=np.float64)
        self._last_check_window = self.num_timesteps // self._check_interval

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.zeros(len(self._ep_current_rewards)))
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, info in enumerate(infos):
            self._ep_current_rewards[i] += float(rewards[i])

        for i, done in enumerate(dones):
            if done:
                self._ep_rewards.append(float(self._ep_current_rewards[i]))
                self._ep_current_rewards[i] = 0.0
                info = infos[i] if i < len(infos) else {}
                self._recent_wins.append(info.get("kill_a", False))

        window = self.num_timesteps // self._check_interval
        if window > self._last_check_window and len(self._ep_rewards) > 0:
            self._last_check_window = window
            recent = self._recent_wins[-100:]
            metrics = {
                "reward": float(np.mean(self._ep_rewards[-100:])),
                "win_rate": sum(recent) / max(len(recent), 1),
            }
            if self._curriculum.should_promote(metrics):
                logger.info("Promotion condition met: %s", metrics)
                self.promoted = True
                self.model.stop_training = True  # type: ignore[attr-defined]
        return True


class CurriculumTrainer:
    """Trains PPO through sequential curriculum stages.

    Each stage uses the same observation space but different opponent behavior
    and reward weights. Weights transfer between stages via model.set_env().

    Supports two construction patterns:

    **Legacy (stages)**: list of dicts with ``"task"`` and ``"timesteps"`` keys.
    **New (phases)**: list of :class:`~aces.curriculum.Phase` objects with
    VecEnv parallelization, TensorBoard logging, checkpoint/resume, and
    integration with :class:`~aces.curriculum.CurriculumManager` and
    :class:`~aces.training.OpponentPool`.
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
        seed: int | None = None,
        use_gpu_env: bool = False,
        gpu_mppi_samples: int = 128,
        gpu_mppi_horizon: int = 15,
        gpu_noise_std: float = 0.03,
    ):
        self._config_dir = config_dir
        self._seed = seed
        self._fpv = fpv
        self._save_dir = Path(save_dir) if save_dir else None
        self._n_envs = n_envs
        self._checkpoint_interval = checkpoint_interval
        self._use_gpu_env = use_gpu_env
        self._gpu_mppi_samples = gpu_mppi_samples
        self._gpu_mppi_horizon = gpu_mppi_horizon
        self._gpu_noise_std = gpu_noise_std
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

        if phases is not None:
            self._phases = phases
        elif stages is not None:
            self._phases = self._stages_to_phases(stages)
        else:
            raise ValueError("Either 'stages' or 'phases' must be provided")

        self.stages = (
            stages
            if stages is not None
            else [{"task": p.task, "timesteps": p.max_timesteps} for p in self._phases]
        )
        self.stage_stats: list[dict] = []

        from aces.curriculum import CurriculumManager

        self.curriculum = CurriculumManager(self._phases)

        self.pool = None
        if pool_dir is not None:
            from aces.training.opponent_pool import OpponentPool

            self.pool = OpponentPool(pool_dir, max_size=pool_max_size)

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
        """Create a single environment."""
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
        """Create a VecEnv with VecNormalize wrapping.

        When ``use_gpu_env=True``, returns a :class:`GpuVecEnv`-backed stack
        regardless of the phase's ``task`` / ``opponent`` settings. The GPU env
        hardcodes MPPI-vs-PPO dogfight; curriculum-specific opponents
        (trajectory controllers, pool opponents, no-opponent) and alternate
        tasks (hover, pursuit_linear, etc.) are NOT supported by that backend
        and will be silently overridden.

        Reward weights: :class:`GpuVecEnv` reads the base ``[reward]`` section
        from ``configs/rules.toml`` (``reward_config=None`` -> auto-resolve).
        Per-task overrides from ``[task_reward_overrides.<task>]`` are NOT
        merged yet — follow-up slice. Matching the CPU path's per-task
        weighting requires passing a merged ``reward_config`` dict here.
        """
        if self._use_gpu_env:
            from stable_baselines3.common.vec_env import VecNormalize

            from aces.training.gpu_vec_env import GpuVecEnv as _GpuVecEnv

            if hasattr(phase_or_task, "task") and phase_or_task.task != "dogfight":
                logger.warning(
                    "Phase task '%s' requested but GpuVecEnv only implements "
                    "dogfight MPPI-vs-MPPI semantics; agent B will be GPU MPPI "
                    "regardless. Curriculum task/opponent specification is "
                    "ignored.",
                    phase_or_task.task,
                )
            if hasattr(phase_or_task, "opponent") and phase_or_task.opponent not in (
                None,
                "mppi",
            ):
                logger.warning(
                    "Phase opponent '%s' requested but GpuVecEnv provides MPPI "
                    "opponent only",
                    phase_or_task.opponent,
                )

            phase_task = (
                getattr(phase_or_task, "task", None)
                if hasattr(phase_or_task, "task")
                else None
            )
            raw_env = _GpuVecEnv(
                n_envs=self._n_envs,
                mppi_samples=self._gpu_mppi_samples,
                mppi_horizon=self._gpu_mppi_horizon,
                noise_std=self._gpu_noise_std,
                seed=self._seed if self._seed is not None else 0,
                task=phase_task,
            )
            return VecNormalize(
                raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0
            )

        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

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

        if opponent in ("policy", "pool"):
            from aces.training.batched_vec_env import BatchedOpponentVecEnv

            vec_env = BatchedOpponentVecEnv(vec_env)

        from stable_baselines3.common.vec_env import VecNormalize

        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        return vec_env

    def _resolve_policy(self) -> tuple[str, dict | None]:
        from aces.training import resolve_policy

        return resolve_policy(self._fpv)

    def train(self) -> PPO:
        """Run curriculum training through all phases."""
        log_dir = (
            Path("logs") / f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401

            tb_log_dir: str | None = str(log_dir / "tensorboard")
        except ImportError:
            tb_log_dir = None

        start_idx = self.curriculum.phase_index
        # Per-phase VecNormalize save paths (phase name is used as key)
        vec_normalize_dir = log_dir / "vec_normalize"

        for i in range(start_idx, len(self._phases)):
            self.curriculum.phase_index = i
            phase = self._phases[i]
            task = phase.task
            timesteps = phase.max_timesteps

            logger.info("=== Phase %d -- %s (%d steps) ===", i, phase.name, timesteps)

            if self._n_envs > 1:
                env = self._make_vec_env(phase)
            else:
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

                raw_env = self._make_env(
                    task, opponent=getattr(phase, "opponent", "random"), phase=phase
                )
                env = VecNormalize(
                    DummyVecEnv([lambda e=raw_env: e]),  # type: ignore[misc]
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                )

            # Restore VecNormalize stats from previous phase using the official API
            if i > start_idx:
                from stable_baselines3.common.vec_env import VecNormalize

                prev_phase = self._phases[i - 1]
                prev_vn_path = vec_normalize_dir / f"{prev_phase.name}_vecnormalize.pkl"
                if isinstance(env, VecNormalize) and prev_vn_path.exists():
                    loaded_vn = VecNormalize.load(str(prev_vn_path), env.venv)
                    env.obs_rms = loaded_vn.obs_rms
                    env.ret_rms = loaded_vn.ret_rms
                    # Warn if norm_reward semantics differ between phases
                    if loaded_vn.norm_reward != env.norm_reward:
                        logger.warning(
                            "Phase %d norm_reward=%s differs from previous phase "
                            "norm_reward=%s; resetting ret_rms to avoid silent breakage",
                            i,
                            env.norm_reward,
                            loaded_vn.norm_reward,
                        )
                        from stable_baselines3.common.running_mean_std import (
                            RunningMeanStd,
                        )

                        env.ret_rms = RunningMeanStd(shape=())
                    logger.info(
                        "Restored VecNormalize stats from phase '%s'", prev_phase.name
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
                    seed=self._seed,
                    **kwargs,  # type: ignore[arg-type]
                )
            else:
                self.model.set_env(env)

            stage_log = log_dir / f"stage{i}_{task}"
            stats_cb = TrainingStatsCallback()
            logger_cb = EpisodeLoggerCallback(log_dir=str(stage_log), verbose=1)
            tb_cb = TensorBoardMetricsCallback()
            window_cb = WindowSummaryCallback(interval=10_000)
            promo_cb = PromotionCheckCallback(self.curriculum, check_interval=5000)
            callbacks: list[BaseCallback] = [
                stats_cb,
                logger_cb,
                tb_cb,
                window_cb,
                promo_cb,
            ]

            if hasattr(phase, "opponent") and phase.opponent == "policy":
                opp_cb = VecOpponentUpdateCallback(update_interval=10000, verbose=1)
                callbacks.append(opp_cb)

            if (
                self.pool is not None
                and hasattr(phase, "opponent")
                and phase.opponent == "pool"
            ):
                pool_cb = PoolOpponentCallback(
                    pool=self.pool, sample_interval=20000, verbose=1
                )
                callbacks.append(pool_cb)

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

            if promo_cb.promoted:
                logger.info("Phase %d promoted early by condition", i)
            elif phase.promote_condition != "steps":
                logger.warning(
                    "Phase %d did not meet promotion condition '%s' within %d steps",
                    i,
                    phase.promote_condition,
                    timesteps,
                )

            self.stage_stats.append(stats_cb.summary())

            # Save VecNormalize stats for next phase using the official API
            if isinstance(env, VecNormalize):
                vn_path = vec_normalize_dir / f"{phase.name}_vecnormalize.pkl"
                vn_path.parent.mkdir(parents=True, exist_ok=True)
                env.save(str(vn_path))

            if self._save_dir:
                self._save_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self._save_dir / f"stage{i}_{task}"))

            if (
                self.pool is not None
                and hasattr(phase, "opponent")
                and phase.opponent == "pool"
            ):
                self.pool.add(self.model, {"phase": phase.name})

            logger.info("Phase %d done: %s", i, stats_cb.summary())

            env.close()

            if not self.curriculum.is_last_phase():
                self.curriculum.promote()

        return self.model  # type: ignore[return-value]

    def save_checkpoint(self, path: str) -> None:
        """Save model, curriculum state, and pool state."""
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
        """Restore model, curriculum state, and pool state."""
        ckpt_dir = Path(path)
        if (ckpt_dir / "model.zip").exists():
            self.model = PPO.load(str(ckpt_dir / "model"))
        if (ckpt_dir / "curriculum_state.json").exists():
            with open(ckpt_dir / "curriculum_state.json") as f:
                self.curriculum.load_state_dict(json.load(f))
        if self.pool and (ckpt_dir / "pool_state.json").exists():
            with open(ckpt_dir / "pool_state.json") as f:
                self.pool.load_state_dict(json.load(f))
