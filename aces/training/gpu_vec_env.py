"""SB3-compatible wrapper around the Rust `aces._core.GpuVecEnv` (GPU MPPI opponent).

The underlying Rust class runs N parallel battles on the GPU with both sides driven
by MPPI. This wrapper:
  - Presents the Gymnasium Box observation_space / action_space that SB3 expects
  - Translates SB3's step_async / step_wait / reset / close protocol
  - Exposes agent A (the learning agent) via `step_with_agent_a`; agent B
    runs GPU MPPI internally
  - Denormalizes SB3's action space [-1, 1] to motor thrusts [0, max_thrust] before
    passing into the Rust side
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

try:
    from aces._core import GpuVecEnv as _RustGpuVecEnv

    _GPU_AVAILABLE = True
except ImportError:
    _RustGpuVecEnv = None
    _GPU_AVAILABLE = False

# Crazyflie default — must match `DroneParams::crazyflie().max_thrust`
# (see configs/drone.toml: max_motor_thrust = 0.15).
MAX_THRUST_PER_MOTOR = 0.15  # N


class GpuVecEnv(VecEnv):
    """SB3 VecEnv backed by the Rust GPU MPPI orchestrator.

    Observation: 21-dim flat float32 vector per env.
    Action: 4-dim float32 vector in [-1, 1] per env; denormalized to
    [0, MAX_THRUST_PER_MOTOR] before being passed to the Rust side.

    Agent A is controlled by the external (learning) policy. Agent B is driven
    by the GPU MPPI opponent.
    """

    def __init__(
        self,
        n_envs: int,
        mppi_samples: int = 128,
        mppi_horizon: int = 15,
        noise_std: float = 0.03,
        max_steps: int = 1000,
        dt_ctrl: float = 0.01,
        substeps: int = 10,
        wind_sigma: float = 0.0,
        seed: int = 42,
    ):
        if not _GPU_AVAILABLE:
            raise RuntimeError(
                "aces._core.GpuVecEnv not available — rebuild with --features gpu"
            )

        observation_space = Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        super().__init__(n_envs, observation_space, action_space)

        self._rust = _RustGpuVecEnv(
            n_envs=n_envs,
            mppi_samples=mppi_samples,
            mppi_horizon=mppi_horizon,
            noise_std=noise_std,
            max_steps=max_steps,
            dt_ctrl=dt_ctrl,
            substeps=substeps,
            wind_sigma=wind_sigma,
            seed=seed,
        )

        self._last_actions: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # VecEnv protocol
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> VecEnvObs:
        # Note: `seed` / `options` exist for VecEnv compatibility. The Rust
        # side is seeded at construction and doesn't accept per-reset seeds.
        del seed, options
        obs: np.ndarray = self._rust.reset()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._last_actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        if self._last_actions is None:
            raise RuntimeError("step_wait called without prior step_async")

        # Denormalize [-1, 1] -> [0, MAX_THRUST_PER_MOTOR]
        raw = (self._last_actions.astype(np.float32) + 1.0) * 0.5 * MAX_THRUST_PER_MOTOR
        raw = np.clip(raw, 0.0, MAX_THRUST_PER_MOTOR).astype(np.float32)

        obs, rewards, dones, infos = self._rust.step_with_agent_a(raw)
        self._last_actions = None

        return (
            obs,
            rewards.astype(np.float32),
            dones.astype(bool),
            list(infos),
        )

    def close(self) -> None:
        # Rust side cleans up on drop.
        self._rust = None

    # ------------------------------------------------------------------
    # Misc VecEnv protocol methods (minimal implementations)
    # ------------------------------------------------------------------

    def get_attr(self, attr_name: str, indices=None) -> list:
        values = [getattr(self, attr_name, None)] * self.num_envs
        if indices is None:
            return values
        return [values[i] for i in indices]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        raise NotImplementedError("set_attr not supported on GpuVecEnv")

    def env_method(
        self, method_name: str, *method_args, indices=None, **method_kwargs
    ) -> list:
        raise NotImplementedError("env_method not supported on GpuVecEnv")

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        if indices is None:
            return [False] * self.num_envs
        return [False] * len(list(indices))

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        # The Rust side is seeded at construction; this is a no-op.
        return [seed] * self.num_envs
