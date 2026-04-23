"""Gymnasium wrapper for training the Neural-Symbolic mode selector."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aces.env.dogfight import DroneDogfightEnv
from aces.perception.neural_symbolic import NeuralSymbolicPolicy, TacticMode


class NeuralSymbolicEnv(gym.Env):
    """Wraps DroneDogfightEnv for hierarchical RL.

    Action space: Dict with
        - mode: Discrete(4) — tactical mode
        - params: Box(5) — continuous control parameters

    The low-level MPPI controller executes the selected mode.
    Decisions are made every ``decision_interval`` steps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env_kwargs: dict | None = None,
        decision_interval: int = 10,
    ):
        super().__init__()
        base_env_kwargs = base_env_kwargs or {}
        self._env = DroneDogfightEnv(**base_env_kwargs)
        self._decision_interval = decision_interval

        # Observation: same as base env (21-dim vector)
        self.observation_space = self._env.observation_space

        # Action: mode (discrete) + params (continuous)
        self.action_space = spaces.Dict(
            {
                "mode": spaces.Discrete(len(TacticMode)),
                "params": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
            }
        )

        self._ns_policy: NeuralSymbolicPolicy | None = None
        self._last_obs: np.ndarray | None = None

    def set_mppi(self, mppi_controller):
        """Set the MPPI controller for low-level execution."""
        self._ns_policy = NeuralSymbolicPolicy(mppi_controller)

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        """Execute high-level action for decision_interval steps."""
        mode = TacticMode(action["mode"])
        params = action["params"]

        total_reward = 0.0
        total_info: dict = {}
        terminated = False
        truncated = False

        for _ in range(self._decision_interval):
            # Get low-level MPPI action based on mode
            self_state = list(self._env._sim.drone_a_state())
            enemy_state = list(self._env._sim.drone_b_state())

            if mode == TacticMode.PURSUE:
                low_action = (
                    self._env._mppi.compute_action(self_state, enemy_state, True)
                    if self._env._mppi
                    else [0.0] * 4
                )
            elif mode == TacticMode.EVADE:
                low_action = (
                    self._env._mppi.compute_action(self_state, enemy_state, False)
                    if self._env._mppi
                    else [0.0] * 4
                )
            elif mode == TacticMode.SEARCH:
                offset = params[:3] * 2.0
                modified = list(enemy_state)
                modified[0] += offset[0]
                modified[1] += offset[1]
                modified[2] += offset[2]
                low_action = (
                    self._env._mppi.compute_action(self_state, modified, True)
                    if self._env._mppi
                    else [0.0] * 4
                )
            else:  # ORBIT
                low_action = (
                    self._env._mppi.compute_action(self_state, enemy_state, False)
                    if self._env._mppi
                    else [0.0] * 4
                )

            # Convert MPPI motor thrusts to normalized actions [-1, 1]
            hover = self._env._hover_thrust
            max_t = self._env._max_thrust
            norm_action = np.array(
                [
                    (m - hover) / (max_t - hover) if max_t > hover else 0.0
                    for m in low_action
                ],
                dtype=np.float32,
            ).clip(-1.0, 1.0)

            obs, reward, terminated, truncated, info = self._env.step(norm_action)
            total_reward += reward
            total_info = info
            self._last_obs = obs

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, total_info

    def close(self):
        self._env.close()
