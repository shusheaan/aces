"""VecEnv wrapper that batches opponent inference across all environments.

Instead of each SubprocVecEnv worker running opponent policy.predict()
independently (N serial CPU inferences), this wrapper collects opponent
observations from all N envs, runs a single batched forward pass on GPU,
and sets the resulting actions back in the envs before stepping.

Usage:
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = BatchedOpponentVecEnv(vec_env)
    vec_env.set_opponent_policy(some_sb3_policy)
    # Now PPO.learn() will automatically use batched opponent inference
"""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class BatchedOpponentVecEnv(VecEnvWrapper):
    """Wraps a VecEnv to batch opponent policy inference in the main process.

    When an opponent policy is set, each call to ``step_async``:
    1. Collects opponent observations from all N envs (via ``env_method``)
    2. Runs a single batched ``policy.predict()`` on the main process (GPU)
    3. Sets the resulting actions in each env (via ``env_method``)
    4. Proceeds with the normal ``step_async`` dispatch

    When no opponent policy is set, acts as a transparent pass-through.
    """

    def __init__(self, venv):
        super().__init__(venv)
        self._opponent_policy = None

    @property
    def has_opponent(self) -> bool:
        return self._opponent_policy is not None

    def set_opponent_policy(self, policy) -> None:
        """Set the opponent policy for batched inference."""
        self._opponent_policy = policy

    def set_opponent_weights(self, state_dict: dict) -> None:
        """Update opponent policy weights (called by training callbacks)."""
        if self._opponent_policy is not None:
            self._opponent_policy.load_state_dict(state_dict)

    def step_async(self, actions: np.ndarray) -> None:
        if self._opponent_policy is not None:
            self._inject_opponent_actions()
        self.venv.step_async(actions)

    def _inject_opponent_actions(self) -> None:
        """Collect opponent obs, batch predict, set actions in envs."""
        # 1. Collect opponent observations from all envs
        opp_obs_list = self.venv.env_method("get_opponent_obs")
        opp_obs = np.stack(opp_obs_list)  # (N, obs_dim)

        # 2. Single batched forward pass
        raw_actions, _ = self._opponent_policy.predict(opp_obs, deterministic=True)

        # 3. Set actions in each env (env maps [-1,1] → motor thrusts)
        for i in range(self.num_envs):
            self.venv.env_method(
                "set_next_opponent_action",
                raw_actions[i].tolist(),
                indices=[i],
            )

    # ------------------------------------------------------------------
    # Pass-through for VecEnv interface methods used by callbacks
    # ------------------------------------------------------------------

    def env_method(
        self, method_name: str, *method_args, indices=None, **method_kwargs
    ) -> list[Any]:
        """Forward env_method calls to the underlying VecEnv.

        Special case: ``set_opponent_weights`` is intercepted to update the
        batched policy instead of (or in addition to) per-env policies.
        """
        if method_name == "set_opponent_weights" and self._opponent_policy is not None:
            # Update the batched policy
            if method_args:
                self.set_opponent_weights(method_args[0])
            return [None] * self.num_envs
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )
