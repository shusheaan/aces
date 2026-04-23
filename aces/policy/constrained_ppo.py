"""Lagrangian PPO: PPO with adaptive constraint penalties."""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class CostValueNetwork(nn.Module):
    """Separate value network for constraint cost estimation."""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(obs)
        return result


class LagrangianPPO:
    """Wraps SB3 PPO with Lagrangian constraint handling.

    Maintains a Lagrange multiplier that scales a cost penalty added
    to the reward. The multiplier is updated via dual gradient ascent
    to satisfy ``E[sum(cost)] <= cost_limit``.
    """

    def __init__(
        self,
        policy: str,
        env,
        cost_limit: float = 25.0,
        lambda_init: float = 0.0,
        lambda_lr: float = 0.005,
        lambda_max: float = 50.0,
        cost_gamma: float = 0.99,
        **ppo_kwargs,
    ):
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.cost_gamma = cost_gamma
        self.lam = lambda_init  # Lagrange multiplier

        self.ppo = PPO(policy, env, **ppo_kwargs)
        self._episode_costs: list[float] = []
        self._current_episode_cost: float = 0.0

    def _compute_cost(self, info: dict) -> float:
        """Extract constraint cost from step info.

        Cost sources:
        - collision: 1.0 if drone collided
        - near_wall: max(0, 0.3 - sdf) / 0.3 if SDF < 0.3m
        - angular_velocity: max(0, |omega| - 10.0) / 10.0 if |omega| > 10 rad/s
        """
        cost = 0.0
        if info.get("collision", False):
            cost += 1.0
        sdf = info.get("nearest_obs_dist", 1.0)
        if sdf < 0.3:
            cost += (0.3 - sdf) / 0.3
        ang_vel = info.get("angular_velocity_norm", 0.0)
        if ang_vel > 10.0:
            cost += (ang_vel - 10.0) / 10.0
        return cost

    def _update_lambda(self):
        """Dual gradient ascent on the Lagrange multiplier."""
        if not self._episode_costs:
            return
        mean_cost = np.mean(self._episode_costs)
        # gradient: (mean_cost - cost_limit)
        self.lam = max(
            0.0,
            min(
                self.lambda_max,
                self.lam + self.lambda_lr * (mean_cost - self.cost_limit),
            ),
        )
        self._episode_costs.clear()

    def learn(self, total_timesteps: int, **kwargs):
        """Train with Lagrangian constraint."""
        cb: BaseCallback = LagrangianCallback(self)
        existing_cb = kwargs.pop("callback", None)
        if existing_cb is not None:
            from stable_baselines3.common.callbacks import CallbackList

            cb = CallbackList([cb, existing_cb])
        self.ppo.learn(total_timesteps=total_timesteps, callback=cb, **kwargs)
        return self

    @property
    def policy(self):
        return self.ppo.policy

    def save(self, path: str):
        self.ppo.save(path)

    @classmethod
    def load(cls, path: str, env, **kwargs):
        instance = cls.__new__(cls)
        instance.ppo = PPO.load(path, env=env)
        instance.cost_limit = kwargs.get("cost_limit", 25.0)
        instance.lambda_lr = kwargs.get("lambda_lr", 0.005)
        instance.lambda_max = kwargs.get("lambda_max", 50.0)
        instance.cost_gamma = kwargs.get("cost_gamma", 0.99)
        instance.lam = kwargs.get("lambda_init", 0.0)
        instance._episode_costs = []
        instance._current_episode_cost = 0.0
        return instance


class LagrangianCallback(BaseCallback):
    """Callback that modifies rewards with Lagrangian cost penalty."""

    def __init__(self, lagr: LagrangianPPO):
        super().__init__()
        self.lagr = lagr
        self._step_costs: list[float] = []
        self._current_episode_costs: np.ndarray = np.array([0.0])

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._current_episode_costs = np.zeros(n, dtype=np.float64)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards")
        if rewards is None:
            return True

        for i, info in enumerate(infos):
            cost = self.lagr._compute_cost(info)
            self._step_costs.append(cost)
            rewards[i] -= self.lagr.lam * cost

            self._current_episode_costs[i] += cost
            if info.get("terminated", False) or info.get("truncated", False):
                self.lagr._episode_costs.append(float(self._current_episode_costs[i]))
                self._current_episode_costs[i] = 0.0

        return True

    def _on_rollout_end(self) -> None:
        self.lagr._update_lambda()
        if self.logger is not None:
            self.logger.record("constraint/lambda", self.lagr.lam)
            if self._step_costs:
                self.logger.record(
                    "constraint/mean_step_cost", np.mean(self._step_costs)
                )
            if self.lagr._episode_costs:
                self.logger.record(
                    "constraint/mean_episode_cost", np.mean(self.lagr._episode_costs)
                )
        self._step_costs.clear()
