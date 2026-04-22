"""Neural-Symbolic policy: small NN mode selector + MPPI execution."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn


class TacticMode(IntEnum):
    PURSUE = 0
    EVADE = 1
    SEARCH = 2
    ORBIT = 3


class ModeSelector(nn.Module):
    """Small MLP that selects tactical mode from observation.

    Input: 21-dim vector observation
    Output: 4-dim logits (pursue, evade, search, orbit)
    """

    def __init__(self, obs_dim: int = 21, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, len(TacticMode)),
        )
        # Also output continuous parameters for the controller
        self.param_head = nn.Sequential(
            nn.Linear(
                hidden, 5
            ),  # [target_offset_x, y, z, aggressiveness, safety_margin]
            nn.Tanh(),
        )
        # Store intermediate for param_head
        self._hidden: torch.Tensor | None = None

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = obs
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        self._hidden = x
        mode_logits: torch.Tensor = list(self.net.children())[-1](x)
        params: torch.Tensor = self.param_head(x)
        return mode_logits, params


class NeuralSymbolicPolicy:
    """Hierarchical policy: NN selects mode, MPPI/PD executes.

    Architecture:
        Observation → ModeSelector (small NN, ~2K params)
            → mode: {PURSUE, EVADE, SEARCH, ORBIT}
            → params: [offset_x, offset_y, offset_z, aggressiveness, safety_margin]
                → MPPI controller (traditional, not learned)
                    → motor commands [f1, f2, f3, f4]
    """

    def __init__(self, mppi_controller, obs_dim: int = 21):
        self.mode_net = ModeSelector(obs_dim=obs_dim)
        self.mppi = mppi_controller
        self._device = torch.device("cpu")

    def to(self, device: str | torch.device) -> "NeuralSymbolicPolicy":
        self._device = torch.device(device)
        self.mode_net = self.mode_net.to(self._device)
        return self

    def select_mode(self, obs: np.ndarray) -> tuple[TacticMode, np.ndarray]:
        """Select tactical mode and continuous parameters."""
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            logits, params = self.mode_net(obs_t)
            mode = TacticMode(logits.argmax(dim=-1).item())
            params_np = params.squeeze(0).cpu().numpy()
        return mode, params_np

    def act(
        self,
        obs: np.ndarray,
        self_state: list[float],
        enemy_state: list[float],
        belief_var: float = 0.0,
    ) -> list[float]:
        """Full neural-symbolic action pipeline."""
        mode, params = self.select_mode(obs)

        # Decode continuous parameters
        target_offset = params[:3] * 2.0  # scale to [-2, 2] meters
        aggressiveness = (params[3] + 1.0) / 2.0  # [0, 1]
        _safety_margin = (
            params[4] * 0.25 + 0.35
        )  # [0.1, 0.6] meters (reserved for future use)

        if mode == TacticMode.PURSUE:
            return list(
                self.mppi.compute_action_with_belief(  # type: ignore[no-any-return]
                    self_state, enemy_state, True, belief_var
                )
            )
        elif mode == TacticMode.EVADE:
            return list(
                self.mppi.compute_action_with_belief(  # type: ignore[no-any-return]
                    self_state, enemy_state, False, belief_var
                )
            )
        elif mode == TacticMode.SEARCH:
            return self._search_action(self_state, enemy_state, target_offset)
        else:  # ORBIT
            return self._orbit_action(self_state, enemy_state, aggressiveness)

    def _search_action(
        self,
        self_state: list[float],
        enemy_state: list[float],
        offset: np.ndarray,
    ) -> list[float]:
        """Search mode: pursue last-known position with offset."""
        modified_enemy = list(enemy_state)
        modified_enemy[0] += offset[0]
        modified_enemy[1] += offset[1]
        modified_enemy[2] += offset[2]
        return list(self.mppi.compute_action(self_state, modified_enemy, True))  # type: ignore[no-any-return]

    def _orbit_action(
        self,
        self_state: list[float],
        enemy_state: list[float],
        aggressiveness: float,
    ) -> list[float]:
        """Orbit mode: maintain distance, circle opponent."""
        # Use evasion MPPI with pursuit=False to maintain distance
        return list(self.mppi.compute_action(self_state, enemy_state, False))  # type: ignore[no-any-return]

    def state_dict(self) -> dict:
        return self.mode_net.state_dict()

    def load_state_dict(self, sd: dict):
        self.mode_net.load_state_dict(sd)

    def parameters(self):
        return self.mode_net.parameters()
