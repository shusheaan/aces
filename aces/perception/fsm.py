"""Symbolic Finite State Machine for tactical drone decision-making.

Reads semantic features from the Perception NN and selects a tactical
mode. Outputs mode, safety margin, and pursuit flag for the MPPI
controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DroneMode(IntEnum):
    HOVER = 0
    PURSUE = 1
    EVADE = 2
    SEARCH = 3
    ORBIT = 4


_PRIORITY = {
    DroneMode.HOVER: 0,
    DroneMode.EVADE: 1,
    DroneMode.PURSUE: 2,
    DroneMode.SEARCH: 3,
    DroneMode.ORBIT: 4,
}


@dataclass
class FsmOutput:
    mode: DroneMode
    d_safe: float
    pursuit: bool


class SymbolicFSM:
    """Priority-based FSM with hysteresis.

    Transition rules (checked in priority order):
        1. collision_risk > 0.7  -> HOVER
        2. threat > 0.7          -> EVADE
        3. opportunity > 0.6 and opponent_distance < 3  -> PURSUE
        4. uncertainty > 0.5     -> SEARCH
        5. else                  -> ORBIT

    Hysteresis: a lower-priority mode must be requested for
    ``hysteresis_ticks`` consecutive ticks before the FSM switches
    to it. Higher-priority modes override immediately.
    """

    def __init__(self, hysteresis_ticks: int = 10) -> None:
        self.mode: DroneMode = DroneMode.ORBIT
        self.hysteresis_ticks = hysteresis_ticks
        self._ticks_requesting: int = 0
        self._requested_mode: DroneMode = DroneMode.ORBIT

    def step(self, features: dict[str, float]) -> FsmOutput:
        desired = self._evaluate(features)

        if desired == self.mode:
            self._ticks_requesting = 0
            self._requested_mode = desired
        elif _PRIORITY[desired] < _PRIORITY[self.mode]:
            self.mode = desired
            self._ticks_requesting = 0
            self._requested_mode = desired
        elif desired == self._requested_mode:
            self._ticks_requesting += 1
            if self._ticks_requesting >= self.hysteresis_ticks:
                self.mode = desired
                self._ticks_requesting = 0
        else:
            self._requested_mode = desired
            self._ticks_requesting = 1

        collision_risk = features.get("collision_risk", 0.0)
        d_safe = 0.3 + 0.3 * collision_risk
        pursuit = self.mode in (DroneMode.PURSUE, DroneMode.SEARCH)

        return FsmOutput(mode=self.mode, d_safe=d_safe, pursuit=pursuit)

    def _evaluate(self, f: dict[str, float]) -> DroneMode:
        if f.get("collision_risk", 0.0) > 0.7:
            return DroneMode.HOVER
        if f.get("threat", 0.0) > 0.7:
            return DroneMode.EVADE
        if f.get("opportunity", 0.0) > 0.6 and f.get("opponent_distance", 99.0) < 3.0:
            return DroneMode.PURSUE
        if f.get("uncertainty", 0.0) > 0.5:
            return DroneMode.SEARCH
        return DroneMode.ORBIT
