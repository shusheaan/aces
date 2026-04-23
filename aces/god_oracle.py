"""God Oracle: computes ground truth semantic labels from simulator state.

Uses omniscient information available in simulation (true positions,
velocities, lock-on state) to produce supervised training labels
for the Perception NN.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class GodOracle:
    """Compute ground truth semantic features from simulation state."""

    def compute(
        self,
        *,
        lock_b_progress: float,
        distance: float,
        opponent_facing_me: float,
        lock_a_progress: float,
        a_sees_b: bool,
        i_face_opponent: float,
        nearest_obs_dist: float,
        speed: float,
        belief_var: float,
        opponent_closing_speed: float,
    ) -> dict[str, float]:
        threat = _clamp(
            0.4 * lock_b_progress
            + 0.3 * max(0.0, 1.0 - distance / 5.0)
            + 0.3 * _clamp(opponent_facing_me, 0.0, 1.0)
        )

        opportunity = _clamp(
            0.3 * lock_a_progress
            + 0.3 * max(0.0, 1.0 - distance / 5.0)
            + 0.2 * _clamp(i_face_opponent, 0.0, 1.0)
            + 0.2 * float(a_sees_b)
        )

        wall_proximity = max(0.0, 1.0 - nearest_obs_dist / 0.5)
        speed_factor = min(1.0, speed / 3.0)
        collision_risk = _clamp(wall_proximity * speed_factor)

        uncertainty = _clamp(belief_var / 5.0)

        opponent_distance = max(0.0, distance)

        if opponent_closing_speed > 0.5:
            opponent_intent = 0  # approach
        elif opponent_closing_speed < -0.5:
            opponent_intent = 1  # flee
        else:
            opponent_intent = 2  # patrol

        return {
            "threat": threat,
            "opportunity": opportunity,
            "collision_risk": collision_risk,
            "uncertainty": uncertainty,
            "opponent_distance": opponent_distance,
            "opponent_intent": opponent_intent,
        }


def extract_oracle_inputs(
    obs: NDArray[np.floating[Any]], info: dict[str, Any]
) -> dict[str, Any]:
    """Extract GodOracle inputs from a 21-dim observation and step info.

    Computes facing angles and closing speed from the observation vector
    so callers don't need to repeat the geometry math.
    """
    rel_pos = obs[6:9]
    own_vel = obs[0:3]
    opp_vel = obs[9:12]
    rel_dist = float(np.linalg.norm(rel_pos))

    if rel_dist > 0.01:
        direction_to_me = -rel_pos / rel_dist
        opp_vel_norm = max(float(np.linalg.norm(opp_vel)), 0.01)
        own_vel_norm = max(float(np.linalg.norm(own_vel)), 0.01)
        opponent_facing_me = max(
            0.0, float(np.dot(opp_vel / opp_vel_norm, direction_to_me))
        )
        i_face_opponent = max(
            0.0, float(np.dot(own_vel / own_vel_norm, rel_pos / rel_dist))
        )
        closing_speed = -float(np.dot(opp_vel, direction_to_me))
    else:
        opponent_facing_me = 0.0
        i_face_opponent = 0.0
        closing_speed = 0.0

    return {
        "lock_b_progress": float(info.get("being_locked_progress", obs[17])),
        "distance": float(info.get("distance", rel_dist)),
        "opponent_facing_me": opponent_facing_me,
        "lock_a_progress": float(info.get("lock_progress", obs[16])),
        "a_sees_b": bool(obs[18] > 0.5),
        "i_face_opponent": i_face_opponent,
        "nearest_obs_dist": float(info.get("nearest_obs_dist", obs[15])),
        "speed": float(np.linalg.norm(own_vel)),
        "belief_var": float(info.get("belief_var", obs[19])),
        "opponent_closing_speed": closing_speed,
    }


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
