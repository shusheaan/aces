"""God Oracle: computes ground truth semantic labels from simulator state.

Uses omniscient information available in simulation (true positions,
velocities, lock-on state) to produce supervised training labels
for the Perception NN.
"""

from __future__ import annotations


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


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
