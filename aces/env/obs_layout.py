"""Observation vector layout helper.

The 21-dim observation vector used throughout ACES has the following layout
(source of truth: crates/batch-sim/src/observation.rs).

Use `describe_obs(obs)` for a human-readable breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBS_DIM = 21


@dataclass(frozen=True)
class ObsField:
    name: str
    start: int
    length: int
    units: str
    description: str

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


OBS_LAYOUT: tuple[ObsField, ...] = (
    ObsField("own_velocity", 0, 3, "m/s", "own velocity (world frame)"),
    ObsField(
        "own_angular_velocity", 3, 3, "rad/s", "own angular velocity (body frame)"
    ),
    ObsField("rel_position", 6, 3, "m", "relative position to opponent (world frame)"),
    ObsField("rel_velocity", 9, 3, "m/s", "relative velocity (world frame)"),
    ObsField(
        "own_attitude_rpy",
        12,
        3,
        "rad",
        "own attitude (roll, pitch, yaw from quaternion)",
    ),
    ObsField("nearest_sdf", 15, 1, "m", "nearest obstacle distance (SDF)"),
    ObsField("lock_progress_self", 16, 1, "[0,1]", "lock-on progress self->opponent"),
    ObsField(
        "lock_progress_enemy", 17, 1, "[0,1]", "being-locked progress opponent->self"
    ),
    ObsField("opponent_visible", 18, 1, "0/1", "opponent is in line-of-sight"),
    ObsField(
        "belief_uncertainty",
        19,
        1,
        "m^2",
        "opponent position belief variance (0 for MPPI-vs-MPPI)",
    ),
    ObsField(
        "time_since_seen",
        20,
        1,
        "s",
        "time since last opponent observation (0 if always visible)",
    ),
)


def _field_by_name(name: str) -> ObsField:
    for f in OBS_LAYOUT:
        if f.name == name:
            return f
    raise KeyError(f"unknown obs field {name!r}")


def get_field(obs: np.ndarray, name: str) -> np.ndarray | float:
    """Extract a named field from an observation vector or batch.

    Args:
        obs: 1-D array of length OBS_DIM, OR 2-D array (batch, OBS_DIM).
        name: one of the OBS_LAYOUT names.

    Returns:
        Array slice. Shape is the input's trailing dim replaced by the field length,
        or the field as a scalar if input is 1-D and field length is 1.
    """
    field = _field_by_name(name)
    if obs.ndim == 1:
        if obs.size != OBS_DIM:
            raise ValueError(f"1-D obs must have len {OBS_DIM}, got {obs.size}")
        result = obs[field.slice]
        return result if field.length > 1 else result.item()
    if obs.ndim == 2:
        if obs.shape[1] != OBS_DIM:
            raise ValueError(
                f"2-D obs trailing dim must be {OBS_DIM}, got {obs.shape[1]}"
            )
        return obs[:, field.slice]
    raise ValueError(f"obs must be 1-D or 2-D, got ndim={obs.ndim}")


def describe_obs(obs: np.ndarray, env_index: int = 0) -> str:
    """Format a single observation (or one env of a batch) for humans.

    Returns a multi-line string, one line per field, with name, range, units,
    and formatted values.
    """
    if obs.ndim == 2:
        if env_index >= obs.shape[0]:
            raise IndexError(f"env_index {env_index} >= batch size {obs.shape[0]}")
        obs_1d = obs[env_index]
    elif obs.ndim == 1:
        obs_1d = obs
    else:
        raise ValueError(f"obs must be 1-D or 2-D, got ndim={obs.ndim}")

    if obs_1d.size != OBS_DIM:
        raise ValueError(f"obs must have {OBS_DIM} elements, got {obs_1d.size}")

    lines = [f"Observation ({OBS_DIM}-dim):"]
    for f in OBS_LAYOUT:
        values = obs_1d[f.slice]
        if f.length == 1:
            val_str = f"{values.item(): .4f}"
        else:
            val_str = "[" + ", ".join(f"{v: .4f}" for v in values) + "]"
        lines.append(
            f"  [{f.start:>2}:{f.end:<2}] {f.name:<22} = {val_str:<40} ({f.units})  # {f.description}"
        )
    return "\n".join(lines)


def verify_layout() -> None:
    """Raises AssertionError if OBS_LAYOUT is inconsistent.

    Useful as a runtime self-check at program startup.
    """
    total = sum(f.length for f in OBS_LAYOUT)
    assert total == OBS_DIM, f"OBS_LAYOUT total {total} != OBS_DIM {OBS_DIM}"

    # Check fields are contiguous and sorted
    for i, f in enumerate(OBS_LAYOUT):
        expected_start = sum(g.length for g in OBS_LAYOUT[:i])
        assert f.start == expected_start, (
            f"field {f.name!r} start {f.start} != expected {expected_start}"
        )

    # Check names are unique
    names = [f.name for f in OBS_LAYOUT]
    assert len(names) == len(set(names)), (
        f"duplicate field names in OBS_LAYOUT: {names}"
    )
