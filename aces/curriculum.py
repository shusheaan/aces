"""TOML-driven curriculum manager for progressive training phases.

Loads a curriculum definition from a TOML file and manages phase
advancement based on configurable promotion conditions (win rate,
reward threshold, or timestep limits).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Regex for "metric>threshold" conditions (e.g. "win_rate>0.3")
_CONDITION_RE = re.compile(r"^(\w+)>([\d.]+)$")


@dataclass
class Phase:
    """A single curriculum phase definition."""

    name: str
    task: str = "dogfight"
    opponent: str = "random"
    wind_sigma: float = 0.0
    obs_noise_std: float = 0.0
    use_fpv: bool = False
    max_timesteps: int = 500_000
    promote_condition: str = "steps"
    promote_window: int = 100


def _project_root() -> Path:
    """Return the project root (directory containing 'configs/')."""
    return Path(__file__).resolve().parent.parent


def load_curriculum(path: str | Path | None = None) -> list[Phase]:
    """Load curriculum phases from a TOML file.

    Parameters
    ----------
    path:
        Path to the curriculum TOML file.  Defaults to
        ``<project_root>/configs/curriculum.toml``.

    Returns
    -------
    list[Phase]
        Ordered list of curriculum phases.
    """
    if path is None:
        path = _project_root() / "configs" / "curriculum.toml"
    else:
        path = Path(path)

    with open(path, "rb") as f:
        data = tomllib.load(f)

    phases: list[Phase] = []
    for entry in data.get("phase", []):
        phases.append(Phase(**entry))
    return phases


class CurriculumManager:
    """Manages progression through curriculum phases.

    Parameters
    ----------
    phases:
        Ordered list of :class:`Phase` objects defining the curriculum.
    """

    def __init__(self, phases: list[Phase]) -> None:
        if not phases:
            raise ValueError("Curriculum must have at least one phase")
        self._phases = phases
        self.phase_index: int = 0

    def current_phase(self) -> Phase:
        """Return the current curriculum phase."""
        return self._phases[self.phase_index]

    def is_last_phase(self) -> bool:
        """Return True if the current phase is the final one."""
        return self.phase_index >= len(self._phases) - 1

    def should_promote(self, stats: dict) -> bool:
        """Check whether promotion condition is satisfied.

        Parameters
        ----------
        stats:
            Dictionary containing at minimum ``"episodes"`` (int) and
            the metric referenced by the promote condition (e.g.
            ``"win_rate"``, ``"reward"``).

        Returns
        -------
        bool
            True if the agent should be promoted to the next phase.
        """
        phase = self.current_phase()
        condition = phase.promote_condition

        # "steps" — always True (caller is responsible for timestep limits)
        if condition == "steps":
            return True

        # "metric>threshold" format
        match = _CONDITION_RE.match(condition)
        if match is None:
            raise ValueError(f"Unknown promote_condition format: {condition!r}")

        metric = match.group(1)
        threshold = float(match.group(2))

        # Need enough episodes before evaluating
        episodes = stats.get("episodes", 0)
        if episodes < phase.promote_window:
            return False

        value = stats.get(metric, 0.0)
        return value > threshold

    def promote(self) -> Phase | None:
        """Advance to the next curriculum phase.

        Returns
        -------
        Phase | None
            The new phase, or None if already at the last phase.
        """
        if self.is_last_phase():
            return None
        self.phase_index += 1
        return self.current_phase()

    def state_dict(self) -> dict:
        """Return serialisable state for checkpointing."""
        return {"phase_index": self.phase_index}

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a checkpoint dictionary."""
        self.phase_index = state["phase_index"]
