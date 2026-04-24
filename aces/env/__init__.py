"""ACES environment subpackage."""

# Trajectory is pure-Python and safe to import eagerly.
from aces.env.trajectory import Trajectory as Trajectory

# DroneDogfightEnv / NeuralSymbolicEnv require the Rust `aces._core` extension
# (built via `maturin develop`). In CPU-only / pre-build environments (e.g.
# unit tests for pure-Python helpers like obs_layout) we still want
# `import aces.env` to succeed, so guard the eager import.
try:
    from aces.env.dogfight import DroneDogfightEnv as DroneDogfightEnv
    from aces.env.ns_env import NeuralSymbolicEnv as NeuralSymbolicEnv
except ImportError:
    DroneDogfightEnv = None  # type: ignore[assignment,misc]
    NeuralSymbolicEnv = None  # type: ignore[assignment,misc]

from aces.env.obs_layout import OBS_DIM as OBS_DIM
from aces.env.obs_layout import OBS_LAYOUT as OBS_LAYOUT
from aces.env.obs_layout import describe_obs as describe_obs
from aces.env.obs_layout import get_field as get_field

__all__ = [
    "DroneDogfightEnv",
    "NeuralSymbolicEnv",
    "Trajectory",
    "OBS_DIM",
    "OBS_LAYOUT",
    "describe_obs",
    "get_field",
]
