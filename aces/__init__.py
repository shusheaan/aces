"""ACES -- Air Combat Engagement Simulation.

Quadrotor drone dogfight with MPPI control and reinforcement learning.
"""

__version__ = "0.1.0"

from aces.config import AcesConfig as AcesConfig
from aces.config import load_configs as load_configs
from aces.curriculum import CurriculumManager as CurriculumManager
from aces.viz import AcesVisualizer as AcesVisualizer

# DroneDogfightEnv requires the Rust `aces._core` extension (built via
# `maturin develop`). In CPU-only / pre-build environments (e.g. unit tests
# for pure-Python helpers) we still want `import aces` to succeed, so guard
# the eager import. Modules that actually need DroneDogfightEnv will import
# `aces.env.dogfight` directly and fail with a clear ImportError there.
try:
    from aces.env import DroneDogfightEnv as DroneDogfightEnv

    _HAS_CORE = True
except ImportError:
    DroneDogfightEnv = None  # type: ignore[assignment,misc]
    _HAS_CORE = False

__all__ = [
    "AcesConfig",
    "load_configs",
    "CurriculumManager",
    "DroneDogfightEnv",
    "AcesVisualizer",
]


def __getattr__(name: str):
    """Lazy import heavy training dependencies."""
    if name == "CurriculumTrainer":
        from aces.training import CurriculumTrainer

        return CurriculumTrainer
    if name == "SelfPlayTrainer":
        from aces.training import SelfPlayTrainer

        return SelfPlayTrainer
    if name == "OpponentPool":
        from aces.training import OpponentPool

        return OpponentPool
    if name == "CnnImuExtractor":
        from aces.policy import CnnImuExtractor

        return CnnImuExtractor
    raise AttributeError(f"module 'aces' has no attribute {name!r}")
