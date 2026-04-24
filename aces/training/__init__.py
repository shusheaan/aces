"""ACES training subpackage — callbacks, evaluation, logging, and utilities."""

from __future__ import annotations

from aces.training.batched_vec_env import BatchedOpponentVecEnv
from aces.training.callbacks import (
    CheckpointResumeCallback,
    EpisodeLoggerCallback,
    OpponentUpdateCallback,
    PoolOpponentCallback,
    StateCallback,
    TensorBoardMetricsCallback,
    TrainingStatsCallback,
    VecOpponentUpdateCallback,
    WindowSummaryCallback,
)
from aces.training.evaluate import evaluate
from aces.training.logging import (
    create_run_dir,
    save_config_snapshot,
    save_run_metadata,
    setup_logging,
)
from aces.training.opponent_pool import OpponentPool, PoolEntry
from aces.training.self_play import SelfPlayTrainer
from aces.training.curriculum_trainer import CurriculumTrainer

try:
    from aces.training.gpu_vec_env import GpuVecEnv
except ImportError:
    GpuVecEnv = None  # type: ignore[assignment,misc]

__all__ = [
    # trainers
    "SelfPlayTrainer",
    "CurriculumTrainer",
    # batched_vec_env
    "BatchedOpponentVecEnv",
    # gpu_vec_env (optional — None if GPU ext not built)
    "GpuVecEnv",
    # callbacks
    "CheckpointResumeCallback",
    "EpisodeLoggerCallback",
    "OpponentUpdateCallback",
    "PoolOpponentCallback",
    "StateCallback",
    "TensorBoardMetricsCallback",
    "TrainingStatsCallback",
    "VecOpponentUpdateCallback",
    "WindowSummaryCallback",
    # evaluate
    "evaluate",
    # logging
    "create_run_dir",
    "save_config_snapshot",
    "save_run_metadata",
    "setup_logging",
    # opponent_pool
    "OpponentPool",
    "PoolEntry",
    # utils
    "resolve_policy",
]


def resolve_policy(fpv: bool) -> tuple[str, dict | None]:
    """Return (policy_name, policy_kwargs) based on observation mode."""
    if fpv:
        from aces.policy.extractors import CnnImuExtractor

        return "MultiInputPolicy", {
            "features_extractor_class": CnnImuExtractor,
            "features_extractor_kwargs": {"features_dim": 192},
        }
    return "MlpPolicy", None
