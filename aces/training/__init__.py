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
from aces.training.evaluate import _pick, evaluate
from aces.training.logging import (
    create_run_dir,
    save_config_snapshot,
    save_run_metadata,
    setup_logging,
)
from aces.training.opponent_pool import OpponentPool, PoolEntry
from aces.training.self_play import SelfPlayTrainer
from aces.training.curriculum_trainer import CurriculumTrainer

__all__ = [
    # trainers
    "SelfPlayTrainer",
    "CurriculumTrainer",
    # batched_vec_env
    "BatchedOpponentVecEnv",
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
    "_pick",
    "evaluate",
    # logging
    "create_run_dir",
    "save_config_snapshot",
    "save_run_metadata",
    "setup_logging",
    # opponent_pool
    "OpponentPool",
    "PoolEntry",
]
