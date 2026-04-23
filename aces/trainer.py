"""Backward-compatibility shim. Import from aces.training instead."""

from aces.training.callbacks import (
    CheckpointResumeCallback as CheckpointResumeCallback,
    EpisodeLoggerCallback as EpisodeLoggerCallback,
    OpponentUpdateCallback as OpponentUpdateCallback,
    PoolOpponentCallback as PoolOpponentCallback,
    StateCallback as StateCallback,
    TensorBoardMetricsCallback as TensorBoardMetricsCallback,
    TrainingStatsCallback as TrainingStatsCallback,
    VecOpponentUpdateCallback as VecOpponentUpdateCallback,
    WindowSummaryCallback as WindowSummaryCallback,
)
from aces.training.curriculum_trainer import CurriculumTrainer as CurriculumTrainer
from aces.training.evaluate import evaluate as evaluate
from aces.training.self_play import SelfPlayTrainer as SelfPlayTrainer
from aces.training.self_play import _pick as _pick
