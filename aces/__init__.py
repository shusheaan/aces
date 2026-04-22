"""ACES -- Air Combat Engagement Simulation.

Quadrotor drone dogfight with MPPI control and reinforcement learning.
"""

__version__ = "0.1.0"

from aces.config import AcesConfig as AcesConfig
from aces.config import load_configs as load_configs
from aces.curriculum import CurriculumManager as CurriculumManager
from aces.env import DroneDogfightEnv as DroneDogfightEnv
from aces.opponent_pool import OpponentPool as OpponentPool
from aces.policy import CnnImuExtractor as CnnImuExtractor
from aces.trainer import CurriculumTrainer as CurriculumTrainer
from aces.trainer import SelfPlayTrainer as SelfPlayTrainer
from aces.viz import AcesVisualizer as AcesVisualizer
