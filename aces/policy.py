"""CNN + MLP hybrid policy for FPV depth-image observations."""

from __future__ import annotations

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CnnImuExtractor(BaseFeaturesExtractor):
    """Feature extractor for Dict observations with depth image + IMU vector.

    Processes:
      - "image": (1, 60, 80) depth image through a 3-layer CNN
      - "vector": (12,) IMU/proprioceptive vector through a small MLP

    Concatenates both feature streams into a single vector.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 192):
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        vector_space = observation_space["vector"]

        n_channels = image_space.shape[0]  # 1
        vector_dim = vector_space.shape[0]  # 12

        # CNN for depth image: (1, 60, 80) -> 128-dim
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with th.no_grad():
            sample = th.zeros(1, *image_space.shape)
            cnn_out_dim = self.cnn(sample).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_out_dim, 128),
            nn.ReLU(),
        )

        # MLP for IMU vector: 12 -> 64
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
        )

        # Verify output dimension matches
        assert 128 + 64 == features_dim

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        image_features = self.cnn_linear(self.cnn(observations["image"]))
        vector_features = self.vector_mlp(observations["vector"])
        return th.cat([image_features, vector_features], dim=1)
