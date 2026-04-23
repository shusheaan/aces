"""Tests for aces.perception — Perception NN and export."""

import struct
from pathlib import Path

import numpy as np
import torch

from aces.perception import PerceptionNet, export_perception


def test_perception_net_output_shape() -> None:
    net = PerceptionNet()
    obs = torch.randn(1, 21)
    continuous, intent_logits = net(obs)
    assert continuous.shape == (1, 5)
    assert intent_logits.shape == (1, 3)


def test_perception_net_batch() -> None:
    net = PerceptionNet()
    obs = torch.randn(32, 21)
    continuous, intent_logits = net(obs)
    assert continuous.shape == (32, 5)
    assert intent_logits.shape == (32, 3)


def test_continuous_outputs_bounded() -> None:
    net = PerceptionNet()
    obs = torch.randn(100, 21)
    continuous, _ = net(obs)
    # First 4 outputs use sigmoid -> [0, 1]
    assert continuous[:, :4].min() >= 0.0
    assert continuous[:, :4].max() <= 1.0
    # 5th output (opponent_dist) uses softplus -> [0, inf)
    assert continuous[:, 4].min() >= 0.0


def test_export_creates_valid_binary(tmp_path: Path) -> None:
    net = PerceptionNet()
    out_path = str(tmp_path / "perception.bin")
    export_perception(net, out_path)
    data = open(out_path, "rb").read()
    num_layers = struct.unpack("<I", data[:4])[0]
    assert num_layers == 3  # 21->64, 64->64, 64->8


def test_train_on_synthetic_data_converges() -> None:
    """Train on easy synthetic data; loss should drop below 0.5."""
    from aces.perception import train_perception_on_data

    rng = np.random.default_rng(42)
    n = 2000
    obs = rng.standard_normal((n, 21)).astype(np.float32)
    continuous = np.column_stack(
        [
            1.0 / (1.0 + np.exp(-obs[:, 0])),
            1.0 / (1.0 + np.exp(-obs[:, 1])),
            1.0 / (1.0 + np.exp(-obs[:, 2])),
            1.0 / (1.0 + np.exp(-obs[:, 3])),
            np.abs(obs[:, 4]) * 5.0,
        ]
    ).astype(np.float32)
    intent = (obs[:, 5] > 0.3).astype(np.int64)
    intent[obs[:, 5] < -0.3] = 1

    final_loss = train_perception_on_data(obs, continuous, intent, epochs=30, lr=1e-3)
    assert final_loss < 0.5


def test_train_and_evaluate_returns_model_and_metrics() -> None:
    """train_and_evaluate returns a working model and metric dict."""
    from aces.perception import train_and_evaluate

    rng = np.random.default_rng(42)
    n = 500
    obs = rng.standard_normal((n, 21)).astype(np.float32)
    continuous = np.column_stack(
        [
            1.0 / (1.0 + np.exp(-obs[:, 0])),
            1.0 / (1.0 + np.exp(-obs[:, 1])),
            1.0 / (1.0 + np.exp(-obs[:, 2])),
            1.0 / (1.0 + np.exp(-obs[:, 3])),
            np.abs(obs[:, 4]) * 5.0,
        ]
    ).astype(np.float32)
    intent = (obs[:, 5] > 0.3).astype(np.int64)
    intent[obs[:, 5] < -0.3] = 1

    net, metrics = train_and_evaluate(obs, continuous, intent, epochs=10, val_split=0.2)

    # Model should be usable
    test_obs = torch.randn(1, 21)
    cont_out, intent_out = net(test_obs)
    assert cont_out.shape == (1, 5)

    # Metrics should have expected keys
    assert "train_loss" in metrics
    assert "val_mse" in metrics
    assert "val_intent_accuracy" in metrics
    assert metrics["val_intent_accuracy"] >= 0.0
