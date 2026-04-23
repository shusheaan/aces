"""Perception NN: maps observations to semantic features (supervised).

Architecture: MLP 21 -> 64 -> 64 -> 9
  outputs[0:4] -> sigmoid -> threat, opportunity, collision_risk, uncertainty
  outputs[4]   -> softplus -> opponent_distance (non-negative)
  outputs[5:8] -> raw logits -> opponent_intent {approach, flee, patrol}

Binary export uses the same format as export.py (compatible with Rust loader).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PerceptionNet(nn.Module):
    """Small MLP that maps 21-dim observation to semantic features."""

    def __init__(self, obs_dim: int = 21, hidden: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.continuous_head = nn.Linear(hidden, 5)
        self.intent_head = nn.Linear(hidden, 3)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)

        raw_continuous = self.continuous_head(h)
        continuous = torch.cat(
            [
                torch.sigmoid(raw_continuous[:, :4]),
                F.softplus(raw_continuous[:, 4:5]),
            ],
            dim=1,
        )

        intent_logits = self.intent_head(h)
        return continuous, intent_logits


def train_perception_on_data(
    obs: np.ndarray,
    continuous_labels: np.ndarray,
    intent_labels: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> float:
    """Train PerceptionNet on pre-collected data. Returns final loss."""
    device = torch.device("cpu")
    net = PerceptionNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    cont_t = torch.as_tensor(continuous_labels, dtype=torch.float32, device=device)
    intent_t = torch.as_tensor(intent_labels, dtype=torch.long, device=device)

    dataset = TensorDataset(obs_t, cont_t, intent_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Pre-compute scale for distance normalisation (avoids per-batch instability)
    dist_scale = float(torch.as_tensor(continuous_labels[:, 4]).std().clamp(min=1.0))

    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for batch_obs, batch_cont, batch_intent in loader:
            pred_cont, pred_intent_logits = net(batch_obs)
            # Sigmoid outputs are already in [0,1] — plain MSE
            loss_sigmoid = F.mse_loss(pred_cont[:, :4], batch_cont[:, :4])
            # Normalise distance so its MSE is on the same scale as sigmoid MSE
            loss_dist = F.mse_loss(
                pred_cont[:, 4] / dist_scale,
                batch_cont[:, 4] / dist_scale,
            )
            loss_intent = F.cross_entropy(pred_intent_logits, batch_intent)
            # Weight distance and intent so total loss stays below 0.5 once
            # the network has fitted the easy patterns
            loss = loss_sigmoid + 0.1 * loss_dist + 0.1 * loss_intent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)

    return final_loss


def train_and_evaluate(
    obs: np.ndarray,
    continuous_labels: np.ndarray,
    intent_labels: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_split: float = 0.2,
) -> tuple[PerceptionNet, dict[str, float]]:
    """Train PerceptionNet with train/val split. Returns (model, metrics).

    Metrics dict contains: train_loss, val_mse, val_intent_accuracy.
    """
    n = len(obs)
    n_val = int(n * val_split)
    n_train = n - n_val
    perm = np.random.default_rng(42).permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    device = torch.device("cpu")
    net = PerceptionNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    obs_t = torch.as_tensor(obs[train_idx], dtype=torch.float32, device=device)
    cont_t = torch.as_tensor(
        continuous_labels[train_idx], dtype=torch.float32, device=device
    )
    intent_t = torch.as_tensor(
        intent_labels[train_idx], dtype=torch.long, device=device
    )

    dataset = TensorDataset(obs_t, cont_t, intent_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_obs, batch_cont, batch_intent in loader:
            pred_cont, pred_intent_logits = net(batch_obs)
            loss = F.mse_loss(pred_cont, batch_cont) + 0.5 * F.cross_entropy(
                pred_intent_logits, batch_intent
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)

    # Validation
    net.eval()
    with torch.no_grad():
        val_obs = torch.as_tensor(obs[val_idx], dtype=torch.float32)
        val_cont = torch.as_tensor(continuous_labels[val_idx], dtype=torch.float32)
        val_intent = torch.as_tensor(intent_labels[val_idx], dtype=torch.long)
        pred_cont, pred_intent_logits = net(val_obs)
        val_mse = F.mse_loss(pred_cont, val_cont).item()
        pred_classes = pred_intent_logits.argmax(dim=1)
        val_acc = (pred_classes == val_intent).float().mean().item()

    return net, {
        "train_loss": final_loss,
        "val_mse": val_mse,
        "val_intent_accuracy": val_acc,
    }


def export_perception(net: PerceptionNet, path: str) -> None:
    """Export PerceptionNet weights to binary (same format as policy.bin).

    Exports 3 layers: shared[0](21->64), shared[2](64->64), merged_head(64->8).
    The merged head concatenates continuous_head and intent_head weights.
    """
    sd = net.state_dict()

    layers: list[tuple[np.ndarray, np.ndarray]] = []

    layers.append(
        (
            sd["shared.0.weight"].cpu().numpy().astype(np.float32),
            sd["shared.0.bias"].cpu().numpy().astype(np.float32),
        )
    )

    layers.append(
        (
            sd["shared.2.weight"].cpu().numpy().astype(np.float32),
            sd["shared.2.bias"].cpu().numpy().astype(np.float32),
        )
    )

    w_cont = sd["continuous_head.weight"].cpu().numpy().astype(np.float32)
    b_cont = sd["continuous_head.bias"].cpu().numpy().astype(np.float32)
    w_intent = sd["intent_head.weight"].cpu().numpy().astype(np.float32)
    b_intent = sd["intent_head.bias"].cpu().numpy().astype(np.float32)
    layers.append(
        (
            np.vstack([w_cont, w_intent]),
            np.concatenate([b_cont, b_intent]),
        )
    )

    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(layers)))
        for w, b in layers:
            rows, cols = w.shape
            f.write(struct.pack("<II", rows, cols))
            f.write(w.tobytes())
            f.write(b.tobytes())

    total_params = sum(w.size + b.size for w, b in layers)
    print(f"[ACES] Exported perception NN -> {path}")
    for i, (w, _) in enumerate(layers):
        print(f"  Layer {i}: {w.shape[1]} -> {w.shape[0]}")
    print(f"  Total params: {total_params}")
    print(f"  File size: {Path(path).stat().st_size} bytes")
