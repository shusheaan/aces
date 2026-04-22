#!/usr/bin/env python3
"""Train Perception NN from collected oracle data.

Usage:
    poetry run python scripts/train_perception.py \
        --data data/oracle_data.npz \
        --output perception.bin \
        --epochs 50
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aces.perception import PerceptionNet, export_perception


def main() -> None:
    parser = argparse.ArgumentParser(description="Train perception NN")
    parser.add_argument("--data", type=str, default="data/oracle_data.npz")
    parser.add_argument("--output", type=str, default="perception.bin")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    print(f"[ACES] Loading data from {args.data}")
    data = np.load(args.data)
    obs = data["observations"]
    cont = data["continuous_labels"]
    intent = data["intent_labels"]
    print(f"  Samples: {len(obs)}, Obs dim: {obs.shape[1]}")

    # Train/val split
    n = len(obs)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    perm = np.random.default_rng(42).permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    print(f"  Train: {n_train}, Val: {n_val}")
    print(f"[ACES] Training for {args.epochs} epochs...")

    device = torch.device("cpu")
    net = PerceptionNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    obs_t = torch.as_tensor(obs[train_idx], dtype=torch.float32, device=device)
    cont_t = torch.as_tensor(cont[train_idx], dtype=torch.float32, device=device)
    intent_t = torch.as_tensor(intent[train_idx], dtype=torch.long, device=device)

    dataset = TensorDataset(obs_t, cont_t, intent_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
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
        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{args.epochs} — loss: {avg_loss:.4f}")

    # Validation metrics
    net.eval()
    with torch.no_grad():
        val_obs = torch.as_tensor(obs[val_idx], dtype=torch.float32)
        val_cont = torch.as_tensor(cont[val_idx], dtype=torch.float32)
        val_intent = torch.as_tensor(intent[val_idx], dtype=torch.long)
        pred_cont, pred_intent_logits = net(val_obs)
        val_mse = F.mse_loss(pred_cont, val_cont).item()
        pred_classes = pred_intent_logits.argmax(dim=1)
        val_acc = (pred_classes == val_intent).float().mean().item()

    print(f"  Val MSE: {val_mse:.4f}")
    print(f"  Val intent accuracy: {val_acc:.2%}")

    export_perception(net, args.output)


if __name__ == "__main__":
    main()
