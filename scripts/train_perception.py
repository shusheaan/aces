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

from aces.perception import export_perception, train_and_evaluate


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

    net, metrics = train_and_evaluate(
        obs,
        cont,
        intent,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    print(f"  Train loss: {metrics['train_loss']:.4f}")
    print(f"  Val MSE: {metrics['val_mse']:.4f}")
    print(f"  Val intent accuracy: {metrics['val_intent_accuracy']:.2%}")

    export_perception(net, args.output)


if __name__ == "__main__":
    main()
