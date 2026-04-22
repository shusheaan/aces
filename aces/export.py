"""Export trained SB3 MLP policy weights for Rust/Bevy inference.

Binary format (little-endian):
    u32: num_layers
    Per layer:
        u32: rows (output dim)
        u32: cols (input dim)
        f32[rows*cols]: weight matrix (row-major)
        f32[rows]: bias vector

The MLP actor network from SB3 PPO MlpPolicy:
    Layer 0: mlp_extractor.policy_net.0  (21 → 64, Tanh)
    Layer 1: mlp_extractor.policy_net.2  (64 → 64, Tanh)
    Layer 2: action_net                  (64 → 4,  Tanh squash)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def export_mlp_policy(model_path: str, output_path: str = "policy.bin") -> None:
    """Export SB3 PPO MlpPolicy actor weights to a flat binary file."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    sd = model.policy.state_dict()

    actor_keys = [
        "mlp_extractor.policy_net.0",
        "mlp_extractor.policy_net.2",
        "action_net",
    ]

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for name in actor_keys:
        w_key = f"{name}.weight"
        b_key = f"{name}.bias"
        if w_key not in sd:
            raise ValueError(
                f"Key '{w_key}' not found in state_dict. "
                "This model may use a CNN policy — only MlpPolicy is supported."
            )
        w = sd[w_key].cpu().numpy().astype(np.float32)
        b = sd[b_key].cpu().numpy().astype(np.float32)
        layers.append((w, b))

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(layers)))
        for w, b in layers:
            rows, cols = w.shape
            f.write(struct.pack("<II", rows, cols))
            f.write(w.tobytes())  # row-major
            f.write(b.tobytes())

    total_params = sum(w.size + b.size for w, b in layers)
    print(f"[ACES] Exported MLP policy → {output_path}")
    for i, (w, _) in enumerate(layers):
        print(f"  Layer {i}: {w.shape[1]} → {w.shape[0]}")
    print(f"  Total params: {total_params}")
    print(f"  File size: {Path(output_path).stat().st_size} bytes")
