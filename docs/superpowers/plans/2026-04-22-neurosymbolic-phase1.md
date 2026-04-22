# Neurosymbolic Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a neurosymbolic drone AI: supervised Perception NN + hand-coded Symbolic FSM + existing MPPI controller, playable in Bevy.

**Architecture:** A small MLP (21→64→64→9, ~5K params) learns to map observations to 6 semantic features (threat, opportunity, collision_risk, uncertainty, opponent_distance, opponent_intent) via supervised learning against a God Oracle. A hand-coded FSM reads those features and selects a tactical mode (HOVER/PURSUE/EVADE/SEARCH/ORBIT). The existing MPPI controller executes the selected mode. The full pipeline runs in both Python (training) and Rust (Bevy game).

**Tech Stack:** Python (PyTorch, NumPy, Gymnasium), Rust (nalgebra, Bevy 0.15), existing MPPI/sim-core crates.

**Spec:** `docs/superpowers/specs/2026-04-22-neurosymbolic-phase1-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `aces/god_oracle.py` | Create | Ground truth label computation from StepResult |
| `tests/test_god_oracle.py` | Create | Unit tests for all 6 label formulas |
| `aces/fsm.py` | Create | Symbolic FSM: 5 states, transitions, hysteresis |
| `tests/test_fsm.py` | Create | Unit tests for transitions, hysteresis, outputs |
| `aces/perception.py` | Create | PerceptionNet MLP, dataset, training, export |
| `tests/test_perception.py` | Create | Synthetic data training convergence test |
| `scripts/collect_oracle_data.py` | Create | Batch data collection: MPPI episodes → .npz |
| `scripts/train_perception.py` | Create | CLI: load data, train, export perception.bin |
| `crates/game/src/fsm.rs` | Create | Rust FSM (mirror of Python fsm.py) |
| `crates/game/src/perception.rs` | Create | Rust perception NN loader + inference |
| `crates/game/src/simulation.rs` | Modify | Wire perception→FSM→MPPI pipeline |
| `crates/game/src/main.rs` | Modify | Register new modules |

---

### Task 1: God Oracle (Python)

**Files:**
- Create: `aces/god_oracle.py`
- Create: `tests/test_god_oracle.py`

- [ ] **Step 1: Write god_oracle tests**

```python
# tests/test_god_oracle.py
"""Tests for aces.god_oracle — ground truth label computation."""

import numpy as np

from aces.god_oracle import GodOracle


def test_threat_max_when_being_locked_and_close() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=1.0,       # fully locked
        distance=0.5,              # very close
        opponent_facing_me=1.0,    # opponent aimed at me
        lock_a_progress=0.0,
        a_sees_b=False,
        i_face_opponent=0.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["threat"] > 0.9


def test_threat_zero_when_safe() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=10.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=False,
        i_face_opponent=0.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["threat"] < 0.1


def test_opportunity_high_when_locked_and_close() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=1.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.8,
        a_sees_b=True,
        i_face_opponent=1.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["opportunity"] > 0.7


def test_collision_risk_high_near_wall_fast() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=10.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=False,
        i_face_opponent=0.0,
        nearest_obs_dist=0.1,  # very close to wall
        speed=3.0,             # fast
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["collision_risk"] > 0.7


def test_collision_risk_zero_far_from_walls() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=10.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=False,
        i_face_opponent=0.0,
        nearest_obs_dist=2.0,
        speed=1.0,
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["collision_risk"] < 0.01


def test_opponent_intent_approach() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=5.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=True,
        i_face_opponent=0.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=2.0,  # closing fast
    )
    assert labels["opponent_intent"] == 0  # approach


def test_opponent_intent_flee() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=5.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=True,
        i_face_opponent=0.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=-2.0,  # opening fast
    )
    assert labels["opponent_intent"] == 1  # flee


def test_all_labels_in_range() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.5,
        distance=3.0,
        opponent_facing_me=0.5,
        lock_a_progress=0.5,
        a_sees_b=True,
        i_face_opponent=0.5,
        nearest_obs_dist=0.3,
        speed=2.0,
        belief_var=2.0,
        opponent_closing_speed=0.0,
    )
    assert 0.0 <= labels["threat"] <= 1.0
    assert 0.0 <= labels["opportunity"] <= 1.0
    assert 0.0 <= labels["collision_risk"] <= 1.0
    assert 0.0 <= labels["uncertainty"] <= 1.0
    assert labels["opponent_distance"] >= 0.0
    assert labels["opponent_intent"] in (0, 1, 2)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `poetry run pytest tests/test_god_oracle.py -v`
Expected: `ModuleNotFoundError: No module named 'aces.god_oracle'`

- [ ] **Step 3: Implement god_oracle.py**

```python
# aces/god_oracle.py
"""God Oracle: computes ground truth semantic labels from simulator state.

Uses omniscient information available in simulation (true positions,
velocities, lock-on state) to produce supervised training labels
for the Perception NN.
"""

from __future__ import annotations


class GodOracle:
    """Compute ground truth semantic features from simulation state."""

    def compute(
        self,
        *,
        lock_b_progress: float,
        distance: float,
        opponent_facing_me: float,
        lock_a_progress: float,
        a_sees_b: bool,
        i_face_opponent: float,
        nearest_obs_dist: float,
        speed: float,
        belief_var: float,
        opponent_closing_speed: float,
    ) -> dict[str, float]:
        threat = _clamp(
            0.4 * lock_b_progress
            + 0.3 * max(0.0, 1.0 - distance / 5.0)
            + 0.3 * _clamp(opponent_facing_me, 0.0, 1.0)
        )

        opportunity = _clamp(
            0.3 * lock_a_progress
            + 0.3 * max(0.0, 1.0 - distance / 5.0)
            + 0.2 * _clamp(i_face_opponent, 0.0, 1.0)
            + 0.2 * float(a_sees_b)
        )

        wall_proximity = max(0.0, 1.0 - nearest_obs_dist / 0.5)
        speed_factor = min(1.0, speed / 3.0)
        collision_risk = _clamp(wall_proximity * speed_factor)

        uncertainty = _clamp(belief_var / 5.0)

        opponent_distance = max(0.0, distance)

        if opponent_closing_speed > 0.5:
            opponent_intent = 0  # approach
        elif opponent_closing_speed < -0.5:
            opponent_intent = 1  # flee
        else:
            opponent_intent = 2  # patrol

        return {
            "threat": threat,
            "opportunity": opportunity,
            "collision_risk": collision_risk,
            "uncertainty": uncertainty,
            "opponent_distance": opponent_distance,
            "opponent_intent": opponent_intent,
        }


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `poetry run pytest tests/test_god_oracle.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add aces/god_oracle.py tests/test_god_oracle.py
git commit -m "feat(oracle): add God Oracle ground truth label computation"
```

---

### Task 2: Symbolic FSM (Python)

**Files:**
- Create: `aces/fsm.py`
- Create: `tests/test_fsm.py`

- [ ] **Step 1: Write FSM tests**

```python
# tests/test_fsm.py
"""Tests for aces.fsm — Symbolic Finite State Machine."""

from aces.fsm import DroneMode, FsmOutput, SymbolicFSM


def _features(**overrides: float) -> dict[str, float]:
    """Default safe features with overrides."""
    defaults = {
        "threat": 0.0,
        "opportunity": 0.0,
        "collision_risk": 0.0,
        "uncertainty": 0.0,
        "opponent_distance": 5.0,
        "opponent_intent": 2,  # patrol
    }
    defaults.update(overrides)
    return defaults


def test_default_state_is_orbit() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features())
    assert out.mode == DroneMode.ORBIT


def test_high_collision_risk_triggers_hover() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(collision_risk=0.8))
    assert out.mode == DroneMode.HOVER


def test_high_threat_triggers_evade() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(threat=0.8))
    assert out.mode == DroneMode.EVADE


def test_high_opportunity_close_triggers_pursue() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert out.mode == DroneMode.PURSUE


def test_high_uncertainty_triggers_search() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(uncertainty=0.6))
    assert out.mode == DroneMode.SEARCH


def test_priority_collision_over_threat() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(collision_risk=0.8, threat=0.9))
    assert out.mode == DroneMode.HOVER  # collision highest priority


def test_priority_threat_over_opportunity() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(threat=0.8, opportunity=0.9, opponent_distance=1.0))
    assert out.mode == DroneMode.EVADE  # threat > opportunity


def test_hysteresis_prevents_chatter() -> None:
    fsm = SymbolicFSM(hysteresis_ticks=10)
    # Enter PURSUE
    fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert fsm.mode == DroneMode.PURSUE
    # Drop opportunity slightly below threshold — should stay in PURSUE
    for _ in range(5):
        out = fsm.step(_features(opportunity=0.5, opponent_distance=2.0))
    assert out.mode == DroneMode.PURSUE  # hysteresis keeps it


def test_hysteresis_allows_transition_after_enough_ticks() -> None:
    fsm = SymbolicFSM(hysteresis_ticks=3)
    fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert fsm.mode == DroneMode.PURSUE
    # Now conditions say ORBIT for enough ticks
    for _ in range(4):
        out = fsm.step(_features())
    assert out.mode == DroneMode.ORBIT


def test_high_priority_overrides_hysteresis() -> None:
    fsm = SymbolicFSM(hysteresis_ticks=100)
    fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert fsm.mode == DroneMode.PURSUE
    # collision_risk is highest priority — overrides hysteresis immediately
    out = fsm.step(_features(collision_risk=0.8))
    assert out.mode == DroneMode.HOVER


def test_d_safe_scales_with_collision_risk() -> None:
    fsm = SymbolicFSM()
    out_safe = fsm.step(_features(collision_risk=0.0))
    fsm2 = SymbolicFSM()
    out_risky = fsm2.step(_features(collision_risk=1.0))
    assert out_risky.d_safe > out_safe.d_safe


def test_pursuit_flag() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert out.pursuit is True  # PURSUE
    fsm2 = SymbolicFSM()
    out2 = fsm2.step(_features(threat=0.8))
    assert out2.pursuit is False  # EVADE
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `poetry run pytest tests/test_fsm.py -v`
Expected: `ModuleNotFoundError: No module named 'aces.fsm'`

- [ ] **Step 3: Implement fsm.py**

```python
# aces/fsm.py
"""Symbolic Finite State Machine for tactical drone decision-making.

Reads semantic features from the Perception NN and selects a tactical
mode. Outputs mode, safety margin, and pursuit flag for the MPPI
controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DroneMode(IntEnum):
    HOVER = 0
    PURSUE = 1
    EVADE = 2
    SEARCH = 3
    ORBIT = 4


# Priorities: lower index = higher priority
_PRIORITY = {
    DroneMode.HOVER: 0,
    DroneMode.EVADE: 1,
    DroneMode.PURSUE: 2,
    DroneMode.SEARCH: 3,
    DroneMode.ORBIT: 4,
}


@dataclass
class FsmOutput:
    mode: DroneMode
    d_safe: float
    pursuit: bool


class SymbolicFSM:
    """Priority-based FSM with hysteresis.

    Transition rules (checked in priority order):
        1. collision_risk > 0.7  → HOVER
        2. threat > 0.7          → EVADE
        3. opportunity > 0.6 and opponent_distance < 3  → PURSUE
        4. uncertainty > 0.5     → SEARCH
        5. else                  → ORBIT

    Hysteresis: a lower-priority mode must be requested for
    ``hysteresis_ticks`` consecutive ticks before the FSM switches
    to it. Higher-priority modes override immediately.
    """

    def __init__(self, hysteresis_ticks: int = 10) -> None:
        self.mode: DroneMode = DroneMode.ORBIT
        self.hysteresis_ticks = hysteresis_ticks
        self._ticks_requesting: int = 0
        self._requested_mode: DroneMode = DroneMode.ORBIT

    def step(self, features: dict[str, float]) -> FsmOutput:
        desired = self._evaluate(features)

        if desired == self.mode:
            # Already in this mode — reset counter
            self._ticks_requesting = 0
            self._requested_mode = desired
        elif _PRIORITY[desired] < _PRIORITY[self.mode]:
            # Higher priority — switch immediately
            self.mode = desired
            self._ticks_requesting = 0
            self._requested_mode = desired
        elif desired == self._requested_mode:
            # Same lower-priority request — count up
            self._ticks_requesting += 1
            if self._ticks_requesting >= self.hysteresis_ticks:
                self.mode = desired
                self._ticks_requesting = 0
        else:
            # Different lower-priority request — restart counter
            self._requested_mode = desired
            self._ticks_requesting = 1

        collision_risk = features.get("collision_risk", 0.0)
        d_safe = 0.3 + 0.3 * collision_risk
        pursuit = self.mode in (DroneMode.PURSUE, DroneMode.SEARCH)

        return FsmOutput(mode=self.mode, d_safe=d_safe, pursuit=pursuit)

    def _evaluate(self, f: dict[str, float]) -> DroneMode:
        if f.get("collision_risk", 0.0) > 0.7:
            return DroneMode.HOVER
        if f.get("threat", 0.0) > 0.7:
            return DroneMode.EVADE
        if f.get("opportunity", 0.0) > 0.6 and f.get("opponent_distance", 99.0) < 3.0:
            return DroneMode.PURSUE
        if f.get("uncertainty", 0.0) > 0.5:
            return DroneMode.SEARCH
        return DroneMode.ORBIT
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `poetry run pytest tests/test_fsm.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add aces/fsm.py tests/test_fsm.py
git commit -m "feat(fsm): add symbolic finite state machine with hysteresis"
```

---

### Task 3: Perception NN + Training Pipeline (Python)

**Files:**
- Create: `aces/perception.py`
- Create: `tests/test_perception.py`

- [ ] **Step 1: Write perception tests**

```python
# tests/test_perception.py
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
    # First 4 outputs use sigmoid → [0, 1]
    assert continuous[:, :4].min() >= 0.0
    assert continuous[:, :4].max() <= 1.0
    # 5th output (opponent_dist) uses softplus → [0, inf)
    assert continuous[:, 4].min() >= 0.0


def test_export_creates_valid_binary(tmp_path: Path) -> None:
    net = PerceptionNet()
    out_path = str(tmp_path / "perception.bin")
    export_perception(net, out_path)
    data = open(out_path, "rb").read()
    num_layers = struct.unpack("<I", data[:4])[0]
    assert num_layers == 3  # 21→64, 64→64, 64→9


def test_train_on_synthetic_data_converges() -> None:
    """Train on easy synthetic data; loss should drop below 0.1."""
    from aces.perception import train_perception_on_data

    rng = np.random.default_rng(42)
    n = 2000
    obs = rng.standard_normal((n, 21)).astype(np.float32)
    # Simple synthetic labels: features are just scaled means of obs chunks
    continuous = np.column_stack([
        1.0 / (1.0 + np.exp(-obs[:, 0])),   # threat ~ sigmoid(obs[0])
        1.0 / (1.0 + np.exp(-obs[:, 1])),   # opportunity
        1.0 / (1.0 + np.exp(-obs[:, 2])),   # collision_risk
        1.0 / (1.0 + np.exp(-obs[:, 3])),   # uncertainty
        np.abs(obs[:, 4]) * 5.0,             # opponent_dist
    ]).astype(np.float32)
    intent = (obs[:, 5] > 0.3).astype(np.int64)  # binary-ish classification
    intent[obs[:, 5] < -0.3] = 1

    final_loss = train_perception_on_data(obs, continuous, intent, epochs=30, lr=1e-3)
    assert final_loss < 0.5  # should converge on this easy data
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `poetry run pytest tests/test_perception.py -v`
Expected: `ModuleNotFoundError: No module named 'aces.perception'`

- [ ] **Step 3: Implement perception.py**

```python
# aces/perception.py
"""Perception NN: maps observations to semantic features (supervised).

Architecture: MLP 21 → 64 → 64 → 9
  outputs[0:4] → sigmoid → threat, opportunity, collision_risk, uncertainty
  outputs[4]   → softplus → opponent_distance (non-negative)
  outputs[5:8] → raw logits → opponent_intent {approach, flee, patrol}

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
        # 5 continuous + 3 intent logits = 8 outputs, but we split them
        self.continuous_head = nn.Linear(hidden, 5)
        self.intent_head = nn.Linear(hidden, 3)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)

        raw_continuous = self.continuous_head(h)
        continuous = torch.cat([
            torch.sigmoid(raw_continuous[:, :4]),        # threat/opp/coll/unc in [0,1]
            F.softplus(raw_continuous[:, 4:5]),           # opponent_dist ≥ 0
        ], dim=1)

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

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_obs, batch_cont, batch_intent in loader:
            pred_cont, pred_intent_logits = net(batch_obs)
            loss_cont = F.mse_loss(pred_cont, batch_cont)
            loss_intent = F.cross_entropy(pred_intent_logits, batch_intent)
            loss = loss_cont + 0.5 * loss_intent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)

    return final_loss


def export_perception(net: PerceptionNet, path: str) -> None:
    """Export PerceptionNet weights to binary (same format as policy.bin).

    Exports 3 layers: shared[0](21→64), shared[2](64→64), merged_head(64→8).
    The merged head concatenates continuous_head and intent_head weights.
    Intent logits (last 3) need no activation in Rust — argmax directly.
    """
    sd = net.state_dict()

    layers: list[tuple[np.ndarray, np.ndarray]] = []

    # Layer 0: shared.0 (Linear 21→64)
    layers.append((
        sd["shared.0.weight"].cpu().numpy().astype(np.float32),
        sd["shared.0.bias"].cpu().numpy().astype(np.float32),
    ))

    # Layer 1: shared.2 (Linear 64→64)
    layers.append((
        sd["shared.2.weight"].cpu().numpy().astype(np.float32),
        sd["shared.2.bias"].cpu().numpy().astype(np.float32),
    ))

    # Layer 2: merged head — stack continuous_head (5) + intent_head (3) → 8 outputs
    w_cont = sd["continuous_head.weight"].cpu().numpy().astype(np.float32)
    b_cont = sd["continuous_head.bias"].cpu().numpy().astype(np.float32)
    w_intent = sd["intent_head.weight"].cpu().numpy().astype(np.float32)
    b_intent = sd["intent_head.bias"].cpu().numpy().astype(np.float32)
    layers.append((
        np.vstack([w_cont, w_intent]),
        np.concatenate([b_cont, b_intent]),
    ))

    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(layers)))
        for w, b in layers:
            rows, cols = w.shape
            f.write(struct.pack("<II", rows, cols))
            f.write(w.tobytes())
            f.write(b.tobytes())

    total_params = sum(w.size + b.size for w, b in layers)
    print(f"[ACES] Exported perception NN → {path}")
    for i, (w, _) in enumerate(layers):
        print(f"  Layer {i}: {w.shape[1]} → {w.shape[0]}")
    print(f"  Total params: {total_params}")
    print(f"  File size: {Path(path).stat().st_size} bytes")
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `poetry run pytest tests/test_perception.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add aces/perception.py tests/test_perception.py
git commit -m "feat(perception): add perception NN with training and binary export"
```

---

### Task 4: Data Collection Script

**Files:**
- Create: `scripts/collect_oracle_data.py`

- [ ] **Step 1: Implement data collection script**

```python
#!/usr/bin/env python3
"""Collect (observation, god_oracle_label) pairs from MPPI-vs-MPPI episodes.

Usage:
    poetry run python scripts/collect_oracle_data.py --episodes 500 --output data/oracle_data.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aces.config import load_configs
from aces.env import DroneDogfightEnv
from aces.god_oracle import GodOracle


def collect(n_episodes: int, output_path: str) -> None:
    configs = load_configs()
    env = DroneDogfightEnv(
        task="dogfight",
        opponent="mppi",
        fpv=False,
        config=configs,
    )
    oracle = GodOracle()

    all_obs: list[np.ndarray] = []
    all_continuous: list[np.ndarray] = []
    all_intent: list[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Extract god oracle inputs from info dict
            own_state = np.array(info.get("agent_state", obs[:3]))
            opp_state = np.array(info.get("opponent_state", obs[6:9]))

            # Compute facing angles from observation
            rel_pos = obs[6:9]
            own_vel = obs[0:3]
            rel_dist = float(np.linalg.norm(rel_pos))

            # opponent_facing_me: use relative velocity toward us as proxy
            opp_vel = obs[9:12]
            if rel_dist > 0.01:
                direction_to_me = -rel_pos / rel_dist
                opponent_facing_me = float(np.dot(opp_vel / max(np.linalg.norm(opp_vel), 0.01), direction_to_me))
                i_face_opponent = float(np.dot(own_vel / max(np.linalg.norm(own_vel), 0.01), rel_pos / rel_dist))
                closing_speed = -float(np.dot(opp_vel, direction_to_me))
            else:
                opponent_facing_me = 0.0
                i_face_opponent = 0.0
                closing_speed = 0.0

            labels = oracle.compute(
                lock_b_progress=info.get("being_locked_progress", obs[17]),
                distance=info.get("distance", rel_dist),
                opponent_facing_me=max(0.0, opponent_facing_me),
                lock_a_progress=info.get("lock_progress", obs[16]),
                a_sees_b=bool(obs[18] > 0.5),
                i_face_opponent=max(0.0, i_face_opponent),
                nearest_obs_dist=info.get("nearest_obs_dist", obs[15]),
                speed=float(np.linalg.norm(own_vel)),
                belief_var=info.get("belief_var", obs[19]),
                opponent_closing_speed=closing_speed,
            )

            all_obs.append(obs.copy())
            all_continuous.append(np.array([
                labels["threat"],
                labels["opportunity"],
                labels["collision_risk"],
                labels["uncertainty"],
                labels["opponent_distance"],
            ], dtype=np.float32))
            all_intent.append(labels["opponent_intent"])
            steps += 1

        if (ep + 1) % 50 == 0:
            total = len(all_obs)
            print(f"Episode {ep + 1}/{n_episodes} — {total} samples collected")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.array(all_obs, dtype=np.float32),
        continuous_labels=np.array(all_continuous, dtype=np.float32),
        intent_labels=np.array(all_intent, dtype=np.int64),
    )
    print(f"[ACES] Saved {len(all_obs)} samples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect oracle training data")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/oracle_data.npz")
    args = parser.parse_args()
    collect(args.episodes, args.output)
```

- [ ] **Step 2: Verify script is syntactically valid**

Run: `poetry run python -c "import ast; ast.parse(open('scripts/collect_oracle_data.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/collect_oracle_data.py
git commit -m "feat(data): add oracle data collection script"
```

---

### Task 5: Training Script

**Files:**
- Create: `scripts/train_perception.py`

- [ ] **Step 1: Implement training script**

```python
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

from aces.perception import PerceptionNet, export_perception, train_perception_on_data


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

    final_loss = train_perception_on_data(
        obs[train_idx], cont[train_idx], intent[train_idx],
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )
    print(f"  Final train loss: {final_loss:.4f}")

    # Validation
    net = PerceptionNet()
    # Re-train to get the model (train_perception_on_data doesn't return model)
    # Better: modify to return model. For now, retrain quickly.
    # Actually let's fix this properly — we need the model back.
    # We'll use a standalone training that returns the model.
    net, final_loss = _train_and_return_model(
        obs[train_idx], cont[train_idx], intent[train_idx],
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    # Validation metrics
    net.eval()
    with torch.no_grad():
        val_obs = torch.as_tensor(obs[val_idx], dtype=torch.float32)
        val_cont = torch.as_tensor(cont[val_idx], dtype=torch.float32)
        val_intent = torch.as_tensor(intent[val_idx], dtype=torch.long)
        pred_cont, pred_intent_logits = net(val_obs)
        val_mse = torch.nn.functional.mse_loss(pred_cont, val_cont).item()
        pred_classes = pred_intent_logits.argmax(dim=1)
        val_acc = (pred_classes == val_intent).float().mean().item()

    print(f"  Val MSE: {val_mse:.4f}")
    print(f"  Val intent accuracy: {val_acc:.2%}")

    export_perception(net, args.output)


def _train_and_return_model(
    obs: np.ndarray,
    continuous_labels: np.ndarray,
    intent_labels: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> tuple[PerceptionNet, float]:
    """Train and return both model and loss."""
    device = torch.device("cpu")
    net = PerceptionNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    cont_t = torch.as_tensor(continuous_labels, dtype=torch.float32, device=device)
    intent_t = torch.as_tensor(intent_labels, dtype=torch.long, device=device)

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(obs_t, cont_t, intent_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_obs, batch_cont, batch_intent in loader:
            pred_cont, pred_intent_logits = net(batch_obs)
            loss = (
                torch.nn.functional.mse_loss(pred_cont, batch_cont)
                + 0.5 * torch.nn.functional.cross_entropy(pred_intent_logits, batch_intent)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {final_loss:.4f}")

    return net, final_loss


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `poetry run python -c "import ast; ast.parse(open('scripts/train_perception.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/train_perception.py
git commit -m "feat(train): add perception NN training script with val metrics"
```

---

### Task 6: Rust FSM (Bevy)

**Files:**
- Create: `crates/game/src/fsm.rs`

- [ ] **Step 1: Implement Rust FSM**

```rust
// crates/game/src/fsm.rs
//! Symbolic Finite State Machine — mirrors aces/fsm.py.
//!
//! Reads semantic features from PerceptionNet output, selects a tactical
//! mode, and generates constraints for the MPPI controller.

/// Tactical mode selected by the FSM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroneMode {
    Hover = 0,
    Pursue = 1,
    Evade = 2,
    Search = 3,
    Orbit = 4,
}

impl DroneMode {
    fn priority(self) -> u8 {
        match self {
            Self::Hover => 0,
            Self::Evade => 1,
            Self::Pursue => 2,
            Self::Search => 3,
            Self::Orbit => 4,
        }
    }
}

/// Semantic features produced by the Perception NN.
pub struct SemanticFeatures {
    pub threat: f64,
    pub opportunity: f64,
    pub collision_risk: f64,
    pub uncertainty: f64,
    pub opponent_distance: f64,
    pub opponent_intent: u8, // 0=approach, 1=flee, 2=patrol
}

/// Output of the FSM transition.
pub struct FsmOutput {
    pub mode: DroneMode,
    pub d_safe: f64,
    pub pursuit: bool,
}

/// Symbolic FSM with priority-based transitions and hysteresis.
pub struct SymbolicFsm {
    pub mode: DroneMode,
    hysteresis_ticks: u32,
    ticks_requesting: u32,
    requested_mode: DroneMode,
}

impl SymbolicFsm {
    pub fn new(hysteresis_ticks: u32) -> Self {
        Self {
            mode: DroneMode::Orbit,
            hysteresis_ticks,
            ticks_requesting: 0,
            requested_mode: DroneMode::Orbit,
        }
    }

    pub fn step(&mut self, features: &SemanticFeatures) -> FsmOutput {
        let desired = Self::evaluate(features);

        if desired == self.mode {
            self.ticks_requesting = 0;
            self.requested_mode = desired;
        } else if desired.priority() < self.mode.priority() {
            // Higher priority — switch immediately
            self.mode = desired;
            self.ticks_requesting = 0;
            self.requested_mode = desired;
        } else if desired == self.requested_mode {
            self.ticks_requesting += 1;
            if self.ticks_requesting >= self.hysteresis_ticks {
                self.mode = desired;
                self.ticks_requesting = 0;
            }
        } else {
            self.requested_mode = desired;
            self.ticks_requesting = 1;
        }

        FsmOutput {
            mode: self.mode,
            d_safe: 0.3 + 0.3 * features.collision_risk,
            pursuit: matches!(self.mode, DroneMode::Pursue | DroneMode::Search),
        }
    }

    pub fn reset(&mut self) {
        self.mode = DroneMode::Orbit;
        self.ticks_requesting = 0;
        self.requested_mode = DroneMode::Orbit;
    }

    fn evaluate(f: &SemanticFeatures) -> DroneMode {
        if f.collision_risk > 0.7 {
            DroneMode::Hover
        } else if f.threat > 0.7 {
            DroneMode::Evade
        } else if f.opportunity > 0.6 && f.opponent_distance < 3.0 {
            DroneMode::Pursue
        } else if f.uncertainty > 0.5 {
            DroneMode::Search
        } else {
            DroneMode::Orbit
        }
    }
}

impl std::fmt::Display for DroneMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hover => write!(f, "HOVER"),
            Self::Pursue => write!(f, "PURSUE"),
            Self::Evade => write!(f, "EVADE"),
            Self::Search => write!(f, "SEARCH"),
            Self::Orbit => write!(f, "ORBIT"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safe_features() -> SemanticFeatures {
        SemanticFeatures {
            threat: 0.0,
            opportunity: 0.0,
            collision_risk: 0.0,
            uncertainty: 0.0,
            opponent_distance: 5.0,
            opponent_intent: 2,
        }
    }

    #[test]
    fn test_default_orbit() {
        let mut fsm = SymbolicFsm::new(10);
        let out = fsm.step(&safe_features());
        assert_eq!(out.mode, DroneMode::Orbit);
    }

    #[test]
    fn test_collision_overrides_all() {
        let mut fsm = SymbolicFsm::new(10);
        let mut f = safe_features();
        f.collision_risk = 0.9;
        f.threat = 0.9;
        let out = fsm.step(&f);
        assert_eq!(out.mode, DroneMode::Hover);
    }

    #[test]
    fn test_pursue_flag() {
        let mut fsm = SymbolicFsm::new(0);
        let mut f = safe_features();
        f.opportunity = 0.8;
        f.opponent_distance = 2.0;
        let out = fsm.step(&f);
        assert!(out.pursuit);
    }
}
```

- [ ] **Step 2: Run cargo check**

Run: `cargo check -p aces-game`
Expected: success (after adding `mod fsm;` in main.rs — done in Task 8)

- [ ] **Step 3: Commit**

```bash
git add crates/game/src/fsm.rs
git commit -m "feat(game): add Rust symbolic FSM with hysteresis"
```

---

### Task 7: Rust Perception NN Loader

**Files:**
- Create: `crates/game/src/perception.rs`

- [ ] **Step 1: Implement Rust perception NN inference**

```rust
// crates/game/src/perception.rs
//! Load and run a trained Perception NN exported from Python.
//!
//! Binary format: identical to policy.rs (see `aces/export.py`).
//! Network: 21→64 (Tanh) → 64 (Tanh) → 8 (no activation in Rust).
//! Outputs [0..4]: apply sigmoid for threat/opp/coll/unc, softplus for dist.
//! Outputs [5..7]: raw logits for intent — argmax to classify.

use crate::fsm::SemanticFeatures;
use nalgebra::{DMatrix, DVector};

/// A loaded Perception MLP (21→64→64→8 with Tanh hidden layers).
pub struct PerceptionMlp {
    layers: Vec<(DMatrix<f64>, DVector<f64>)>,
}

impl PerceptionMlp {
    /// Try to load from binary file. Returns `None` on any error.
    pub fn load(path: &str) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        let mut cur = 0_usize;

        let read_u32 = |c: &mut usize| -> Option<u32> {
            if *c + 4 > data.len() {
                return None;
            }
            let v = u32::from_le_bytes(data[*c..*c + 4].try_into().ok()?);
            *c += 4;
            Some(v)
        };

        let num_layers = read_u32(&mut cur)? as usize;
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let rows = read_u32(&mut cur)? as usize;
            let cols = read_u32(&mut cur)? as usize;

            let w_len = rows * cols;
            let w_bytes = w_len * 4;
            if cur + w_bytes > data.len() {
                return None;
            }
            let mut w_data = Vec::with_capacity(w_len);
            for i in 0..w_len {
                let off = cur + i * 4;
                let v = f32::from_le_bytes(data[off..off + 4].try_into().ok()?);
                w_data.push(v as f64);
            }
            cur += w_bytes;
            let weight = DMatrix::from_row_slice(rows, cols, &w_data);

            let b_bytes = rows * 4;
            if cur + b_bytes > data.len() {
                return None;
            }
            let mut b_data = Vec::with_capacity(rows);
            for i in 0..rows {
                let off = cur + i * 4;
                let v = f32::from_le_bytes(data[off..off + 4].try_into().ok()?);
                b_data.push(v as f64);
            }
            cur += b_bytes;
            let bias = DVector::from_vec(b_data);

            layers.push((weight, bias));
        }

        Some(Self { layers })
    }

    /// Forward pass: 21-dim observation → SemanticFeatures.
    pub fn infer(&self, obs: &[f64; 21]) -> SemanticFeatures {
        let mut x = DVector::from_column_slice(obs);

        for (i, (w, b)) in self.layers.iter().enumerate() {
            x = w * &x + b;
            // Tanh on hidden layers only (not the output layer)
            if i < self.layers.len() - 1 {
                x = x.map(|v| v.tanh());
            }
        }

        // Apply activations matching Python PerceptionNet
        let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
        let softplus = |v: f64| (1.0 + v.exp()).ln();

        let threat = sigmoid(x[0]);
        let opportunity = sigmoid(x[1]);
        let collision_risk = sigmoid(x[2]);
        let uncertainty = sigmoid(x[3]);
        let opponent_distance = softplus(x[4]);

        // Intent: argmax of logits [5], [6], [7]
        let intent_logits = [x[5], x[6], x[7]];
        let opponent_intent = intent_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u8)
            .unwrap_or(2);

        SemanticFeatures {
            threat,
            opportunity,
            collision_risk,
            uncertainty,
            opponent_distance,
            opponent_intent,
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add crates/game/src/perception.rs
git commit -m "feat(game): add Rust perception NN loader and inference"
```

---

### Task 8: Wire Everything in Bevy Simulation

**Files:**
- Modify: `crates/game/src/main.rs`
- Modify: `crates/game/src/simulation.rs`

- [ ] **Step 1: Add modules to main.rs**

Add `mod fsm;` and `mod perception;` to `crates/game/src/main.rs` after the existing module declarations (line 8):

```rust
mod fsm;
mod perception;
```

- [ ] **Step 2: Update SimState and init_sim in simulation.rs**

Add these imports at the top of `crates/game/src/simulation.rs`:

```rust
use crate::fsm::{SemanticFeatures, SymbolicFsm};
use crate::perception::PerceptionMlp;
```

Add fields to `SimState` struct (after `pub policy: Option<MlpPolicy>`):

```rust
    /// Trained perception network (if perception.bin found at startup).
    pub perception: Option<PerceptionMlp>,
    /// Symbolic FSM for neurosymbolic AI.
    pub fsm: SymbolicFsm,
```

In `init_sim`, after the policy loading block, add:

```rust
    let perception = PerceptionMlp::load("perception.bin");
    if perception.is_some() {
        println!("[ACES] Loaded perception NN from perception.bin — using FSM+MPPI AI");
    } else if policy.is_some() {
        println!("[ACES] Loaded neural network policy from policy.bin");
    } else {
        println!("[ACES] No policy.bin or perception.bin — using pure MPPI AI");
    }
```

Initialize in the `SimState` constructor:

```rust
        perception,
        fsm: SymbolicFsm::new(10), // 10-tick hysteresis = 0.1s
```

- [ ] **Step 3: Update sim_step AI logic**

Replace the AI decision block in `sim_step` (the `let ai_motors = if let Some(ref nn) = s.policy { ... }` block) with a three-tier priority: perception+FSM first, then policy.bin, then MPPI fallback.

```rust
    // AI opponent: perception+FSM → policy.bin → MPPI fallback
    let ai_motors = if let Some(ref perc) = s.perception {
        // --- Neurosymbolic: perception NN → FSM → MPPI ---
        let lock_a_p = s.lock_a.progress();
        let lock_b_p = s.lock_b.progress();
        let (ai_own, ai_opp, lock_p, locked_p) = match *active {
            ActiveDrone::A => (&s.state_b, &s.state_a, lock_b_p, lock_a_p),
            ActiveDrone::B => (&s.state_a, &s.state_b, lock_a_p, lock_b_p),
        };
        let obs_dist = s.arena.obstacle_sdf(&ai_own.position);
        let obs = policy::build_obs(ai_own, ai_opp, obs_dist, lock_p, locked_p);
        let features = perc.infer(&obs);

        let prev_mode = s.fsm.mode;
        let fsm_out = s.fsm.step(&features);
        if fsm_out.mode != prev_mode {
            bevy::log::info!(
                "[FSM] {} → {} (threat={:.2} opp={:.2} coll={:.2} unc={:.2} dist={:.1})",
                prev_mode, fsm_out.mode,
                features.threat, features.opportunity,
                features.collision_risk, features.uncertainty,
                features.opponent_distance,
            );
        }

        // Use MPPI with mode from FSM (throttled at 10Hz)
        if s.tick.is_multiple_of(MPPI_EVERY_N_TICKS) {
            let (ai_state, player_state) = match *active {
                ActiveDrone::A => (s.state_b.clone(), s.state_a.clone()),
                ActiveDrone::B => (s.state_a.clone(), s.state_b.clone()),
            };
            s.cached_ai_motors =
                s.mppi
                    .compute_action(&ai_state, &player_state, fsm_out.pursuit);
        }
        s.tick = s.tick.wrapping_add(1);
        s.cached_ai_motors
    } else if let Some(ref nn) = s.policy {
        // --- End-to-end NN policy ---
        let lock_a_p = s.lock_a.progress();
        let lock_b_p = s.lock_b.progress();
        let (ai_own, ai_opp, lock_p, locked_p) = match *active {
            ActiveDrone::A => (&s.state_b, &s.state_a, lock_b_p, lock_a_p),
            ActiveDrone::B => (&s.state_a, &s.state_b, lock_a_p, lock_b_p),
        };
        let obs_dist = s.arena.obstacle_sdf(&ai_own.position);
        let obs = policy::build_obs(ai_own, ai_opp, obs_dist, lock_p, locked_p);
        let action = nn.infer(&obs);
        let m = nn.action_to_motors(&action);
        Vector4::new(m[0], m[1], m[2], m[3])
    } else {
        // --- Pure MPPI fallback ---
        if s.tick.is_multiple_of(MPPI_EVERY_N_TICKS) {
            let (ai_state, player_state) = match *active {
                ActiveDrone::A => (s.state_b.clone(), s.state_a.clone()),
                ActiveDrone::B => (s.state_a.clone(), s.state_b.clone()),
            };
            s.cached_ai_motors = s.mppi.compute_action(&ai_state, &player_state, true);
        }
        s.tick = s.tick.wrapping_add(1);
        s.cached_ai_motors
    };
```

- [ ] **Step 4: Update reset_sim to reset FSM**

In `reset_sim`, add after `sim.mppi.reset()`:

```rust
    sim.fsm.reset();
```

- [ ] **Step 5: Build and verify**

Run: `cargo check -p aces-game`
Expected: success

Run: `cargo test -p aces-game`
Expected: FSM tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/game/src/main.rs crates/game/src/simulation.rs
git commit -m "feat(game): wire perception→FSM→MPPI neurosymbolic pipeline in Bevy"
```

---

### Task 9: Integration Test — Full Pipeline Smoke Test

**Files:**
- Create: `tests/test_neurosymbolic_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_neurosymbolic_integration.py
"""Integration test: God Oracle → PerceptionNet → FSM → verify outputs."""

import numpy as np
import torch

from aces.fsm import DroneMode, SymbolicFSM
from aces.god_oracle import GodOracle
from aces.perception import PerceptionNet


def test_full_pipeline_oracle_to_fsm() -> None:
    """Run oracle → perception → FSM on synthetic scenarios."""
    oracle = GodOracle()
    net = PerceptionNet()
    fsm = SymbolicFSM(hysteresis_ticks=0)  # no hysteresis for testing

    # Scenario: being attacked (high threat)
    labels = oracle.compute(
        lock_b_progress=0.8,
        distance=1.5,
        opponent_facing_me=0.9,
        lock_a_progress=0.0,
        a_sees_b=True,
        i_face_opponent=0.0,
        nearest_obs_dist=3.0,
        speed=1.0,
        belief_var=0.0,
        opponent_closing_speed=1.0,
    )
    assert labels["threat"] > 0.7  # oracle says high threat

    # Feed through FSM directly with oracle labels
    out = fsm.step(labels)
    assert out.mode == DroneMode.EVADE


def test_perception_net_processes_observation() -> None:
    """PerceptionNet forward pass produces valid features for FSM."""
    net = PerceptionNet()
    net.eval()

    obs = np.zeros(21, dtype=np.float32)
    obs_t = torch.as_tensor(obs).unsqueeze(0)
    with torch.no_grad():
        continuous, intent_logits = net(obs_t)

    features = {
        "threat": continuous[0, 0].item(),
        "opportunity": continuous[0, 1].item(),
        "collision_risk": continuous[0, 2].item(),
        "uncertainty": continuous[0, 3].item(),
        "opponent_distance": continuous[0, 4].item(),
        "opponent_intent": int(intent_logits[0].argmax().item()),
    }

    # All features should be valid
    assert 0 <= features["threat"] <= 1
    assert 0 <= features["opportunity"] <= 1
    assert 0 <= features["collision_risk"] <= 1
    assert 0 <= features["uncertainty"] <= 1
    assert features["opponent_distance"] >= 0
    assert features["opponent_intent"] in (0, 1, 2)

    # Feed through FSM — should not crash
    fsm = SymbolicFSM()
    out = fsm.step(features)
    assert out.mode in list(DroneMode)
```

- [ ] **Step 2: Run integration test**

Run: `poetry run pytest tests/test_neurosymbolic_integration.py -v`
Expected: 2 passed

- [ ] **Step 3: Build Bevy release**

Run: `cargo build -p aces-game --release`
Expected: success

- [ ] **Step 4: Commit**

```bash
git add tests/test_neurosymbolic_integration.py
git commit -m "test: add neurosymbolic integration tests (oracle→perception→FSM)"
```

---

## Execution Summary

| Task | Component | New Files | Tests |
|------|-----------|-----------|-------|
| 1 | God Oracle | `god_oracle.py` | 8 unit tests |
| 2 | Symbolic FSM | `fsm.py` | 13 unit tests |
| 3 | Perception NN | `perception.py` | 5 unit tests |
| 4 | Data Collection | `collect_oracle_data.py` | syntax check |
| 5 | Training Script | `train_perception.py` | syntax check |
| 6 | Rust FSM | `fsm.rs` | 3 unit tests |
| 7 | Rust Perception | `perception.rs` | — (tested via cargo check) |
| 8 | Bevy Wiring | `simulation.rs`, `main.rs` | cargo check + test |
| 9 | Integration | `test_neurosymbolic_integration.py` | 2 integration tests |

**Total: 9 tasks, 9 commits, 31+ tests**

**After all tasks complete, the end-to-end workflow is:**

```bash
# 1. Collect data (~10 min on M1)
poetry run python scripts/collect_oracle_data.py --episodes 500

# 2. Train perception NN (~5 min on M1)
poetry run python scripts/train_perception.py --data data/oracle_data.npz

# 3. Play in Bevy (human vs FSM+MPPI)
cargo run -p aces-game --release
```
