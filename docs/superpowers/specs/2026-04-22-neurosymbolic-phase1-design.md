# Neurosymbolic Controller — Phase 1 Design Spec

**Date**: 2026-04-22
**Scope**: MLP Perception NN + Symbolic FSM + MPPI Execution
**Goal**: Drone AI that uses a small supervised NN for perception, hand-coded FSM for decisions, and existing MPPI for control. Playable in Bevy as human vs AI.

## Architecture

```
Observation (21-dim vector, 100Hz)
        │
        ▼
┌─ Perception NN (supervised, MLP ~5K params) ─┐
│  Input:  21-dim vector observation            │
│  Output: 6 semantic features                  │
│    threat          ∈ [0,1]                    │
│    opportunity     ∈ [0,1]                    │
│    collision_risk  ∈ [0,1]                    │
│    uncertainty     ∈ [0,1]                    │
│    opponent_dist   ∈ ℝ+                       │
│    opponent_intent ∈ {approach,flee,patrol}    │
│  Training: supervised vs God Oracle labels     │
│  Architecture: 21 → 64 → 64 → 9              │
│    (6 continuous sigmoid + 3-class softmax)    │
└──────────────────┬────────────────────────────┘
                   │
                   ▼
┌─ Symbolic FSM (hand-coded rules) ────────────┐
│  States: HOVER, PURSUE, EVADE, SEARCH, ORBIT │
│                                               │
│  Transitions (priority high→low):             │
│    collision_risk > 0.7       → HOVER         │
│    threat > 0.7               → EVADE         │
│    opportunity > 0.6 & d < 3  → PURSUE        │
│    uncertainty > 0.5          → SEARCH         │
│    else                       → ORBIT          │
│                                               │
│  Hysteresis: stay in state ≥ 10 ticks (0.1s)  │
│    before allowing transition (prevent chatter)│
│                                               │
│  Output:                                      │
│    mode: enum {Hover,Pursue,Evade,Search,Orbit}│
│    d_safe: 0.3 + 0.3 * collision_risk         │
│    pursuit flag: bool (true for Pursue/Search) │
└──────────────────┬────────────────────────────┘
                   │
                   ▼
┌─ MPPI Controller (existing) ─────────────────┐
│  mode → cost function selection               │
│  d_safe → obstacle safety margin              │
│  pursuit → pursuit_cost vs evasion_cost       │
│  belief_var → belief-weighted cost scaling    │
│  + chance constraint P(collision) ≤ δ         │
│                                               │
│  Output: [f1, f2, f3, f4] motor thrusts       │
└───────────────────────────────────────────────┘
```

## God Oracle

Runs in simulation with access to ground truth. Computes labels for supervised training.

### Label Definitions

**threat** ∈ [0,1]: How endangered is the agent.
```
threat = clamp(
    0.4 * lock_b_progress              # opponent locking onto me
  + 0.3 * max(0, 1 - distance / 5.0)   # proximity danger
  + 0.3 * opponent_facing_me,           # opponent's forward dot my direction
  0, 1)
```
`opponent_facing_me`: dot product of opponent forward vector with (my_pos - opponent_pos).normalized(). Clamped to [0,1].

**opportunity** ∈ [0,1]: How favorable for attack.
```
opportunity = clamp(
    0.3 * lock_a_progress              # my lock-on progress
  + 0.3 * max(0, 1 - distance / 5.0)   # close enough to attack
  + 0.2 * i_face_opponent              # my forward dot toward opponent
  + 0.2 * float(a_sees_b),             # I can see them
  0, 1)
```

**collision_risk** ∈ [0,1]: Risk of hitting obstacle.
```
sdf = nearest_obs_dist_a
speed = norm(velocity_a)
collision_risk = clamp(max(0, 1 - sdf / 0.5) * min(1, speed / 3.0), 0, 1)
```

**uncertainty** ∈ [0,1]: How uncertain is opponent position.
```
uncertainty = clamp(belief_b_var_from_a / 5.0, 0, 1)
```

**opponent_distance** ∈ ℝ+: True distance.
```
opponent_distance = step_result.distance
```

**opponent_intent** ∈ {0=approach, 1=flee, 2=patrol}: What opponent is doing.
```
closing_speed = -dot(opponent_vel, (my_pos - opponent_pos).normalized())
if closing_speed > 0.5:  approach
elif closing_speed < -0.5:  flee
else:  patrol
```

## Data Pipeline

### Step 1: Collect Data
- Run MPPI-vs-MPPI and MPPI-vs-trajectory for 1000 episodes
- At each step, record `(observation_21dim, god_oracle_labels)`
- Store as NumPy `.npz` file
- Expected: ~500K samples, ~100MB

### Step 2: Train Perception NN
- Standard PyTorch supervised training
- Architecture: `Linear(21,64) → Tanh → Linear(64,64) → Tanh → Linear(64,9)`
  - outputs[0:5] → sigmoid → continuous features (threat, opportunity, collision_risk, uncertainty, opponent_dist_normalized)
  - outputs[5:8] → softmax → opponent_intent classification
- Loss: `MSE(continuous) + 0.5 * CrossEntropy(intent)`
- Optimizer: Adam, lr=1e-3, 50 epochs, batch_size=256
- Validation split: 80/20
- Expected: <5 minutes on M1 CPU

### Step 3: Export
- Export trained perception NN to `perception.bin` (same binary format as `policy.bin`)
- Export FSM config to `fsm_config.toml` (thresholds, hysteresis)

### Step 4: Deploy in Bevy
- Bevy loads `perception.bin` + FSM config at startup
- Each frame: observation → perception NN → FSM → MPPI → motors
- Fallback: if no `perception.bin`, use raw observation heuristics (no NN)

## File Plan

### New Python Files

**aces/god_oracle.py** — God Oracle label computation
- `class GodOracle`: takes StepResult fields, returns label dict
- Pure math, no dependencies beyond numpy

**aces/perception.py** — Perception NN definition + training
- `class PerceptionNet(nn.Module)`: MLP 21→64→64→9
- `class PerceptionDataset(Dataset)`: loads .npz data
- `train_perception(data_path, save_path, epochs=50)`: training loop
- `export_perception(model, path)`: binary export (reuse export.py format)

**aces/fsm.py** — Symbolic Finite State Machine
- `class DroneState(Enum)`: HOVER, PURSUE, EVADE, SEARCH, ORBIT
- `class SymbolicFSM`: transition logic + hysteresis + constraint generation
- `fsm.step(features) → (mode, d_safe, pursuit_flag)`

**scripts/collect_oracle_data.py** — Data collection script
- Runs N episodes of MPPI-vs-MPPI
- Records (observation, labels) pairs
- Saves to `data/oracle_data.npz`

**scripts/train_perception.py** — Training script
- Loads data, trains PerceptionNet, exports to `perception.bin`

### Modified Files

**aces/env.py** — Add `opponent="fsm_mppi"` mode
- FSM+MPPI agent as opponent option for RL training

**crates/game/src/perception.rs** — NEW: Rust perception NN inference
- Same binary format loader as policy.rs
- `PerceptionNet::infer(&obs) → SemanticFeatures`

**crates/game/src/fsm.rs** — NEW: Rust FSM
- `enum DroneMode { Hover, Pursue, Evade, Search, Orbit }`
- `struct SymbolicFsm { state, hysteresis_counter, thresholds }`
- `fn transition(&mut self, features: &SemanticFeatures) → FsmOutput`

**crates/game/src/simulation.rs** — Wire FSM+MPPI pipeline
- Replace `nn.infer()` → `perception.infer() → fsm.transition() → mppi.compute_action()`

**crates/game/src/main.rs** — Register FSM plugin

## Testing

- **god_oracle.py**: Unit tests for each label formula (boundary values, symmetry)
- **perception.py**: Train on synthetic data, verify loss converges
- **fsm.py**: Unit tests for each transition rule, hysteresis behavior
- **Integration**: Run FSM+MPPI in env for 100 episodes, verify no crashes, positive reward

## Success Criteria

1. `cargo run -p aces-game --release` starts with FSM+MPPI AI opponent
2. AI opponent visibly switches between pursue/evade/search behaviors
3. Human player can play against FSM+MPPI agent using keyboard
4. Console log shows FSM state transitions: `[FSM] ORBIT → PURSUE (opportunity=0.72)`
5. Perception NN achieves <0.1 MSE on validation set for continuous features
6. Perception NN achieves >80% accuracy on opponent_intent classification
