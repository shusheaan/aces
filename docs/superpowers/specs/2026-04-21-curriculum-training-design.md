# Curriculum Training — Design Spec

## Goal

Add parameterized difficulty stages to the existing `DroneDogfightEnv` so that a single PPO agent can be trained incrementally: first pursuing a predictable target, then an evasive one, then searching through occlusion, and finally full 1v1 dogfight. Each stage loads the previous stage's weights. All stages share the same 21-dim (or FPV Dict) observation space so weights transfer directly.

## Stages

Four stages, numbered 2–5 (Stage 1 "goto fixed point" deliberately omitted — too trivial for this agent).

### Stage 2 — `pursuit_linear`

**Opponent**: follows a pre-computed trajectory inside the arena. No MPPI, no evasion. Possible trajectories:

| Trajectory | Parameters | Description |
|------------|-----------|-------------|
| `circle` | center, radius, altitude, speed | Horizontal circle at fixed height |
| `lemniscate` | center, scale, altitude, speed | Figure-8, stays in arena center |
| `patrol` | waypoints[], speed | Linear segments between waypoints |

Trajectory is selected at `reset()` — either fixed per episode or randomly sampled from the set. The opponent applies motor thrusts to track its next waypoint using a simple PD position controller (not MPPI — too expensive for a non-adversarial target):

```
error = waypoint - current_position
thrust_cmd = hover + kp * error + kd * (error - prev_error)
→ clamp to [0, max_thrust] per motor via inverse mixing
```

PD gains `kp`, `kd` are tuned once to produce smooth, non-aggressive flight within the arena. The opponent ignores the agent entirely.

**Spawn**: both drones visible to each other (existing spawn positions).

**Reward emphasis**: `approach_reward` weight increased to 0.2 (vs 0.05 default) to strongly reward closing distance. `info_gain_reward` and `lost_contact_penalty` set to 0 (opponent always visible, no search needed).

**Termination**: kill (lock-on complete), collision, OOB, or timeout.

**Learning objective**: basic flight control + pursuit + lock-on mechanics.

### Stage 3 — `pursuit_evasive`

**Opponent**: MPPI controller with `pursuit=False` (evasion mode). Uses the existing `MppiController.compute_action(state_b, state_a, pursuit=False)`. The opponent actively tries to maximize distance and avoid being locked on.

**Spawn**: both drones visible (existing spawn positions).

**Reward**: default weights from `rules.toml [reward]`. Identical to current dogfight rewards minus opponent's ability to fight back.

**Termination**: same as Stage 2.

**Learning objective**: adversarial pursuit against a reactive opponent.

### Stage 4 — `search_pursuit`

**Opponent**: same as Stage 3 (MPPI evasion).

**Spawn**: **occluded** — the two drones spawn on opposite sides of an obstacle so that `check_line_of_sight(arena, pos_a, pos_b) == Occluded` at `t=0`. Selection algorithm:

```
1. Pick spawn_a from existing spawn position
2. For spawn_b, try candidate positions behind each pillar:
   offset opponent spawn to the far side of a randomly chosen obstacle
3. Verify check_line_of_sight returns Occluded
4. If no valid occluded pair found after N attempts, fall back to
   maximum-distance spawn (worst case: visible but far)
```

The `reset()` method in env.py handles this — the Rust `Simulation.reset(pos_a, pos_b)` already accepts arbitrary positions.

**Reward**: `info_gain_reward` increased to 0.1, `lost_contact_penalty` increased to 0.02. These rewards leverage existing infrastructure:

- `belief_b_var_from_a` (particle filter variance) — decreases as agent gets information
- `time_since_a_saw_b` — accumulates while opponent is hidden

No explicit state machine needed. The reward gradient naturally produces:
1. **Search phase**: high `belief_var` → `info_gain_reward` drives agent toward predicted opponent location
2. **Transition**: `a_sees_b` flips to True → `belief_var` drops to 0, `approach_reward` takes over
3. **Pursuit phase**: standard pursuit + lock-on behavior
4. **Re-search**: if opponent ducks behind pillar, `lost_contact_penalty` kicks in, agent re-enters search

The agent's observation already contains `opponent_visible` (obs[18]), `belief_uncertainty` (obs[19]), and `time_since_last_seen` (obs[20]) — sufficient for the policy to learn mode-switching internally.

**Learning objective**: exploration under uncertainty + pursuit.

### Stage 5 — `dogfight`

**Opponent**: current behavior — MPPI pursuit (`pursuit=True`) or self-play policy. The opponent actively tries to lock onto the agent while evading.

**Spawn**: random (existing jitter logic).

**Reward**: full default weights including `killed_penalty` (opponent can kill agent).

**Learning objective**: complete attack + defense.

## Implementation: Changes to `DroneDogfightEnv`

### New constructor parameter

```python
class DroneDogfightEnv(gym.Env):
    def __init__(
        self,
        ...,
        task: str = "dogfight",  # NEW: "pursuit_linear" | "pursuit_evasive" | "search_pursuit" | "dogfight"
    ):
```

The `task` parameter overrides three things internally:
1. `_opponent_action()` dispatch
2. `reset()` spawn logic
3. Reward weight overrides

### Opponent behavior dispatch

```python
def _opponent_action(self) -> list[float]:
    if self._task == "pursuit_linear":
        return self._trajectory_action()       # NEW: PD controller on trajectory
    elif self._task == "pursuit_evasive":
        return self._mppi_evasion_action()     # NEW: MPPI pursuit=False
    elif self._task == "search_pursuit":
        return self._mppi_evasion_action()     # same as evasive
    elif self._task == "dogfight":
        return self._current_opponent_action() # existing logic (random/mppi/policy)
```

New methods:

- `_trajectory_action()`: Computes PD-controlled motor thrusts to follow the current trajectory waypoint. Trajectory state (current waypoint index, phase) stored as instance variables, advanced each step.
- `_mppi_evasion_action()`: Wraps existing `MppiController.compute_action(state_b, state_a, pursuit=False)`. Requires constructing an MPPI controller internally (same as `opponent="mppi"` but with `pursuit=False`).

### Spawn logic in `reset()`

```python
def reset(self, ...):
    if self._task == "search_pursuit":
        pos_a, pos_b = self._occluded_spawn()  # NEW
    else:
        pos_a, pos_b = self._visible_spawn()   # existing jitter logic
```

`_occluded_spawn()`:
1. Place drone A at its configured spawn + jitter
2. Pick a random obstacle from `self._obstacles`
3. Place drone B on the far side of that obstacle (opposite direction from A)
4. Verify occlusion via `self._sim.arena_sdf()` or reset and retry
5. The Rust side's `check_line_of_sight` is not directly exposed to Python yet — add a thin wrapper: `Simulation.check_los(pos_a, pos_b) -> bool`

### Reward weight overrides

Stored as a dict per task, applied in `__init__`:

```python
TASK_REWARD_OVERRIDES = {
    "pursuit_linear": {
        "approach_reward": 0.2,
        "info_gain_reward": 0.0,
        "lost_contact_penalty": 0.0,
    },
    "pursuit_evasive": {},  # use defaults
    "search_pursuit": {
        "info_gain_reward": 0.1,
        "lost_contact_penalty": 0.02,
    },
    "dogfight": {},  # use defaults
}
```

Overrides are merged into `self._reward_cfg` after loading from TOML.

### Trajectory system (Stage 2 only)

Defined in a new file `aces/trajectory.py`:

```python
class Trajectory:
    """Generates time-parameterized 3D waypoints inside the arena."""
    
    @staticmethod
    def circle(center, radius, altitude, speed, t) -> np.ndarray: ...
    
    @staticmethod
    def lemniscate(center, scale, altitude, speed, t) -> np.ndarray: ...
    
    @staticmethod
    def patrol(waypoints, speed, t) -> tuple[np.ndarray, int]: ...
    
    @staticmethod
    def random_trajectory(arena_bounds, rng) -> callable: ...
```

Each returns a `(x, y, z)` target position given elapsed time `t`. The PD controller in `_trajectory_action()` tracks this target.

Arena-safety: trajectories are parameterized to stay within `bounds - margin` (margin = 1.0m from walls). Obstacle avoidance is not needed — trajectories are designed to pass between pillars, not through them.

## Implementation: `CurriculumTrainer`

New class in `aces/trainer.py`:

```python
class CurriculumTrainer:
    def __init__(
        self,
        stages: list[dict],       # [{"task": "pursuit_linear", "timesteps": 200000}, ...]
        config_dir: str | None = None,
        fpv: bool = False,
        **ppo_kwargs,
    ): ...
    
    def train(self) -> PPO:
        """Run all stages sequentially, loading weights between stages."""
        model = None
        for i, stage in enumerate(self.stages):
            env = DroneDogfightEnv(task=stage["task"], ...)
            if model is None:
                model = PPO("MlpPolicy", env, ...)
            else:
                model.set_env(env)  # swap environment, keep weights
            model.learn(total_timesteps=stage["timesteps"], ...)
            model.save(f"curriculum_stage_{i}")
        return model
```

Key detail: `model.set_env(env)` swaps the environment without resetting the policy network. Since all stages use the same observation space (21-dim or FPV Dict), the weights transfer seamlessly.

### CLI integration

```bash
# Single stage
python scripts/run.py --mode train --task pursuit_linear --timesteps 200000

# Full curriculum
python scripts/run.py --mode curriculum --timesteps 200000,300000,300000,500000
```

### Logging

Each stage appends to the same `logs/` directory with stage prefixed:

```
logs/curriculum_20260421_150000/
├── stage2_pursuit_linear_episodes.csv
├── stage3_pursuit_evasive_episodes.csv
├── stage4_search_pursuit_episodes.csv
├── stage5_dogfight_episodes.csv
└── models/
    ├── stage2.zip
    ├── stage3.zip
    ├── stage4.zip
    └── stage5.zip
```

## PyO3 Bridge Addition

One new method on `Simulation`:

```rust
/// Check line-of-sight between two arbitrary points.
fn check_los(&self, pos_a: [f64; 3], pos_b: [f64; 3]) -> bool {
    check_line_of_sight(&self.arena, &v3(pos_a), &v3(pos_b)) == Visibility::Visible
}
```

Needed by `_occluded_spawn()` in Stage 4 to verify spawn positions are actually occluded.

## Pitfalls and Mitigations

### 1. PD controller instability (Stage 2)
The simple PD position controller can oscillate or overshoot. **Mitigation**: tune `kp=2.0, kd=1.0` conservatively, clamp output aggressively, test with all three trajectory types. The opponent doesn't need to fly perfectly — slightly wobbly pursuit targets are fine for training.

### 2. MPPI evasion quality (Stage 3)
`compute_action(pursuit=False)` uses `evasion_cost` which maximizes distance. The evader might fly into walls if obstacle avoidance weight is too low. **Mitigation**: verify `w_obs=1000` is sufficient in evasion mode. If not, create a separate `evasion_weights` config.

### 3. Occluded spawn validation (Stage 4)
Random placement might fail to produce occluded pairs if obstacles are small relative to the arena. **Mitigation**: try up to 50 candidates. With 5 pillars (1m x 1m) in a 10m x 10m arena, geometric analysis shows at least 3–4 valid occluded positions per pillar. Fallback: place drones at maximum distance corners.

### 4. Value function mismatch between stages
The value head trained on Stage 2 rewards (scale ~0–100) may produce poor estimates when Stage 3 rewards have different dynamics. **Mitigation**: keep reward scales consistent across stages (all use kill_reward=100, collision_penalty=-50). The shaping rewards differ in magnitude but are small relative to terminal rewards.

### 5. Observation space consistency
All stages must produce the exact same 21-dim observation. For Stage 2, the "opponent" is not a real drone with its own lock-on — but the observation still includes `being_locked_progress` (which will be 0 since the trajectory opponent can't lock). This is fine: the agent learns to ignore irrelevant obs dimensions.

### 6. Self-play in Stage 5
The current `OpponentUpdateCallback` copies agent weights to the opponent. In curriculum mode, the opponent starts from Stage 4 weights (which learned evasion but not attack). Self-play will naturally bootstrap from there. No special handling needed.

## What This Does NOT Include

- Automatic stage promotion based on win rate (manual timestep budgets for now)
- Adaptive difficulty within a stage (e.g., gradually increasing opponent MPPI samples)
- Multi-agent (>2 drones)
- FPV-specific trajectory rendering (FPV uses the same Rust depth camera, which will see the trajectory opponent as a normal drone)

## Success Criteria

1. `python scripts/run.py --mode train --task pursuit_linear` trains successfully
2. `python scripts/run.py --mode curriculum` runs all 4 stages end-to-end
3. Each stage's model can be exported and loaded in Bevy
4. Stage 4 agent demonstrably searches for occluded opponent (visible in Rerun viz)
5. Stage 5 agent achieves >30% win rate vs MPPI in evaluation
6. All existing 50 tests continue to pass (backward compatible)
7. `episodes.csv` logs include a `stage` column for curriculum runs
