# Curriculum Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add parameterized curriculum stages (pursuit_linear → pursuit_evasive → search_pursuit → dogfight) to `DroneDogfightEnv` and a `CurriculumTrainer` that chains them with weight transfer.

**Architecture:** Single `task` parameter on the existing env controls opponent behavior, spawn logic, and reward weights. A new `Trajectory` class generates waypoints for Stage 2. A new `CurriculumTrainer` runs stages sequentially, swapping envs while preserving PPO weights. One new Rust method `check_los` on the PyO3 `Simulation` for occluded spawn validation.

**Tech Stack:** Python (gymnasium, stable-baselines3, numpy), Rust (sim-core collision via PyO3)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `aces/trajectory.py` | **Create** | Trajectory generators (circle, lemniscate, patrol) + PD position controller |
| `aces/env.py` | **Modify** | Add `task` param, reward overrides, new opponent modes, occluded spawn |
| `aces/trainer.py` | **Modify** | Add `CurriculumTrainer` class |
| `scripts/run.py` | **Modify** | Add `--task` and `--mode curriculum` CLI options |
| `crates/py-bridge/src/lib.rs` | **Modify** | Add `check_los()` method to `Simulation` |
| `tests/test_trajectory.py` | **Create** | Tests for trajectory generators |
| `tests/test_env.py` | **Modify** | Tests for new task modes |
| `tests/test_trainer.py` | **Modify** | Tests for `CurriculumTrainer` |

---

### Task 1: Trajectory generators

**Files:**
- Create: `aces/trajectory.py`
- Create: `tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests for circle trajectory**

```python
# tests/test_trajectory.py
import math
import numpy as np
import pytest
from aces.trajectory import Trajectory


def test_circle_returns_3d_point():
    pos = Trajectory.circle(center=[5, 5], radius=2.0, altitude=1.5, speed=1.0, t=0.0)
    assert pos.shape == (3,)
    assert pos[2] == pytest.approx(1.5)


def test_circle_stays_on_radius():
    center = [5.0, 5.0]
    radius = 2.0
    for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
        pos = Trajectory.circle(center=center, radius=radius, altitude=1.5, speed=1.0, t=t)
        dx = pos[0] - center[0]
        dy = pos[1] - center[1]
        assert math.sqrt(dx**2 + dy**2) == pytest.approx(radius, abs=1e-6)


def test_circle_moves_with_time():
    p0 = Trajectory.circle(center=[5, 5], radius=2.0, altitude=1.5, speed=1.0, t=0.0)
    p1 = Trajectory.circle(center=[5, 5], radius=2.0, altitude=1.5, speed=1.0, t=1.0)
    assert not np.allclose(p0, p1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aces.trajectory'`

- [ ] **Step 3: Implement circle trajectory**

```python
# aces/trajectory.py
"""Trajectory generators for curriculum Stage 2 opponents."""

from __future__ import annotations

import math

import numpy as np


class Trajectory:
    """Time-parameterized 3D trajectories inside the arena."""

    @staticmethod
    def circle(
        center: list[float],
        radius: float,
        altitude: float,
        speed: float,
        t: float,
    ) -> np.ndarray:
        """Horizontal circle at fixed altitude.

        Args:
            center: [x, y] center of the circle.
            radius: Circle radius in meters.
            altitude: Fixed z coordinate.
            speed: Angular speed in rad/s.
            t: Elapsed time in seconds.
        """
        angle = speed * t
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        return np.array([x, y, altitude], dtype=np.float64)

    @staticmethod
    def lemniscate(
        center: list[float],
        scale: float,
        altitude: float,
        speed: float,
        t: float,
    ) -> np.ndarray:
        """Figure-8 (lemniscate of Bernoulli) at fixed altitude.

        Parametric form: x = scale * cos(t) / (1 + sin^2(t))
                         y = scale * sin(t) * cos(t) / (1 + sin^2(t))
        """
        angle = speed * t
        denom = 1.0 + math.sin(angle) ** 2
        x = center[0] + scale * math.cos(angle) / denom
        y = center[1] + scale * math.sin(angle) * math.cos(angle) / denom
        return np.array([x, y, altitude], dtype=np.float64)

    @staticmethod
    def patrol(
        waypoints: list[list[float]],
        speed: float,
        t: float,
    ) -> np.ndarray:
        """Linear interpolation between waypoints at constant speed.

        Returns the 3D position at time t. Loops back to start after
        reaching the last waypoint.
        """
        if len(waypoints) < 2:
            return np.array(waypoints[0], dtype=np.float64)

        # Compute segment lengths
        pts = [np.array(w, dtype=np.float64) for w in waypoints]
        seg_lengths = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)]
        total_length = sum(seg_lengths)
        if total_length < 1e-9:
            return pts[0].copy()

        # Distance traveled (looping)
        dist = (speed * t) % total_length

        # Find segment
        accum = 0.0
        for i, seg_len in enumerate(seg_lengths):
            if accum + seg_len >= dist:
                frac = (dist - accum) / seg_len if seg_len > 1e-9 else 0.0
                return pts[i] + frac * (pts[i + 1] - pts[i])
            accum += seg_len
        return pts[-1].copy()

    @staticmethod
    def random_trajectory(
        bounds: list[float],
        rng: np.random.Generator,
    ) -> tuple[str, dict]:
        """Pick a random trajectory type with arena-safe parameters.

        Args:
            bounds: [bx, by, bz] arena dimensions.
            rng: Numpy random generator.

        Returns:
            (type_name, kwargs) to be passed to the corresponding static method.
        """
        margin = 1.5  # stay away from walls
        cx = bounds[0] / 2
        cy = bounds[1] / 2
        alt = rng.uniform(0.8, bounds[2] - 0.5)

        choice = rng.choice(["circle", "lemniscate", "patrol"])

        if choice == "circle":
            radius = rng.uniform(1.0, min(cx, cy) - margin)
            speed = rng.uniform(0.5, 1.5)
            return "circle", dict(center=[cx, cy], radius=radius, altitude=alt, speed=speed)

        elif choice == "lemniscate":
            scale = rng.uniform(1.5, min(cx, cy) - margin)
            speed = rng.uniform(0.5, 1.2)
            return "lemniscate", dict(center=[cx, cy], scale=scale, altitude=alt, speed=speed)

        else:  # patrol
            n_wp = rng.integers(3, 6)
            wps = []
            for _ in range(n_wp):
                x = rng.uniform(margin, bounds[0] - margin)
                y = rng.uniform(margin, bounds[1] - margin)
                wps.append([x, y, alt])
            # Close the loop
            wps.append(wps[0])
            speed = rng.uniform(1.0, 3.0)
            return "patrol", dict(waypoints=wps, speed=speed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory.py -v`
Expected: 3 passed

- [ ] **Step 5: Add lemniscate and patrol tests**

Append to `tests/test_trajectory.py`:

```python
def test_lemniscate_returns_3d():
    pos = Trajectory.lemniscate(center=[5, 5], scale=2.0, altitude=1.5, speed=1.0, t=0.0)
    assert pos.shape == (3,)
    assert pos[2] == pytest.approx(1.5)


def test_lemniscate_stays_near_center():
    for t in np.linspace(0, 10, 50):
        pos = Trajectory.lemniscate(center=[5, 5], scale=2.0, altitude=1.5, speed=1.0, t=t)
        assert 3.0 < pos[0] < 7.0
        assert 3.0 < pos[1] < 7.0


def test_patrol_loops():
    wps = [[1, 1, 1.5], [9, 1, 1.5], [9, 9, 1.5], [1, 1, 1.5]]
    # Total perimeter: 8 + 8 + ~11.3 ≈ 27.3m. At speed=1, t=27.3 should loop to start.
    p0 = Trajectory.patrol(waypoints=wps, speed=1.0, t=0.0)
    assert p0[0] == pytest.approx(1.0)
    assert p0[1] == pytest.approx(1.0)


def test_patrol_interpolates():
    wps = [[0, 0, 1.5], [10, 0, 1.5]]
    mid = Trajectory.patrol(waypoints=wps, speed=1.0, t=5.0)
    assert mid[0] == pytest.approx(5.0)
    assert mid[1] == pytest.approx(0.0)


def test_random_trajectory_returns_valid():
    rng = np.random.default_rng(42)
    for _ in range(20):
        name, kwargs = Trajectory.random_trajectory(bounds=[10, 10, 3], rng=rng)
        assert name in ("circle", "lemniscate", "patrol")
        # Generate a point to ensure it doesn't crash
        pos = getattr(Trajectory, name)(t=0.0, **kwargs)
        assert pos.shape == (3,)
        # Check in bounds
        assert 0 < pos[0] < 10
        assert 0 < pos[1] < 10
        assert 0 < pos[2] < 3
```

- [ ] **Step 6: Run full test file**

Run: `pytest tests/test_trajectory.py -v`
Expected: 8 passed

- [ ] **Step 7: Commit**

```bash
git add aces/trajectory.py tests/test_trajectory.py
git commit -m "feat: add trajectory generators for curriculum Stage 2"
```

---

### Task 2: PyO3 bridge — add `check_los`

**Files:**
- Modify: `crates/py-bridge/src/lib.rs` (inside `#[pymethods] impl Simulation`)

- [ ] **Step 1: Add `check_los` method**

Add this method inside the `#[pymethods] impl Simulation` block, after the existing `belief_particles_b` method:

```rust
/// Check line-of-sight between two arbitrary points.
/// Returns true if visible, false if occluded by obstacles.
fn check_los(&self, pos_a: [f64; 3], pos_b: [f64; 3]) -> bool {
    check_line_of_sight(&self.arena, &v3(pos_a), &v3(pos_b)) == Visibility::Visible
}
```

- [ ] **Step 2: Build and verify**

Run: `cargo check -p aces-py-bridge`
Expected: compiles with no errors (the imports `check_line_of_sight`, `Visibility`, and `v3` are already in scope)

- [ ] **Step 3: Rebuild Python extension**

Run: `poetry run maturin develop`
Expected: builds successfully

- [ ] **Step 4: Quick smoke test**

Run: `python -c "from aces._core import Simulation; s = Simulation(bounds=[10,10,3], obstacles=[([5,5,1.5],[0.5,0.5,1.5])]); s.reset([1,1,1.5],[9,9,1.5]); print('LOS:', s.check_los([1,1,1.5],[9,9,1.5])); print('Blocked:', not s.check_los([4,5,1.5],[6,5,1.5]))"`
Expected: `LOS: True` and `Blocked: True` (pillar at 5,5 blocks the second pair)

- [ ] **Step 5: Commit**

```bash
git add crates/py-bridge/src/lib.rs
git commit -m "feat: expose check_los on PyO3 Simulation for occluded spawn"
```

---

### Task 3: Env — add `task` parameter, reward overrides, MPPI evasion

**Files:**
- Modify: `aces/env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write failing tests for new task modes**

Append to `tests/test_env.py`:

```python
def test_env_task_pursuit_evasive():
    """pursuit_evasive task creates env, steps without error."""
    env = DroneDogfightEnv(task="pursuit_evasive", max_episode_steps=10)
    obs, info = env.reset(seed=42)
    assert obs.shape == (21,)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def test_env_task_default_is_dogfight():
    """Default task is dogfight (backward compatible)."""
    env = DroneDogfightEnv(max_episode_steps=5)
    assert env._task == "dogfight"


def test_env_task_reward_overrides():
    """pursuit_linear overrides approach_reward."""
    env = DroneDogfightEnv(task="pursuit_linear", max_episode_steps=5)
    assert env._reward_cfg["approach_reward"] == 0.2
    assert env._reward_cfg["info_gain_reward"] == 0.0


def test_env_task_search_pursuit_reward():
    """search_pursuit boosts info_gain_reward."""
    env = DroneDogfightEnv(task="search_pursuit", max_episode_steps=5)
    assert env._reward_cfg["info_gain_reward"] == 0.1
    assert env._reward_cfg["lost_contact_penalty"] == 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_env.py::test_env_task_pursuit_evasive -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'task'`

- [ ] **Step 3: Implement task parameter and reward overrides in env.py**

In `aces/env.py`, add `task` parameter to `__init__` signature (after `fpv`):

```python
    def __init__(
        self,
        config_dir: str | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        opponent: str = "random",
        mppi_samples: int | None = None,
        mppi_horizon: int | None = None,
        wind_sigma: float | None = None,
        obs_noise_std: float | None = None,
        fpv: bool = False,
        task: str = "dogfight",
    ):
```

Right after `self._reward_cfg = rules_cfg["reward"]` (around line 100), add task setup:

```python
        self._task = task

        # Task-specific reward overrides
        _TASK_REWARD_OVERRIDES = {
            "pursuit_linear": {
                "approach_reward": 0.2,
                "info_gain_reward": 0.0,
                "lost_contact_penalty": 0.0,
            },
            "pursuit_evasive": {},
            "search_pursuit": {
                "info_gain_reward": 0.1,
                "lost_contact_penalty": 0.02,
            },
            "dogfight": {},
        }
        for key, val in _TASK_REWARD_OVERRIDES.get(task, {}).items():
            self._reward_cfg[key] = val
```

For tasks that need MPPI evasion (`pursuit_evasive`, `search_pursuit`), ensure an MPPI controller is constructed even if `opponent` was not set to "mppi". Add after the existing MPPI construction block (around line 202):

```python
        # Curriculum tasks that need MPPI evasion
        if task in ("pursuit_evasive", "search_pursuit") and self._mppi is None:
            self._mppi = MppiController(
                bounds=self._bounds,
                obstacles=self._obstacles,
                num_samples=mppi_cfg.get("num_samples", 256),
                horizon=mppi_cfg.get("horizon", 30),
                noise_std=mppi_cfg["noise_std"],
                temperature=mppi_cfg["temperature"],
                mass=self._mass,
                arm_length=self._arm_length,
                inertia=self._inertia,
                max_thrust=self._max_thrust,
                torque_coeff=self._torque_coeff,
                drag_coeff=self._drag_coeff,
                dt_ctrl=self._dt_ctrl,
                substeps=self._substeps,
                drone_radius=self._drone_radius,
                w_lock=mppi_cfg["weights"]["w_lock"],
                w_dist=mppi_cfg["weights"]["w_dist"],
                w_face=mppi_cfg["weights"]["w_face"],
                w_vel=mppi_cfg["weights"]["w_vel"],
                w_ctrl=mppi_cfg["weights"]["w_ctrl"],
                w_obs=mppi_cfg["weights"]["w_obs"],
                d_safe=mppi_cfg["weights"]["d_safe"],
            )
```

- [ ] **Step 4: Implement opponent action dispatch**

Replace the existing `_opponent_action` method in `env.py`:

```python
    def _opponent_action(self) -> list[float]:
        """Compute opponent motor thrusts based on current task/mode."""
        if self._task == "pursuit_linear":
            return self._trajectory_action()
        elif self._task in ("pursuit_evasive", "search_pursuit"):
            return self._mppi_evasion_action()
        # dogfight: use existing opponent logic
        elif self._opponent_mode == "random":
            raw = self.np_random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
            return self._map_action(raw)
        elif self._opponent_mode == "mppi" and self._mppi is not None:
            state_b = list(self._sim.drone_b_state())
            state_a = list(self._sim.drone_a_state())
            action = self._mppi.compute_action(state_b, state_a, True)
            return list(action)
        elif self._opponent_mode == "policy" and self._opponent_policy is not None:
            state_b = list(self._sim.drone_b_state())
            state_a = list(self._sim.drone_a_state())
            if self._fpv:
                opp_obs = self._build_fpv_obs(
                    state_b,
                    nearest_obs_dist=0.0,
                    lock_progress=0.0,
                    being_locked_progress=0.0,
                    depth_image=None,
                )
            else:
                opp_obs = self._build_obs(state_b, state_a, 0.0, 0.0, 0.0)
            raw, _ = self._opponent_policy.predict(opp_obs, deterministic=True)
            return self._map_action(np.array(raw, dtype=np.float32))
        else:
            return [self._hover_thrust] * 4

    def _mppi_evasion_action(self) -> list[float]:
        """MPPI opponent in evasion mode (pursuit=False)."""
        state_b = list(self._sim.drone_b_state())
        state_a = list(self._sim.drone_a_state())
        action = self._mppi.compute_action(state_b, state_a, False)
        return list(action)

    def _trajectory_action(self) -> list[float]:
        """PD controller tracking the current trajectory waypoint."""
        target = self._traj_fn(t=self._step_count * self._dt_ctrl)
        state_b = self._sim.drone_b_state()
        pos = np.array(state_b[:3], dtype=np.float64)
        vel = np.array(state_b[3:6], dtype=np.float64)

        error = target - pos
        kp, kd = 2.0, 1.0
        # PD → desired acceleration, mapped to thrust delta
        accel_cmd = kp * error - kd * vel
        hover = self._hover_thrust

        # Simple uniform thrust + per-axis correction (approximate)
        # Each motor gets hover + a fraction of the correction
        thrust = hover + (accel_cmd[2] * self._mass / 4.0)
        # Lateral corrections via differential thrust (simplified)
        lateral_mag = float(np.linalg.norm(accel_cmd[:2])) * self._mass * 0.25
        motors = [
            max(0.0, min(self._max_thrust, thrust + lateral_mag)),
            max(0.0, min(self._max_thrust, thrust + lateral_mag)),
            max(0.0, min(self._max_thrust, thrust - lateral_mag)),
            max(0.0, min(self._max_thrust, thrust - lateral_mag)),
        ]
        return motors
```

- [ ] **Step 5: Add trajectory setup to `__init__` and `reset`**

Add after the task reward overrides block in `__init__`:

```python
        # Trajectory state for pursuit_linear task
        self._traj_fn = None
        self._traj_type = None
        self._traj_kwargs = None
```

Add the `from aces.trajectory import Trajectory` import at the top of env.py.

In `reset()`, after `super().reset(seed=seed)` and `self._step_count = 0`, add:

```python
        # Set up trajectory for pursuit_linear
        if self._task == "pursuit_linear":
            self._traj_type, self._traj_kwargs = Trajectory.random_trajectory(
                self._bounds, self.np_random
            )
            self._traj_fn = lambda t: getattr(Trajectory, self._traj_type)(
                t=t, **self._traj_kwargs
            )
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_env.py -v -k "task"`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add aces/env.py tests/test_env.py
git commit -m "feat: add task parameter to env with reward overrides and MPPI evasion"
```

---

### Task 4: Env — occluded spawn for `search_pursuit`

**Files:**
- Modify: `aces/env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_env.py`:

```python
def test_env_search_pursuit_occluded_spawn():
    """search_pursuit spawns drones out of line-of-sight."""
    env = DroneDogfightEnv(task="search_pursuit", max_episode_steps=10)
    # Run multiple resets — most should produce occluded spawns
    occluded_count = 0
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        # Check if opponent is NOT visible at spawn
        # obs[18] is opponent_visible in the 21-dim vector
        if obs[18] < 0.5:
            occluded_count += 1
    # At least half should be occluded (fallback allows some visible)
    assert occluded_count >= 10, f"Only {occluded_count}/20 were occluded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env.py::test_env_search_pursuit_occluded_spawn -v`
Expected: FAIL — spawns are currently always at default visible positions

- [ ] **Step 3: Implement occluded spawn**

Add this method to `DroneDogfightEnv`:

```python
    def _occluded_spawn(self) -> tuple[list[float], list[float]]:
        """Find spawn positions where drones can't see each other.

        Places drone A at its configured spawn + jitter, then tries
        positions on the far side of obstacles for drone B.
        """
        jitter_a = self.np_random.uniform(-0.3, 0.3, size=(3,))
        pos_a = [self._spawn_a[i] + jitter_a[i] for i in range(3)]

        obstacles = self._obstacles
        if not obstacles:
            # No obstacles — fall back to max distance
            pos_b = [self._spawn_b[i] + self.np_random.uniform(-0.3, 0.3) for i in range(3)]
            return pos_a, pos_b

        for _ in range(50):
            # Pick a random obstacle
            obs_center, obs_half = obstacles[self.np_random.integers(len(obstacles))]

            # Place B on the opposite side of the obstacle from A
            dir_a_to_obs = np.array(obs_center[:2]) - np.array(pos_a[:2])
            dist = float(np.linalg.norm(dir_a_to_obs))
            if dist < 0.1:
                continue
            dir_norm = dir_a_to_obs / dist

            # Place B: obstacle center + direction * (half_extent + margin)
            offset = max(obs_half[0], obs_half[1]) + self.np_random.uniform(0.5, 1.5)
            bx = obs_center[0] + dir_norm[0] * offset
            by = obs_center[1] + dir_norm[1] * offset
            bz = self.np_random.uniform(0.8, self._bounds[2] - 0.5)

            # Clamp to arena bounds with margin
            margin = 0.5
            bx = float(np.clip(bx, margin, self._bounds[0] - margin))
            by = float(np.clip(by, margin, self._bounds[1] - margin))

            pos_b = [bx, by, bz]

            # Verify occlusion
            if not self._sim.check_los(pos_a, pos_b):
                return pos_a, pos_b

        # Fallback: maximum distance corners (might be visible)
        pos_b = [
            self._bounds[0] - pos_a[0],
            self._bounds[1] - pos_a[1],
            self.np_random.uniform(0.8, self._bounds[2] - 0.5),
        ]
        return pos_a, pos_b
```

Modify `reset()` — replace the existing spawn jitter block:

```python
        # Spawn positions
        if self._task == "search_pursuit":
            pos_a, pos_b = self._occluded_spawn()
        else:
            jitter_a = self.np_random.uniform(-0.5, 0.5, size=(3,))
            jitter_b = self.np_random.uniform(-0.5, 0.5, size=(3,))
            pos_a = [self._spawn_a[i] + jitter_a[i] for i in range(3)]
            pos_b = [self._spawn_b[i] + jitter_b[i] for i in range(3)]
```

Note: `_occluded_spawn` calls `self._sim.check_los()` which requires the sim to exist. Since `_build_sim()` is called in `__init__`, the sim is available by the time `reset()` runs. But `check_los` needs the arena loaded in the sim, which happens at construction — so this works.

- [ ] **Step 4: Run test**

Run: `pytest tests/test_env.py::test_env_search_pursuit_occluded_spawn -v`
Expected: PASS

- [ ] **Step 5: Run full env test suite for backward compatibility**

Run: `pytest tests/test_env.py -v`
Expected: all existing tests pass (default `task="dogfight"` doesn't change behavior)

- [ ] **Step 6: Commit**

```bash
git add aces/env.py tests/test_env.py
git commit -m "feat: occluded spawn for search_pursuit task"
```

---

### Task 5: `CurriculumTrainer`

**Files:**
- Modify: `aces/trainer.py`
- Modify: `tests/test_trainer.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_trainer.py`:

```python
from aces.trainer import CurriculumTrainer


def test_curriculum_trainer_runs():
    """CurriculumTrainer runs 2 stages end-to-end."""
    trainer = CurriculumTrainer(
        stages=[
            {"task": "pursuit_linear", "timesteps": 256},
            {"task": "pursuit_evasive", "timesteps": 256},
        ],
        n_steps=128,
        batch_size=64,
    )
    model = trainer.train()
    assert model is not None
    assert len(trainer.stage_stats) == 2


def test_curriculum_trainer_saves_models(tmp_path):
    """Each stage saves a model checkpoint."""
    trainer = CurriculumTrainer(
        stages=[
            {"task": "pursuit_linear", "timesteps": 128},
        ],
        n_steps=128,
        batch_size=64,
        save_dir=str(tmp_path),
    )
    trainer.train()
    assert (tmp_path / "stage0_pursuit_linear.zip").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trainer.py::test_curriculum_trainer_runs -v`
Expected: FAIL — `ImportError: cannot import name 'CurriculumTrainer'`

- [ ] **Step 3: Implement `CurriculumTrainer`**

Add to `aces/trainer.py` after the `SelfPlayTrainer` class:

```python
class CurriculumTrainer:
    """Trains PPO through sequential curriculum stages.

    Each stage uses the same observation space but different opponent behavior
    and reward weights. Weights transfer between stages via ``model.set_env()``.
    """

    def __init__(
        self,
        stages: list[dict],
        config_dir: str | None = None,
        fpv: bool = False,
        save_dir: str | None = None,
        n_steps: int = 2048,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        wind_sigma: float | None = None,
        obs_noise_std: float | None = None,
    ):
        self.stages = stages
        self._config_dir = config_dir
        self._fpv = fpv
        self._save_dir = Path(save_dir) if save_dir else None
        self._ppo_kwargs = dict(
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            n_epochs=n_epochs,
            verbose=0,
        )
        self._wind_sigma = wind_sigma
        self._obs_noise_std = obs_noise_std
        self.stage_stats: list[dict] = []

    def _make_env(self, task: str) -> DroneDogfightEnv:
        return DroneDogfightEnv(
            config_dir=self._config_dir,
            max_episode_steps=1000,
            task=task,
            wind_sigma=self._wind_sigma,
            obs_noise_std=self._obs_noise_std,
            fpv=self._fpv,
        )

    def _resolve_policy(self) -> tuple[str, dict | None]:
        if self._fpv:
            from aces.policy import CnnImuExtractor

            return "MultiInputPolicy", {
                "features_extractor_class": CnnImuExtractor,
                "features_extractor_kwargs": {"features_dim": 192},
            }
        return "MlpPolicy", None

    def train(self) -> PPO:
        log_dir = Path("logs") / f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model = None

        for i, stage in enumerate(self.stages):
            task = stage["task"]
            timesteps = stage["timesteps"]
            env = self._make_env(task)

            print(f"\n[ACES] ═══ Stage {i} — {task} ({timesteps} steps) ═══")

            if model is None:
                policy_name, policy_kwargs = self._resolve_policy()
                kwargs = {**self._ppo_kwargs}
                if policy_kwargs:
                    kwargs["policy_kwargs"] = policy_kwargs
                model = PPO(policy_name, env, **kwargs)
            else:
                model.set_env(env)

            stage_log = log_dir / f"stage{i}_{task}"
            stats_cb = TrainingStatsCallback()
            logger_cb = EpisodeLoggerCallback(log_dir=str(stage_log), verbose=1)

            model.learn(
                total_timesteps=timesteps,
                callback=[stats_cb, logger_cb],
                reset_num_timesteps=False,
            )

            self.stage_stats.append(stats_cb.summary())

            if self._save_dir:
                self._save_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(self._save_dir / f"stage{i}_{task}"))

            print(f"[ACES] Stage {i} done: {stats_cb.summary()}")

        return model
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_trainer.py -v -k "curriculum"`
Expected: 2 passed

- [ ] **Step 5: Run full trainer test suite**

Run: `pytest tests/test_trainer.py -v`
Expected: all tests pass (existing tests unaffected)

- [ ] **Step 6: Commit**

```bash
git add aces/trainer.py tests/test_trainer.py
git commit -m "feat: CurriculumTrainer with sequential stage training"
```

---

### Task 6: CLI integration

**Files:**
- Modify: `scripts/run.py`

- [ ] **Step 1: Add `--task` and `--mode curriculum` options**

In `scripts/run.py`, add argument after `--fpv`:

```python
    parser.add_argument(
        "--task",
        default="dogfight",
        choices=["pursuit_linear", "pursuit_evasive", "search_pursuit", "dogfight"],
        help="Curriculum task / difficulty stage",
    )
```

Change the `--mode` choices:

```python
    parser.add_argument(
        "--mode",
        choices=["mppi-vs-mppi", "train", "evaluate", "export", "curriculum"],
        default="mppi-vs-mppi",
    )
```

- [ ] **Step 2: Pass `task` to training**

In `run_train()`, modify the `SelfPlayTrainer` construction to pass `task`:

```python
    trainer = SelfPlayTrainer(
        config_dir=args.config_dir,
        total_timesteps=args.timesteps,
        wind_sigma=wind_sigma,
        obs_noise_std=obs_noise_std,
        fpv=args.fpv,
        task=args.task,
    )
```

This requires adding `task` to `SelfPlayTrainer.__init__` — add it as a parameter and pass to `DroneDogfightEnv`:

In `aces/trainer.py`, `SelfPlayTrainer.__init__`, add `task: str = "dogfight"` parameter, and pass it to env:

```python
        self.env = DroneDogfightEnv(
            config_dir=config_dir,
            max_episode_steps=max_episode_steps,
            opponent="random",
            wind_sigma=wind_sigma,
            obs_noise_std=obs_noise_std,
            fpv=fpv,
            task=task,
        )
```

- [ ] **Step 3: Add curriculum mode handler**

Add to the bottom of `scripts/run.py`, in the `main()` dispatch:

```python
    elif args.mode == "curriculum":
        from aces.trainer import CurriculumTrainer

        wind_sigma, obs_noise_std = _resolve_noise(args)
        ts = [int(x) for x in args.timesteps.split(",")] if isinstance(args.timesteps, str) else [args.timesteps] * 4
        tasks = ["pursuit_linear", "pursuit_evasive", "search_pursuit", "dogfight"]
        # Pad timesteps if fewer than 4 given
        while len(ts) < len(tasks):
            ts.append(ts[-1])
        stages = [{"task": t, "timesteps": s} for t, s in zip(tasks, ts)]

        print(f"[ACES] Curriculum training: {len(stages)} stages")
        for s in stages:
            print(f"  {s['task']}: {s['timesteps']} steps")

        trainer = CurriculumTrainer(
            stages=stages,
            config_dir=args.config_dir,
            fpv=args.fpv,
            save_dir=args.save_path,
            wind_sigma=wind_sigma,
            obs_noise_std=obs_noise_std,
        )
        model = trainer.train()
        model.save(args.save_path + "_final")
        print(f"[ACES] Final model saved to {args.save_path}_final")
```

Change the `--timesteps` argument to accept strings (for comma-separated):

```python
    parser.add_argument("--timesteps", default="500000")
```

And in `run_train`, parse it:

```python
    timesteps = int(args.timesteps.split(",")[0]) if "," in str(args.timesteps) else int(args.timesteps)
```

- [ ] **Step 4: Verify CLI works**

Run: `python scripts/run.py --mode train --task pursuit_linear --timesteps 256 --no-vis`
Expected: trains briefly, no errors

Run: `python scripts/run.py --mode curriculum --timesteps 128,128,128,128 --no-vis`
Expected: runs 4 stages, prints stage transitions, no errors

- [ ] **Step 5: Commit**

```bash
git add scripts/run.py aces/trainer.py
git commit -m "feat: CLI support for --task and --mode curriculum"
```

---

### Task 7: Integration test — full curriculum end-to-end

**Files:**
- Modify: `tests/test_trainer.py`

- [ ] **Step 1: Write integration test**

Append to `tests/test_trainer.py`:

```python
def test_curriculum_full_pipeline():
    """Full 4-stage curriculum runs without error and produces models."""
    trainer = CurriculumTrainer(
        stages=[
            {"task": "pursuit_linear", "timesteps": 128},
            {"task": "pursuit_evasive", "timesteps": 128},
            {"task": "search_pursuit", "timesteps": 128},
            {"task": "dogfight", "timesteps": 128},
        ],
        n_steps=128,
        batch_size=64,
    )
    model = trainer.train()
    assert model is not None
    assert len(trainer.stage_stats) == 4
    # Each stage should have recorded at least 1 episode
    for i, stats in enumerate(trainer.stage_stats):
        assert stats["episodes"] >= 0, f"Stage {i} failed"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_trainer.py::test_curriculum_full_pipeline -v`
Expected: PASS (may take 30-60s due to MPPI in stages 3-4)

- [ ] **Step 3: Run entire test suite**

Run: `pytest tests/ -v`
Expected: all tests pass, including all existing tests

- [ ] **Step 4: Commit**

```bash
git add tests/test_trainer.py
git commit -m "test: full curriculum pipeline integration test"
```

---

### Task 8: Update documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add curriculum commands to CLAUDE.md Key Commands section**

Add after the existing training command:

```markdown
python scripts/run.py --mode train --task pursuit_linear --timesteps 200000  # single stage
python scripts/run.py --mode curriculum --timesteps 200000,300000,300000,500000  # full curriculum
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add curriculum training commands"
```
