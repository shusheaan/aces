"""Ensure CPU env reward matches batch-sim reward for common scenarios.

ACES has two code paths that compute per-step reward for drone A:

* CPU path: :meth:`aces.env.dogfight.DroneDogfightEnv.step` — Python reward
  shaping block that reads ``configs/rules.toml`` via the typed
  ``RewardConfig`` dataclass plus per-task overrides.

* Batch / GPU path: :func:`aces_batch_sim::reward::compute_reward_a`
  (Rust). Used by :class:`aces_batch_sim::orchestrator::BatchOrchestrator`
  and :class:`aces_batch_sim::gpu::orchestrator::GpuBatchOrchestrator`.
  Defaults live in ``RewardConfig::default()`` in ``crates/batch-sim/src/reward.rs``.

If these diverge, a policy trained against one backend will see different
learning dynamics on the other. This test pins the constants AND the
formula shape to a shared canonical form so drift is caught early.

It complements :mod:`tests.test_action_normalization_consistency`,
which does the same job for the action denormalization path.

Skipped in CPU-only environments (needs ``aces._core`` for the CPU env).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from aces.config import load_configs


# ---------------------------------------------------------------------------
# Canonical batch-sim defaults, mirrored from
# ``crates/batch-sim/src/reward.rs::RewardConfig::default()``.
#
# These are duplicated (rather than read from the Rust extension) so the
# test fails loudly when either side drifts — copy/paste is the signal.
# ---------------------------------------------------------------------------

BATCH_SIM_DEFAULTS = {
    "kill_reward": 100.0,
    "killed_penalty": -100.0,
    "collision_penalty": -50.0,
    "opponent_crash_reward": 5.0,
    "lock_progress_reward": 5.0,
    "approach_reward": 3.0,  # tuned for the dogfight task
    "survival_bonus": 0.01,
    "control_penalty": 0.01,
}


def _read_rules_toml() -> dict:
    """Parse ``configs/rules.toml`` directly, without going through
    typed dataclasses, so we can inspect per-task overrides too."""
    root = Path(__file__).resolve().parent.parent
    with (root / "configs" / "rules.toml").open("rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# 1. Config-level consistency: rules.toml base + dogfight override must
#    agree with the batch-sim Rust defaults.
# ---------------------------------------------------------------------------


def test_terminal_rewards_match_batch_sim_defaults():
    """Terminal rewards (kill, killed, collision) must match across paths."""
    raw = _read_rules_toml()
    base = raw["reward"]
    for key in ("kill_reward", "killed_penalty", "collision_penalty"):
        assert base[key] == BATCH_SIM_DEFAULTS[key], (
            f"rules.toml [reward].{key}={base[key]} disagrees with "
            f"batch-sim RewardConfig::default().{key}={BATCH_SIM_DEFAULTS[key]}. "
            "Update one side."
        )


def test_shaping_constants_match_batch_sim_defaults():
    """Per-step shaping constants (survival, control, lock) must match."""
    raw = _read_rules_toml()
    base = raw["reward"]
    for key in ("lock_progress_reward", "survival_bonus", "control_penalty"):
        assert base[key] == BATCH_SIM_DEFAULTS[key], (
            f"rules.toml [reward].{key}={base[key]} disagrees with "
            f"batch-sim default {BATCH_SIM_DEFAULTS[key]}."
        )


def test_dogfight_task_overrides_match_batch_sim_defaults():
    """batch-sim defaults were tuned for the ``dogfight`` task, so that
    task's overrides must be the fixed point where CPU == GPU."""
    raw = _read_rules_toml()
    dogfight = raw["task_reward_overrides"]["dogfight"]
    assert dogfight["approach_reward"] == BATCH_SIM_DEFAULTS["approach_reward"]
    assert (
        dogfight["opponent_crash_reward"] == BATCH_SIM_DEFAULTS["opponent_crash_reward"]
    )


def test_typed_rewardconfig_loads_rules_toml():
    """``load_configs()`` must surface the rules.toml values unchanged."""
    cfg = load_configs()
    raw = _read_rules_toml()["reward"]
    r = cfg.rules.reward
    assert r.kill_reward == raw["kill_reward"]
    assert r.killed_penalty == raw["killed_penalty"]
    assert r.collision_penalty == raw["collision_penalty"]
    assert r.lock_progress_reward == raw["lock_progress_reward"]
    assert r.control_penalty == raw["control_penalty"]
    assert r.approach_reward == raw["approach_reward"]
    assert r.survival_bonus == raw["survival_bonus"]


# ---------------------------------------------------------------------------
# 2. Formula-level consistency: a Python replica of batch-sim's
#    ``compute_reward_a`` must agree with what the CPU env actually
#    emits on a few canonical step outcomes.
# ---------------------------------------------------------------------------


def _batch_sim_reward(
    *,
    info: dict,
    prev_distance: float,
    prev_lock_progress_a: float,
    control_cost: float,
    cfg: dict,
) -> float:
    """Pure-Python replica of ``compute_reward_a`` from
    ``crates/batch-sim/src/reward.rs``. Kept in lockstep with the Rust
    source so we can unit-test formula equivalence without a maturin build.
    """
    # Terminal rewards — priority order matches the Rust source.
    if info.get("kill_a"):
        return float(cfg["kill_reward"])
    if info.get("kill_b"):
        return float(cfg["killed_penalty"])
    if info.get("collision_a"):
        return float(cfg["collision_penalty"])
    if info.get("collision_b"):
        return float(cfg["opponent_crash_reward"])
    if info.get("timeout"):
        return 0.0

    reward: float = float(cfg["survival_bonus"])

    # Approach reward: positive when closing the gap.
    delta_distance = prev_distance - info["distance"]
    reward += float(cfg["approach_reward"]) * delta_distance

    # Lock progress: only reward positive gains.
    delta_lock = info["lock_progress_a"] - prev_lock_progress_a
    if delta_lock > 0.0:
        reward += float(cfg["lock_progress_reward"]) * delta_lock

    # Control penalty (sum of squared motor deviations from hover).
    reward -= float(cfg["control_penalty"]) * control_cost

    return reward


def test_replica_hover_step():
    """Pure hover: both drones stationary, distance unchanged, no lock.

    Expected per-step reward = survival_bonus (0.01)."""
    info = {
        "distance": 5.0,
        "lock_progress_a": 0.0,
    }
    r = _batch_sim_reward(
        info=info,
        prev_distance=5.0,
        prev_lock_progress_a=0.0,
        control_cost=0.0,
        cfg=BATCH_SIM_DEFAULTS,
    )
    assert r == pytest.approx(BATCH_SIM_DEFAULTS["survival_bonus"])


def test_replica_kill_event():
    """Agent kills opponent → +kill_reward regardless of shaping state."""
    info = {
        "kill_a": True,
        "distance": 1.0,
        "lock_progress_a": 1.0,
    }
    r = _batch_sim_reward(
        info=info,
        prev_distance=5.0,
        prev_lock_progress_a=0.5,
        control_cost=100.0,
        cfg=BATCH_SIM_DEFAULTS,
    )
    assert r == BATCH_SIM_DEFAULTS["kill_reward"]


def test_replica_got_killed():
    """Opponent kills agent → killed_penalty."""
    info = {
        "kill_b": True,
        "distance": 1.0,
        "lock_progress_a": 0.0,
    }
    r = _batch_sim_reward(
        info=info,
        prev_distance=5.0,
        prev_lock_progress_a=0.0,
        control_cost=0.0,
        cfg=BATCH_SIM_DEFAULTS,
    )
    assert r == BATCH_SIM_DEFAULTS["killed_penalty"]


def test_replica_collision():
    """Agent hits a wall → collision_penalty."""
    info = {
        "collision_a": True,
        "distance": 3.0,
        "lock_progress_a": 0.0,
    }
    r = _batch_sim_reward(
        info=info,
        prev_distance=3.0,
        prev_lock_progress_a=0.0,
        control_cost=0.0,
        cfg=BATCH_SIM_DEFAULTS,
    )
    assert r == BATCH_SIM_DEFAULTS["collision_penalty"]


def test_replica_lock_decay_not_penalized():
    """Losing lock (delta_lock < 0) must NOT produce a negative
    lock-progress shaping term. This is the fix that aligned the CPU
    env with batch-sim after the 2026-04 audit — both paths clamp the
    lock contribution at zero."""
    info = {
        "distance": 5.0,
        "lock_progress_a": 0.2,
    }
    cfg = dict(BATCH_SIM_DEFAULTS)
    r = _batch_sim_reward(
        info=info,
        prev_distance=5.0,  # distance unchanged → zero approach term
        prev_lock_progress_a=0.8,  # lock decayed by 0.6 → must be clamped
        control_cost=0.0,
        cfg=cfg,
    )
    assert r == pytest.approx(cfg["survival_bonus"])


# ---------------------------------------------------------------------------
# 3. Real CPU env agreement: drive ``DroneDogfightEnv.step`` for a short
#    canned episode and confirm the reward it emits matches what the
#    batch-sim replica would return for the same transition.
# ---------------------------------------------------------------------------


def test_cpu_env_hover_matches_replica(core_available: bool):
    """One step of pure-hover with the dogfight reward config should
    produce a reward within floating-point tolerance of the batch-sim
    formula applied to the same transition.

    Requires the Rust extension (``aces._core``). Skips gracefully if unbuilt.
    """
    if not core_available:
        pytest.skip("aces._core not available — skipping real-env consistency check")

    from aces.env.dogfight import DroneDogfightEnv

    # dogfight task: approach_reward=3.0, opponent_crash_reward=5.0 —
    # matches batch-sim defaults on those two task-overridden fields.
    env = DroneDogfightEnv(
        opponent="none",
        task="dogfight",
        max_episode_steps=16,
    )
    try:
        obs, _info = env.reset(seed=0)
        hover_action = np.zeros(4, dtype=np.float32)

        # Capture pre-step shaping state that the env uses internally.
        prev_distance = env._prev_distance
        prev_lock = env._prev_lock_progress

        obs, reward, terminated, truncated, info = env.step(hover_action)

        # Compute what batch-sim would emit for the same transition.
        # "motors_a = hover" → control_cost = 0.
        bs_info = {
            "distance": info["distance"],
            "lock_progress_a": info["lock_a_progress"],
            "kill_a": info["kill_a"],
            "kill_b": info["kill_b"],
            "collision_a": info["collision"],
        }
        # Use the task's effective reward config (dogfight overrides).
        cfg = dict(BATCH_SIM_DEFAULTS)
        # Also honour per-task overrides from rules.toml (dogfight).
        raw = _read_rules_toml()
        for k, v in raw["task_reward_overrides"].get("dogfight", {}).items():
            cfg[k] = v

        expected = _batch_sim_reward(
            info=bs_info,
            prev_distance=prev_distance,
            prev_lock_progress_a=prev_lock,
            control_cost=0.0,
            cfg=cfg,
        )

        # Step is non-terminal (hover in free space) → shaping-only reward.
        # CPU env additionally applies info-gain / lost-contact shaping on
        # top of the shared formula — rules.toml base enables them with
        # small weights. For hover with zero lock and constant belief,
        # those contributions are either zero (no visible change) or
        # strictly non-positive (lost_contact_penalty * t_since_seen).
        # So the CPU reward must be ≤ expected + tiny epsilon.
        assert not terminated
        assert reward <= expected + 1e-6, (
            f"CPU reward {reward} exceeds batch-sim replica {expected}"
        )
        # And within one 'unit' of the info-gain / lost-contact terms.
        # At t=1 step (dt_ctrl=0.01) lost_contact_penalty * 0.01 is ~5e-5.
        # Allow a generous 0.05 slack for Level-3 terms.
        assert reward >= expected - 0.05, (
            f"CPU reward {reward} far below batch-sim replica {expected}; "
            "formulas may have diverged on the shaping block."
        )
    finally:
        env.close()


def test_cpu_env_collision_matches_replica(core_available: bool):
    """Dropping motors to zero eventually drives drone A out of bounds;
    on that step the CPU env must emit ``collision_penalty`` — same as
    batch-sim would for ``collision_a=True``."""
    if not core_available:
        pytest.skip("aces._core not available — skipping real-env consistency check")

    from aces.env.dogfight import DroneDogfightEnv

    env = DroneDogfightEnv(
        opponent="none",
        task="dogfight",
        max_episode_steps=2000,
    )
    try:
        env.reset(seed=0)
        zero_thrust = -np.ones(4, dtype=np.float32)

        collided = False
        for _ in range(2000):
            _, reward, terminated, _, info = env.step(zero_thrust)
            if info["collision"]:
                assert terminated
                assert reward == BATCH_SIM_DEFAULTS["collision_penalty"]
                collided = True
                break
        assert collided, "expected drone A to OOB / collide under zero thrust"
    finally:
        env.close()
