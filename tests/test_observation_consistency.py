"""Ensure CPU env observation matches batch-sim canonical build_observation.

ACES has two code paths that assemble the 21-dim observation vector:

* CPU path: :meth:`aces.env.dogfight.DroneDogfightEnv._build_obs` — Python
  code that reads :class:`aces._core.StepResult` fields and concatenates
  them into a numpy array.

* Batch / GPU path: :func:`aces_batch_sim::observation::build_observation`
  (Rust). Used by :class:`aces_batch_sim::orchestrator::BatchOrchestrator`
  and :class:`aces_batch_sim::gpu::orchestrator::GpuBatchOrchestrator`.
  Declared canonical per the 2026-04-23 audit.

If these diverge, a policy trained against one backend will see different
inputs on the other (silent failure — the vector dims match, just the
semantics drift). This test pins the layout by independently computing the
expected observation in pure Python and comparing it element-wise to what
the CPU env emits.

It complements :mod:`tests.test_action_normalization_consistency` and
:mod:`tests.test_reward_consistency`, which do the same job for the action
denormalization path and the reward formula respectively.

Skipped in CPU-only environments (needs ``aces._core`` for the CPU env).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aces.env.obs_layout import OBS_DIM, OBS_LAYOUT


def _quat_to_euler_zyx(
    w: float, x: float, y: float, z: float
) -> tuple[float, float, float]:
    """Replica of ``crates/batch-sim/src/observation.rs::quaternion_to_euler``.

    Returns (roll, pitch, yaw) for a unit quaternion ``(w, x, y, z)``.
    Matches nalgebra's ZYX intrinsic Euler convention used by sim-core.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _batch_sim_build_observation(
    self_state: list[float],
    opponent_state: list[float],
    arena_sdf_at_self: float,
    lock_progress: float,
    being_locked_progress: float,
    opponent_visible: bool,
) -> np.ndarray:
    """Pure-Python replica of ``build_observation`` in batch-sim.

    Matches ``crates/batch-sim/src/observation.rs`` line-for-line.
    Belief uncertainty and time-since-last-seen are hardcoded to 0.0 there
    (MPPI-vs-MPPI has no particle filter), so this replica also fixes them.
    """
    # self/opponent state layout from DroneState::to_array:
    #   [0:3]   position
    #   [3:6]   velocity (world)
    #   [6:10]  quaternion (w, x, y, z)
    #   [10:13] angular velocity (body)
    own_pos = np.asarray(self_state[0:3], dtype=np.float64)
    own_vel = np.asarray(self_state[3:6], dtype=np.float64)
    own_quat = self_state[6:10]
    own_angvel = np.asarray(self_state[10:13], dtype=np.float64)

    opp_pos = np.asarray(opponent_state[0:3], dtype=np.float64)
    opp_vel = np.asarray(opponent_state[3:6], dtype=np.float64)

    rel_pos = opp_pos - own_pos
    rel_vel = opp_vel - own_vel

    roll, pitch, yaw = _quat_to_euler_zyx(*own_quat)

    return np.array(
        [
            own_vel[0],
            own_vel[1],
            own_vel[2],
            own_angvel[0],
            own_angvel[1],
            own_angvel[2],
            rel_pos[0],
            rel_pos[1],
            rel_pos[2],
            rel_vel[0],
            rel_vel[1],
            rel_vel[2],
            roll,
            pitch,
            yaw,
            arena_sdf_at_self,
            lock_progress,
            being_locked_progress,
            1.0 if opponent_visible else 0.0,
            0.0,  # belief_uncertainty (MPPI-vs-MPPI)
            0.0,  # time_since_last_seen (MPPI-vs-MPPI)
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Layout doc sanity: the shared OBS_LAYOUT must span all 21 slots exactly.
# ---------------------------------------------------------------------------


def test_obs_layout_covers_21_slots():
    """Sanity: each of the 21 slots belongs to exactly one named field."""
    assert sum(f.length for f in OBS_LAYOUT) == OBS_DIM
    covered = [False] * OBS_DIM
    for f in OBS_LAYOUT:
        for i in range(f.start, f.end):
            assert not covered[i], f"slot {i} covered twice"
            covered[i] = True
    assert all(covered)


# ---------------------------------------------------------------------------
# Replica sanity: hand-check a canonical state.
# ---------------------------------------------------------------------------


def test_replica_identity_quaternion_euler_zero():
    """Identity quaternion (w=1, x=y=z=0) must produce Euler (0, 0, 0)."""
    roll, pitch, yaw = _quat_to_euler_zyx(1.0, 0.0, 0.0, 0.0)
    assert roll == pytest.approx(0.0, abs=1e-12)
    assert pitch == pytest.approx(0.0, abs=1e-12)
    assert yaw == pytest.approx(0.0, abs=1e-12)


def test_replica_canonical_state():
    """Known state -> known observation vector (hand-computed)."""
    self_state = [
        1.0,
        2.0,
        3.0,  # position
        0.5,
        -0.5,
        0.25,  # velocity (world)
        1.0,
        0.0,
        0.0,
        0.0,  # quaternion identity
        0.1,
        -0.2,
        0.3,  # angular velocity (body)
    ]
    opp_state = [
        4.0,
        6.0,
        3.0,  # position
        1.0,
        0.0,
        0.0,  # velocity
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    obs = _batch_sim_build_observation(
        self_state,
        opp_state,
        arena_sdf_at_self=1.23,
        lock_progress=0.4,
        being_locked_progress=0.1,
        opponent_visible=True,
    )

    expected = np.array(
        [
            0.5,
            -0.5,
            0.25,  # own velocity
            0.1,
            -0.2,
            0.3,  # own angular velocity
            3.0,
            4.0,
            0.0,  # rel_pos = opp - self
            0.5,
            0.5,
            -0.25,  # rel_vel = opp - self
            0.0,
            0.0,
            0.0,  # roll, pitch, yaw from identity quat
            1.23,  # SDF
            0.4,
            0.1,  # lock progress self, enemy
            1.0,  # visible
            0.0,
            0.0,  # belief var, time_since_seen
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(obs, expected, atol=1e-12)
    assert obs.shape == (OBS_DIM,)


# ---------------------------------------------------------------------------
# Real CPU env agreement: drive DroneDogfightEnv.reset + one step and
# confirm the observation it emits matches the batch-sim replica applied
# to the same simulation state.
# ---------------------------------------------------------------------------


def _extract_env_cpu_state(env) -> tuple[list[float], list[float]]:
    state_a = list(env._sim.drone_a_state())
    state_b = list(env._sim.drone_b_state())
    return state_a, state_b


def test_cpu_env_reset_obs_matches_batch_sim_replica(core_available: bool):
    """Observation returned by ``reset()`` must match the batch-sim layout.

    At reset drones start at hover with identity quaternion, so roll/pitch/yaw
    are exactly (0, 0, 0) and angular velocity is zero. Velocities are also
    zero. The replica is therefore a strict equality check for most slots.
    """
    if not core_available:
        pytest.skip("aces._core not available — skipping CPU-env consistency check")

    from aces.env.dogfight import DroneDogfightEnv

    env = DroneDogfightEnv(
        opponent="none",
        task="dogfight",
        max_episode_steps=4,
    )
    try:
        obs, _info = env.reset(seed=123)
        assert obs.shape == (OBS_DIM,)

        state_a, state_b = _extract_env_cpu_state(env)
        # For opponent="none" CPU env the step path uses the real opponent
        # state but sets visible=0. At reset, however, the env DOES use the
        # real opponent state (not zeroed) and sets visible via check_los —
        # match that here.
        sdf = float(env._sim.arena_sdf(list(state_a[:3])))
        visible = bool(env._sim.check_los(list(state_a[:3]), list(state_b[:3])))
        # task != "hover" and opponent_mode "none" DOES zero opponent state.
        # mirror that exactly.
        zero_opp = [0.0] * 13
        expected = _batch_sim_build_observation(
            state_a,
            zero_opp,
            arena_sdf_at_self=sdf,
            lock_progress=0.0,
            being_locked_progress=0.0,
            opponent_visible=False,  # opponent="none" initial_visible = 0
        )
        # obs is float32; compare with f32 tolerance.
        np.testing.assert_allclose(obs, expected, atol=1e-5, rtol=0)
        # Unused to silence linter in case visible tooling shifts.
        _ = visible
    finally:
        env.close()


def test_cpu_env_step_obs_matches_batch_sim_replica(core_available: bool):
    """One hover step: CPU env observation must match the batch-sim replica.

    The CPU env emits float32 while the replica is float64; tolerance is
    loosened to 1e-5 to absorb the quaternion->Euler round-trip. This test
    is the main regression guard against layout drift.
    """
    if not core_available:
        pytest.skip("aces._core not available — skipping CPU-env consistency check")

    from aces.env.dogfight import DroneDogfightEnv

    env = DroneDogfightEnv(
        opponent="none",
        task="dogfight",
        max_episode_steps=4,
    )
    try:
        env.reset(seed=42)

        hover_action = np.zeros(4, dtype=np.float32)
        obs, _reward, terminated, _truncated, info = env.step(hover_action)
        assert not terminated
        assert obs.shape == (OBS_DIM,)

        # Recompute the canonical observation from the post-step physics
        # state. CPU env with opponent="none" zeros the opponent slots.
        state_a, _state_b = _extract_env_cpu_state(env)
        expected = _batch_sim_build_observation(
            state_a,
            [0.0] * 13,  # zero opponent state (task=dogfight, opp=none)
            arena_sdf_at_self=info["nearest_obs_dist"],
            lock_progress=0.0,
            being_locked_progress=0.0,
            opponent_visible=False,
        )
        np.testing.assert_allclose(obs, expected, atol=1e-5, rtol=0)
    finally:
        env.close()


def test_cpu_env_step_obs_with_opponent_matches_replica(core_available: bool):
    """Dogfight with a real opponent: rel_pos/rel_vel slots must use
    (opponent - self), and opponent_visible must track LOS. This covers
    the non-trivial opponent case that ``opponent="none"`` skips."""
    if not core_available:
        pytest.skip("aces._core not available — skipping CPU-env consistency check")

    from aces.env.dogfight import DroneDogfightEnv

    env = DroneDogfightEnv(
        opponent="mppi",
        task="dogfight",
        max_episode_steps=4,
        # Force noise-free observations so the replica (which has no noise
        # model) stays equivalent to what the CPU env emits. The default
        # value is read from rules.toml [noise] (0.1), which would break
        # the assertion below.
        obs_noise_std=0.0,
    )
    try:
        env.reset(seed=7)
        hover_action = np.zeros(4, dtype=np.float32)
        obs, _r, terminated, _t, info = env.step(hover_action)
        assert not terminated

        state_a = list(env._sim.drone_a_state())
        state_b = list(env._sim.drone_b_state())

        # The CPU env feeds EKF estimates or belief means when obs_noise_std
        # is nonzero or visibility is lost; with default config and one
        # hover step starting from LOS, we expect obs_noise_std=0 AND
        # a_sees_b=True, so opp_for_obs == state_b. Guard that invariant
        # explicitly to keep the replica equivalence strict.
        assert env._obs_noise_std == 0.0, (
            "this test assumes default obs_noise_std=0; update if config changed"
        )
        # This test only verifies the visible-opponent path. If LOS is blocked
        # (e.g. pillar between default spawns (1,1,1.5) and (9,9,1.5) at (5,5)),
        # the CPU env feeds EKF belief instead of raw state_b and the replica
        # would not match — that's a separate code path not covered here.
        if not info["opponent_visible"]:
            pytest.skip("no LOS at this step; visible-opponent path not exercised")

        expected = _batch_sim_build_observation(
            state_a,
            state_b,
            arena_sdf_at_self=info["nearest_obs_dist"],
            lock_progress=info["lock_a_progress"],
            being_locked_progress=info["lock_b_progress"],
            opponent_visible=info["opponent_visible"],
        )
        # CPU env passes belief_uncertainty = belief_b_var_from_a and
        # time_since_last_seen = time_since_a_saw_b into slots [19] and [20].
        # These are 0.0 in batch-sim. Accept the CPU value for those slots
        # (since batch-sim intentionally zeroes them) by masking before
        # comparison — but still require slots [0..18] to match strictly.
        np.testing.assert_allclose(obs[:19], expected[:19], atol=1e-5, rtol=0)
        # Slots 19 and 20 are allowed to differ (known batch-sim limitation).
    finally:
        env.close()
