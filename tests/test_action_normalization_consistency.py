"""End-to-end consistency check between the CPU and GPU action-denorm paths.

ACES has two code paths that map SB3 actions in ``[-1, 1]^4`` to motor thrusts
in ``[0, max_thrust]^4``:

  * CPU path: :meth:`aces.env.dogfight.DroneDogfightEnv._map_action`
  * GPU path: :func:`aces.training.gpu_vec_env.denormalize_action`

A policy trained against one environment must behave identically when rolled
out in the other. The GPU implementation was originally a simple bilinear
``(action + 1) / 2 * max_thrust`` map while the CPU implementation has always
been the hover-centered ``hover + action * (max - hover)`` form. This test
pins both to the CPU convention and guards against future drift.

The pure-NumPy direction is exercised directly. The Rust-backed CPU env is
constructed through the ``core_available`` fixture from ``conftest.py`` —
when the Rust extension is not built, the real-env check skips cleanly and
only the pure-NumPy parity assertion runs.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from aces.training.gpu_vec_env import (
    HOVER_THRUST_PER_MOTOR,
    MAX_THRUST_PER_MOTOR,
    denormalize_action,
)


def _cpu_map_reference(action: np.ndarray) -> np.ndarray:
    """Pure-NumPy replica of ``DroneDogfightEnv._map_action``.

    Kept here rather than imported from ``aces.env.dogfight`` because the
    env module requires ``aces._core`` (the Rust extension) at import time.
    This stand-in mirrors the exact formula in that method so we can unit-test
    the math on Python-only machines.
    """
    hover = HOVER_THRUST_PER_MOTOR
    motors = hover + action.astype(np.float32) * (MAX_THRUST_PER_MOTOR - hover)
    clipped = np.clip(motors, 0.0, MAX_THRUST_PER_MOTOR).astype(np.float32)
    return cast(np.ndarray, clipped)


def test_cpu_and_gpu_agree_on_random_batch():
    """Random actions in [-1, 1] produce element-wise identical motor thrusts."""
    rng = np.random.default_rng(seed=0xACE5)
    actions = rng.uniform(-1.0, 1.0, size=(256, 4)).astype(np.float32)

    cpu = _cpu_map_reference(actions)
    gpu = denormalize_action(actions)

    np.testing.assert_allclose(cpu, gpu, atol=1e-6, rtol=0.0)


def test_cpu_and_gpu_agree_on_clamp_extremes():
    """Inputs outside [-1, 1] should clamp identically in both paths."""
    actions = np.array(
        [
            [-5.0, -1.0, 0.0, 1.0],
            [1.0, 5.0, -1.5, 0.25],
            [-0.9999, 0.9999, 0.5, -0.5],
        ],
        dtype=np.float32,
    )
    cpu = _cpu_map_reference(actions)
    gpu = denormalize_action(actions)
    np.testing.assert_allclose(cpu, gpu, atol=1e-6, rtol=0.0)
    # Both must respect the motor-thrust bounds.
    assert np.all(cpu >= 0.0) and np.all(cpu <= MAX_THRUST_PER_MOTOR)
    assert np.all(gpu >= 0.0) and np.all(gpu <= MAX_THRUST_PER_MOTOR)


def test_hover_is_zero_action():
    """action = 0 must correspond to hover thrust on both paths.

    This is the behavioural contract the CPU env was built around
    (hover-centered action) and it is the reason the GPU path must NOT use the
    old ``(action+1)/2 * max`` bilinear form, which would send action=0 to
    ``max/2`` instead of hover.
    """
    zero = np.zeros((1, 4), dtype=np.float32)
    gpu = denormalize_action(zero)
    np.testing.assert_allclose(gpu, HOVER_THRUST_PER_MOTOR, atol=1e-6)


def test_full_throttle_is_plus_one_action():
    ones = np.ones((1, 4), dtype=np.float32)
    gpu = denormalize_action(ones)
    np.testing.assert_allclose(gpu, MAX_THRUST_PER_MOTOR, atol=1e-6)


def test_cpu_env_map_action_matches_gpu_denormalize(core_available: bool):
    """Exercises the real CPU env's ``_map_action`` against the GPU denorm.

    Requires the Rust extension (``aces._core``). Skips gracefully if unbuilt.
    """
    if not core_available:
        pytest.skip("aces._core not available — skipping real-env consistency check")

    from aces.env.dogfight import DroneDogfightEnv

    # Minimal env: hover task, no opponent, no MPPI. Keeps construction light
    # but still exercises the real ``_map_action`` with the Rust-side hover
    # thrust plumbed in.
    env = DroneDogfightEnv(
        opponent="none",
        task="hover",
        max_episode_steps=1,
    )
    try:
        # Sanity: the env's cached hover thrust must equal our module constant
        # up to float rounding — otherwise the two paths silently diverge
        # whenever configs/drone.toml drifts.
        np.testing.assert_allclose(env._hover_thrust, HOVER_THRUST_PER_MOTOR, atol=1e-6)
        np.testing.assert_allclose(env._max_thrust, MAX_THRUST_PER_MOTOR, atol=1e-6)

        rng = np.random.default_rng(seed=0xC0FFEE)
        for _ in range(32):
            a = rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
            cpu = np.array(env._map_action(a), dtype=np.float32)
            gpu = denormalize_action(a)
            np.testing.assert_allclose(cpu, gpu, atol=1e-6, rtol=0.0)
    finally:
        # DroneDogfightEnv doesn't hold GPU resources — nothing to close, but
        # keep the try/finally for symmetry in case that changes.
        pass
