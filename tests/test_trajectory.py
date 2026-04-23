"""Tests for aces.trajectory module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aces.env import Trajectory


# ---------------------------------------------------------------------------
# circle
# ---------------------------------------------------------------------------


def test_circle_returns_3d_point():
    pt = Trajectory.circle(
        center=[0.0, 0.0], radius=2.0, altitude=1.5, speed=1.0, t=0.0
    )
    assert pt.shape == (3,)
    assert pt[2] == pytest.approx(1.5)


def test_circle_stays_on_radius():
    center = [1.0, 2.0]
    radius = 3.0
    for t in np.linspace(0, 2 * math.pi, 50):
        pt = Trajectory.circle(
            center=center, radius=radius, altitude=1.5, speed=1.0, t=t
        )
        dist = math.hypot(pt[0] - center[0], pt[1] - center[1])
        assert dist == pytest.approx(radius, rel=1e-9)


def test_circle_moves_with_time():
    p0 = Trajectory.circle(
        center=[0.0, 0.0], radius=2.0, altitude=1.5, speed=1.0, t=0.0
    )
    p1 = Trajectory.circle(
        center=[0.0, 0.0], radius=2.0, altitude=1.5, speed=1.0, t=1.0
    )
    assert not np.allclose(p0, p1)


# ---------------------------------------------------------------------------
# lemniscate
# ---------------------------------------------------------------------------


def test_lemniscate_returns_3d():
    pt = Trajectory.lemniscate(
        center=[0.0, 0.0], scale=2.0, altitude=1.2, speed=1.0, t=0.0
    )
    assert pt.shape == (3,)
    assert pt[2] == pytest.approx(1.2)


def test_lemniscate_stays_near_center():
    center = [5.0, 5.0]
    scale = 2.0
    for t in np.linspace(0, 10, 200):
        pt = Trajectory.lemniscate(
            center=center, scale=scale, altitude=1.5, speed=1.0, t=t
        )
        assert pt[0] == pytest.approx(center[0], abs=scale + 1e-9)
        assert pt[1] == pytest.approx(center[1], abs=scale + 1e-9)


# ---------------------------------------------------------------------------
# patrol
# ---------------------------------------------------------------------------


def test_patrol_loops():
    """After traversing the full loop at speed=1 m/s, position should match t=0."""
    waypoints = [[0.0, 0.0, 1.5], [4.0, 0.0, 1.5], [4.0, 3.0, 1.5], [0.0, 3.0, 1.5]]
    speed = 1.0
    # Total perimeter: 4 + 3 + 5 + 3 = 15? No: 4+3+4+3=14 m (back to start not included).
    # Segments: 0->1: 4, 1->2: 3, 2->3: 4, 3->0: 3 → total 14
    total = 4.0 + 3.0 + 4.0 + 3.0
    p0 = Trajectory.patrol(waypoints=waypoints, speed=speed, t=0.0)
    p_loop = Trajectory.patrol(waypoints=waypoints, speed=speed, t=total)
    assert np.allclose(p0, p_loop, atol=1e-9)


def test_patrol_interpolates():
    """Midpoint of a 10m straight segment at speed=1 m/s should be at t=5."""
    waypoints = [[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]]
    pt = Trajectory.patrol(waypoints=waypoints, speed=1.0, t=5.0)
    assert pt[0] == pytest.approx(5.0, abs=1e-9)
    assert pt[1] == pytest.approx(0.0, abs=1e-9)
    assert pt[2] == pytest.approx(1.5, abs=1e-9)


# ---------------------------------------------------------------------------
# random_trajectory
# ---------------------------------------------------------------------------


def test_random_trajectory_returns_valid():
    """20 random trajectories should all produce valid 3D points within bounds."""
    bounds = [10.0, 10.0, 3.0]
    rng = np.random.default_rng(seed=0)

    for _ in range(20):
        traj_type, kwargs = Trajectory.random_trajectory(bounds=bounds, rng=rng)
        assert traj_type in ("circle", "lemniscate", "patrol")

        # Sample a few time steps and verify points are finite and within bounds
        for t in [0.0, 1.0, 5.0, 10.0]:
            pt = getattr(Trajectory, traj_type)(t=t, **kwargs)
            assert pt.shape == (3,), f"Expected 3D point, got shape {pt.shape}"
            assert np.all(np.isfinite(pt)), f"Non-finite point: {pt}"
            assert 0.0 <= pt[0] <= bounds[0], f"x={pt[0]} out of bounds"
            assert 0.0 <= pt[1] <= bounds[1], f"y={pt[1]} out of bounds"
            assert 0.0 <= pt[2] <= bounds[2], f"z={pt[2]} out of bounds"
