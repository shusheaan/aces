"""Tests for Rerun visualization (headless)."""

import rerun as rr


def test_visualizer_creation():
    from aces.viz import AcesVisualizer

    rr.init("test_viz", spawn=False)
    vis = AcesVisualizer.__new__(AcesVisualizer)
    vis.trail_a = []
    vis.trail_b = []
    vis.max_trail = 100
    assert vis is not None


def test_log_step_no_crash():
    """Verify logging doesn't crash (headless, no viewer)."""
    from aces.viz import AcesVisualizer

    rr.init("test_viz_step", spawn=False)
    vis = AcesVisualizer.__new__(AcesVisualizer)
    vis.trail_a = []
    vis.trail_b = []
    vis.max_trail = 100

    class FakeResult:
        drone_a_state = [1.0, 1.0, 1.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        drone_b_state = [9.0, 9.0, 1.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        drone_a_forward = [1.0, 0.0, 0.0]
        drone_b_forward = [-1.0, 0.0, 0.0]
        lock_a_progress = 0.0
        lock_b_progress = 0.0
        distance = 10.0

    vis.log_step(0, FakeResult())
    vis.log_step(1, FakeResult())
