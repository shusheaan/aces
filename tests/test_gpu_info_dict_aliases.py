"""
Tests that the GPU VecEnv info dict contains CPU-env-compatible alias keys.
See docs/architecture.md §11.25.
"""

import numpy as np
import pytest


@pytest.fixture
def gpu_available():
    try:
        import aces._core as _core  # noqa: F401

        return hasattr(_core, "GpuVecEnv")
    except ImportError:
        return False


def test_gpu_info_dict_aliases(gpu_available):
    if not gpu_available:
        pytest.skip("GPU feature not available")

    import aces._core as _core

    env = _core.GpuVecEnv(n_envs=2, mppi_samples=8, mppi_horizon=4)
    env.reset()

    # Hover actions: thrust ~= hover_thrust, zero torques
    # 4 controls per drone: [thrust, tx, ty, tz] normalized; use 0.5 for thrust
    actions = np.zeros((2, 4), dtype=np.float32)
    actions[:, 0] = 0.5  # hover-ish thrust

    _obs, _rewards, _dones, infos = env.step(actions)

    info = infos[0]

    # Verify alias keys exist
    assert "collision" in info, "missing CPU-compat alias 'collision'"
    assert "truncated" in info, "missing CPU-compat alias 'truncated'"
    assert "lock_a_progress" in info, "missing CPU-compat alias 'lock_a_progress'"

    # Verify alias values match the granular keys they shadow
    assert info["collision"] == info["collision_a"], (
        f"collision alias mismatch: {info['collision']} != {info['collision_a']}"
    )
    assert info["truncated"] == info["timeout"], (
        f"truncated alias mismatch: {info['truncated']} != {info['timeout']}"
    )
    assert info["lock_a_progress"] == info["lock_a"], (
        f"lock_a_progress alias mismatch: {info['lock_a_progress']} != {info['lock_a']}"
    )
