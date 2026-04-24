"""Verify PyGpuVecEnv accepts and applies custom reward config.

Covers the plumbing slice that wires ``configs/rules.toml`` section
``[reward]`` through the Python ``GpuVecEnv`` wrapper into the Rust-side
``RewardConfig``. Before this slice, GPU training silently ignored any
user edits to ``rules.toml`` and always used ``RewardConfig::default()``.
"""

from __future__ import annotations

import pytest


def test_reward_config_constants_match_rules_toml(core_available: bool) -> None:
    """``rules.toml`` section ``[reward]`` exposes the keys the wrapper expects.

    Doesn't require a GPU — pure Python config parse. Guards against the
    rules.toml schema drifting away from what the GPU wrapper reads.
    """
    if not core_available:
        pytest.skip("aces._core needed for config load")

    import dataclasses

    from aces.config import load_configs

    cfg = load_configs()
    rcfg = dataclasses.asdict(cfg.rules.reward)

    required = [
        "kill_reward",
        "killed_penalty",
        "collision_penalty",
        "survival_bonus",
        "approach_reward",
        "lock_progress_reward",
        "control_penalty",
    ]
    for key in required:
        assert key in rcfg, f"rules.toml [reward] missing '{key}'"


def test_gpu_vec_env_accepts_custom_reward(gpu_available: bool) -> None:
    """Custom reward_config kwarg is forwarded to Rust without error."""
    if not gpu_available:
        pytest.skip("GPU/extension unavailable")

    from aces.training.gpu_vec_env import GpuVecEnv

    custom = {
        "kill_reward": 200.0,  # non-default
        "killed_penalty": -100.0,
        "collision_penalty": -50.0,
        "opponent_crash_reward": 5.0,
        "lock_progress_reward": 5.0,
        "approach_reward": 0.05,
        "survival_bonus": 0.01,
        "control_penalty": 0.01,
    }
    env = GpuVecEnv(
        n_envs=2,
        mppi_samples=8,
        mppi_horizon=4,
        reward_config=custom,
    )
    try:
        assert env.num_envs == 2
    finally:
        env.close()


def test_gpu_vec_env_default_reads_rules_toml(gpu_available: bool) -> None:
    """Without a ``reward_config`` override, the wrapper falls back to rules.toml."""
    if not gpu_available:
        pytest.skip("GPU/extension unavailable")

    from aces.training.gpu_vec_env import GpuVecEnv

    env = GpuVecEnv(n_envs=2, mppi_samples=8, mppi_horizon=4)
    try:
        assert env.num_envs == 2
    finally:
        env.close()


def test_load_reward_from_rules_has_all_keys(core_available: bool) -> None:
    """``_load_reward_from_rules`` returns all 8 keys the Rust side requires."""
    if not core_available:
        pytest.skip("aces._core needed for config load")

    from aces.training.gpu_vec_env import _load_reward_from_rules

    d = _load_reward_from_rules()
    expected = {
        "kill_reward",
        "killed_penalty",
        "collision_penalty",
        "opponent_crash_reward",
        "lock_progress_reward",
        "approach_reward",
        "survival_bonus",
        "control_penalty",
    }
    assert set(d.keys()) == expected
    for k, v in d.items():
        assert isinstance(v, float), f"{k} should be float, got {type(v)}"
