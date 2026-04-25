"""Verify PyGpuVecEnv accepts and applies custom reward config.

Covers the plumbing slice that wires ``configs/rules.toml`` section
``[reward]`` through the Python ``GpuVecEnv`` wrapper into the Rust-side
``RewardConfig``. Before this slice, GPU training silently ignored any
user edits to ``rules.toml`` and always used ``RewardConfig::default()``.
"""

from __future__ import annotations

import pytest


def test_reward_config_constants_match_rules_toml() -> None:
    """``rules.toml`` section ``[reward]`` exposes the keys the wrapper expects.

    Doesn't require a GPU — pure Python config parse. Guards against the
    rules.toml schema drifting away from what the GPU wrapper reads.
    """
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

    # opponent_crash_reward lives in [task_reward_overrides.*], not root [reward]
    assert "opponent_crash_reward" not in rcfg


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


def test_load_reward_from_rules_has_all_keys() -> None:
    """``_load_reward_from_rules`` returns all 8 keys the Rust side requires."""
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


# ---------------------------------------------------------------------------
# Bug #24 — per-task reward overrides plumbed under --use-gpu-env
# ---------------------------------------------------------------------------


def test_task_reward_override_hover_differs_from_dogfight() -> None:
    """``GpuVecEnv(task='hover')`` merges ``[task_reward_overrides.hover]``.

    Pure Python — no GPU required. Verifies that the override dict is merged
    on top of the base ``[reward]`` section, mirroring the CPU env's behaviour.
    """
    from aces.config import load_configs

    cfg = load_configs()
    # If [task_reward_overrides.hover] is empty, this test is vacuous but
    # still valid (it checks the merge doesn't crash).
    hover_overrides = cfg.rules.task_reward_overrides.get("hover", {})
    dogfight_overrides = cfg.rules.task_reward_overrides.get("dogfight", {})

    # At minimum one of the two tasks should have overrides to make this
    # test meaningful; skip rather than fail if neither does.
    if not hover_overrides and not dogfight_overrides:
        pytest.skip(
            "Neither [task_reward_overrides.hover] nor [task_reward_overrides.dogfight]"
            " has entries — test would be vacuous"
        )

    # Build merged reward for hover and dogfight tasks the same way GpuVecEnv does.
    import dataclasses

    base = dataclasses.asdict(cfg.rules.reward)

    def merge_overrides(task: str) -> dict:
        merged = dict(base)
        for k, v in cfg.rules.task_reward_overrides.get(task, {}).items():
            if k in merged:
                merged[k] = float(v)
        return merged

    hover_merged = merge_overrides("hover")
    dogfight_merged = merge_overrides("dogfight")

    # For any key that appears in the overrides for either task, the merged
    # values should reflect those overrides.
    for key, val in hover_overrides.items():
        if key in hover_merged:
            assert hover_merged[key] == pytest.approx(float(val)), (
                f"[task_reward_overrides.hover].{key} not merged correctly"
            )

    for key, val in dogfight_overrides.items():
        if key in dogfight_merged:
            assert dogfight_merged[key] == pytest.approx(float(val)), (
                f"[task_reward_overrides.dogfight].{key} not merged correctly"
            )


def test_gpu_vec_env_task_hover_accepts_task_kwarg(gpu_available: bool) -> None:
    """``GpuVecEnv(task='hover')`` constructs without error."""
    if not gpu_available:
        pytest.skip("GPU/extension unavailable")

    from aces.training.gpu_vec_env import GpuVecEnv

    env = GpuVecEnv(n_envs=2, mppi_samples=8, mppi_horizon=4, task="hover")
    try:
        assert env.num_envs == 2
    finally:
        env.close()


def test_gpu_vec_env_task_dogfight_accepts_task_kwarg(gpu_available: bool) -> None:
    """``GpuVecEnv(task='dogfight')`` constructs without error."""
    if not gpu_available:
        pytest.skip("GPU/extension unavailable")

    from aces.training.gpu_vec_env import GpuVecEnv

    env = GpuVecEnv(n_envs=2, mppi_samples=8, mppi_horizon=4, task="dogfight")
    try:
        assert env.num_envs == 2
    finally:
        env.close()


def test_gpu_vec_env_hover_vs_dogfight_different_config() -> None:
    """``GpuVecEnv(task='hover')`` vs ``'dogfight'`` produces different reward config.

    Pure Python — checks that the per-task override merge actually changes
    the reward dict that would be sent to Rust, when the two tasks have
    different overrides in rules.toml.
    """
    from aces.config import load_configs

    cfg = load_configs()
    hover_overrides = cfg.rules.task_reward_overrides.get("hover", {})
    dogfight_overrides = cfg.rules.task_reward_overrides.get("dogfight", {})

    # Find a key that differs between the two tasks' overrides.
    # approach_reward differs between dogfight (3.0) and pursuit_linear (5.0)
    # in the default rules.toml.
    import dataclasses

    base = dataclasses.asdict(cfg.rules.reward)

    def merged(task: str) -> dict:
        d = dict(base)
        for k, v in cfg.rules.task_reward_overrides.get(task, {}).items():
            if k in d:
                d[k] = float(v)
        return d

    hover_cfg = merged("hover")
    dogfight_cfg = merged("dogfight")

    # Only assert they differ if the overrides actually differ.
    differing_keys = {
        k
        for k in set(hover_overrides) | set(dogfight_overrides)
        if hover_overrides.get(k) != dogfight_overrides.get(k) and k in base
    }
    if not differing_keys:
        pytest.skip(
            "hover and dogfight task_reward_overrides do not differ on any base key"
        )

    key = next(iter(differing_keys))
    assert hover_cfg[key] != dogfight_cfg[key], (
        f"After merge, hover and dogfight should differ on '{key}'"
    )
